import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
} from '@features/generation/core/contracts';
import type { FamilyPipelineInput } from '@features/generation/core/familyGraph';
import type { CompiledGenerateGraph, ImageWithDims, MainModelConfig } from '@features/generation/core/types';

import { type EditBase, getInstructionEditSupport } from '@features/generation/core/baseGenerationPolicies';
import {
  addFluxKontextImages,
  addFlux3ReferenceImages,
  addQwenImage21ReferenceImages,
  buildFlux2Pipeline,
  buildFlux3Pipeline,
  buildQwenImage21Pipeline,
  getFlux2PipelineMetadata,
} from '@features/generation/core/graph';
import { addEdge, addMetadata, addOutputPostprocessors, toGraphContract } from '@features/generation/core/graphBuilder';

import {
  EDIT_FAMILY_INFO,
  type EditGraphSettings,
  getEditReferenceImages,
  getEditValidationIssues,
  pickEditSettings,
} from './editContract';

export { type EditGraphSettings } from './editContract';

export interface EditGraphInput {
  model: MainModelConfig;
  settings: EditGraphSettings;
  /** The image being edited; the model reads it as reference image 1. */
  source: ImageWithDims;
  /** Further references in order, after the source. */
  referenceImages: readonly ImageWithDims[];
  /** The size of the result, on the model's grid. */
  size: { width: number; height: number };
  projectSettings: GenerationProjectSettings;
  randDevice?: string;
}

interface EditFamilyInput {
  model: MainModelConfig;
  settings: EditGraphSettings;
  size: { width: number; height: number };
  /** The source first, then the references, as the model reads them. */
  images: readonly ImageWithDims[];
  useCpuNoise: boolean;
}

/** A family's graph before its metadata record is added. */
interface EditFamilyGraph {
  graph: BackendGraphContract;
  /** The decode node whose image is the result. */
  output: BackendInvocationContract;
  /** `output`, or an upstream node that passes the record on to it (Qwen-Image 2.1's denoise adds its own keys). */
  metadataNode: BackendInvocationContract;
  /** The family's keys of the metadata record. */
  metadata: Record<string, unknown>;
}

interface EditFamily {
  /** The family's text-to-image pipeline with the references added, started from noise. */
  build: (input: EditFamilyInput) => EditFamilyGraph;
}

const toPipelineInput = ({ model, settings, size, useCpuNoise }: EditFamilyInput): FamilyPipelineInput => ({
  denoiseSize: size,
  denoisingStart: 0,
  model,
  multiDiffusion: null,
  output: { id: 'canvas_output', isIntermediate: false },
  settings,
  useCpuNoise,
  vaeTileSize: null,
});

/** An entry per base in `EDIT_BASES`; a base added there fails to compile until it has its family here. */
const EDIT_FAMILIES: Record<EditBase, EditFamily> = {
  flux2: {
    build: (input) => {
      const pipeline = buildFlux2Pipeline(toPipelineInput(input));

      addFluxKontextImages(
        pipeline.graph,
        input.images.map((image) => image.image_name),
        pipeline.denoise,
        'flux2_reference_image'
      );

      return {
        graph: pipeline.graph,
        metadata: getFlux2PipelineMetadata(input.model, input.settings, pipeline),
        metadataNode: pipeline.output,
        output: pipeline.output,
      };
    },
  },
  flux3: {
    build: (input) => {
      const pipeline = buildFlux3Pipeline(toPipelineInput(input));
      const { components, graph, output, vaeModel } = pipeline;

      addFlux3ReferenceImages(
        graph,
        { ...pipeline, upsamplerModel: null },
        input.images.map((image) => image.image_name)
      );

      return {
        graph,
        metadata: {
          // FLUX.3 offers no scheduler: its denoise node always runs the reference Euler sampler.
          scheduler: 'euler',
          qwen3_vl_encoder: components.qwen3VlEncoderModel ?? undefined,
          vae: vaeModel ?? undefined,
        },
        metadataNode: output,
        output,
      };
    },
  },
  'qwen-image-2-1': {
    build: (input) => {
      const pipeline = buildQwenImage21Pipeline(toPipelineInput(input), {
        visionTower: input.settings.qwen3VLVisionTowerModel,
      });
      const { components, conditionings, denoise, graph, modelLoader, output, vaeModel } = pipeline;

      addQwenImage21ReferenceImages(
        graph,
        { conditionings, denoise, modelLoader },
        input.images.map((image) => ({ image_name: image.image_name }))
      );

      return {
        graph,
        metadata: {
          qwen3_vl_encoder: components.qwen3VlEncoderModel ?? undefined,
          qwen3_vl_vision_tower: components.qwen3VlVisionTowerModel ?? undefined,
          vae: vaeModel ?? undefined,
        },
        // The denoise adds its prefix-cache mode to the record, so metadata passes through it.
        metadataNode: denoise,
        output,
      };
    },
  },
};

/**
 * The instruction-edit graph: the family's text-to-image pipeline with the source and references as the family's
 * reference conditioning, started from noise. The variants of one step are submission batches over this single graph.
 * Results are saved images (never intermediates), which the edit tree refers to by name; the board is the queue item's.
 */
export const compileEditGraph = ({
  model,
  projectSettings,
  randDevice = projectSettings.useCpuNoise ? 'cpu' : 'cuda',
  referenceImages,
  settings: given,
  size,
  source,
}: EditGraphInput): CompiledGenerateGraph => {
  const support = getInstructionEditSupport(model);

  if (!support) {
    throw new Error(`${model.name} cannot edit images from instructions.`);
  }

  if (support.img2imgStart) {
    throw new Error(`${model.name} starts editing from the source latent, which Edit does not support yet.`);
  }

  const family = EDIT_FAMILIES[model.base as EditBase];
  const { mode } = EDIT_FAMILY_INFO[model.base as EditBase];
  const settings = pickEditSettings(given);
  const images = [source, ...referenceImages];
  const [invalid] = getEditValidationIssues({ images, model, settings, size });

  if (invalid) {
    throw new Error(invalid.message);
  }

  const runtimeProjectSettings = { ...projectSettings, randDevice };
  const { graph, metadata, metadataNode, output } = family.build({
    images,
    model,
    settings,
    size,
    useCpuNoise: projectSettings.useCpuNoise,
  });

  addMetadata(
    graph,
    metadataNode,
    { ...settings, ...size, seamlessXAxis: false, seamlessYAxis: false },
    model,
    mode,
    runtimeProjectSettings,
    {
      ...metadata,
      edit_source_image_name: source.image_name,
      // The source first, as the model reads it, so Generate recalls the run from these alone.
      ref_images: getEditReferenceImages(model.base as EditBase, images),
    }
  );

  if (metadataNode !== output) {
    addEdge(graph, metadataNode, 'metadata', output, 'metadata');
  }

  addOutputPostprocessors(graph, runtimeProjectSettings, false);

  return {
    backendGraph: graph,
    graph: toGraphContract(graph, `${model.name} edit`),
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};

export { fitToOptimalArea } from './editContract';
