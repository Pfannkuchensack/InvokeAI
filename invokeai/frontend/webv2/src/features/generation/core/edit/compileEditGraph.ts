import type { GenerationProjectSettings } from '@features/generation/core/contracts';
import type { FamilyPipelineSettings } from '@features/generation/core/familyGraph';
import type {
  CompiledGenerateGraph,
  GenerateReferenceImage,
  ImageWithDims,
  MainModelConfig,
} from '@features/generation/core/types';

import {
  createReferenceImageId,
  getDefaultGenerateSettings,
  getGenerationDimensions,
  getGenerationValidationReasons,
  getInstructionEditSupport,
} from '@features/generation/core/baseGenerationPolicies';
import { addFluxKontextImages, buildFlux2Pipeline, getFlux2PipelineMetadata } from '@features/generation/core/graph';
import { addMetadata, addOutputPostprocessors, toGraphContract } from '@features/generation/core/graphBuilder';
import { calculateNewSize } from '@features/generation/core/settings';

/** The settings an edit step reads; prompt and seed enter at submission through the graph's `positive_prompt` and `seed` nodes. */
export type EditGraphSettings = FamilyPipelineSettings;

/** Every field the edit graph reads; the record type fails to compile until a new pipeline setting is placed here. */
const EDIT_SETTINGS = {
  cfgRescaleMultiplier: true,
  cfgScale: true,
  clipEmbedModel: true,
  clipSkip: true,
  componentSourceModel: true,
  loras: true,
  mistralEncoderModel: true,
  qwen35EncoderModel: true,
  qwen3EncoderModel: true,
  qwen3VLEncoderModel: true,
  qwenVLEncoderModel: true,
  scheduler: true,
  steps: true,
  t5EncoderModel: true,
  vae: true,
  vaePrecision: true,
} as const satisfies Record<keyof EditGraphSettings, true>;

/** Only what the edit graph reads, so a wider object (Generate's settings) never reaches the validator or the graph. */
const pickEditSettings = (settings: EditGraphSettings): EditGraphSettings =>
  Object.fromEntries(
    Object.keys(EDIT_SETTINGS).map((key) => [key, settings[key as keyof EditGraphSettings]])
  ) as unknown as EditGraphSettings;

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

/**
 * The instruction-edit graph: the family's text-to-image pipeline with the source and references as Kontext
 * conditioning, started from noise. The variants of one step are submission batches over this single graph. Results are
 * saved images (never intermediates), which the edit tree refers to by name; the board is the queue item's.
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

  if (model.base !== 'flux2' || !support) {
    throw new Error(`${model.name} cannot edit images from instructions.`);
  }

  if (support.img2imgStart) {
    throw new Error(`${model.name} starts editing from the source latent, which Edit does not support yet.`);
  }

  const settings = pickEditSettings(given);
  // What Generate would run: the same rules for the size, the references and the components the loader needs.
  const images = [source, ...referenceImages];
  const refs: GenerateReferenceImage[] = images.map((image) => ({
    config: { image: { original: { image } }, type: 'flux2_reference_image' },
    id: createReferenceImageId(),
    isEnabled: true,
  }));
  const [invalid] = getGenerationValidationReasons(model, {
    ...getDefaultGenerateSettings(model),
    ...settings,
    ...size,
    referenceImages: refs,
  });

  if (invalid) {
    throw new Error(invalid);
  }

  const runtimeProjectSettings = { ...projectSettings, randDevice };
  const pipeline = buildFlux2Pipeline({
    denoiseSize: size,
    denoisingStart: 0,
    model,
    multiDiffusion: null,
    output: { id: 'canvas_output', isIntermediate: false },
    settings,
    useCpuNoise: projectSettings.useCpuNoise,
    vaeTileSize: null,
  });
  const { denoise, graph, output } = pipeline;

  addFluxKontextImages(
    graph,
    images.map((image) => image.image_name),
    denoise,
    'flux2_reference_image'
  );
  addMetadata(
    graph,
    output,
    { ...settings, ...size, seamlessXAxis: false, seamlessYAxis: false },
    model,
    'flux2_edit',
    runtimeProjectSettings,
    {
      ...getFlux2PipelineMetadata(model, settings, pipeline),
      edit_source_image_name: source.image_name,
      // The source first, as the model reads it, so Generate recalls the run from these alone.
      ref_images: refs,
    }
  );
  addOutputPostprocessors(graph, runtimeProjectSettings, false);

  return {
    backendGraph: graph,
    graph: toGraphContract(graph, `${model.name} edit`),
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};

/**
 * The size an edit result takes from a source image: its aspect ratio at the model's ideal area on the model's grid,
 * whether the source is larger or smaller.
 */
export const fitToOptimalArea = (
  source: { width: number; height: number },
  model: Pick<MainModelConfig, 'base' | 'type' | 'variant'>
): { width: number; height: number } => {
  const { grid, optimal } = getGenerationDimensions(model);

  return calculateNewSize(source.width / source.height, optimal * optimal, grid);
};
