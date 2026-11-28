import type { BackendGraphContract } from '@features/generation/core/contracts';
import type { ModelShape } from '@features/generation/core/graphCoverage.testing';
import type { ImageWithDims } from '@features/generation/core/types';

import { getGenerationDimensions } from '@features/generation/core/baseGenerationPolicies';
import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { compileEditGraph, fitToOptimalArea } from './compileEditGraph';

seedArchitectureCapabilities();

const DEV: ModelShape = { label: 'dev', overrides: { format: 'diffusers', variant: 'dev' } };
const KLEIN: ModelShape = { label: 'klein', overrides: { format: 'gguf_quantized', variant: 'klein_9b' } };
const PROJECT = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };

const reference = (name: string): ImageWithDims => ({ height: 640, image_name: name, width: 480 });

const compile = (
  shape: ModelShape,
  referenceImages: ImageWithDims[] = [],
  projectSettings = PROJECT
): BackendGraphContract => {
  const { model, settings } = satisfiedSettingsFor('flux2', shape);

  return compileEditGraph({
    model,
    projectSettings,
    referenceImages,
    settings,
    size: { height: 768, width: 1024 },
    source: { height: 512, image_name: 'source.png', width: 512 },
  }).backendGraph;
};

/** The image names the denoise receives, first conditioning first, read back along the collector chain. */
const conditionedImages = (graph: BackendGraphContract): string[] => {
  const nodeFor = (id: string) => graph.nodes[id]!;
  const edgeInto = (nodeId: string, field: string) =>
    graph.edges.find((edge) => edge.destination.node_id === nodeId && edge.destination.field === field);
  const names: string[] = [];
  let collect = edgeInto('denoise_latents', 'kontext_conditioning')?.source.node_id;

  while (collect) {
    const item = edgeInto(collect, 'item')!;
    names.unshift((nodeFor(item.source.node_id).image as { image_name: string }).image_name);
    collect = edgeInto(collect, 'collection')?.source.node_id;
  }

  return names;
};

const coreMetadata = (graph: BackendGraphContract) =>
  Object.values(graph.nodes).find((node) => node.type === 'core_metadata')!;

describe('compileEditGraph', () => {
  it('gives the model the source first, then the references in order', () => {
    const graph = compile(DEV, [reference('b.png'), reference('a.png')]);

    expect(conditionedImages(graph)).toEqual(['source.png', 'b.png', 'a.png']);
  });

  it('edits with the source alone', () => {
    expect(conditionedImages(compile(DEV))).toEqual(['source.png']);
  });

  it('takes the source and four references and refuses a sixth image', () => {
    const four = ['1', '2', '3', '4'].map((name) => reference(`${name}.png`));

    expect(conditionedImages(compile(DEV, four))).toHaveLength(5);
    expect(() => compile(DEV, [...four, reference('5.png')])).toThrow(/at most 5 reference images/);
  });

  it('starts from noise: no source latent is encoded', () => {
    const types = Object.values(compile(DEV, [reference('a.png')]).nodes).map((node) => node.type);

    expect(types).not.toContain('flux2_vae_encode');
    expect(types).toContain('flux2_vae_decode');
  });

  it('records the edit mode, the source and the references the way Generate records references', () => {
    const metadata = coreMetadata(compile(DEV, [reference('a.png')]));

    expect(metadata.generation_mode).toBe('flux2_edit');
    expect(metadata.edit_source_image_name).toBe('source.png');
    expect(metadata.width).toBe(1024);
    expect(metadata.height).toBe(768);
    expect(metadata.ref_images).toMatchObject([
      {
        config: { image: { original: { image: { height: 512, image_name: 'source.png', width: 512 } } } },
        isEnabled: true,
      },
      {
        config: {
          image: { original: { image: { height: 640, image_name: 'a.png', width: 480 } } },
          type: 'flux2_reference_image',
        },
        isEnabled: true,
      },
    ]);
  });

  it('saves the result as a gallery image, also behind the output filters', () => {
    const plain = compile(DEV);
    const filtered = compile(DEV, [], { ...PROJECT, useNsfwChecker: true, useWatermark: true });

    expect(plain.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'flux2_vae_decode' });
    expect(filtered.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'img_watermark' });
    expect(Object.values(filtered.nodes).filter((node) => node.type === 'flux2_vae_decode')).toMatchObject([
      { is_intermediate: true },
    ]);
  });

  it('runs a Klein model by the Generate rules: no negative conditioning, guidance 1 on the denoise', () => {
    const graph = compile(KLEIN, [reference('a.png')]);

    expect(conditionedImages(graph)).toEqual(['source.png', 'a.png']);
    expect(graph.nodes.denoise_latents).toMatchObject({ cfg_scale: 1, type: 'flux2_denoise' });
    expect(Object.values(graph.nodes).some((node) => node.type === 'flux2_klein_text_encoder')).toBe(true);
    expect(coreMetadata(graph).generation_mode).toBe('flux2_edit');
  });

  it('refuses what Generate would refuse: a size off the grid, a missing component', () => {
    const { model, settings } = satisfiedSettingsFor('flux2', DEV);
    const run = (change: object) =>
      compileEditGraph({
        model,
        projectSettings: PROJECT,
        referenceImages: [],
        settings: { ...settings, ...change },
        size: { height: 1024, width: 1000 },
        source: { height: 512, image_name: 'source.png', width: 512 },
      });

    expect(() => run({})).toThrow(/multiple of 16/);
    expect(() => run({ mistralEncoderModel: null, componentSourceModel: null, vae: null })).toThrow();
  });

  it('ignores Generate-only settings of a wider object: PiD does not constrain the edit size', () => {
    const { model, settings } = satisfiedSettingsFor('flux2', DEV);
    const wider = { ...settings, pidMode: 'native' as const, pidSteps: 0 };
    const graph = compileEditGraph({
      model,
      projectSettings: PROJECT,
      referenceImages: [],
      settings: wider,
      size: { height: 880, width: 1184 },
      source: { height: 512, image_name: 'source.png', width: 512 },
    }).backendGraph;

    expect(graph.nodes.denoise_latents).toMatchObject({ height: 880, width: 1184 });
    expect(Object.values(graph.nodes).some((node) => node.type.includes('pid'))).toBe(false);
  });

  it('rejects a base that cannot edit from instructions', () => {
    const { settings } = satisfiedSettingsFor('flux2', DEV);
    const { model: sdxl } = satisfiedSettingsFor('sdxl', { label: 'sdxl', overrides: { format: 'diffusers' } });

    expect(() =>
      compileEditGraph({
        model: sdxl,
        projectSettings: PROJECT,
        referenceImages: [],
        settings,
        size: { height: 1024, width: 1024 },
        source: { height: 512, image_name: 'source.png', width: 512 },
      })
    ).toThrow(/cannot edit images from instructions/);
  });
});

const QWEN21_GGUF: ModelShape = { ggufEncoder: true, label: 'gguf', overrides: { format: 'gguf_quantized' } };
const DIFFUSERS: ModelShape = { label: 'diffusers', overrides: { format: 'diffusers' } };
type GraphNode = BackendGraphContract['nodes'][string];

const compileBase = (
  base: 'flux3' | 'qwen-image-2-1',
  shape: ModelShape,
  referenceImages: ImageWithDims[] = [],
  patch: Record<string, unknown> = {}
) => {
  const { model, settings } = satisfiedSettingsFor(base, shape);

  return compileEditGraph({
    model,
    projectSettings: PROJECT,
    referenceImages,
    settings: { ...settings, ...patch },
    size: { height: 1024, width: 1024 },
    source: { height: 512, image_name: 'source.png', width: 512 },
  }).backendGraph;
};

const edgeInto = (graph: BackendGraphContract, nodeId: string, field: string) =>
  graph.edges.find((edge) => edge.destination.node_id === nodeId && edge.destination.field === field);

/** The item sources of a collect chain, first item first. */
const chainItems = (graph: BackendGraphContract, collectId: string | undefined): GraphNode[] => {
  const items: GraphNode[] = [];
  let collect = collectId;

  while (collect) {
    items.unshift(graph.nodes[edgeInto(graph, collect, 'item')!.source.node_id]!);
    collect = edgeInto(graph, collect, 'collection')?.source.node_id;
  }

  return items;
};

const refMetadata = (graph: BackendGraphContract) =>
  (
    coreMetadata(graph).ref_images as {
      config: { image: { original: { image: { image_name: string } } }; type: string };
    }[]
  ).map((ref) => ({ name: ref.config.image.original.image.image_name, type: ref.config.type }));

describe('compileEditGraph for Qwen-Image 2.1', () => {
  const refs = [reference('b.png'), reference('a.png')];

  it('shows the encoder every image and gives the denoise their latents, the source first', () => {
    const graph = compileBase('qwen-image-2-1', DIFFUSERS, refs);
    const shown = chainItems(graph, edgeInto(graph, 'pos_cond', 'reference_images')?.source.node_id);
    const latents = chainItems(graph, edgeInto(graph, 'denoise_latents', 'reference_latents')?.source.node_id);

    expect(shown.map((node) => (node.image as { image_name: string }).image_name)).toEqual([
      'source.png',
      'b.png',
      'a.png',
    ]);
    expect(latents).toHaveLength(3);
    expect(graph.nodes.denoise_latents).toMatchObject({ height: 1024, type: 'qwen_image_2_1_denoise', width: 1024 });
  });

  it('records the mode on the denoise record the decode node carries, and the references as Generate does', () => {
    const graph = compileBase('qwen-image-2-1', DIFFUSERS, refs);

    expect(coreMetadata(graph)).toMatchObject({
      edit_source_image_name: 'source.png',
      generation_mode: 'qwen_image_2_1_edit',
    });
    expect(edgeInto(graph, 'denoise_latents', 'metadata')?.source.node_id).toBe(coreMetadata(graph).id);
    expect(edgeInto(graph, 'canvas_output', 'metadata')?.source.node_id).toBe('denoise_latents');
    expect(graph.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'qwen_image_2_1_l2i' });
    expect(refMetadata(graph)).toEqual([
      { name: 'source.png', type: 'qwen_image_2_1_reference_image' },
      { name: 'b.png', type: 'qwen_image_2_1_reference_image' },
      { name: 'a.png', type: 'qwen_image_2_1_reference_image' },
    ]);
  });

  it('takes the source and three references and refuses a fifth image', () => {
    const three = ['1', '2', '3'].map((name) => reference(`${name}.png`));

    expect(() => compileBase('qwen-image-2-1', DIFFUSERS, three)).not.toThrow();
    expect(() => compileBase('qwen-image-2-1', DIFFUSERS, [...three, reference('4.png')])).toThrow(
      /at most 4 reference images/
    );
  });

  it("needs the GGUF encoder's vision tower, and loads it when chosen", () => {
    const withTower = compileBase('qwen-image-2-1', QWEN21_GGUF);

    expect(withTower.nodes.model_loader?.qwen3_vl_vision_tower_model).toBeTruthy();
    expect(coreMetadata(withTower).qwen3_vl_vision_tower).toBeTruthy();
    expect(() => compileBase('qwen-image-2-1', QWEN21_GGUF, [], { qwen3VLVisionTowerModel: null })).toThrow(
      /Vision Tower/
    );
  });

  it('refuses a size off the 32 grid', () => {
    const { model, settings } = satisfiedSettingsFor('qwen-image-2-1', DIFFUSERS);

    expect(() =>
      compileEditGraph({
        model,
        projectSettings: PROJECT,
        referenceImages: [],
        settings,
        size: { height: 1024, width: 1040 },
        source: { height: 512, image_name: 'source.png', width: 512 },
      })
    ).toThrow(/multiple of 32/);
  });
});

describe('compileEditGraph for FLUX.3', () => {
  const KLEIN: ModelShape = { label: 'klein', overrides: { format: 'diffusers', variant: 'flux3_klein' } };
  const BASE: ModelShape = { label: 'base', overrides: { format: 'diffusers', variant: 'flux3_klein_base' } };

  it('gives the denoise the source first, then the references in order, and marks the boxes as edit targets', () => {
    const graph = compileBase('flux3', KLEIN, [reference('b.png'), reference('a.png')]);

    expect(conditionedImages(graph)).toEqual(['source.png', 'b.png', 'a.png']);
    expect(graph.nodes.layout_prompt).toMatchObject({ edit_targets: true, type: 'flux3_layout_prompt' });
    expect(Object.values(graph.nodes).some((node) => node.type === 'flux3_prompt_upsampler')).toBe(false);
  });

  it('records the mode, the source and the references with the FLUX.2 reference type Generate uses', () => {
    const graph = compileBase('flux3', KLEIN, [reference('a.png')]);

    expect(coreMetadata(graph)).toMatchObject({
      edit_source_image_name: 'source.png',
      generation_mode: 'flux3_edit',
      scheduler: 'euler',
    });
    expect(refMetadata(graph)).toEqual([
      { name: 'source.png', type: 'flux2_reference_image' },
      { name: 'a.png', type: 'flux2_reference_image' },
    ]);
    expect(graph.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'flux2_vae_decode' });
  });

  it('takes the source and nine references and refuses an eleventh image', () => {
    const nine = Array.from({ length: 9 }, (_, index) => reference(`${index}.png`));

    expect(() => compileBase('flux3', KLEIN, nine)).not.toThrow();
    expect(() => compileBase('flux3', KLEIN, [...nine, reference('9.png')])).toThrow(/at most 10 reference images/);
  });

  it('runs Klein without negative conditioning and Klein Base with CFG, as Generate does', () => {
    const klein = compileBase('flux3', KLEIN);
    const base = compileBase('flux3', BASE);

    expect(klein.nodes.denoise_latents).toMatchObject({ cfg_scale: 1, type: 'flux3_denoise' });
    expect(klein.nodes.neg_cond).toBeUndefined();
    expect(base.nodes.denoise_latents?.cfg_scale).toBeGreaterThan(1);
    expect(base.nodes.neg_cond).toMatchObject({ type: 'flux3_text_encoder' });
  });
});

describe('fitToOptimalArea', () => {
  const model = { base: 'flux2', type: 'main' as const, variant: 'dev' };
  const { grid, optimal } = getGenerationDimensions(model);

  it('shrinks a large source to the ideal area on the grid, keeping its aspect ratio', () => {
    const size = fitToOptimalArea({ height: 3000, width: 4000 }, model);

    expect(size.width % grid).toBe(0);
    expect(size.height % grid).toBe(0);
    expect(size).toEqual({ height: 880, width: 1184 });
    expect(Math.abs(size.width / size.height - 4 / 3)).toBeLessThan(0.02);
    expect(Math.abs(size.width * size.height - optimal * optimal) / (optimal * optimal)).toBeLessThan(0.03);
  });

  it('grows a small source the same way', () => {
    expect(fitToOptimalArea({ height: 256, width: 256 }, model)).toEqual({ height: optimal, width: optimal });
    expect(fitToOptimalArea({ height: 200, width: 100 }, model)).toEqual({ height: 1456, width: 720 });
  });
});
