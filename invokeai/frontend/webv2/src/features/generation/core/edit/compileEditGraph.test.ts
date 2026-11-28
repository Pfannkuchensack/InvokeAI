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
