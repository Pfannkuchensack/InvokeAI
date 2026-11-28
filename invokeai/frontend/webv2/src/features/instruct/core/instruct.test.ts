import type { EditGraphSettings } from '@features/generation/editGraph';
import type { ModelConfig } from '@features/models';

import { addConcept } from '@features/generation/components';
import { compileEditGraph } from '@features/generation/core/edit/compileEditGraph';
import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import {
  EDIT_BASES,
  fitToOptimalArea,
  getDefaultGenerateSettings,
  getEditValidationIssues,
  getGenerationDimensions,
  toEditMessage,
  getInstructionEditSupport,
} from '@features/generation/settings';
import architectureCapabilitiesFixture from '@features/models/core/__fixtures__/architectureCapabilities.json';
import { setArchitectureCapabilities } from '@features/models/core/architectureCapabilities';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { InstructNode, InstructTreeView, InstructWidgetValues } from './types';

import { getInstructRoute, getInstructSteps, getInstructTip } from './history';
import { addInstructReferences, getReferenceNumber, getReferenceRoom, removeInstructReference } from './references';
import { resolveEditOutputSize } from './size';
import {
  createDefaultInstructValues,
  getInstructStepSettings,
  getInstructReferenceTag,
  getInstructValidationReasons,
  INSTRUCT_PIPELINE_KEYS,
  isInstructEditModel,
  normalizeInstructValues,
  resolveInstructPatch,
  selectInstructModel,
  syncInstructValuesWithModels,
} from './values';

seedArchitectureCapabilities();

const GGUF_SHAPE = { ggufEncoder: true, label: 'gguf', overrides: { format: 'gguf_quantized' } } as const;

const image = (name: string) => ({ height: 100, image_name: name, width: 100 });

const modelOf = (base: 'flux2' | 'flux3' | 'qwen-image-2-1') =>
  satisfiedSettingsFor(base, {
    label: 'm',
    overrides: base === 'flux2' ? { format: 'diffusers', variant: 'dev' } : { format: 'diffusers' },
  }).model;

describe('output size', () => {
  const limits = getGenerationDimensions(modelOf('qwen-image-2-1'));
  const flux2 = modelOf('flux2');
  const qwen = modelOf('qwen-image-2-1');
  const base = { manual: null, model: flux2, source: { height: 300, width: 200 } };

  it('fits the source for the first step and inherits the parent size after it, so a chain never drifts', () => {
    expect(resolveEditOutputSize({ ...base, parent: { height: 300, isSource: true, width: 200 } })).toEqual(
      fitToOptimalArea({ height: 300, width: 200 }, flux2)
    );
    expect(resolveEditOutputSize({ ...base, parent: { height: 832, isSource: false, width: 1216 } })).toEqual({
      height: 832,
      width: 1216,
    });
  });

  it('takes a manual size over both', () => {
    expect(
      resolveEditOutputSize({
        ...base,
        manual: { height: 512, width: 768 },
        parent: { height: 1, isSource: false, width: 1 },
      })
    ).toEqual({ height: 512, width: 768 });
  });

  it('snaps an inherited size to the grid of a model switched to: 1184x880 is 1184x896 on the 32 grid', () => {
    expect(
      resolveEditOutputSize({ ...base, model: qwen, parent: { height: 880, isSource: false, width: 1184 } })
    ).toEqual({ height: 896, width: 1184 });
  });

  it('snaps a manual size to the nearest multiple, halves rounding up, within the limits: 1008 is 1024', () => {
    const manual = (width: number, height: number) =>
      resolveEditOutputSize({
        ...base,
        manual: { height, width },
        model: qwen,
        parent: { height: 1, isSource: false, width: 1 },
      });

    expect(manual(1008, 1008)).toEqual({ height: 1024, width: 1024 });
    expect(manual(1, 100000)).toEqual({ height: limits.max, width: limits.min });
  });
});

describe('extra references', () => {
  it('numbers them after the source, which is Image 1', () => {
    expect([0, 1, 2].map(getReferenceNumber)).toEqual([2, 3, 4]);
  });

  it('leaves room for the model maximum minus the source', () => {
    expect(getReferenceRoom(5, [image('a'), image('b')])).toBe(2);
    expect(getReferenceRoom(1, [])).toBe(0);
  });

  it('adds what fits, skipping the source and repeats, and removes by name', () => {
    const added = addInstructReferences([image('a')], [image('a'), image('src'), image('b'), image('c'), image('d')], {
      maxReferenceImages: 4,
      sourceImageName: 'src',
    });

    expect(added.map((reference) => reference.image_name)).toEqual(['a', 'b', 'c']);
    expect(removeInstructReference(added, 'b').map((reference) => reference.image_name)).toEqual(['a', 'c']);
  });
});

describe('Edit values', () => {
  it('reads a FLUX.2 dev checkpoint as an edit model and keeps only the edit-scoped fields in a node snapshot', () => {
    const { model, settings } = satisfiedSettingsFor('flux2', {
      label: 'dev',
      overrides: { format: 'diffusers', variant: 'dev' },
    });
    const values = normalizeInstructValues({ model, settings, variants: 9 });
    const snapshot = getInstructStepSettings(values);

    expect(isInstructEditModel(model)).toBe(true);
    expect(values.variants).toBe(4);
    // Every pipeline field the step reads and the model holds, as an identifier where it is a model.
    expect(Object.keys(snapshot).sort()).toEqual(
      [...INSTRUCT_PIPELINE_KEYS, 'model']
        .filter(
          (key) =>
            key === 'model' ||
            ((values.settings as never)[key] !== null && (values.settings as never)[key] !== undefined)
        )
        .sort()
    );
    // Models are kept by identifier, not as the full config.
    expect(JSON.stringify(snapshot)).not.toContain('"format"');
    expect(JSON.stringify(snapshot).length).toBeLessThan(1500);
  });

  it('reads Qwen-Image 2.1 as an edit model and holds the vision tower it reads', () => {
    const qwen = satisfiedSettingsFor('qwen-image-2-1', { label: 'diffusers', overrides: { format: 'diffusers' } });
    expect(isInstructEditModel(qwen.model)).toBe(true);
    const gguf = satisfiedSettingsFor('qwen-image-2-1', GGUF_SHAPE);

    expect(gguf.settings.qwen3VLVisionTowerModel).toBeTruthy();
    expect(normalizeInstructValues({ model: gguf.model, settings: gguf.settings }).settings).toMatchObject({
      qwen3VLVisionTowerModel: { key: gguf.settings.qwen3VLVisionTowerModel!.key },
    });
    expect(normalizeInstructValues({ model: qwen.model, settings: qwen.settings }).settings).toMatchObject({
      qwen3VLVisionTowerModel: null,
    });
  });

  it('starts without a model when none is installed', () => {
    expect(createDefaultInstructValues([])).toMatchObject({ model: null, settings: null, variants: 1 });
  });
});

const node = (
  id: string,
  parentId: string | null,
  stepId: string,
  extra: Partial<InstructNode> = {}
): InstructNode => ({
  height: 10,
  id,
  imageName: `${id}.png`,
  instruction: id,
  parentId,
  referenceImages: [],
  seed: 0,
  status: 'completed',
  stepId,
  width: 10,
  ...extra,
});

describe('history of the shown branch', () => {
  const nodes = [
    node('root', null, 'root'),
    node('a1', 'root', 's1'),
    node('a2', 'root', 's1'),
    node('b', 'root', 's2'),
    node('c', 'b', 's3'),
  ];
  const tree: InstructTreeView = {
    nodes,
    path: [nodes[0]!, nodes[3]!, nodes[4]!],
    rootNodeId: 'root',
    selection: {},
  };

  it('groups the variants of each step and lists the sibling steps for the switcher', () => {
    const [first, second] = getInstructSteps(tree);

    expect(first).toMatchObject({ stepId: 's2' });
    expect(first?.siblingSteps).toEqual([
      { firstNodeId: 'a1', stepId: 's1' },
      { firstNodeId: 'b', stepId: 's2' },
    ]);
    expect(first?.variants.map((variant) => variant.id)).toEqual(['b']);
    expect(second?.parent.id).toBe('b');
  });

  it('continues from the chosen node while it exists, else from the end of the branch', () => {
    expect(getInstructTip(tree, { continueFromNodeId: 'a1' }).id).toBe('a1');
    expect(getInstructTip(tree, { continueFromNodeId: 'gone' }).id).toBe('c');
    expect(getInstructTip(tree, { continueFromNodeId: null }).id).toBe('c');
  });

  it('routes from the root to any node', () => {
    expect(getInstructRoute(tree, 'c').map((entry) => entry.id)).toEqual(['root', 'b', 'c']);
  });
});

describe('the default edit model before the catalog and the architecture table have arrived', () => {
  const klein = satisfiedSettingsFor('flux2', {
    label: 'klein',
    overrides: { format: 'gguf_quantized', variant: 'klein_9b' },
  }).model;

  it('shows no model for an empty list, then the first edit model once it is listed', () => {
    const stored = normalizeInstructValues({});

    expect(syncInstructValuesWithModels(stored, []).model).toBeNull();
    expect(syncInstructValuesWithModels(stored, [klein]).model?.key).toBe(klein.key);
  });

  it('shows no model while the architecture table is empty, then the model once it is there', () => {
    const stored = normalizeInstructValues({});

    setArchitectureCapabilities([]);
    expect(syncInstructValuesWithModels(stored, [klein]).model).toBeNull();
    setArchitectureCapabilities(architectureCapabilitiesFixture as never);
    expect(syncInstructValuesWithModels(stored, [klein]).model?.key).toBe(klein.key);
  });
});

describe('patches computed against the shown values', () => {
  it('reads back every setting changed on a fresh project that has not stored a model yet', () => {
    const { model, settings } = satisfiedSettingsFor('flux2', {
      label: 'klein',
      overrides: { format: 'gguf_quantized', variant: 'klein_9b' },
    });
    const lora = { base: 'flux2', hash: 'h', key: 'lora-1', name: 'Style', type: 'lora' } as unknown as ModelConfig;
    const models = [model, lora, ...[settings.mistralEncoderModel].filter((entry) => entry !== null)];
    let stored: Record<string, unknown> = {};
    const apply = (patch: (current: InstructWidgetValues) => Partial<InstructWidgetValues>) => {
      stored = { ...stored, ...resolveInstructPatch(stored, patch, models) };
    };
    const change = (fields: Partial<EditGraphSettings>) => (current: InstructWidgetValues) => ({
      settings: current.settings ? { ...current.settings, ...fields } : undefined,
    });

    apply(change({ steps: 7 }));
    apply(change({ cfgScale: 3 }));
    apply((current) => ({
      settings: current.settings
        ? { ...current.settings, loras: addConcept(current.settings.loras, lora as never, current.model) }
        : undefined,
    }));
    apply(change({ scheduler: 'lcm' }));
    apply(change({ mistralEncoderModel: settings.mistralEncoderModel }));
    apply((current) => ({
      references: addInstructReferences(current.references, [image('extra.png')], {
        maxReferenceImages: 5,
        sourceImageName: null,
      }),
    }));

    const shown = syncInstructValuesWithModels(normalizeInstructValues(stored), models);

    expect(shown.model?.key).toBe(model.key);
    expect(shown.settings).toMatchObject({
      cfgScale: 3,
      mistralEncoderModel: settings.mistralEncoderModel,
      scheduler: 'lcm',
      steps: 7,
    });
    expect(shown.settings?.loras.map((entry) => entry.model.key)).toEqual(['lora-1']);
    expect(shown.references.map((reference) => reference.image_name)).toEqual(['extra.png']);
  });
});

describe('what keeps a step from being queued', () => {
  const kleinShape = { label: 'klein', overrides: { format: 'gguf_quantized', variant: 'klein_9b' } } as const;

  it('names the missing encoder and VAE as Edit, then clears once they are picked', () => {
    const { model, settings } = satisfiedSettingsFor('flux2', kleinShape);
    const bare = { ...settings, qwen3EncoderModel: null, vae: null };
    const reasons = getInstructValidationReasons(
      { ...createDefaultInstructValues([model]), instruction: 'brighter', settings: bare },
      undefined
    );

    expect(reasons).toHaveLength(2);
    expect(reasons.every((reason) => reason.startsWith('Edit needs a '))).toBe(true);
    expect(
      getInstructValidationReasons(
        { ...createDefaultInstructValues([model]), instruction: 'brighter', settings },
        undefined
      )
    ).toEqual([]);
  });

  it('fills the model default components when a model is first chosen, like Generate', () => {
    const { model, settings } = satisfiedSettingsFor('flux2', kleinShape);
    const encoder = settings.qwen3EncoderModel!;
    const withDefaults = {
      ...model,
      default_settings: { components: { qwen3_encoder: encoder.key } },
    } as unknown as ModelConfig;
    const shown = syncInstructValuesWithModels(normalizeInstructValues({}), [withDefaults, encoder as ModelConfig]);

    expect(shown.settings?.qwen3EncoderModel?.key).toBe(encoder.key);
  });
});

describe('what Send allows is what compiles', () => {
  const PROJECT = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };
  const SOURCE = { height: 512, image_name: 'source.png', width: 512 };

  /** The reasons before Send for these values, and the message compile refuses with (both worded as Edit). */
  const both = (
    base: 'flux3' | 'qwen-image-2-1',
    overrides: Record<string, unknown>,
    patch: Record<string, unknown>,
    size = { height: 1024, width: 1024 }
  ) => {
    const { model, settings } = satisfiedSettingsFor(base, {
      ...(overrides.format === 'gguf_quantized' ? { ggufEncoder: true } : {}),
      label: 'm',
      overrides,
    } as never);
    const picked = { ...settings, ...patch } as EditGraphSettings;
    const reasons = getInstructValidationReasons(
      { ...createDefaultInstructValues([model]), instruction: 'brighter', settings: picked },
      undefined,
      size
    );
    let refusal = '';

    try {
      compileEditGraph({
        model,
        projectSettings: PROJECT,
        referenceImages: [],
        settings: picked,
        size,
        source: SOURCE,
      });
    } catch (error) {
      refusal = (error as Error).message.replace(/^Generate /, 'Edit ');
    }

    return { reasons, refusal };
  };

  it('refuses a GGUF encoder without its Vision Tower for Qwen-Image 2.1, in the same words', () => {
    const base = 'qwen-image-2-1';
    const { reasons, refusal } = both(base, { format: 'gguf_quantized' }, { qwen3VLVisionTowerModel: null });

    expect(refusal).toMatch(/Vision Tower/);
    expect(reasons).toContain(refusal);
    expect(both(base, { format: 'gguf_quantized' }, {}).reasons).toEqual([]);
  });

  it('refuses FLUX.3 Klein with CFG above 1, as after switching from Klein Base, in the same words', () => {
    const { reasons, refusal } = both('flux3', { format: 'diffusers', variant: 'flux3_klein' }, { cfgScale: 3.5 });

    expect(refusal).toMatch(/Klein samples without CFG/);
    expect(reasons).toContain(refusal);
  });

  it('refuses a size off the model grid in the same words', () => {
    const { reasons, refusal } = both('qwen-image-2-1', { format: 'diffusers' }, {}, { height: 1024, width: 1040 });

    expect(refusal).toMatch(/multiple of 32/);
    expect(reasons).toContain(refusal);
  });

  it('points the Vision Tower issue at its Components slot', () => {
    const { model, settings } = satisfiedSettingsFor('qwen-image-2-1', GGUF_SHAPE);
    const issues = getEditValidationIssues({
      images: [SOURCE],
      model,
      settings: { ...settings, qwen3VLVisionTowerModel: null } as EditGraphSettings,
    });

    expect(issues.map((issue) => issue.target.field)).toContain('qwen3VLVisionTowerModel');
  });
});

describe('edit bases', () => {
  const shapes = {
    flux2: { label: 'dev', overrides: { format: 'diffusers', variant: 'dev' } },
    flux3: { label: 'klein', overrides: { format: 'diffusers', variant: 'flux3_klein' } },
  } as const;
  const modelFor = (base: 'flux2' | 'flux3' | 'krea-2' | 'qwen-image-2-1' | 'sdxl' | 'flux' | 'z-image') =>
    satisfiedSettingsFor(
      base,
      (shapes as unknown as Record<string, never>)[base] ?? { label: 'd', overrides: { format: 'diffusers' } }
    ).model;

  it('are exactly the models the picker accepts', () => {
    const accepted = (['flux2', 'flux3', 'krea-2', 'qwen-image-2-1', 'sdxl', 'flux', 'z-image'] as const).filter(
      (base) => isInstructEditModel(modelFor(base))
    );

    expect([...accepted].sort()).toEqual([...EDIT_BASES].sort());
  });

  it('describe FLUX.3 as editing from noise with up to ten images', () => {
    expect(getInstructionEditSupport(modelFor('flux3'))).toEqual({
      img2imgStart: false,
      maxReferenceImages: 10,
    });
  });

  it('name FLUX.3 images by tag, one number below the one Edit shows, and no other model', () => {
    expect([1, 2, 10].map((number) => getInstructReferenceTag(modelFor('flux3'), number))).toEqual([
      '<ref_image_0>',
      '<ref_image_1>',
      '<ref_image_9>',
    ]);
    expect(getInstructReferenceTag(modelFor('flux2'), 1)).toBeUndefined();
    expect(getInstructReferenceTag(null, 1)).toBeUndefined();
  });
});

describe('switching the Edit model', () => {
  const models = (...picked: ReturnType<typeof modelOf>[]) => picked as unknown as ModelConfig[];

  it('gives a value still at the previous default the new default, and keeps one the user changed', () => {
    const klein = satisfiedSettingsFor('flux3', {
      label: 'klein',
      overrides: { format: 'diffusers', variant: 'flux3_klein' },
    });
    const target = satisfiedSettingsFor('qwen-image-2-1', { label: 'd', overrides: { format: 'diffusers' } });
    const kleinDefaults = getDefaultGenerateSettings(klein.model);
    const targetDefaults = getDefaultGenerateSettings(target.model);
    const untouched = normalizeInstructValues({
      model: klein.model,
      settings: {
        ...klein.settings,
        cfgScale: kleinDefaults.cfgScale,
        scheduler: kleinDefaults.scheduler,
        steps: kleinDefaults.steps,
      },
    });
    const changed = normalizeInstructValues({ model: klein.model, settings: { ...klein.settings, steps: 23 } });

    expect(kleinDefaults.steps).not.toBe(targetDefaults.steps);
    expect(selectInstructModel(untouched, target.model, models(target.model)).settings).toMatchObject({
      cfgScale: targetDefaults.cfgScale,
      scheduler: targetDefaults.scheduler,
      steps: targetDefaults.steps,
    });
    expect(selectInstructModel(changed, target.model, models(target.model)).settings?.steps).toBe(23);
  });
});

describe('Edit wording of a refusal', () => {
  it('counts the image being edited in a reference limit and names Edit for a missing component', () => {
    const { model, settings } = satisfiedSettingsFor('qwen-image-2-1', {
      label: 'd',
      overrides: { format: 'diffusers' },
    });
    const issues = getEditValidationIssues({
      images: Array.from({ length: 5 }, (_, index) => image(`${index}.png`)),
      model,
      settings: settings as EditGraphSettings,
    });

    expect(issues.map(({ message }) => message)).toEqual([
      expect.stringMatching(/takes at most 4 reference images, the image you edit counted as one\. Remove the extra/),
    ]);
    expect(toEditMessage('Generate needs a VAE for FLUX models.')).toBe('Edit needs a VAE for FLUX models.');
  });
});
