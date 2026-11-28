import type { EditGraphSettings } from '@features/generation/editGraph';
import type { ModelConfig } from '@features/models';

import { addConcept } from '@features/generation/components';
import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
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
  getInstructValidationReasons,
  isInstructEditModel,
  normalizeInstructValues,
  resolveInstructPatch,
  syncInstructValuesWithModels,
} from './values';

seedArchitectureCapabilities();

const fit = ({ height, width }: { width: number; height: number }) => ({ height: height * 2, width: width * 2 });
const image = (name: string) => ({ height: 100, image_name: name, width: 100 });

describe('output size', () => {
  const base = { fit, manual: null, source: { height: 300, width: 200 } };

  it('fits the source for the first step and inherits the parent size after it, so a chain never drifts', () => {
    expect(resolveEditOutputSize({ ...base, parent: { height: 300, isSource: true, width: 200 } })).toEqual({
      height: 600,
      width: 400,
    });
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
    expect(Object.keys(snapshot).sort()).toEqual([
      'cfgScale',
      'loras',
      'mistralEncoderModel',
      'model',
      'scheduler',
      'steps',
      'vae',
    ]);
    expect(JSON.stringify(snapshot).length).toBeLessThan(1500);
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
