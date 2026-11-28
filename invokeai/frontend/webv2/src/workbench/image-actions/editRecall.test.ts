import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { Project } from '@workbench/projectContracts';

import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import { fitToOptimalArea } from '@features/generation/editGraph';
import { takeInstructEntry } from '@features/instruct/core/entryRequests';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { createEditTree, readStoredEditTree } from '@workbench/projectEditTree';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { recallIntoEdit, type EditRecallInput } from './editRecall';

seedArchitectureCapabilities();

const edit = satisfiedSettingsFor('flux2', { label: 'dev', overrides: { format: 'diffusers', variant: 'dev' } });
const sdxl = satisfiedSettingsFor('sdxl', { label: 'base', overrides: { format: 'checkpoint' } });
const SOURCE = { height: 640, image_name: 'source.png', width: 480 };
const reference = (id: string, imageName: string) => ({
  config: {
    image: { original: { image: { height: 64, image_name: imageName, width: 64 } } },
    type: 'flux2_reference_image',
  },
  id,
  isEnabled: true,
});

const inputFor = (
  overrides: Partial<EditRecallInput> = {},
  values: Partial<GenerateWidgetValues> = {}
): EditRecallInput =>
  ({
    kind: 'all',
    source: SOURCE,
    sourceName: 'source.png',
    values: {
      ...edit.settings,
      height: 1024,
      model: edit.model,
      modelKey: edit.model.key,
      positivePrompt: 'make the sky evening',
      referenceImages: [reference('r1', 'source.png'), reference('r2', 'extra.png')],
      seed: 7,
      seedMode: 'fixed',
      width: 768,
      ...values,
    } as unknown as GenerateWidgetValues,
    ...overrides,
  }) as EditRecallInput;

const treeOf = (imageName: string) =>
  createEditTree({ height: SOURCE.height, imageName, width: SOURCE.width }, '2026-11-28T00:00:00.000Z', 'root');

const NO_CHAT = { widgetInstances: {}, widgetRegions: { center: { activeInstanceId: null, instanceIds: [] } } };
const CHAT_SHOWN = {
  widgetInstances: { 'instruct:center': { typeId: 'instruct' } },
  widgetRegions: { center: { activeInstanceId: 'instruct:center', instanceIds: ['instruct:center'] } },
};

const setup = (project: Record<string, unknown> = {}, { isActive = true }: { isActive?: boolean } = {}) => {
  const patchValues = vi.fn();
  const activatePreset = vi.fn();
  const select = vi.fn();
  const target = {
    commands: { layout: { activatePreset }, widgets: { patchValues, select } } as never,
    getProject: () => ({ layout: { presetId: 'compose' }, ...NO_CHAT, ...project }) as unknown as Project,
    isActiveProject: () => isActive,
    projectId: 'project-1',
  };

  return { activatePreset, patchValues, select, target };
};

describe('recallIntoEdit', () => {
  beforeEach(() => {
    takeInstructEntry('project-1');
  });

  it('puts the step into the composer and keeps the source as the continue point of the history it is in', async () => {
    const { activatePreset, patchValues, target } = setup({ editTree: treeOf('source.png') });

    await expect(recallIntoEdit(target, inputFor())).resolves.toBe('recalled');

    expect(takeInstructEntry('project-1')).toMatchObject({
      continueFromImage: true,
      image: SOURCE,
      values: {
        instruction: 'make the sky evening',
        model: { key: edit.model.key },
        // The record's list is the source, then the extra references: the source is not an extra.
        references: [{ height: 64, image_name: 'extra.png', width: 64 }],
        seed: 7,
        seedMode: 'fixed',
      },
    });
    expect(patchValues).not.toHaveBeenCalled();
    expect(activatePreset).toHaveBeenCalledExactlyOnceWith('instruct');
  });

  it('offers the source as the start of a new history when the current one does not hold it, or there is none', async () => {
    for (const editTree of [treeOf('another.png'), undefined]) {
      const { patchValues, target } = setup({ editTree });

      await recallIntoEdit(target, inputFor());

      expect(takeInstructEntry('project-1')).toMatchObject({ continueFromImage: true, image: SOURCE });
      expect(patchValues).not.toHaveBeenCalled();
    }
  });

  it('only restores the values when the source is gone or the stored history cannot be read', async () => {
    const gone = setup({ editTree: treeOf('source.png') });

    await recallIntoEdit(gone.target, inputFor({ source: null }));

    expect(takeInstructEntry('project-1')).toBeNull();
    expect(gone.patchValues).toHaveBeenCalledWith(
      'instruct',
      expect.objectContaining({ instruction: 'make the sky evening' }),
      'project-1'
    );

    const unreadable = setup({ editTree: readStoredEditTree({ nodes: 'newer format', rootNodeId: 'x' }) });

    await recallIntoEdit(unreadable.target, inputFor());

    expect(takeInstructEntry('project-1')).toBeNull();
    expect(unreadable.patchValues).toHaveBeenCalledTimes(1);
  });

  it('leaves the layout alone while the chat is the view on screen', async () => {
    const { activatePreset, target } = setup({
      editTree: treeOf('source.png'),
      layout: { presetId: 'instruct' },
      ...CHAT_SHOWN,
    });

    await recallIntoEdit(target, inputFor());

    expect(activatePreset).not.toHaveBeenCalled();
  });

  it('refuses a record whose model Edit cannot run, so Generate takes it', async () => {
    const { activatePreset, patchValues, target } = setup();

    await expect(
      recallIntoEdit(target, inputFor({}, { model: sdxl.model, modelKey: sdxl.model.key } as never))
    ).resolves.toBe('declined');

    expect(takeInstructEntry('project-1')).toBeNull();
    expect(patchValues).not.toHaveBeenCalled();
    expect(activatePreset).not.toHaveBeenCalled();
  });

  it('brings a hidden chat forward without switching the preset', async () => {
    const { activatePreset, select, target } = setup({
      layout: { presetId: 'custom' },
      widgetInstances: { 'instruct:center': { typeId: 'instruct' }, preview: { typeId: 'preview' } },
      widgetRegions: { center: { activeInstanceId: 'preview', instanceIds: ['instruct:center', 'preview'] } },
    });

    await recallIntoEdit(target, inputFor());

    expect(select).toHaveBeenCalledExactlyOnceWith({
      projectId: 'project-1',
      region: 'center',
      widgetId: 'instruct:center',
    });
    expect(activatePreset).not.toHaveBeenCalled();
  });

  it('leaves Edit alone when the project is no longer the active one', async () => {
    const { activatePreset, patchValues, target } = setup({ editTree: treeOf('source.png') }, { isActive: false });

    await expect(recallIntoEdit(target, inputFor())).resolves.toBe('stale');

    expect(takeInstructEntry('project-1')).toBeNull();
    expect(patchValues).not.toHaveBeenCalled();
    expect(activatePreset).not.toHaveBeenCalled();
  });

  it('keeps the seed of Edit when remixing, and takes the recorded one for Recall All', async () => {
    const { target } = setup({ editTree: treeOf('source.png') });

    await recallIntoEdit(target, inputFor({ kind: 'remix' }));
    const remix = takeInstructEntry('project-1')?.values;

    expect(remix).not.toHaveProperty('seed');
    expect(remix).not.toHaveProperty('seedMode');

    await recallIntoEdit(target, inputFor());

    expect(takeInstructEntry('project-1')?.values).toMatchObject({ seed: 7, seedMode: 'fixed' });
  });

  it('sizes against the tip of the kept history when the source is not in it', async () => {
    const { target } = setup({ editTree: treeOf('another.png') });

    await recallIntoEdit(target, inputFor());

    const entry = takeInstructEntry('project-1');

    expect(entry?.valuesIfKept).toBeDefined();
    expect(entry?.valuesIfKept).toMatchObject({ instruction: 'make the sky evening' });
  });

  describe('size', () => {
    // Read in the test: the model's grid comes from the capabilities seeded per test.
    const getFitted = () => fitToOptimalArea({ height: SOURCE.height, width: SOURCE.width }, edit.model);

    it('stays automatic when the record has the size the source gives', async () => {
      const { target } = setup({ editTree: treeOf('source.png') });

      await recallIntoEdit(target, inputFor({}, getFitted()));

      expect(takeInstructEntry('project-1')?.values).toMatchObject({ sizeMode: 'auto' });
    });

    it('becomes manual when the record has another size', async () => {
      const { target } = setup({ editTree: treeOf('source.png') });
      const fitted = getFitted();

      await recallIntoEdit(target, inputFor({}, { height: fitted.height + 64, width: fitted.width }));

      expect(takeInstructEntry('project-1')?.values).toMatchObject({
        height: fitted.height + 64,
        sizeMode: 'manual',
        width: fitted.width,
      });
    });
  });
});
