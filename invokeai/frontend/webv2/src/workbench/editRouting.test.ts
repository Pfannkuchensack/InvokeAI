import type { ModelShape } from '@features/generation/core/graphCoverage.testing';
import type { InstructWidgetValues } from '@features/instruct';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import { resolveInstructPatch } from '@features/instruct';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { EditStepPlan } from './editSubmission';

import { getBrowserId } from './browserId';
import { prepareEditStep } from './editSubmission';
import { settleStaleEditNodes } from './editTreeRouting';
import { captureInvocationRuntimeInputs } from './invocationPreparation';
import { getEditableEditTree, getEditNode, readStoredEditTree } from './projectEditTree';
import { getProjectWidgetValues } from './widgetState';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

seedArchitectureCapabilities();

const DEV: ModelShape = { label: 'dev', overrides: { format: 'diffusers', variant: 'dev' } };
const SOURCE = { height: 640, imageName: 'source.png', width: 480 };

const getActive = (state: WorkbenchState): Project =>
  state.projects.find((project) => project.id === state.activeProjectId)!;
const getTree = (state: WorkbenchState) => getEditableEditTree(getActive(state).editTree)!;

/** A project on its source image with the Edit widget set up for `variants` images per step. */
const createEditState = (variants = 2): WorkbenchState => {
  const { model, settings } = satisfiedSettingsFor('flux2', DEV);
  let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'instruct', type: 'applyPreset' });

  state = workbenchReducer(state, { projectId: state.activeProjectId, source: SOURCE, type: 'startEditTree' });

  return workbenchReducer(state, {
    projectId: state.activeProjectId,
    type: 'patchWidgetValues',
    values: { instruction: 'make the sky evening', model, seedMode: 'increment', settings, variants },
    widgetId: 'instruct',
  });
};

const submitStep = async (state: WorkbenchState): Promise<{ plan: EditStepPlan; state: WorkbenchState }> => {
  const project = getActive(state);
  const plan = await prepareEditStep({
    defaults: {},
    models: undefined,
    project,
    runtime: { noiseDevice: 'cpu' },
  });
  const next = workbenchReducer(state, {
    backendSupportsCancellation: true,
    editStep: plan,
    projectId: project.id,
    route: { ...project.invocation, destination: 'gallery', sourceId: 'instruct' },
    type: 'submitResolvedInvocationSnapshot',
  });

  return { plan, state: next };
};

const image = (imageName: string, sourceQueueItemId: string) => ({
  height: 768,
  imageName,
  imageUrl: `/${imageName}`,
  queuedAt: '2026-11-28T00:00:00Z',
  sourceQueueItemId,
  thumbnailUrl: `/${imageName}`,
  width: 1024,
});

const getItem = (state: WorkbenchState) => getActive(state).queue.items[0]!;

describe('Edit step routing', () => {
  it('queues the step with its nodes pending and links the run to them', async () => {
    const { plan, state } = await submitStep(createEditState());
    const item = getItem(state);
    const tree = getTree(state);

    expect(item.snapshot.sourceId).toBe('instruct');
    expect(item.snapshot.backendSubmission).toMatchObject({
      batchCount: 2,
      edit: { nodeIds: plan.step.variants.map((variant) => variant.id), rootNodeId: tree.rootNodeId },
      kind: 'generate',
    });
    expect(plan.step.variants.map((variant) => getEditNode(tree, variant.id)?.status)).toEqual(['pending', 'pending']);
    // The instruction goes with the step.
    expect(getActive(state).widgetInstances.instruct?.state.values.instruction).toBe('');
  });

  it('writes the run images into the nodes in order', async () => {
    const { plan, state } = await submitStep(createEditState());
    const item = getItem(state);
    const done = workbenchReducer(state, {
      images: [image('a.png', item.id), image('b.png', item.id)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(done), variant.id))).toMatchObject([
      { imageName: 'a.png', status: 'completed' },
      { imageName: 'b.png', status: 'completed' },
    ]);
  });

  it('fails a variant the run produced no image for', async () => {
    const { plan, state } = await submitStep(createEditState());
    const item = getItem(state);
    const done = workbenchReducer(state, {
      images: [image('a.png', item.id)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(getEditNode(getTree(done), plan.step.variants[1]!.id)).toMatchObject({ status: 'failed' });
  });

  it('stores the error of a failed run and cancels a cancelled one', async () => {
    const { plan, state } = await submitStep(createEditState());
    const item = getItem(state);
    const failed = workbenchReducer(state, {
      error: 'CUDA out of memory',
      projectId: getActive(state).id,
      queueItemId: item.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
    const cancelled = workbenchReducer(state, {
      notice: null,
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'cancelQueueItem',
    });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(failed), variant.id))).toMatchObject([
      { error: 'CUDA out of memory', status: 'failed' },
      { error: 'CUDA out of memory', status: 'failed' },
    ]);
    expect(plan.step.variants.map((variant) => getEditNode(getTree(cancelled), variant.id)?.status)).toEqual([
      'canceled',
      'canceled',
    ]);
  });

  it('takes no result for a tree that was replaced after the run was queued', async () => {
    const { state } = await submitStep(createEditState());
    const item = getItem(state);
    const replaced = workbenchReducer(state, {
      projectId: getActive(state).id,
      source: { height: 512, imageName: 'other.png', width: 512 },
      type: 'startEditTree',
    });
    const done = workbenchReducer(replaced, {
      images: [image('a.png', item.id), image('b.png', item.id)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(getTree(done).nodes).toHaveLength(1);
    expect(getTree(done).nodes[0]?.imageName).toBe('other.png');
  });

  it('refuses a step compiled for a tree that was replaced meanwhile, and says so', async () => {
    const queued = createEditState();
    const project = getActive(queued);
    const plan = await prepareEditStep({ defaults: {}, models: undefined, project, runtime: { noiseDevice: 'cpu' } });
    const replaced = workbenchReducer(queued, {
      projectId: project.id,
      source: { height: 512, imageName: 'other.png', width: 512 },
      type: 'startEditTree',
    });
    const next = workbenchReducer(replaced, {
      backendSupportsCancellation: true,
      editStep: plan,
      projectId: project.id,
      route: { ...project.invocation, destination: 'gallery', sourceId: 'instruct' },
      type: 'submitResolvedInvocationSnapshot',
    });

    expect(getActive(next).queue.items).toHaveLength(0);
    expect(getTree(next).nodes).toHaveLength(1);
    expect(next.notifications[0]).toMatchObject({ kind: 'error' });
  });

  it('keeps the tree and reports when a tree operation refuses its input', async () => {
    const { plan, state } = await submitStep(createEditState());
    const item = getItem(state);
    // An image name the tree parser refuses (empty) makes the completion throw inside the reducer.
    const done = workbenchReducer(state, {
      images: [image('', item.id), image('', item.id)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(done), variant.id)?.status)).toEqual([
      'pending',
      'pending',
    ]);
    expect(done.notifications.some((notification) => notification.kind === 'error')).toBe(true);
  });
});

const withBackendItems = (state: WorkbenchState, queueItemId: string, ids: number[]): WorkbenchState =>
  workbenchReducer(state, {
    backendItemIds: ids,
    projectId: getActive(state).id,
    queueItemId,
    type: 'markQueueItemBackendSubmitted',
  });

const imageFrom = (imageName: string, queueItemId: string, backendItemId: number) =>
  ({ ...image(imageName, queueItemId), sourceBackendItemId: backendItemId }) as ReturnType<typeof image>;

describe('Edit step settings', () => {
  it('submits the settings changed on a project that never stored a model', async () => {
    const { model } = satisfiedSettingsFor('flux2', DEV);
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'instruct', type: 'applyPreset' });

    state = workbenchReducer(state, { projectId: state.activeProjectId, source: SOURCE, type: 'startEditTree' });

    for (const patch of [
      (current: InstructWidgetValues) => ({ settings: { ...current.settings!, steps: 7 } }),
      (current: InstructWidgetValues) => ({ settings: { ...current.settings!, cfgScale: 2 } }),
    ]) {
      const stored = getProjectWidgetValues(getActive(state), 'instruct');

      state = workbenchReducer(state, {
        projectId: state.activeProjectId,
        type: 'patchWidgetValues',
        values: resolveInstructPatch(stored, patch, [model]),
        widgetId: 'instruct',
      });
    }
    state = workbenchReducer(state, {
      projectId: state.activeProjectId,
      type: 'patchWidgetValues',
      values: { instruction: 'brighter' },
      widgetId: 'instruct',
    });

    const plan = await prepareEditStep({
      defaults: {},
      models: [model as never],
      project: getActive(state),
      runtime: { noiseDevice: 'cpu' },
    });

    expect(plan.step.settings).toMatchObject({ cfgScale: 2, steps: 7 });
  });
});

describe('Edit step results by backend item', () => {
  it('puts each image in the node of the item that made it, and cancels the variant whose item was cancelled', async () => {
    const { plan, state: queued } = await submitStep(createEditState(3));
    const item = getItem(queued);
    let state = withBackendItems(queued, item.id, [11, 12, 13]);

    state = workbenchReducer(state, {
      backendItemId: 12,
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'markQueueItemBackendCancelled',
    });
    state = workbenchReducer(state, {
      images: [imageFrom('first.png', item.id, 11), imageFrom('third.png', item.id, 13)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(state), variant.id))).toMatchObject([
      { imageName: 'first.png', status: 'completed' },
      { status: 'canceled' },
      { imageName: 'third.png', status: 'completed' },
    ]);
  });

  it('keeps the images a failed run delivered and fails only the variants without one', async () => {
    const { plan, state: queued } = await submitStep(createEditState(2));
    const item = getItem(queued);
    let state = withBackendItems(queued, item.id, [11, 12]);

    state = {
      ...state,
      projects: state.projects.map((project) => ({
        ...project,
        queue: {
          items: project.queue.items.map((entry) => ({ ...entry, resultImages: [imageFrom('b.png', item.id, 12)] })),
        },
      })),
    };
    state = workbenchReducer(state, {
      error: 'out of memory',
      projectId: getActive(state).id,
      queueItemId: item.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(state), variant.id))).toMatchObject([
      { error: 'out of memory', status: 'failed' },
      { imageName: 'b.png', status: 'completed' },
    ]);
  });

  it('follows a variant through repeated retries, also when the original item is retried again', async () => {
    const { plan, state: queued } = await submitStep(createEditState(2));
    const item = getItem(queued);
    let state = withBackendItems(queued, item.id, [11, 12]);
    const fail = () =>
      workbenchReducer(state, {
        error: 'boom',
        projectId: getActive(state).id,
        queueItemId: item.id,
        status: 'failed',
        type: 'setQueueItemStatus',
      });
    const retry = (sourceItemId: number, itemId: number) =>
      workbenchReducer(state, {
        projectId: getActive(state).id,
        queueItemId: item.id,
        retriedItems: [{ itemId, sourceItemId }],
        type: 'reopenQueueItemForRetry',
      });

    state = fail();
    state = retry(12, 22);
    state = fail();
    state = retry(12, 32);
    expect(getEditNode(getTree(state), plan.step.variants[1]!.id)?.status).toBe('pending');

    state = workbenchReducer(state, {
      images: [imageFrom('again.png', item.id, 32)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(getEditNode(getTree(state), plan.step.variants[1]!.id)).toMatchObject({ imageName: 'again.png' });
    expect(getEditNode(getTree(state), plan.step.variants[1]!.id)).not.toHaveProperty('origin');
  });

  it('brings a retried variant back to pending and lands the retry result in it', async () => {
    const { plan, state: queued } = await submitStep(createEditState(2));
    const item = getItem(queued);
    let state = withBackendItems(queued, item.id, [11, 12]);

    state = workbenchReducer(state, {
      error: 'boom',
      projectId: getActive(state).id,
      queueItemId: item.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
    expect(getEditNode(getTree(state), plan.step.variants[1]!.id)?.status).toBe('failed');

    state = workbenchReducer(state, {
      projectId: getActive(state).id,
      queueItemId: item.id,
      retriedItems: [{ itemId: 22, sourceItemId: 12 }],
      type: 'reopenQueueItemForRetry',
    });
    expect(getEditNode(getTree(state), plan.step.variants[1]!.id)).toMatchObject({ status: 'pending' });

    state = workbenchReducer(state, {
      images: [imageFrom('retry.png', item.id, 22)],
      projectId: getActive(state).id,
      queueItemId: item.id,
      type: 'routeQueueItemResults',
    });

    expect(getEditNode(getTree(state), plan.step.variants[1]!.id)).toMatchObject({
      imageName: 'retry.png',
      status: 'completed',
    });
  });
});

describe('stopping the wait for a step started elsewhere', () => {
  it('cancels the pending nodes of a step with no live run here, and leaves one that has a run', async () => {
    const { plan, state } = await submitStep(createEditState());
    const projectId = getActive(state).id;
    const elsewhere: WorkbenchState = {
      ...state,
      projects: state.projects.map((project) => ({ ...project, queue: { items: [] } })),
    };
    const abandoned = workbenchReducer(elsewhere, { projectId, stepId: plan.step.stepId, type: 'abandonEditStep' });
    const kept = workbenchReducer(state, { projectId, stepId: plan.step.stepId, type: 'abandonEditStep' });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(abandoned), variant.id)?.status)).toEqual([
      'canceled',
      'canceled',
    ]);
    expect(plan.step.variants.map((variant) => getEditNode(getTree(kept), variant.id)?.status)).toEqual([
      'pending',
      'pending',
    ]);
  });
});

describe('stale pending nodes', () => {
  it('cancels a pending node with no live run once the project queue is known, and keeps one that has a run', async () => {
    const { plan, state } = await submitStep(createEditState());
    const projectId = getActive(state).id;
    // A reload: the nodes come back, the queue holds nothing.
    const reloaded: WorkbenchState = {
      ...state,
      projects: state.projects.map((project) => ({ ...project, queue: { items: [] } })),
    };
    const settled = workbenchReducer(reloaded, { origin: getBrowserId(), projectId, type: 'settleStaleEditNodes' });
    const kept = workbenchReducer(state, { origin: getBrowserId(), projectId, type: 'settleStaleEditNodes' });
    // Another browser's step: its run is in a journal this one cannot read, so the rule leaves it.
    const foreign = workbenchReducer(reloaded, { origin: 'another-browser', projectId, type: 'settleStaleEditNodes' });

    expect(plan.step.variants.map((variant) => getEditNode(getTree(settled), variant.id)?.status)).toEqual([
      'canceled',
      'canceled',
    ]);
    expect(plan.step.variants.map((variant) => getEditNode(getTree(kept), variant.id)?.status)).toEqual([
      'pending',
      'pending',
    ]);
    expect(plan.step.variants.map((variant) => getEditNode(getTree(foreign), variant.id)?.status)).toEqual([
      'pending',
      'pending',
    ]);
  });

  it('delivers a run that ended while the project was unloaded before applying the rule', async () => {
    const { plan, state } = await submitStep(createEditState());
    const item = getItem(state);
    const ended = {
      ...getActive(state),
      queue: {
        items: [
          { ...item, resultImages: [image('a.png', item.id), image('b.png', item.id)], status: 'completed' as const },
        ],
      },
    };

    expect(
      plan.step.variants.map(
        (variant) => getEditNode(getTree({ ...state, projects: [ended] } as WorkbenchState), variant.id)?.status
      )
    ).toEqual(['pending', 'pending']);

    const settled = settleStaleEditNodes(ended, getBrowserId()).project;

    expect(
      plan.step.variants.map((variant) => getEditNode(getEditableEditTree(settled.editTree)!, variant.id))
    ).toMatchObject([
      { imageName: 'a.png', status: 'completed' },
      { imageName: 'b.png', status: 'completed' },
    ]);
  });
});

describe('Deleting images an Edit history uses', () => {
  const ref = { height: 64, image_name: 'ref.png', width: 64 };
  const keep = { height: 64, image_name: 'keep.png', width: 64 };

  /** Two steps: source > a, b > two children of a; the composer holds two extra references. */
  const createHistory = async () => {
    let state = createEditState();
    const projectId = state.activeProjectId;
    const patch = (values: Record<string, unknown>) =>
      (state = workbenchReducer(state, { projectId, type: 'patchWidgetValues', values, widgetId: 'instruct' }));
    const complete = (images: string[]) => {
      const item = getActive(state).queue.items[0]!;

      state = workbenchReducer(state, {
        images: images.map((name) => image(name, item.id)),
        projectId,
        queueItemId: item.id,
        type: 'routeQueueItemResults',
      });
    };

    patch({ references: [ref, keep] });
    state = (await submitStep(state)).state;
    complete(['a.png', 'b.png']);
    patch({ instruction: 'and add birds' });
    state = (await submitStep(state)).state;

    return state;
  };

  const remove = (state: WorkbenchState, ...names: string[]) =>
    workbenchReducer(state, { itemKeys: names.map((name) => `image:${name}` as const), type: 'removeGalleryItems' });

  it('keeps a deleted result as a placeholder with its instruction, settings and children', async () => {
    const before = await createHistory();
    const treeBefore = getTree(before);
    const a = treeBefore.nodes.find((node) => node.imageName === 'a.png')!;
    const after = getTree(remove(before, 'a.png'));
    const placeholder = getEditNode(after, a.id)!;

    expect(placeholder).toMatchObject({
      imageName: null,
      instruction: a.instruction,
      parentId: a.parentId,
      settings: a.settings,
      status: 'deleted',
    });
    expect(after.nodes).toHaveLength(treeBefore.nodes.length);
    expect(after.nodes.filter((node) => node.parentId === a.id)).toEqual(
      treeBefore.nodes.filter((node) => node.parentId === a.id)
    );
    expect(getEditNode(after, treeBefore.nodes.find((node) => node.imageName === 'b.png')!.id)?.status).toBe(
      'completed'
    );
  });

  it('marks a deleted source and keeps the names of deleted step references', async () => {
    const before = await createHistory();
    const after = getTree(remove(before, 'source.png', 'ref.png'));

    expect(getEditNode(after, after.rootNodeId)).toMatchObject({ imageName: null, status: 'deleted' });
    expect(after.nodes.flatMap((node) => node.referenceImages.map((reference) => reference.imageName))).toContain(
      'ref.png'
    );
  });

  it('removes deleted images from the composer references only', async () => {
    const after = remove(await createHistory(), 'ref.png');

    expect(getProjectWidgetValues(getActive(after), 'instruct').references).toEqual([keep]);
  });

  it('waits for the confirmed removal: an optimistic one that is rolled back leaves the history and references intact', async () => {
    const before = await createHistory();
    const optimistic = workbenchReducer(before, {
      itemKeys: ['image:a.png', 'image:ref.png', 'image:source.png'],
      optimistic: true,
      type: 'removeGalleryItems',
    });

    expect(getActive(optimistic).editTree).toBe(getActive(before).editTree);
    expect(getProjectWidgetValues(getActive(optimistic), 'instruct').references).toEqual([ref, keep]);

    const confirmed = workbenchReducer(optimistic, {
      itemKeys: ['image:a.png'],
      type: 'removeGalleryItems',
    });

    expect(getTree(confirmed).nodes.filter((node) => node.status === 'deleted')).toHaveLength(1);
  });

  it('tells which reference was removed when later ones are renumbered, and only then', async () => {
    const messages = (state: WorkbenchState) =>
      state.notifications.filter((notice) => notice.titleKey === 'widgets.instruct.referencesChanged');

    expect(messages(remove(await createHistory(), 'ref.png'))).toMatchObject([
      { message: 'Image 2 was removed from the references; check your instruction.' },
    ]);
    expect(messages(remove(await createHistory(), 'keep.png'))).toEqual([]);
  });

  it('refuses to start over an edit history this build cannot read', () => {
    const state = createEditState();
    const unreadable = readStoredEditTree({ nodes: 'newer format', rootNodeId: 'x' });
    const held = {
      ...state,
      projects: state.projects.map((project) => ({ ...project, editTree: unreadable })),
    } as WorkbenchState;
    const after = workbenchReducer(held, { projectId: held.activeProjectId, source: SOURCE, type: 'startEditTree' });

    expect(getActive(after).editTree).toBe(unreadable);
    expect(after.notifications.at(-1)).toMatchObject({ kind: 'error', titleKey: 'widgets.instruct.historyNotUpdated' });
  });

  it('leaves the history and the references alone when none of the images is theirs', async () => {
    const before = await createHistory();
    const after = remove(before, 'unrelated.png');

    expect(getActive(after).editTree).toBe(getActive(before).editTree);
    expect(getActive(after).widgetInstances).toBe(getActive(before).widgetInstances);
  });
});

void captureInvocationRuntimeInputs;
