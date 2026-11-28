import { takeInstructEntry } from '@features/instruct/core/entryRequests';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { openImageInEdit, type EditEntryTarget } from './editEntry';
import { createInitialWorkbenchState } from './workbenchState';
import { createWorkbenchStore } from './workbenchStore';

const IMAGE = { height: 480, image_name: 'sent.png', width: 640 };

const createTarget = (store: ReturnType<typeof createWorkbenchStore>, projectId?: string): EditEntryTarget => {
  const id = projectId ?? store.getSnapshot().activeProject.id;

  return {
    commands: store.commands,
    getProject: (candidate) => store.getSnapshot().projects.find((project) => project.id === candidate),
    isActiveProject: (candidate) => store.getSnapshot().activeProject.id === candidate,
    projectId: id,
  };
};

const getCenter = (store: ReturnType<typeof createWorkbenchStore>) =>
  store.getSnapshot().activeProject.widgetRegions.center;

// The real store and layout commands: the entry must leave the Edit chat as the view on screen.
describe('openImageInEdit', () => {
  let store: ReturnType<typeof createWorkbenchStore>;

  beforeEach(() => {
    store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutPresetWidgets: () => Promise.resolve() });
    takeInstructEntry(store.getSnapshot().activeProject.id);
  });

  it('brings the chat forward in the Edit preset when Preview is the center view', async () => {
    const projectId = store.getSnapshot().activeProject.id;

    await store.commands.layout.activatePreset('instruct');
    store.commands.widgets.select({ projectId, region: 'center', widgetId: 'preview' });
    expect(getCenter(store).activeInstanceId).toBe('preview');

    openImageInEdit(createTarget(store), IMAGE);

    expect(getCenter(store).activeInstanceId).toBe('instruct:center');
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('instruct');
    expect(takeInstructEntry(projectId)).toEqual({ image: IMAGE, projectId });
  });

  it('switches to the Edit preset from a layout without the chat', async () => {
    openImageInEdit(createTarget(store), IMAGE);
    await vi.waitFor(() => expect(store.getSnapshot().activeProject.layout.presetId).toBe('instruct'));

    expect(getCenter(store).activeInstanceId).toBe('instruct:center');
  });

  it('opens a closed chat on the Edit preset without applying the preset again', async () => {
    const projectId = store.getSnapshot().activeProject.id;
    const activate = vi.spyOn(store.commands.layout, 'activatePreset');

    await store.commands.layout.activatePreset('instruct');
    activate.mockClear();
    store.commands.widgets.toggle({ projectId, region: 'center', widgetId: 'instruct:center' });
    expect(getCenter(store).instanceIds).not.toContain('instruct:center');

    openImageInEdit(createTarget(store), IMAGE);

    const center = getCenter(store);
    const active = store.getSnapshot().activeProject.widgetInstances[center.activeInstanceId!];

    expect(active?.typeId).toBe('instruct');
    expect(activate).not.toHaveBeenCalled();
    expect(takeInstructEntry(projectId)).toEqual({ image: IMAGE, projectId });
  });

  it('opens a center chat on the Edit preset when only a floating instance is left', async () => {
    const projectId = store.getSnapshot().activeProject.id;
    const activate = vi.spyOn(store.commands.layout, 'activatePreset');

    await store.commands.layout.activatePreset('instruct');
    store.commands.widgets.float('instruct:center');
    expect(Object.keys(store.getSnapshot().activeProject.floatingWidgets ?? {})).toContain('instruct:center');
    activate.mockClear();

    openImageInEdit(createTarget(store), IMAGE);

    const center = getCenter(store);

    expect(store.getSnapshot().activeProject.widgetInstances[center.activeInstanceId!]?.typeId).toBe('instruct');
    expect(center.activeInstanceId).not.toBe('instruct:center');
    expect(activate).not.toHaveBeenCalled();
    expect(takeInstructEntry(projectId)).toEqual({ image: IMAGE, projectId });
  });

  it('does nothing to the layout when a chat is already the center view', async () => {
    const projectId = store.getSnapshot().activeProject.id;

    await store.commands.layout.activatePreset('instruct');
    const before = store.getSnapshot().activeProject.widgetRegions;

    openImageInEdit(createTarget(store), IMAGE);

    expect(store.getSnapshot().activeProject.widgetRegions).toBe(before);
    expect(takeInstructEntry(projectId)).not.toBeNull();
  });

  it('leaves a project the user has moved on from alone', () => {
    const first = store.getSnapshot().activeProject.id;
    const target = createTarget(store, first);

    store.commands.projects.switchTo(store.commands.projects.create().id);
    openImageInEdit(target, IMAGE);

    expect(store.getSnapshot().projects.find((project) => project.id === first)?.layout.presetId).not.toBe('instruct');
    expect(takeInstructEntry(first)).toBeNull();
  });
});
