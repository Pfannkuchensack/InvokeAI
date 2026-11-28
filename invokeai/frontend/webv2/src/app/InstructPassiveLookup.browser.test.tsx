import type * as GalleryModule from '@features/gallery';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { useInstructUi } from '@features/instruct/ui/InstructUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { addEditStep, completeEditNode, createEditTree } from '@workbench/projectEditTree';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, createRef, useImperativeHandle, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { InstructUiAdapterProvider } from './InstructUiAdapter';

const gallery = vi.hoisted(() => ({ resolveMany: vi.fn() }));

vi.mock('@features/gallery', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryModule>()),
  galleryImages: { resolveMany: gallery.resolveMany },
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getModelsSnapshot: () => ({ models: [], status: 'loading' }),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: [], status: 'loading' }),
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => () => undefined }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(useSyncExternalStore(store.subscribe, store.getSnapshot).activeProject),
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) => {
    const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot);

    return selector(
      getResolvedProjectWidgetValues(snapshot.activeProject, widgetId as 'gallery', snapshot.scopedDefaults)
    );
  },
  useWorkbenchCommands: () => store.commands,
  useWorkbenchQueries: () => store.queries,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let store: WorkbenchInternalStore;
let host: HTMLDivElement;
let root: Root;
const seen = createRef<ReturnType<typeof useInstructUi>>();

const Probe = () => {
  const ui = useInstructUi();

  useImperativeHandle(seen, () => ui, [ui]);

  return null;
};

beforeEach(() => {
  accountLifecycle.activate('user-a');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const buildTree = () => {
  const root0 = createEditTree({ height: 64, imageName: 'source.png', width: 64 }, '2026-11-28T00:00:00.000Z', 'root');
  const step = {
    createdAt: '2026-11-28T00:00:00.000Z',
    height: 64,
    instruction: 'x',
    referenceImages: [{ imageName: 'private-ref.png' }],
    settings: {},
    stepId: 'step',
    variants: [{ id: 'v1', seed: 1 }],
    width: 64,
  };

  return completeEditNode(addEditStep(root0, 'root', step), 'v1', 'result.png');
};

const mount = async (references: unknown[] = []) => {
  const initial = createInitialWorkbenchState();
  const tree = buildTree();

  store = createWorkbenchStore({
    ...initial,
    projects: initial.projects.map((project) => ({ ...project, editTree: tree })),
  });
  store.commands.widgets.patchValues('instruct', { instruction: 'make it red', references });
  await act(async () => {
    root.render(
      <QueryClientProvider client={new QueryClient()}>
        <InstructUiAdapterProvider>
          <Probe />
        </InstructUiAdapterProvider>
      </QueryClientProvider>
    );
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 20);
    });
  });
};

const errors = () => store.getSnapshot().notifications.map((notice) => notice.messageKey ?? notice.message);

describe('the Edit adapter looks images up passively', () => {
  it('remembers what the gallery does not return for this session and leaves the stored history untouched', async () => {
    // The gallery answers with none of them: deleted, or not shown to this user, which it cannot tell apart.
    gallery.resolveMany.mockResolvedValue([]);
    await mount();
    const before = JSON.stringify(store.getSnapshot().activeProject.editTree);

    await vi.waitFor(() =>
      expect([...seen.current!.missingImages].sort()).toEqual(['private-ref.png', 'result.png', 'source.png'])
    );
    expect(JSON.stringify(store.getSnapshot().activeProject.editTree)).toBe(before);
    expect(seen.current!.tree?.nodes.every((node) => node.status !== 'deleted')).toBe(true);
  });

  it('looks the composer references up too, and sends nothing while one is not available', async () => {
    gallery.resolveMany.mockResolvedValue([]);
    await mount([{ height: 64, image_name: 'composer-ref.png', width: 64 }]);
    await vi.waitFor(() => expect(seen.current!.missingImages.has('composer-ref.png')).toBe(true));

    await act(() => seen.current!.submit());

    expect(errors()).toContain('widgets.instruct.referenceUnavailable');
    expect(store.getSnapshot().activeProject.queue.items).toHaveLength(0);
    // The reference stays where it is: the image may only be hidden from this user.
    expect(seen.current!.values.references).toHaveLength(1);
  });

  it('refuses to run a step again whose reference does not resolve, instead of dropping it', async () => {
    gallery.resolveMany.mockResolvedValue([]);
    await mount();

    await act(() => seen.current!.runAgain('v1'));

    expect(errors()).toContain('widgets.instruct.rerunReferenceUnavailable');
    expect(store.getSnapshot().activeProject.queue.items).toHaveLength(0);
  });
});
