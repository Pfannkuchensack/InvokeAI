import type * as GalleryModule from '@features/gallery';

import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it, vi } from 'vitest';

import { prepareEditStep } from './editSubmission';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const gallery = vi.hoisted(() => ({ resolveMany: vi.fn() }));

vi.mock('@features/gallery', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryModule>()),
  galleryImages: { resolveMany: gallery.resolveMany },
}));

seedArchitectureCapabilities();

const createProject = () => {
  const { model, settings } = satisfiedSettingsFor('flux2', {
    label: 'dev',
    overrides: { format: 'diffusers', variant: 'dev' },
  });
  let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'instruct', type: 'applyPreset' });

  state = workbenchReducer(state, {
    projectId: state.activeProjectId,
    source: { height: 640, imageName: 'source.png', width: 480 },
    type: 'startEditTree',
  });
  state = workbenchReducer(state, {
    projectId: state.activeProjectId,
    type: 'patchWidgetValues',
    values: { instruction: 'make it evening', model, settings },
    widgetId: 'instruct',
  });

  return state.projects.find((project) => project.id === state.activeProjectId)!;
};

const prepare = () =>
  prepareEditStep({ defaults: {}, models: undefined, project: createProject(), runtime: { noiseDevice: 'cpu' } });

describe('prepareEditStep image check', () => {
  it('refuses a parent whose image the gallery no longer has', async () => {
    gallery.resolveMany.mockResolvedValue([]);

    await expect(prepare()).rejects.toThrow('The image to continue from is not available.');
  });

  it('goes on when the image exists, or when the lookup itself fails', async () => {
    gallery.resolveMany.mockResolvedValueOnce([{ imageName: 'source.png' }]);
    await expect(prepare()).resolves.toMatchObject({ parentId: expect.any(String) });

    gallery.resolveMany.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(prepare()).resolves.toMatchObject({ parentId: expect.any(String) });
  });
});
