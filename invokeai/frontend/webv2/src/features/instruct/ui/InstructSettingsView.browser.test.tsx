/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryHostProvider, type GalleryHost } from '@features/gallery/picker';
import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import { GenerationUiProvider, type GenerationUiAdapter } from '@features/generation/react';
import { normalizeInstructValues } from '@features/instruct/core/values';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { InstructUiAdapter } from './InstructUiContext';

import { InstructSettingsView } from './InstructSettingsView';
import { InstructUiProvider } from './InstructUiContext';

seedArchitectureCapabilities();

const i18n = createInstance();

void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: { en: {} } });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const galleryHost: GalleryHost = {
  galleryValues: {} as GalleryHost['galleryValues'],
  notifications: { add: vi.fn(), reportError: vi.fn() },
  projectName: '',
};
const queryClient = new QueryClient();
/** The Generate building blocks of the section read only the panel-section preferences from the adapter. */
const GENERATION_UI = {
  sectionPreferences: { sectionsOpen: { 'instruct-advanced': true }, setSectionsOpen: vi.fn() },
} as unknown as GenerationUiAdapter;

let container: HTMLDivElement;
let root: Root;

const renderWith = (base: 'flux2' | 'qwen-image', overrides: Record<string, unknown>) => {
  const { model, settings } = satisfiedSettingsFor(base, { label: 'm', overrides } as never);
  const values = normalizeInstructValues({ instruction: '', model, settings });
  const patchValues = vi.fn();
  const adapter: InstructUiAdapter = {
    abandonStep: vi.fn(),
    cancelStep: vi.fn(),
    hasUnreadableHistory: false,
    missingImages: new Set<string>(),
    openImage: vi.fn(),
    pasteShortcut: 'Ctrl+V',
    patchValues,
    projectId: 'project-1',
    readValues: () => values,
    reportError: vi.fn(),
    runAgain: vi.fn(() => Promise.resolve()),
    selectChild: vi.fn(),
    startFromImage: vi.fn(),
    stepRuns: {},
    submit: vi.fn(() => Promise.resolve()),
    tree: null,
    uploadFile: vi.fn(() => Promise.resolve(null)),
    values,
  };

  act(() => {
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <GalleryHostProvider host={galleryHost}>
              <GenerationUiProvider adapter={GENERATION_UI}>
                <InstructUiProvider adapter={adapter}>
                  <DndContext>
                    <InstructSettingsView />
                  </DndContext>
                </InstructUiProvider>
              </GenerationUiProvider>
            </GalleryHostProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return { patchValues, values };
};

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe('Edit settings Advanced section', () => {
  it('offers Qwen-Image-Edit its schedule shift and encoder quantization', () => {
    renderWith('qwen-image', { format: 'diffusers', variant: 'edit' });

    expect(container.textContent).toContain('widgets.generate.timestepShift');
    expect(container.textContent).toContain('widgets.generate.encoderQuantization');
  });

  it('sets the shift override and the encoder quantization in the step settings', async () => {
    const { patchValues, values } = renderWith('qwen-image', { format: 'diffusers', variant: 'edit' });
    const patched = () => {
      const patch = patchValues.mock.calls.at(-1)?.[0] as (current: typeof values) => {
        settings: { qwenImageEncoderQuantization: string; qwenImageShift: number | null };
      };

      return patch(values).settings;
    };

    expect(values.settings).toMatchObject({ qwenImageEncoderQuantization: 'none', qwenImageShift: null });

    // The shift switch starts the override at the Lightning value.
    await userEvent.click(page.getByText('widgets.generate.timestepShift'));
    expect(patched().qwenImageShift).toBe(3);

    // The encoder quantization picks a listed choice.
    await userEvent.click(page.getByRole('combobox', { name: 'widgets.generate.encoderQuantization' }));
    await userEvent.click(page.getByRole('option', { name: 'widgets.generate.encoderQuantizationNf4' }));
    expect(patched().qwenImageEncoderQuantization).toBe('nf4');
  });

  it('is not shown for the other families', () => {
    renderWith('flux2', { format: 'diffusers', variant: 'dev' });

    expect(container.textContent).not.toContain('widgets.generate.timestepShift');
    expect(container.textContent).not.toContain('widgets.generate.encoderQuantization');
  });
});
