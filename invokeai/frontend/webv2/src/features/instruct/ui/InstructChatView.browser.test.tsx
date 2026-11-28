import type { InstructNode, InstructTreeView } from '@features/instruct/core/types';

/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryHostProvider, type GalleryHost } from '@features/gallery/picker';
import { satisfiedSettingsFor } from '@features/generation/core/graphCoverage.testing';
import { normalizeInstructValues } from '@features/instruct/core/values';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { InstructUiAdapter } from './InstructUiContext';

import { InstructChatView } from './InstructChatView';
import { InstructUiProvider } from './InstructUiContext';

seedArchitectureCapabilities();

const i18n = createInstance();

void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: { en: {} } });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const { model, settings } = satisfiedSettingsFor('flux2', {
  label: 'dev',
  overrides: { format: 'diffusers', variant: 'dev' },
});
const EDIT_VALUES = { instruction: '', model, seedMode: 'fixed', settings, variants: 1 };

const node = (
  id: string,
  parentId: string | null,
  stepId: string,
  extra: Partial<InstructNode> = {}
): InstructNode => ({
  height: 64,
  id,
  imageName: `${id}.png`,
  instruction: `instruction ${stepId}`,
  parentId,
  referenceImages: [],
  seed: 1,
  status: 'completed',
  stepId,
  width: 64,
  ...extra,
});

const createTree = (nodes: InstructNode[], pathIds: string[]): InstructTreeView => ({
  nodes,
  path: pathIds.map((id) => nodes.find((candidate) => candidate.id === id)!),
  rootNodeId: nodes[0]!.id,
  selection: {},
});

const galleryHost: GalleryHost = {
  galleryValues: {} as GalleryHost['galleryValues'],
  notifications: { add: vi.fn(), reportError: vi.fn() },
  projectName: '',
};
const queryClient = new QueryClient();

let container: HTMLDivElement;
let root: Root;
let adapter: InstructUiAdapter;

const values = (extra: Record<string, unknown> = {}) => normalizeInstructValues({ ...EDIT_VALUES, ...extra });

const createAdapter = (overrides: Partial<InstructUiAdapter> = {}): InstructUiAdapter => ({
  abandonStep: vi.fn(),
  cancelStep: vi.fn(),
  openImage: vi.fn(),
  patchValues: vi.fn(),
  projectId: 'project-1',
  pasteShortcut: 'Ctrl+V',
  readValues: () => values(),
  reportError: vi.fn(),
  runAgain: vi.fn(() => Promise.resolve()),
  selectChild: vi.fn(),
  startFromImage: vi.fn(),
  stepRuns: {},
  submit: vi.fn(() => Promise.resolve()),
  tree: null,
  uploadFile: vi.fn(() => Promise.resolve(null)),
  values: values(),
  ...overrides,
});

const render = (overrides: Partial<InstructUiAdapter> = {}) => {
  adapter = createAdapter(overrides);
  act(() => {
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <GalleryHostProvider host={galleryHost}>
              <InstructUiProvider adapter={adapter}>
                <DndContext>
                  <InstructChatView />
                </DndContext>
              </InstructUiProvider>
            </GalleryHostProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const getComposer = () => container.querySelector<HTMLTextAreaElement>('textarea')!;

const type = (textarea: HTMLTextAreaElement, text: string) => {
  const setValue = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!;

  act(() => {
    setValue.call(textarea, text);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
  });
};

const press = (textarea: HTMLTextAreaElement, init: KeyboardEventInit) =>
  act(() => {
    textarea.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }));
  });

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe('Edit composer', () => {
  const finished = [node('root', null, 'root'), node('a', 'root', 's1')];

  it('sends the typed instruction on Enter and keeps Shift+Enter for a new line', () => {
    render({ tree: createTree(finished, ['root', 'a']) });
    const textarea = getComposer();

    type(textarea, 'make it red');
    press(textarea, { key: 'Enter', shiftKey: true });

    expect(adapter.submit).not.toHaveBeenCalled();

    press(textarea, { key: 'Enter' });

    // The instruction reaches the project before the step is queued.
    expect(adapter.patchValues).toHaveBeenCalledWith({ instruction: 'make it red' });
    expect(adapter.submit).toHaveBeenCalledTimes(1);
    expect(vi.mocked(adapter.patchValues).mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(adapter.submit).mock.invocationCallOrder[0]!
    );
  });

  const single = satisfiedSettingsFor('flux2', {
    label: 'klein',
    overrides: { format: 'gguf_quantized', variant: 'klein_9b' },
  });

  it('does not send while a component the model needs is missing', () => {
    render({
      tree: createTree(finished, ['root', 'a']),
      values: values({ model: single.model, settings: { ...single.settings, qwen3EncoderModel: null, vae: null } }),
    });
    const textarea = getComposer();

    type(textarea, 'make it red');
    press(textarea, { key: 'Enter' });

    expect(adapter.submit).not.toHaveBeenCalled();
    const send = container.querySelector('button[aria-label="widgets.instruct.send"]')!;
    expect(send.getAttribute('aria-disabled')).toBe('true');
    expect(document.getElementById(send.getAttribute('aria-describedby')!)?.textContent).toMatch(/^Edit needs a /);
  });

  it('is locked while the step at the end of the branch runs, with its progress and a way to cancel', () => {
    render({
      stepRuns: { s1: 'running' },
      tree: createTree([finished[0]!, node('a', 'root', 's1', { imageName: null, status: 'pending' })], ['root', 'a']),
    });
    const textarea = getComposer();

    type(textarea, 'ignored');
    press(textarea, { key: 'Enter' });

    expect(textarea.readOnly).toBe(true);
    expect(adapter.submit).not.toHaveBeenCalled();

    const cancel = Array.from(container.querySelectorAll('button')).filter(
      (button) => button.textContent === 'common.cancel'
    );

    expect(cancel.length).toBeGreaterThan(0);
    act(() => cancel[0]!.click());
    expect(adapter.cancelStep).toHaveBeenCalledWith('s1');
  });

  it('stops waiting for a step this browser holds no run for only after a confirmation', async () => {
    render({
      tree: createTree([finished[0]!, node('a', 'root', 's1', { imageName: null, status: 'pending' })], ['root', 'a']),
    });
    const cancel = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'common.cancel'
    )!;

    act(() => cancel.click());
    expect(adapter.cancelStep).not.toHaveBeenCalled();
    expect(adapter.abandonStep).not.toHaveBeenCalled();

    await vi.waitFor(() =>
      expect(
        Array.from(document.querySelectorAll('button')).some(
          (button) => button.textContent === 'widgets.instruct.abandonConfirm'
        )
      ).toBe(true)
    );
    const confirm = Array.from(document.querySelectorAll('button')).find(
      (button) => button.textContent === 'widgets.instruct.abandonConfirm'
    )!;

    act(() => confirm.click());
    expect(adapter.abandonStep).toHaveBeenCalledWith('s1');
  });

  it('stays unlocked on another branch while a different step runs', () => {
    const nodes = [
      node('root', null, 'root'),
      node('a', 'root', 's1'),
      node('b', 'root', 's2', { imageName: null, status: 'pending' }),
    ];

    render({ values: values({ continueFromNodeId: 'a' }), tree: createTree(nodes, ['root', 'a']) });

    expect(getComposer().readOnly).toBe(false);
  });
});

describe('Edit history', () => {
  it('moves among the steps under one image with the switcher', () => {
    const nodes = [node('root', null, 'root'), node('a', 'root', 's1'), node('b', 'root', 's2')];

    render({ tree: createTree(nodes, ['root', 'b']) });
    const previous = container.querySelector<HTMLButtonElement>('[aria-label="widgets.instruct.previousStep"]')!;

    expect(container.textContent).toContain('2 / 2');
    act(() => previous.click());
    expect(adapter.selectChild).toHaveBeenCalledWith('root', 'a');
  });

  it('shows a failed variant with its error and runs it again from its parent', () => {
    const nodes = [
      node('root', null, 'root'),
      node('a', 'root', 's1', { error: 'out of memory', imageName: null, status: 'failed' }),
    ];

    render({ tree: createTree(nodes, ['root', 'a']) });
    const again = Array.from(container.querySelectorAll('button')).find((button) =>
      button.textContent?.includes('widgets.instruct.runAgain')
    )!;

    expect(container.textContent).toContain('out of memory');
    act(() => again.click());
    expect(adapter.runAgain).toHaveBeenCalledWith('a');
  });
});

describe('Edit empty state', () => {
  it('starts the history from a pasted image', async () => {
    const image = { height: 480, image_name: 'pasted.png', width: 640 };

    render({ uploadFile: vi.fn(() => Promise.resolve(image)) });
    const transfer = new DataTransfer();

    transfer.items.add(new File(['x'], 'pasted.png', { type: 'image/png' }));
    await act(async () => {
      document.body.dispatchEvent(
        new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: transfer })
      );
      await Promise.resolve();
    });

    expect(adapter.uploadFile).toHaveBeenCalledTimes(1);
    expect(adapter.startFromImage).toHaveBeenCalledWith(image);
  });
});
