/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { InstructUiAdapter } from './InstructUiContext';

import { InstructHeaderActions } from './InstructHeaderActions';
import { InstructUiProvider } from './InstructUiContext';

const i18n = createInstance();

void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: { en: {} } });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const render = (region: string, overrides: Partial<InstructUiAdapter> = {}) => {
  const adapter = { canReset: true, hasUnreadableHistory: false, reset: vi.fn(), stepRuns: {}, ...overrides };

  act(() => {
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <InstructUiProvider adapter={adapter as unknown as InstructUiAdapter}>
            <InstructHeaderActions region={region} />
          </InstructUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return adapter;
};

const getButton = () => container.querySelector<HTMLButtonElement>('button[aria-label="widgets.instruct.resetTitle"]');
const click = (element: Element | null | undefined) =>
  act(() => element?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
const getDialogButton = (key: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button, [role="dialog"] button')].find(
    (button) => button.textContent === key
  );

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe('Edit reset action', () => {
  it('asks first, then resets; cancel leaves everything alone', async () => {
    const adapter = render('center');

    click(getButton());
    await vi.waitFor(() => expect(document.body.textContent).toContain('widgets.instruct.resetBody'));
    click(getDialogButton('common.cancel'));

    await vi.waitFor(() => expect(getDialogButton('widgets.instruct.resetConfirm')).toBeUndefined());
    expect(adapter.reset).not.toHaveBeenCalled();

    click(getButton());
    await vi.waitFor(() => expect(getDialogButton('widgets.instruct.resetConfirm')).toBeDefined());
    click(getDialogButton('widgets.instruct.resetConfirm'));

    expect(adapter.reset).toHaveBeenCalledTimes(1);
  });

  it('warns about steps still running', async () => {
    render('center', { stepRuns: { step: 'running' } });
    click(getButton());

    await vi.waitFor(() => expect(document.body.textContent).toContain('widgets.instruct.resetBodyRunning'));
  });

  it('is disabled with nothing to reset, and absent from the settings panel', () => {
    render('center', { canReset: false });
    expect(getButton()?.disabled).toBe(true);

    render('left');
    expect(getButton()).toBeNull();
  });

  it('reports an unreadable history without asking', () => {
    const adapter = render('center', { hasUnreadableHistory: true });

    click(getButton());

    expect(adapter.reset).toHaveBeenCalledTimes(1);
  });
});
