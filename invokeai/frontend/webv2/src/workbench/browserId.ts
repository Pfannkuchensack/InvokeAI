import { createUuid } from '@platform/browser/randomUuid';

const STORAGE_KEY = 'invokeai-webv2-browser-id';

let memoryId: string | undefined;

/** Names this browser, which all its tabs share, so a pending Edit node can say whose run will settle it. */
export const getBrowserId = (): string => {
  try {
    const stored = globalThis.localStorage.getItem(STORAGE_KEY);

    if (stored) {
      return stored;
    }

    const created = createUuid();

    globalThis.localStorage.setItem(STORAGE_KEY, created);

    return created;
  } catch {
    memoryId ??= createUuid();

    return memoryId;
  }
};
