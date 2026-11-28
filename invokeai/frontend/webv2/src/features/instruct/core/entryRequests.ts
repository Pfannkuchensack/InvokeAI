import type { ImageWithDims } from '@features/generation/contracts';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

import type { InstructWidgetValues } from './types';

/**
 * A request from outside the Edit view (a gallery action, a canvas bbox, a recall) to open an image in the Edit history.
 * The view answers it: it starts, jumps or asks before replacing, exactly as for an image chosen in the view.
 */
export interface InstructEntryRequest {
  /** The image the history opens on, or the one an existing history jumps to. */
  image: ImageWithDims;
  projectId: string;
  /** Also makes the image's node the continue point, which a recall needs even at the end of a branch. */
  continueFromImage?: boolean;
  /** Written once the history is settled, so a restart cannot clear them; also when the user keeps the old history. */
  values?: Partial<InstructWidgetValues>;
  /** What applies instead when the user keeps the history the image is not in (a size computed against its tip). */
  valuesIfKept?: Partial<InstructWidgetValues>;
}

let pending: InstructEntryRequest | null = null;
const listeners = new Set<() => void>();

registerAccountOwnedResource({
  clear: () => {
    pending = null;
  },
  name: 'instruct-entry-request',
});

/** Held until an Edit view of that project takes it; a newer request replaces an untaken one. */
export const requestInstructEntry = (request: InstructEntryRequest): void => {
  pending = request;
  listeners.forEach((listener) => listener());
};

export const takeInstructEntry = (projectId: string): InstructEntryRequest | null => {
  const taken = pending;

  // A view only shows the active project, so a request for another one is stale: its project was left or closed.
  pending = null;

  return taken?.projectId === projectId ? taken : null;
};

export const subscribeInstructEntry = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => listeners.delete(listener);
};
