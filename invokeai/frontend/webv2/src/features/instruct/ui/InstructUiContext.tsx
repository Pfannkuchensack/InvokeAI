import type { ImageWithDims } from '@features/generation/contracts';
import type { InstructTreeView, InstructWidgetValues } from '@features/instruct/core/types';
import type { ReactNode } from 'react';

import { createContext, use, useMemo } from 'react';

/** What the workbench fills in; the callbacks are stable for a project, so tiles and steps can be memoized on them. */
export interface InstructUiActions {
  /** Stops waiting for a step whose run this browser does not hold; the run itself is not stopped. */
  abandonStep(stepId: string): void;
  /** Stops the run of a step still waiting or running. */
  cancelStep(stepId: string): void;
  /** Reveals an edit result in the viewer. */
  openImage(imageName: string): void;
  patchValues(
    values: Partial<InstructWidgetValues> | ((current: InstructWidgetValues) => Partial<InstructWidgetValues>)
  ): void;
  /** The widget's values as they are now, without subscribing, for writes that complete later. */
  readValues(): InstructWidgetValues;
  reportError(message: string): void;
  /** Runs a step's own instruction and references again under its parent, as a new step; the composer is left alone. */
  runAgain(nodeId: string): Promise<void>;
  selectChild(branchId: string, childId: string): void;
  /** Starts the history from an image, replacing the one there (the caller has confirmed). */
  startFromImage(image: ImageWithDims): void;
  /** Queues the instruction in the composer as the next step. */
  submit(): Promise<void>;
  /** Uploads a pasted or dropped file to the gallery's current board; null when it was refused. */
  uploadFile(file: File): Promise<ImageWithDims | null>;
}

/** Workbench supplies this port: the edit tree, the queue and the project belong to it, and features cannot import it. */
export interface InstructUiAdapter extends InstructUiActions {
  /** How the paste shortcut reads on this platform. */
  pasteShortcut: string;
  projectId: string;
  /** Where each unfinished step's run stands. */
  stepRuns: Readonly<Record<string, 'queued' | 'running'>>;
  /** The readable edit tree; null while there is none or the stored one is kept unread. */
  tree: InstructTreeView | null;
  /** The widget's values following the installed models (an uninstalled model gives way to an installed one). */
  values: InstructWidgetValues;
}

const InstructUiContext = createContext<InstructUiAdapter | null>(null);
/** Separate from the values so components that only act do not render on every edit. */
const InstructActionsContext = createContext<InstructUiActions | null>(null);

export const InstructUiProvider = ({ adapter, children }: { adapter: InstructUiAdapter; children: ReactNode }) => {
  const {
    abandonStep,
    cancelStep,
    openImage,
    patchValues,
    readValues,
    reportError,
    runAgain,
    selectChild,
    startFromImage,
    submit,
    uploadFile,
  } = adapter;
  const actions = useMemo<InstructUiActions>(
    () => ({
      abandonStep,
      cancelStep,
      openImage,
      patchValues,
      readValues,
      reportError,
      runAgain,
      selectChild,
      startFromImage,
      submit,
      uploadFile,
    }),
    [
      abandonStep,
      cancelStep,
      openImage,
      patchValues,
      readValues,
      reportError,
      runAgain,
      selectChild,
      startFromImage,
      submit,
      uploadFile,
    ]
  );

  return (
    <InstructActionsContext value={actions}>
      <InstructUiContext value={adapter}>{children}</InstructUiContext>
    </InstructActionsContext>
  );
};

export const useInstructUi = (): InstructUiAdapter => {
  const adapter = use(InstructUiContext);

  if (!adapter) {
    throw new Error('Edit UI requires an App-composed InstructUiProvider.');
  }

  return adapter;
};

export const useInstructActions = (): InstructUiActions => {
  const actions = use(InstructActionsContext);

  if (!actions) {
    throw new Error('Edit UI requires an App-composed InstructUiProvider.');
  }

  return actions;
};
