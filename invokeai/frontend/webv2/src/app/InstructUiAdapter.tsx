import type { ImageWithDims } from '@features/generation/contracts';
import type { InstructUiAdapter } from '@features/instruct';
import type { EditRun } from '@workbench/editSubmission';
import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';
import type { ReactNode } from 'react';

import { getGalleryAutoAddBoardId } from '@features/gallery/contracts';
import { uploadGalleryFiles } from '@features/gallery/picker';
import { canAddToCachedGalleryBoard, ensureGalleryBoardDestinations } from '@features/gallery/queries';
import {
  getReferenceNumber,
  InstructUiProvider,
  createDefaultInstructValues,
  normalizeInstructValues,
  resolveInstructPatch,
  syncInstructValuesWithModels,
} from '@features/instruct';
import { getModelsSnapshot, useModelsSelector } from '@features/models';
import {
  getArchitectureCapabilitiesSnapshot,
  useArchitectureCapabilitiesSelector,
} from '@features/models/architecturesStore';
import { useQueryClient } from '@tanstack/react-query';
import { submitActiveInvocation } from '@workbench/activeInvocationSubmission';
import { hasUnreadableEditHistory } from '@workbench/editEntry';
import { toInstructTreeView } from '@workbench/editSubmission';
import { formatHotkeyLabel } from '@workbench/hotkeys/keys';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { getEditableEditTree, getEditNode, listEditImageNames } from '@workbench/projectEditTree';
import { describeCanvasInvocationFailure } from '@workbench/widgets/canvas/invoke/canvasInvocationFailure';
import { getProjectWidgetInstance, getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectSelector,
  useWidgetValuesSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type StepRuns = InstructUiAdapter['stepRuns'];

const EMPTY_RUNS: StepRuns = {};
const EMPTY_NAMES: ReadonlySet<string> = new Set();
const PASTE_SHORTCUT = formatHotkeyLabel('mod+v');

/** Each unfinished Edit step's run: waiting for a slot or executing. */
const getStepRuns = (items: readonly WorkbenchQueueItem[]): StepRuns => {
  const runs: Record<string, 'queued' | 'running'> = {};

  for (const item of items) {
    const submission = (item.snapshot as Partial<WorkbenchQueueItem['snapshot']>).backendSubmission;

    if (
      submission?.kind === 'generate' &&
      submission.edit &&
      (item.status === 'pending' || item.status === 'running')
    ) {
      runs[submission.edit.stepId] = item.status === 'running' ? 'running' : 'queued';
    }
  }

  return Object.keys(runs).length === 0 ? EMPTY_RUNS : runs;
};

const areRunsEqual = (left: StepRuns, right: StepRuns): boolean => {
  const keys = Object.keys(left);

  return keys.length === Object.keys(right).length && keys.every((key) => left[key] === right[key]);
};

/** Which models can edit depends on the architecture table too, so nothing is judged before both have arrived. */
const getLoadedModels = () => {
  const snapshot = getModelsSnapshot();

  return snapshot.status === 'loaded' && getArchitectureCapabilitiesSnapshot().status === 'loaded'
    ? snapshot.models
    : undefined;
};

/** The stored values as the installed models allow them; before the catalog loads, as stored. */
const readSyncedValues = (raw: Record<string, unknown>) => {
  const values = normalizeInstructValues(raw);
  const models = getLoadedModels();

  return models ? syncInstructValuesWithModels(values, models) : values;
};

/** Edit owns its instruction, references and settings in widget values and its history in the project's edit tree. */
export const InstructUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector(
    (activeProject) => ({
      editTree: activeProject.editTree,
      projectId: activeProject.id,
      rawValues: getProjectWidgetInstance(activeProject, 'instruct')?.state.values ?? {},
      stepRuns: getStepRuns(activeProject.queue.items),
    }),
    (left, right) =>
      left.projectId === right.projectId &&
      left.rawValues === right.rawValues &&
      left.editTree === right.editTree &&
      areRunsEqual(left.stepRuns, right.stepRuns)
  );
  const { projectId, rawValues, stepRuns } = project;
  const galleryValues = useWidgetValuesSelector('gallery', (values) => values, Object.is);
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const capabilitiesStatus = useArchitectureCapabilitiesSelector((snapshot) => snapshot.status);
  const findInGallery = useFindGalleryItem();
  const { t } = useTranslation();
  const tree = useMemo(() => {
    const editable = getEditableEditTree(project.editTree);

    return editable ? toInstructTreeView(editable) : null;
  }, [project.editTree]);
  const values = useMemo(() => {
    const stored = normalizeInstructValues(rawValues);

    return modelsStatus === 'loaded' && capabilitiesStatus === 'loaded'
      ? syncInstructValuesWithModels(stored, models)
      : stored;
  }, [capabilitiesStatus, models, modelsStatus, rawValues]);

  // What the panel shows right after a reset: the stored defaults, followed to the installed models like any value.
  const canReset = useMemo(() => {
    const defaults =
      modelsStatus === 'loaded' && capabilitiesStatus === 'loaded'
        ? syncInstructValuesWithModels(createDefaultInstructValues(), models)
        : createDefaultInstructValues();

    return project.editTree !== undefined || JSON.stringify(values) !== JSON.stringify(defaults);
  }, [capabilitiesStatus, models, modelsStatus, project.editTree, values]);
  const reset = useCallback<InstructUiAdapter['reset']>(
    () => commands.edit.reset({ projectId }),
    [commands, projectId]
  );
  const patchValues = useCallback<InstructUiAdapter['patchValues']>(
    (patch) =>
      commands.widgets.patchValues(
        'instruct',
        // The panel shows the synced values, so a patch computed from them starts from them too: from the stored ones,
        // a fresh project (no model stored yet) would drop every settings change.
        typeof patch === 'function'
          ? (current) => ({ ...resolveInstructPatch(current, patch, getLoadedModels()) })
          : patch,
        projectId
      ),
    [commands, projectId]
  );
  const readValues = useCallback<InstructUiAdapter['readValues']>(() => {
    const target = queries.getProject(projectId);

    return readSyncedValues(target ? getProjectWidgetValues(target, 'instruct') : {});
  }, [projectId, queries]);
  const reportError = useCallback<InstructUiAdapter['reportError']>(
    (message) => commands.notifications.reportError({ area: 'instruct', message, namespace: 'generation' }),
    [commands]
  );
  const [missing, setMissing] = useState<{ names: ReadonlySet<string>; projectId: string }>({
    names: new Set(),
    projectId,
  });
  const missingImages = missing.projectId === projectId ? missing.names : EMPTY_NAMES;
  const submitRoute = useCallback(
    (editRun?: EditRun) =>
      submitActiveInvocation({
        commands,
        ...(editRun ? { editRun } : {}),
        formatCanvasInvocationFailure: (failure) => describeCanvasInvocationFailure(t, failure),
        formatUpscaleMetadataFailure: (name) => t('widgets.gallery.itemActions.upscale.metadataFailed', { name }),
        getModels: getLoadedModels,
        queries,
        sourceId: 'instruct',
      }),
    [commands, queries, t]
  );
  const submit = useCallback<InstructUiAdapter['submit']>(async () => {
    const target = queries.getProject(projectId);
    const references = (target ? readSyncedValues(getProjectWidgetValues(target, 'instruct')) : values).references;
    const unavailable = references.findIndex((reference) => missingImages.has(reference.image_name));

    // Leaving one out would shift the numbers the instruction refers to ("Image 3"), so the step waits for the user.
    if (unavailable >= 0) {
      reportError(t('widgets.instruct.referenceUnavailable', { number: getReferenceNumber(unavailable) }));

      return;
    }

    await submitRoute();
  }, [missingImages, projectId, queries, reportError, submitRoute, t, values]);
  const cancelStep = useCallback<InstructUiAdapter['cancelStep']>(
    (stepId) => {
      const item = queries.getProject(projectId)?.queue.items.find((candidate) => {
        const submission = (candidate.snapshot as Partial<WorkbenchQueueItem['snapshot']>).backendSubmission;

        return submission?.kind === 'generate' && submission.edit?.stepId === stepId;
      });

      if (item) {
        commands.queue.cancel(projectId, item.id, null);
      }
    },
    [commands, projectId, queries]
  );
  const runAgain = useCallback<InstructUiAdapter['runAgain']>(
    async (nodeId) => {
      const target = queries.getProject(projectId);
      const editable = target ? getEditableEditTree(target.editTree) : null;
      const node = editable ? getEditNode(editable, nodeId) : undefined;

      if (!node?.parentId) {
        return;
      }

      // The step's own references; their sizes are read again.
      let references: ImageWithDims[] = [];

      try {
        const found = node.referenceImages.length
          ? await import('@features/gallery').then(({ galleryImages }) =>
              galleryImages.resolveMany(node.referenceImages.map((reference) => reference.imageName))
            )
          : [];

        references = found.map((entry) => ({ height: entry.height, image_name: entry.imageName, width: entry.width }));

        // A reference that does not resolve is never dropped: the instruction numbers the references it was written with.
        const unavailable = node.referenceImages.findIndex(
          (reference) => !found.some((entry) => entry.imageName === reference.imageName)
        );

        if (unavailable >= 0) {
          reportError(t('widgets.instruct.rerunReferenceUnavailable', { number: getReferenceNumber(unavailable) }));

          return;
        }
      } catch {
        reportError(t('widgets.instruct.rerunFailed'));

        return;
      }

      // The step is built from the node itself; the composer, its references and its continue point are not touched.
      // The settings are the ones the panel holds now.
      await submitRoute({ instruction: node.instruction, parentId: node.parentId, references });
    },
    [projectId, queries, reportError, submitRoute, t]
  );
  const getUploadBoardId = useCallback(async () => {
    const pending = ensureGalleryBoardDestinations(queryClient);

    if (pending) {
      await pending;
    }

    return (
      getGalleryAutoAddBoardId(galleryValues, (boardId) => canAddToCachedGalleryBoard(boardId, queryClient)) ?? 'none'
    );
  }, [galleryValues, queryClient]);
  const uploadFile = useCallback<InstructUiAdapter['uploadFile']>(
    async (file) => {
      const selectedBoardId = await getUploadBoardId();
      const [item] = await uploadGalleryFiles([file], {
        boards: [],
        notifications: {
          add: (notification) => commands.notifications.add(notification),
          reportError: commands.notifications.reportError,
        },
        queryClient,
        selectedBoardId,
        t,
      });

      return item?.kind === 'image' ? { height: item.height, image_name: item.name, width: item.width } : null;
    },
    [commands, getUploadBoardId, queryClient, t]
  );
  const openImage = useCallback<InstructUiAdapter['openImage']>(
    (imageName) => findInGallery({ kind: 'image', name: imageName }),
    [findInGallery]
  );
  const abandonStep = useCallback<InstructUiAdapter['abandonStep']>(
    (stepId) => commands.edit.abandonStep({ projectId, stepId }),
    [commands, projectId]
  );
  const selectChild = useCallback<InstructUiAdapter['selectChild']>(
    (branchId, childId) => commands.edit.selectChild({ branchId, childId, projectId }),
    [commands, projectId]
  );
  const startFromImage = useCallback<InstructUiAdapter['startFromImage']>(
    (image) =>
      commands.edit.start({
        projectId,
        source: { height: image.height, imageName: image.image_name, width: image.width },
      }),
    [commands, projectId]
  );
  // Names already looked up per project, so a name is asked about once, however often the history changes.
  const lookedUp = useRef(new Map<string, Set<string>>());
  const namesKey = useMemo(() => {
    const editable = getEditableEditTree(project.editTree);
    const names = new Set([
      ...(editable ? listEditImageNames(editable) : []),
      // The composer's references too: one that is not available blocks the step.
      ...values.references.map((reference) => reference.image_name),
    ]);

    return [...names].join('|');
  }, [project.editTree, values.references]);

  // Images the gallery does not return (deleted elsewhere, or not shown to this user: the lookup cannot tell) are only
  // remembered for this session. Nothing is written to the project: a hidden image is not a deleted one.
  useEffect(() => {
    const done = lookedUp.current.get(projectId) ?? new Set<string>();
    const names = namesKey ? namesKey.split('|').filter((name) => !done.has(name)) : [];

    if (names.length === 0) {
      return;
    }

    names.forEach((name) => done.add(name));
    lookedUp.current.set(projectId, done);
    void import('@features/gallery')
      .then(async ({ GALLERY_REQUEST_MAX_NAMES, galleryImages }) => {
        const found = new Set<string>();

        for (let offset = 0; offset < names.length; offset += GALLERY_REQUEST_MAX_NAMES) {
          const batch = await galleryImages.resolveMany(names.slice(offset, offset + GALLERY_REQUEST_MAX_NAMES));

          batch.forEach((entry) => found.add(entry.imageName));
        }

        const gone = names.filter((name) => !found.has(name));

        if (gone.length > 0) {
          setMissing((previous) => ({
            names: new Set([...(previous.projectId === projectId ? previous.names : []), ...gone]),
            projectId,
          }));
        }
      })
      // A failed lookup says nothing about the images: ask again when the names next change.
      .catch(() => names.forEach((name) => done.delete(name)));
  }, [namesKey, projectId]);

  const hasUnreadableHistory = hasUnreadableEditHistory({ editTree: project.editTree });
  const adapter = useMemo<InstructUiAdapter>(
    () => ({
      abandonStep,
      canReset,
      hasUnreadableHistory,
      missingImages,
      cancelStep,
      openImage,
      pasteShortcut: PASTE_SHORTCUT,
      patchValues,
      projectId,
      readValues,
      reportError,
      reset,
      runAgain,
      selectChild,
      startFromImage,
      stepRuns,
      submit,
      tree,
      uploadFile,
      values,
    }),
    [
      abandonStep,
      canReset,
      hasUnreadableHistory,
      missingImages,
      cancelStep,
      openImage,
      patchValues,
      projectId,
      readValues,
      reportError,
      reset,
      runAgain,
      selectChild,
      startFromImage,
      stepRuns,
      submit,
      tree,
      uploadFile,
      values,
    ]
  );

  return <InstructUiProvider adapter={adapter}>{children}</InstructUiProvider>;
};
