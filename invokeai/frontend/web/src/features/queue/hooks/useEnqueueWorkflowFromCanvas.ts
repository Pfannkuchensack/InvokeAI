import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { buildGraphFromWorkflow } from 'features/nodes/util/graph/buildGraphFromWorkflow';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { populateCanvasSessionInfoNodes } from 'features/nodes/util/graph/populateCanvasSessionInfo';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import { workflowsApi } from 'services/api/endpoints/workflows';
import type { EnqueueBatchArg } from 'services/api/types';
import { zWorkflowV3 } from 'features/nodes/types/workflow';
import { serializeError } from 'serialize-error';

const log = logger('generation');

/**
 * Enqueues the currently loaded workflow from the canvas, with canvas session info auto-populated.
 */
const enqueueCurrentWorkflowFromCanvas = async (
  store: AppStore,
  templates: Templates,
  prepend: boolean
) => {
  const { dispatch, getState } = store;
  const state = getState();

  // Build graph from the currently loaded workflow in the editor
  const graph = buildNodesGraph(state, templates);

  // Auto-populate canvas_session_info nodes with current canvas state
  populateCanvasSessionInfoNodes(graph, state);

  const nodesState = selectNodesSlice(state);
  const workflow = buildWorkflowWithValidation(nodesState);
  if (workflow) {
    delete workflow.id;
  }

  const batchConfig: EnqueueBatchArg = {
    batch: {
      graph,
      workflow: workflow ?? undefined,
      runs: 1,
      origin: 'canvas',
      destination: 'gallery',
    },
    prepend,
  };

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
      ...enqueueMutationFixedCacheKeyOptions,
      track: false,
    })
  );

  const enqueueResult = await req.unwrap();
  return { batchConfig, enqueueResult };
};

/**
 * Fetches a saved workflow by ID, builds a graph from it, and enqueues with canvas session info.
 */
const enqueueSavedWorkflowFromCanvas = async (
  store: AppStore,
  templates: Templates,
  workflowId: string,
  prepend: boolean
) => {
  const { dispatch, getState } = store;

  // Fetch the workflow definition from the API
  const workflowRecord = await dispatch(
    workflowsApi.endpoints.getWorkflow.initiate(workflowId)
  ).unwrap();

  // Parse the workflow
  const parseResult = zWorkflowV3.safeParse(workflowRecord.workflow);
  if (!parseResult.success) {
    throw new Error('Failed to parse saved workflow');
  }
  const workflow = parseResult.data;

  const state = getState();

  // Build graph from the saved workflow
  const graph = buildGraphFromWorkflow(workflow, templates, state);

  // Auto-populate canvas_session_info nodes with current canvas state
  populateCanvasSessionInfoNodes(graph, state);

  // Remove the workflow ID so it gets embedded rather than referenced
  const workflowForBatch = { ...workflow };
  delete workflowForBatch.id;

  const batchConfig: EnqueueBatchArg = {
    batch: {
      graph,
      workflow: workflowForBatch,
      runs: 1,
      origin: 'canvas',
      destination: 'gallery',
    },
    prepend,
  };

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
      ...enqueueMutationFixedCacheKeyOptions,
      track: false,
    })
  );

  const enqueueResult = await req.unwrap();
  return { batchConfig, enqueueResult };
};

export const useEnqueueWorkflowFromCanvas = () => {
  const store = useAppStore();

  const enqueueCurrentWorkflow = useCallback(
    async (prepend: boolean = false) => {
      try {
        await enqueueCurrentWorkflowFromCanvas(store, $templates.get(), prepend);
        toast({ title: 'Workflow enqueued', status: 'success' });
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Failed to enqueue current workflow from canvas');
        toast({ title: 'Failed to enqueue workflow', status: 'error' });
      }
    },
    [store]
  );

  const enqueueSavedWorkflow = useCallback(
    async (workflowId: string, prepend: boolean = false) => {
      try {
        await enqueueSavedWorkflowFromCanvas(store, $templates.get(), workflowId, prepend);
        toast({ title: 'Workflow enqueued', status: 'success' });
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Failed to enqueue saved workflow from canvas');
        toast({ title: 'Failed to enqueue workflow', status: 'error' });
      }
    },
    [store]
  );

  return { enqueueCurrentWorkflow, enqueueSavedWorkflow };
};
