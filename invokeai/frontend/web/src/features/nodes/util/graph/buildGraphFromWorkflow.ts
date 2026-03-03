import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { omit, reduce } from 'es-toolkit/compat';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import type { Templates } from 'features/nodes/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { BoardFieldInputInstance } from 'features/nodes/types/field';
import { isBoardFieldInputInstance, isBoardFieldInputTemplate } from 'features/nodes/types/field';
import { isBatchNodeType, isGeneratorNodeType } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { isWorkflowInvocationNode } from 'features/nodes/types/workflow';
import type { AnyInvocation, Graph } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

const log = logger('workflows');

const getBoardFieldFromValue = (value: unknown, state: RootState): BoardField | undefined => {
  if (value === 'auto' || !value) {
    const autoAddBoardId = selectAutoAddBoardId(state);
    if (autoAddBoardId === 'none') {
      return undefined;
    }
    return { board_id: autoAddBoardId };
  }
  if (value === 'none') {
    return undefined;
  }
  return value as BoardField;
};

/**
 * Builds a graph from a WorkflowV3 object and templates, without requiring Redux state
 * for the nodes/edges. This allows running saved workflows without loading them into the editor.
 *
 * @param workflow The workflow to build a graph from
 * @param templates The node templates for type resolution
 * @param state The Redux state for resolving board fields and other state-dependent values
 */
export const buildGraphFromWorkflow = (
  workflow: WorkflowV3,
  templates: Templates,
  state: RootState
): Required<Graph> => {
  const { nodes, edges } = workflow;

  // Filter to invocation nodes only, excluding batch and generator nodes
  const invocationNodes = nodes.filter(isWorkflowInvocationNode);
  const executableNodes = invocationNodes.filter(
    (node) => !isBatchNodeType(node.data.type) && !isGeneratorNodeType(node.data.type)
  );

  // Reduce workflow nodes into invocation graph nodes
  const parsedNodes = executableNodes.reduce<NonNullable<Graph['nodes']>>((nodesAccumulator, node) => {
    const { id, data } = node;
    const { type, inputs, isIntermediate } = data;

    const nodeTemplate = templates[type];
    if (!nodeTemplate) {
      log.warn({ id, type }, 'Node template not found for workflow node!');
      return nodesAccumulator;
    }

    // Transform each node's inputs to simple key-value pairs
    const transformedInputs = reduce(
      inputs,
      (inputsAccumulator, input, name) => {
        const fieldTemplate = nodeTemplate.inputs[name];
        if (!fieldTemplate) {
          log.warn({ id, name }, 'Field template not found!');
          return inputsAccumulator;
        }
        if (isBoardFieldInputTemplate(fieldTemplate) && isBoardFieldInputInstance(input as BoardFieldInputInstance)) {
          inputsAccumulator[name] = getBoardFieldFromValue((input as BoardFieldInputInstance).value, state);
        } else {
          inputsAccumulator[name] = (input as { value: unknown }).value;
        }
        return inputsAccumulator;
      },
      {} as Record<Exclude<string, 'id' | 'type'>, unknown>
    );

    // Add reserved use_cache
    transformedInputs['use_cache'] = data.useCache;

    const graphNode = {
      type,
      id,
      ...transformedInputs,
      is_intermediate: isIntermediate,
    };

    Object.assign(nodesAccumulator, { [id]: graphNode });
    return nodesAccumulator;
  }, {});

  const executableNodeIds = executableNodes.map(({ id }) => id);

  // Filter edges: skip collapsed edges and edges to/from non-executable nodes
  const filteredEdges = edges.filter(
    (edge) =>
      edge.type !== 'collapsed' && executableNodeIds.includes(edge.source) && executableNodeIds.includes(edge.target)
  );

  // Convert edges to graph format
  const parsedEdges = filteredEdges.reduce<NonNullable<Graph['edges']>>((edgesAccumulator, edge) => {
    const { source, target, sourceHandle, targetHandle } = edge;

    if (!sourceHandle || !targetHandle) {
      log.warn({ source, target, sourceHandle, targetHandle }, 'Missing source or target handle for edge');
      return edgesAccumulator;
    }

    edgesAccumulator.push({
      source: { node_id: source, field: sourceHandle },
      destination: { node_id: target, field: targetHandle },
    });
    return edgesAccumulator;
  }, []);

  // Omit inputs that have edges connected (edge value takes precedence)
  parsedEdges.forEach((edge) => {
    const destinationNode = parsedNodes[edge.destination.node_id];
    if (destinationNode) {
      const field = edge.destination.field;
      parsedNodes[edge.destination.node_id] = omit(destinationNode, field) as AnyInvocation;
    }
  });

  return {
    id: uuidv4(),
    nodes: parsedNodes,
    edges: parsedEdges,
  };
};
