import type { QueueEditLink } from '@features/queue/contracts';

import type { Project } from './projectContracts';
import type { ProjectEditTree } from './projectEditTree';
import type { WorkbenchQueueItem } from './queueHistoryContracts';

import {
  cancelEditNode,
  completeEditNode,
  failEditNode,
  getEditableEditTree,
  getEditNode,
  reopenEditNode,
} from './projectEditTree';

/**
 * Settles the Edit tree's pending nodes from the queue. A run is tied to its step by `snapshot.backendSubmission.edit`,
 * fenced by project id and tree root: a tree replaced since the run was queued takes none of its results. Tree
 * operations throw for input the tree parser would refuse; nothing here lets that escape, the tree stays as it was
 * and the failure comes back to be reported.
 */

export interface EditRoutingResult {
  /** Set when a tree operation refused its input; the tree is unchanged by that settle. */
  failure?: string;
  project: Project;
}

const NO_RESULT_ERROR = 'The run finished without an image for this variant.';
const FAILED_ERROR = 'The run failed.';

const getEditLink = (item: WorkbenchQueueItem): QueueEditLink | undefined => {
  // Read defensively, as the other snapshot readers are: a snapshot may lack its submission.
  const submission = (item.snapshot as Partial<WorkbenchQueueItem['snapshot']>).backendSubmission;

  return submission?.kind === 'generate' ? submission.edit : undefined;
};

const isLiveQueueItem = (item: WorkbenchQueueItem): boolean =>
  item.status === 'pending' || item.status === 'running' || item.cancellationPending === true;

const guard = (project: Project, change: (tree: ProjectEditTree) => Project): EditRoutingResult => {
  const tree = getEditableEditTree(project.editTree);

  if (!tree) {
    return { project };
  }

  try {
    return { project: change(tree) };
  } catch (error) {
    return { failure: error instanceof Error ? error.message : String(error), project };
  }
};

const withTree = (project: Project, tree: ProjectEditTree): Project =>
  tree === project.editTree ? project : { ...project, editTree: tree };

/**
 * What one ended queue item settles. Each image goes to the node of the backend item that made it; a variant without
 * an image is canceled where its item was, else failed.
 */
const settleQueueItem = (tree: ProjectEditTree, item: WorkbenchQueueItem): ProjectEditTree => {
  const link = getEditLink(item);

  if (!link || link.rootNodeId !== tree.rootNodeId || item.status === 'pending' || item.status === 'running') {
    return tree;
  }

  const pendingIndexes = link.nodeIds.flatMap((nodeId, index) =>
    getEditNode(tree, nodeId)?.status === 'pending' ? [index] : []
  );

  if (pendingIndexes.length === 0) {
    return tree;
  }

  const taken = new Set(tree.nodes.flatMap((node) => (node.imageName ? [node.imageName] : [])));
  const images = (item.resultImages ?? []).filter((image) => !taken.has(image.imageName));
  const itemIds = link.variantItemIds;
  const imageFor = (index: number, order: number): string | undefined =>
    itemIds
      ? images.find(
          (image) => image.sourceBackendItemId !== undefined && itemIds[index]?.includes(image.sourceBackendItemId)
        )?.imageName
      : // Without the backend item ids (a run recorded before they were kept) the images follow the variants' order.
        images[order]?.imageName;

  return pendingIndexes.reduce((next, index, order) => {
    const nodeId = link.nodeIds[index]!;
    const imageName = imageFor(index, order);

    if (imageName !== undefined) {
      return completeEditNode(next, nodeId, imageName);
    }

    const backendItemId = itemIds?.[index]?.at(-1);
    const isCancelled =
      item.status === 'cancelled' ||
      (backendItemId !== undefined && item.cancelledBackendItemIds?.includes(backendItemId));

    return isCancelled
      ? cancelEditNode(next, nodeId)
      : failEditNode(next, nodeId, item.status === 'failed' ? (item.error ?? FAILED_ERROR) : NO_RESULT_ERROR);
  }, tree);
};

/** Settles every pending node whose run has ended: images into completed nodes, the error into failed ones. */
export const settleEditTreeFromQueue = (project: Project): EditRoutingResult =>
  guard(project, (tree) =>
    tree.nodes.some((node) => node.status === 'pending')
      ? withTree(project, project.queue.items.reduce(settleQueueItem, tree))
      : project
  );

/**
 * A pending node needs a run that can still deliver it. Once the project's queue state is known, this browser's pending
 * nodes without a live queue item become canceled. Another browser's nodes (the tree is shared) and nodes written
 * without an origin are never touched: their runs are in a journal this browser cannot read.
 */
export const settleStaleEditNodes = (project: Project, origin: string): EditRoutingResult => {
  // Ended runs deliver first, so a result that arrived while the project was unloaded is not lost to the rule.
  const settled = settleEditTreeFromQueue(project);

  if (settled.failure) {
    return settled;
  }

  const live = new Set(
    settled.project.queue.items.flatMap((item) => {
      const link = getEditLink(item);

      return link && isLiveQueueItem(item) ? link.nodeIds : [];
    })
  );

  return guard(settled.project, (tree) =>
    withTree(
      settled.project,
      tree.nodes.reduce(
        (next, node) =>
          node.status === 'pending' && node.origin === origin && !live.has(node.id)
            ? cancelEditNode(next, node.id)
            : next,
        tree
      )
    )
  );
};

/**
 * The user stops waiting for a step whose run this browser does not hold (another browser started it, or its id was
 * lost): its pending nodes without a live queue item here become canceled. The run, if it still exists, is not stopped.
 */
export const abandonEditStep = (project: Project, stepId: string): EditRoutingResult => {
  const live = new Set(
    project.queue.items.flatMap((item) => {
      const link = getEditLink(item);

      return link && isLiveQueueItem(item) ? link.nodeIds : [];
    })
  );

  return guard(project, (tree) =>
    withTree(
      project,
      tree.nodes.reduce(
        (next, node) =>
          node.status === 'pending' && node.stepId === stepId && !live.has(node.id)
            ? cancelEditNode(next, node.id)
            : next,
        tree
      )
    )
  );
};

const updateLink = (project: Project, queueItemId: string, change: (link: QueueEditLink) => QueueEditLink): Project => {
  const item = project.queue.items.find((candidate) => candidate.id === queueItemId);
  const link = item && getEditLink(item);

  return item && link
    ? {
        ...project,
        queue: {
          ...project.queue,
          items: project.queue.items.map((candidate) =>
            candidate === item
              ? {
                  ...item,
                  snapshot: {
                    ...item.snapshot,
                    backendSubmission: {
                      ...item.snapshot.backendSubmission,
                      edit: change(link),
                    } as typeof item.snapshot.backendSubmission,
                  },
                }
              : candidate
          ),
        },
      }
    : project;
};

/** Notes which backend item makes each variant, once the backend has accepted the run. */
export const recordEditVariantItems = (project: Project, queueItemId: string): Project => {
  const ids = project.queue.items.find((candidate) => candidate.id === queueItemId)?.backendItemIds;

  return updateLink(project, queueItemId, (link) =>
    link.variantItemIds || !ids || ids.length !== link.nodeIds.length
      ? link
      : { ...link, variantItemIds: ids.map((id) => [id]) }
  );
};

/** A retried run brings its step's failed and canceled variants back to pending and follows them to their new items. */
export const reopenEditItemNodes = (
  project: Project,
  queueItemId: string,
  retriedItems: readonly { itemId: number; sourceItemId: number }[],
  origin?: string
): EditRoutingResult => {
  const item = project.queue.items.find((candidate) => candidate.id === queueItemId);
  const link = item && getEditLink(item);

  if (!item || !link?.variantItemIds || item.status !== 'running') {
    return { project };
  }

  const itemIds = link.variantItemIds.map((ids) => [...ids]);
  const indexes: number[] = [];

  for (const { itemId, sourceItemId } of retriedItems) {
    const index = itemIds.findIndex((ids) => ids.includes(sourceItemId));

    if (index >= 0) {
      itemIds[index]!.push(itemId);
      indexes.push(index);
    }
  }

  const relinked = updateLink(project, queueItemId, (current) => ({ ...current, variantItemIds: itemIds }));

  return guard(relinked, (tree) =>
    link.rootNodeId !== tree.rootNodeId
      ? relinked
      : withTree(
          relinked,
          indexes.reduce((next, index) => reopenEditNode(next, link.nodeIds[index]!, origin), tree)
        )
  );
};
