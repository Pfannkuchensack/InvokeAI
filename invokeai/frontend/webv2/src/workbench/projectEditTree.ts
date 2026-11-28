/**
 * The Edit workspace's history of one source image: a tree of steps, each step's variants as sibling nodes. Nodes
 * persist in the project content; which variant each branch shows (`selection`) is the member's workspace, so
 * `splitStoredEditTree` and `joinStoredEditTree` follow the halves. Every image reference uses the key `imageName`,
 * which the media index reads.
 *
 * Operations return the same tree when they change nothing and throw for input `parseEditTree` would refuse, so a tree
 * they produce always reads back. A stored tree this build cannot read (damaged, or written by a build that knows more)
 * is held as `UnreadableEditTree`: kept as stored and written back unchanged, never edited, never dropped. Unknown
 * fields on a readable tree, its nodes and its references survive the round trip.
 */

export type EditNodeStatus = 'pending' | 'completed' | 'failed' | 'canceled' | 'deleted';

export interface EditTreeReference {
  imageName: string;
  [field: string]: unknown;
}

export interface EditTreeNode {
  id: string;
  /** Null for the root, which is the source image. */
  parentId: string | null;
  /** The variants of one step share it; the root's is its own id. */
  stepId: string;
  /** Null while pending, failed or canceled, and once the image was deleted. */
  imageName: string | null;
  status: EditNodeStatus;
  instruction: string;
  /** The step's generation settings as the Edit widget wrote them, JSON only; this module does not read it. */
  settings: Record<string, unknown>;
  /** The references beside the parent's image, in order. */
  referenceImages: EditTreeReference[];
  seed: number;
  width: number;
  height: number;
  createdAt: string;
  /** Why a failed node failed, as the run reported it. */
  error?: string;
  /** The browser whose run settles a pending node; absent on nodes written without it, which are never auto-canceled. */
  origin?: string;
  [field: string]: unknown;
}

export interface ProjectEditTree {
  rootNodeId: string;
  /** Parents before children, siblings in creation order. */
  nodes: EditTreeNode[];
  /** The active child of each branching node by node id; absent follows the default child. */
  selection: Record<string, string>;
  [field: string]: unknown;
}

/** A stored tree this build does not understand, exactly as stored. */
export interface UnreadableEditTree {
  unreadableEditTree: unknown;
}

/** What `Project.editTree` holds. */
export type StoredEditTree = ProjectEditTree | UnreadableEditTree;

export interface EditStepInput {
  stepId: string;
  instruction: string;
  settings: Record<string, unknown>;
  referenceImages: { imageName: string }[];
  width: number;
  height: number;
  createdAt: string;
  origin?: string;
  /** One pending node per variant. */
  variants: { id: string; seed: number }[];
}

const createEditId = (prefix: string): string =>
  `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

export const createEditNodeId = (): string => createEditId('edit_node');
export const createEditStepId = (): string => createEditId('edit_step');

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;

const isPositiveInteger = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) > 0;

const STATUSES: ReadonlySet<unknown> = new Set<EditNodeStatus>([
  'pending',
  'completed',
  'failed',
  'canceled',
  'deleted',
]);

const parseNode = (candidate: unknown): EditTreeNode | null => {
  if (
    !isRecord(candidate) ||
    !isNonEmptyString(candidate.id) ||
    !(candidate.parentId === null || isNonEmptyString(candidate.parentId)) ||
    !isNonEmptyString(candidate.stepId) ||
    !(candidate.imageName === null || isNonEmptyString(candidate.imageName)) ||
    !STATUSES.has(candidate.status) ||
    typeof candidate.instruction !== 'string' ||
    !isRecord(candidate.settings) ||
    !Array.isArray(candidate.referenceImages) ||
    !candidate.referenceImages.every((reference) => isRecord(reference) && isNonEmptyString(reference.imageName)) ||
    typeof candidate.seed !== 'number' ||
    !Number.isSafeInteger(candidate.seed) ||
    !isPositiveInteger(candidate.width) ||
    !isPositiveInteger(candidate.height) ||
    typeof candidate.createdAt !== 'string' ||
    !(candidate.error === undefined || typeof candidate.error === 'string') ||
    !(candidate.origin === undefined || typeof candidate.origin === 'string')
  ) {
    return null;
  }

  return candidate as EditTreeNode;
};

/** A node an operation built, which must read back; the message names what the operation was doing. */
const assertNode = (candidate: unknown, doing: string): EditTreeNode => {
  const node = parseNode(candidate);

  if (!node) {
    throw new TypeError(`Cannot ${doing}: the node would not read back.`);
  }

  return node;
};

/**
 * Reads a whole tree (nodes, root and selection as one record): one root first, then children after their parents with
 * unique ids; null otherwise. A completed node must name its image, since the tree is how the Edit widget finds it
 * again. Selection entries that name no child of their branch are dropped.
 */
export const parseEditTree = (candidate: unknown): ProjectEditTree | null => {
  if (!isRecord(candidate) || !isNonEmptyString(candidate.rootNodeId) || !Array.isArray(candidate.nodes)) {
    return null;
  }

  const nodes: EditTreeNode[] = [];
  const parentOf = new Map<string, string | null>();

  for (const raw of candidate.nodes) {
    const node = parseNode(raw);
    const isRoot = nodes.length === 0;

    if (
      !node ||
      parentOf.has(node.id) ||
      (isRoot
        ? node.parentId !== null || node.id !== candidate.rootNodeId
        : node.parentId === null || !parentOf.has(node.parentId)) ||
      (node.status === 'completed' && node.imageName === null)
    ) {
      return null;
    }

    parentOf.set(node.id, node.parentId);
    nodes.push(node);
  }

  if (nodes.length === 0) {
    return null;
  }

  const selection = isRecord(candidate.selection)
    ? Object.fromEntries(
        Object.entries(candidate.selection).filter(
          (entry): entry is [string, string] => typeof entry[1] === 'string' && parentOf.get(entry[1]) === entry[0]
        )
      )
    : {};

  return { ...candidate, nodes, rootNodeId: candidate.rootNodeId, selection };
};

/** Wrappers are told apart by identity, never by a key, since stored data may hold any key. */
const unreadableTrees = new WeakSet<object>();

export const isUnreadableEditTree = (stored: StoredEditTree): stored is UnreadableEditTree =>
  unreadableTrees.has(stored);

/** Reads what a document or session holds: a readable tree, or the stored value kept as it is. Absent stays absent. */
export const readStoredEditTree = (stored: unknown): StoredEditTree | undefined => {
  if (stored === undefined) {
    return undefined;
  }

  if (typeof stored === 'object' && stored !== null && unreadableTrees.has(stored)) {
    return stored as UnreadableEditTree;
  }

  const tree = parseEditTree(stored);

  if (tree) {
    return tree;
  }

  const wrapper: UnreadableEditTree = { unreadableEditTree: stored };
  unreadableTrees.add(wrapper);

  return wrapper;
};

/** The tree for editing, or null where the stored one is unreadable and must stay untouched. */
export const getEditableEditTree = (stored: StoredEditTree | undefined): ProjectEditTree | null =>
  stored && !isUnreadableEditTree(stored) ? stored : null;

/** What the document stores for it. */
export const toStoredEditTreeValue = (stored: StoredEditTree): unknown =>
  isUnreadableEditTree(stored) ? stored.unreadableEditTree : stored;

/** Content takes everything but the selection, workspace the selection, whatever else the stored value holds. */
export const splitStoredEditTree = (value: unknown): { content: unknown; workspace: { selection: unknown } | null } => {
  if (!isRecord(value)) {
    return { content: value, workspace: null };
  }

  const { selection, ...content } = value;

  return { content, workspace: selection === undefined ? null : { selection } };
};

/** The inverse of `splitStoredEditTree`. */
export const joinStoredEditTree = (content: unknown, workspace: unknown): unknown =>
  isRecord(content) && isRecord(workspace) && workspace.selection !== undefined
    ? { ...content, selection: workspace.selection }
    : content;

/** A new tree holding only the source image. The caller confirms before it replaces an existing tree. */
export const createEditTree = (
  source: { imageName: string; width: number; height: number },
  createdAt: string,
  id: string = createEditNodeId()
): ProjectEditTree => ({
  nodes: [
    assertNode(
      {
        createdAt,
        height: source.height,
        id,
        imageName: source.imageName,
        instruction: '',
        parentId: null,
        referenceImages: [],
        seed: 0,
        settings: {},
        status: 'completed',
        stepId: id,
        width: source.width,
      },
      'start an edit tree'
    ),
  ],
  rootNodeId: id,
  selection: {},
});

export const getEditNode = (tree: ProjectEditTree, nodeId: string): EditTreeNode | undefined =>
  tree.nodes.find((node) => node.id === nodeId);

export const getEditChildren = (tree: ProjectEditTree, nodeId: string): EditTreeNode[] =>
  tree.nodes.filter((node) => node.parentId === nodeId);

/** The child a branch shows without a choice: the first variant of its newest step. */
const getDefaultChild = (children: readonly EditTreeNode[]): EditTreeNode | undefined => {
  const newestStepId = children.at(-1)?.stepId;

  return children.find((child) => child.stepId === newestStepId);
};

/**
 * Adds the step's variants under `parentId` and shows the first one. Unchanged for an unknown parent or no variants;
 * throws for an id already in the tree, a step id used under another parent, or a node that would not read back.
 */
export const addEditStep = (tree: ProjectEditTree, parentId: string, step: EditStepInput): ProjectEditTree => {
  if (!getEditNode(tree, parentId) || step.variants.length === 0) {
    return tree;
  }

  const taken = new Set(tree.nodes.map((node) => node.id));

  if (tree.nodes.some((node) => node.stepId === step.stepId && node.parentId !== parentId)) {
    throw new TypeError(`Cannot add an edit step: step id "${step.stepId}" belongs to another branch.`);
  }

  const added = step.variants.map(({ id, seed }) => {
    if (taken.has(id)) {
      throw new TypeError(`Cannot add an edit step: node id "${id}" is empty or already in the tree.`);
    }
    taken.add(id);

    return assertNode(
      {
        createdAt: step.createdAt,
        height: step.height,
        id,
        imageName: null,
        instruction: step.instruction,
        parentId,
        ...(step.origin === undefined ? {} : { origin: step.origin }),
        referenceImages: step.referenceImages,
        seed,
        settings: step.settings,
        status: 'pending',
        stepId: step.stepId,
        width: step.width,
      },
      'add an edit step'
    );
  });
  const nodes = [...tree.nodes, ...added];
  const shown = getDefaultChild(nodes.filter((node) => node.parentId === parentId));

  return { ...tree, nodes, selection: { ...tree.selection, [parentId]: shown!.id } };
};

const updateNode = (
  tree: ProjectEditTree,
  nodeId: string,
  doing: string,
  change: (node: EditTreeNode) => EditTreeNode | null
): ProjectEditTree => {
  const index = tree.nodes.findIndex((node) => node.id === nodeId);
  const next = index < 0 ? null : change(tree.nodes[index]!);

  return next
    ? { ...tree, nodes: tree.nodes.map((node, at) => (at === index ? assertNode(next, doing) : node)) }
    : tree;
};

/** A settled node no longer waits for any browser's run, so the id that named one does not stay in shared content. */
const withoutOrigin = (node: EditTreeNode): EditTreeNode => {
  const { origin: _origin, ...rest } = node;

  return rest as EditTreeNode;
};

const settlePending = (status: 'failed' | 'canceled', error?: string) => (node: EditTreeNode) =>
  node.status === 'pending' ? { ...withoutOrigin(node), ...(error === undefined ? {} : { error }), status } : null;

/** A result for a node that was already canceled or failed is dropped. */
export const completeEditNode = (tree: ProjectEditTree, nodeId: string, imageName: string): ProjectEditTree =>
  updateNode(tree, nodeId, 'complete an edit node', (node) =>
    node.status === 'pending' ? { ...withoutOrigin(node), imageName, status: 'completed' } : null
  );

export const failEditNode = (tree: ProjectEditTree, nodeId: string, error?: string): ProjectEditTree =>
  updateNode(tree, nodeId, 'fail an edit node', settlePending('failed', error));

/** A failed or canceled node whose run is being retried is pending again. */
export const reopenEditNode = (tree: ProjectEditTree, nodeId: string, origin?: string): ProjectEditTree =>
  updateNode(tree, nodeId, 'reopen an edit node', (node) => {
    if (node.status !== 'failed' && node.status !== 'canceled') {
      return null;
    }

    const { error: _error, ...rest } = node;

    return { ...rest, ...(origin === undefined ? {} : { origin }), status: 'pending' };
  });

export const cancelEditNode = (tree: ProjectEditTree, nodeId: string): ProjectEditTree =>
  updateNode(tree, nodeId, 'cancel an edit node', settlePending('canceled'));

/** The image is gone but its node and descendants stay, so the history keeps its shape. */
export const markEditImageDeleted = (tree: ProjectEditTree, imageName: string): ProjectEditTree => {
  if (!tree.nodes.some((node) => node.imageName === imageName)) {
    return tree;
  }

  return {
    ...tree,
    nodes: tree.nodes.map((node) =>
      node.imageName === imageName ? { ...node, imageName: null, status: 'deleted' } : node
    ),
  };
};

/** Makes `childId` the shown child of `branchId`. Unchanged unless it is a child of it. */
export const selectEditChild = (tree: ProjectEditTree, branchId: string, childId: string): ProjectEditTree =>
  getEditNode(tree, childId)?.parentId !== branchId || tree.selection[branchId] === childId
    ? tree
    : { ...tree, selection: { ...tree.selection, [branchId]: childId } };

/** The nodes from the root along each branch's selected child, or its default child where none is selected. */
export const getEditActivePath = (tree: ProjectEditTree): EditTreeNode[] => {
  const childrenOf = new Map<string, EditTreeNode[]>();

  for (const node of tree.nodes) {
    if (node.parentId !== null) {
      childrenOf.set(node.parentId, [...(childrenOf.get(node.parentId) ?? []), node]);
    }
  }

  const path: EditTreeNode[] = [];
  let current = getEditNode(tree, tree.rootNodeId);

  while (current) {
    path.push(current);
    const children = childrenOf.get(current.id) ?? [];
    const selectedId = Object.hasOwn(tree.selection, current.id) ? tree.selection[current.id] : undefined;
    current = children.find((child) => child.id === selectedId) ?? getDefaultChild(children);
  }

  return path;
};

/** Which node an image belongs to, for jumping to it from the gallery. */
export const findEditNodeByImageName = (tree: ProjectEditTree, imageName: string): EditTreeNode | undefined =>
  tree.nodes.find((node) => node.imageName === imageName);

/** Every image the tree names, nodes and references alike, each once. */
export const listEditImageNames = (tree: ProjectEditTree): string[] => {
  const names = new Set<string>();

  for (const node of tree.nodes) {
    if (node.imageName) {
      names.add(node.imageName);
    }
    for (const reference of node.referenceImages) {
      names.add(reference.imageName);
    }
  }

  return [...names];
};
