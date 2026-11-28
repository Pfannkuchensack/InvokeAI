import type { InstructNode, InstructTreeView, InstructWidgetValues } from './types';

/** One step of the shown branch: the variants it made under one parent, and which of them the branch follows. */
export interface InstructStepView {
  parent: InstructNode;
  /** The variant the shown branch follows. */
  selected: InstructNode;
  /** The steps under the same parent, oldest first, each with the variant the switcher shows when it moves there. */
  siblingSteps: { firstNodeId: string; stepId: string }[];
  stepId: string;
  variants: InstructNode[];
}

/** The shown branch as steps: for each node after the root, its sibling variants and sibling steps. */
export const getInstructSteps = (tree: InstructTreeView): InstructStepView[] =>
  tree.path.slice(1).map((selected, index) => {
    const parent = tree.path[index]!;
    const children = tree.nodes.filter((node) => node.parentId === parent.id);
    const siblingSteps = [...new Set(children.map((node) => node.stepId))].map((stepId) => ({
      firstNodeId: children.find((node) => node.stepId === stepId)!.id,
      stepId,
    }));

    return {
      parent,
      selected,
      siblingSteps,
      stepId: selected.stepId,
      variants: children.filter((node) => node.stepId === selected.stepId),
    };
  });

/** Where the next step goes under: the node chosen with "Continue from here" while it is in the tree, else the end. */
export const getInstructTip = (tree: InstructTreeView, values: Pick<InstructWidgetValues, 'continueFromNodeId'>) =>
  tree.nodes.find((node) => node.id === values.continueFromNodeId) ?? tree.path.at(-1)!;

/** The nodes from the root down to `nodeId`, for making the shown branch lead to it. */
export const getInstructRoute = (tree: InstructTreeView, nodeId: string): InstructNode[] => {
  const route: InstructNode[] = [];
  let current = tree.nodes.find((node) => node.id === nodeId);

  while (current) {
    route.unshift(current);
    const parentId: string | null = current.parentId;
    current = parentId === null ? undefined : tree.nodes.find((node) => node.id === parentId);
  }

  return route;
};

/** The closest image above `node` that exists, for continuing when the node itself has none. */
export const getNearestCompletedAncestor = (tree: InstructTreeView, node: InstructNode): InstructNode | undefined =>
  getInstructRoute(tree, node.id)
    .slice(0, -1)
    .reverse()
    .find((candidate) => candidate.status === 'completed' && candidate.imageName !== null);
