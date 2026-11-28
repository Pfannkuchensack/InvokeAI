import { describe, expect, it } from 'vitest';

import {
  addEditStep,
  cancelEditNode,
  completeEditNode,
  createEditTree,
  type EditStepInput,
  failEditNode,
  findEditNodeByImageName,
  getEditableEditTree,
  getEditActivePath,
  getEditChildren,
  getEditNode,
  isUnreadableEditTree,
  joinStoredEditTree,
  listEditImageNames,
  markEditImageDeleted,
  parseEditTree,
  type ProjectEditTree,
  readStoredEditTree,
  selectEditChild,
  splitStoredEditTree,
  toStoredEditTreeValue,
} from './projectEditTree';

const AT = '2026-11-27T10:00:00.000Z';

const step = (stepId: string, variantIds: string[], referenceImages: string[] = []): EditStepInput => ({
  createdAt: AT,
  height: 1024,
  instruction: `instruction ${stepId}`,
  referenceImages: referenceImages.map((imageName) => ({ imageName })),
  settings: { steps: 28 },
  stepId,
  variants: variantIds.map((id, index) => ({ id, seed: 100 + index })),
  width: 768,
});

const start = () => createEditTree({ height: 1024, imageName: 'source.png', width: 768 }, AT, 'root');

/** root -> {a, b} (step s1), b -> {c} (step s2), all completed, `b` shown. */
const branched = (): ProjectEditTree => {
  let tree = addEditStep(start(), 'root', step('s1', ['a', 'b']));
  tree = completeEditNode(completeEditNode(tree, 'a', 'a.png'), 'b', 'b.png');
  tree = selectEditChild(tree, 'root', 'b');
  tree = addEditStep(tree, 'b', step('s2', ['c'], ['ref.png']));

  return completeEditNode(tree, 'c', 'c.png');
};

const ids = (nodes: { id: string }[]) => nodes.map((node) => node.id);

describe('starting a tree', () => {
  it('holds the source image as a completed root without an instruction', () => {
    expect(start().nodes).toEqual([
      {
        createdAt: AT,
        height: 1024,
        id: 'root',
        imageName: 'source.png',
        instruction: '',
        parentId: null,
        referenceImages: [],
        seed: 0,
        settings: {},
        status: 'completed',
        stepId: 'root',
        width: 768,
      },
    ]);
    expect(ids(getEditActivePath(start()))).toEqual(['root']);
  });
});

describe('adding steps', () => {
  it('adds the variants of one step as pending siblings sharing the step id and showing the first', () => {
    const tree = addEditStep(start(), 'root', step('s1', ['a', 'b']));
    const variants = getEditChildren(tree, 'root');

    expect(ids(variants)).toEqual(['a', 'b']);
    expect(variants.map((node) => [node.stepId, node.status, node.imageName, node.seed])).toEqual([
      ['s1', 'pending', null, 100],
      ['s1', 'pending', null, 101],
    ]);
    expect(ids(getEditActivePath(tree))).toEqual(['root', 'a']);
  });

  it('ignores an unknown parent and a step without variants', () => {
    const tree = start();

    expect(addEditStep(tree, 'missing', step('s1', ['a']))).toBe(tree);
    expect(addEditStep(tree, 'root', step('s1', []))).toBe(tree);
  });
});

describe('the active path', () => {
  it('follows the selected child at each branch', () => {
    const tree = branched();

    expect(ids(getEditActivePath(tree))).toEqual(['root', 'b', 'c']);
    expect(ids(getEditActivePath(selectEditChild(tree, 'root', 'a')))).toEqual(['root', 'a']);
  });

  it('shows the first variant of the newest step where nothing is selected, as adding a step does', () => {
    const added = addEditStep(branched(), 'root', step('s3', ['x', 'y']));

    expect(ids(getEditActivePath(added))).toEqual(['root', 'x']);
    expect(ids(getEditActivePath({ ...added, selection: {} }))).toEqual(['root', 'x']);
    expect(ids(getEditActivePath({ ...branched(), selection: {} }))).toEqual(['root', 'a']);
  });

  it('only selects a child of the branch it is selected on', () => {
    const tree = branched();

    expect(selectEditChild(tree, 'root', 'c')).toBe(tree);
    expect(selectEditChild(tree, 'root', 'missing')).toBe(tree);
  });

  it('keeps a deleted image on the path', () => {
    const deleted = markEditImageDeleted(branched(), 'b.png');

    expect(getEditActivePath(deleted).map((node) => [node.id, node.status])).toEqual([
      ['root', 'completed'],
      ['b', 'deleted'],
      ['c', 'completed'],
    ]);
  });
});

describe('settling a node', () => {
  it('completes, fails or cancels a pending node and keeps the others', () => {
    const tree = addEditStep(start(), 'root', step('s1', ['a', 'b', 'c']));
    const settled = cancelEditNode(failEditNode(completeEditNode(tree, 'a', 'a.png'), 'b'), 'c');

    expect(settled.nodes.slice(1).map((node) => [node.status, node.imageName])).toEqual([
      ['completed', 'a.png'],
      ['failed', null],
      ['canceled', null],
    ]);
  });

  it('drops a result that arrives after the node was canceled or settled', () => {
    const canceled = cancelEditNode(addEditStep(start(), 'root', step('s1', ['a'])), 'a');

    expect(completeEditNode(canceled, 'a', 'late.png')).toBe(canceled);
    expect(failEditNode(canceled, 'a')).toBe(canceled);
    expect(completeEditNode(canceled, 'missing', 'x.png')).toBe(canceled);
  });
});

describe('finding images', () => {
  it('jumps from an image name to its node', () => {
    const tree = branched();

    expect(findEditNodeByImageName(tree, 'c.png')?.id).toBe('c');
    expect(findEditNodeByImageName(tree, 'source.png')?.id).toBe('root');
    expect(findEditNodeByImageName(tree, 'nope.png')).toBeUndefined();
  });

  it('forgets a deleted image but keeps its node and descendants', () => {
    const tree = markEditImageDeleted(branched(), 'b.png');

    expect(findEditNodeByImageName(tree, 'b.png')).toBeUndefined();
    expect(getEditNode(tree, 'b')).toMatchObject({ imageName: null, status: 'deleted' });
    expect(getEditNode(tree, 'c')).toMatchObject({ parentId: 'b', status: 'completed' });
    expect(markEditImageDeleted(tree, 'b.png')).toBe(tree);
  });

  it('lists every image once: nodes and references', () => {
    expect(listEditImageNames(branched()).sort()).toEqual(['a.png', 'b.png', 'c.png', 'ref.png', 'source.png']);
  });
});

describe('operations never write what the reader refuses', () => {
  const serializeAndParse = (tree: ProjectEditTree) => parseEditTree(JSON.parse(JSON.stringify(tree)));

  it('reads back the result of every operation', () => {
    const pending = addEditStep(start(), 'root', step('s1', ['a', 'b', 'c', 'd']));
    const results = [
      start(),
      pending,
      completeEditNode(pending, 'a', 'a.png'),
      failEditNode(pending, 'b'),
      cancelEditNode(pending, 'c'),
      markEditImageDeleted(completeEditNode(pending, 'a', 'a.png'), 'a.png'),
      selectEditChild(pending, 'root', 'd'),
      branched(),
    ];

    for (const tree of results) {
      expect(serializeAndParse(tree)).toEqual(tree);
    }
  });

  it.each([
    ['an empty source image name', () => createEditTree({ height: 1, imageName: '', width: 1 }, AT, 'r')],
    ['a fractional width', () => createEditTree({ height: 8, imageName: 's.png', width: 8.5 }, AT, 'r')],
    ['a non-finite height', () => createEditTree({ height: Infinity, imageName: 's.png', width: 8 }, AT, 'r')],
    ['an empty root id', () => createEditTree({ height: 8, imageName: 's.png', width: 8 }, AT, '')],
    ['an empty node id', () => addEditStep(start(), 'root', step('s1', ['']))],
    ['a variant id already in the tree', () => addEditStep(branched(), 'root', step('s9', ['c']))],
    ['the same variant id twice', () => addEditStep(start(), 'root', step('s1', ['a', 'a']))],
    ['an empty step id', () => addEditStep(start(), 'root', step('', ['a']))],
    ['a step id of another branch', () => addEditStep(branched(), 'c', step('s1', ['z']))],
    [
      'an unsafe seed',
      () => addEditStep(start(), 'root', { ...step('s1', ['a']), variants: [{ id: 'a', seed: 2 ** 60 }] }),
    ],
    ['a fractional step size', () => addEditStep(start(), 'root', { ...step('s1', ['a']), height: 10.5 })],
    ['an empty reference name', () => addEditStep(start(), 'root', step('s1', ['a'], ['']))],
    ['an empty result image name', () => completeEditNode(addEditStep(start(), 'root', step('s1', ['a'])), 'a', '')],
  ])('refuses %s', (_name, run) => {
    expect(run).toThrow(TypeError);
  });
});

describe('persisted form', () => {
  const stored = () => JSON.parse(JSON.stringify(branched())) as Record<string, unknown>;
  const withNodes = (change: (node: ProjectEditTree['nodes'][number]) => object) => ({
    ...stored(),
    nodes: branched().nodes.map(change),
  });

  it('survives a JSON round trip through its two halves', () => {
    const { content, workspace } = splitStoredEditTree(branched());
    const joined = joinStoredEditTree(JSON.parse(JSON.stringify(content)), JSON.parse(JSON.stringify(workspace)));

    expect(readStoredEditTree(joined)).toEqual(branched());
  });

  it('keeps the image key the media index reads', () => {
    expect(JSON.stringify(splitStoredEditTree(branched()).content)).toContain('"imageName":"c.png"');
  });

  it('keeps selections in the workspace half only', () => {
    const { content, workspace } = splitStoredEditTree(branched());

    expect(Object.keys(content as object).sort()).toEqual(['nodes', 'rootNodeId']);
    expect(workspace).toEqual({ selection: { b: 'c', root: 'b' } });
  });

  it('drops selections that name no child of their branch', () => {
    expect(parseEditTree({ ...stored(), selection: { root: 'c', gone: 'a', b: 7 } })?.selection).toEqual({});
    expect(parseEditTree({ ...stored(), selection: undefined })?.selection).toEqual({});
  });

  it('keeps unknown fields on the tree, its nodes and its references', () => {
    const future = {
      ...stored(),
      futureTreeField: { a: 1 },
      nodes: branched().nodes.map((node) => ({
        ...node,
        futureNodeField: node.id,
        referenceImages: node.referenceImages.map((reference) => ({ ...reference, crop: [1, 2] })),
      })),
    };
    const read = readStoredEditTree(future) as ProjectEditTree;
    const rewritten = JSON.parse(JSON.stringify(toStoredEditTreeValue(read)));

    expect(isUnreadableEditTree(read)).toBe(false);
    expect(rewritten).toEqual(future);
    expect(rewritten.nodes[3].referenceImages).toEqual([{ crop: [1, 2], imageName: 'ref.png' }]);
    expect(completeEditNode(read, 'a', 'a.png').futureTreeField).toEqual({ a: 1 });
  });

  it.each([
    ['not an object', 'tree'],
    ['no nodes', { nodes: [], rootNodeId: 'root' }],
    ['a root id that is not the first node', { ...stored(), rootNodeId: 'a' }],
    ['a child before its parent', { ...stored(), nodes: [...branched().nodes].reverse() }],
    ['a repeated node id', { ...stored(), nodes: [...branched().nodes, branched().nodes[1]] }],
    ['a status of a later build', withNodes((node) => ({ ...node, status: 'running' }))],
    ['a completed node without an image', withNodes((node) => (node.id === 'c' ? { ...node, imageName: null } : node))],
    ['a settings snapshot that is not an object', withNodes((node) => ({ ...node, settings: 'x' }))],
    ['a reference without a name', withNodes((node) => ({ ...node, referenceImages: [{ imageName: '' }] }))],
    ['a second root', withNodes((node) => (node.id === 'a' ? { ...node, parentId: null } : node))],
  ])('keeps a stored tree with %s exactly as stored, unavailable for editing', (_name, candidate) => {
    expect(parseEditTree(candidate)).toBeNull();

    const read = readStoredEditTree(candidate)!;

    expect(isUnreadableEditTree(read)).toBe(true);
    expect(getEditableEditTree(read)).toBeNull();
    expect(toStoredEditTreeValue(read)).toBe(candidate);
  });

  it('tells the in-memory wrapper by identity: stored data with an unreadableEditTree key round-trips whole', () => {
    const readable = { ...stored(), unreadableEditTree: { keep: true } };
    const damaged = { unreadableEditTree: 1, nodes: 'x' };
    const readableRead = readStoredEditTree(readable)!;
    const damagedRead = readStoredEditTree(damaged)!;

    expect(isUnreadableEditTree(readableRead)).toBe(false);
    expect(JSON.stringify(toStoredEditTreeValue(readableRead))).toBe(JSON.stringify(readable));
    expect(isUnreadableEditTree(damagedRead)).toBe(true);
    expect(toStoredEditTreeValue(damagedRead)).toBe(damaged);
  });

  it('leaves an already read tree or wrapper unchanged when it is read again', () => {
    const wrapper = readStoredEditTree({ nodes: 'x' })!;
    const tree = readStoredEditTree(stored())!;

    expect(readStoredEditTree(wrapper)).toBe(wrapper);
    expect(readStoredEditTree(tree)).toEqual(tree);
    expect(toStoredEditTreeValue(readStoredEditTree(wrapper)!)).toEqual({ nodes: 'x' });
  });

  it('leaves an absent tree absent', () => {
    expect(readStoredEditTree(undefined)).toBeUndefined();
  });
});
