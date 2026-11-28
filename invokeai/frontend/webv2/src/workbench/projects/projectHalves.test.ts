import type { Project } from '@workbench/projectContracts';

import {
  addEditStep,
  completeEditNode,
  createEditTree,
  getEditableEditTree,
  selectEditChild,
} from '@workbench/projectEditTree';
import { addProjectWorkflow, createBlankWorkflowDocument, recordProjectWorkflowRun } from '@workbench/projectWorkflows';
import { normalizeWorkbenchProject } from '@workbench/workbenchState';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import type { ProjectWithWorkspaceDTO } from './api';

import { serializeProjectDocumentV3, serializeProjectDocumentV3Json } from './projectDocument';
import {
  findRepeatedWorkflowItem,
  joinProjectDocument,
  PROJECT_DOCUMENT_HALVES,
  splitProjectDocument,
} from './projectHalves';
import { joinProjectRecord } from './projectHydration';
import { deserializeProjectDocument } from './syncedPersistence';

const RUN = { completedAt: '2026-10-10T00:00:02.000Z', imageName: 'run.png', submittedAt: '2026-10-10T00:00:01.000Z' };

/** A project whose workspace differs from a fresh one: a second, active workflow with a run, and a staged image. */
const workedOnProject = (): Project => {
  const project = createInitialWorkbenchState().projects[0]!;
  const second = createBlankWorkflowDocument();
  const withSecond = recordProjectWorkflowRun(addProjectWorkflow(project, second), second.id, RUN);
  return {
    ...withSecond,
    canvas: {
      ...withSecond.canvas,
      stagingArea: { ...withSecond.canvas.stagingArea, isVisible: false, selectedImageIndex: 2 },
    },
  };
};

/** A source, one step of two variants (the second shown), and one step below the second. */
const editedTree = () => {
  const step = (id: string, variants: string[]) => ({
    createdAt: '2026-11-27T10:00:00.000Z',
    height: 1024,
    instruction: 'make it night',
    referenceImages: [{ imageName: 'ref.png' }],
    settings: { steps: 28 },
    stepId: id,
    variants: variants.map((variant, index) => ({ id: variant, seed: index + 1 })),
    width: 1024,
  });
  let tree = createEditTree({ height: 1024, imageName: 'source.png', width: 1024 }, '2026-11-27T09:00:00.000Z', 'root');

  tree = addEditStep(tree, 'root', step('s1', ['a', 'b']));
  tree = completeEditNode(completeEditNode(tree, 'a', 'a.png'), 'b', 'b.png');
  tree = selectEditChild(tree, 'root', 'b');

  return addEditStep(tree, 'b', step('s2', ['c']));
};

const defaultsOf = (project: Project) => () => {
  const { documentSchemaVersion: _schema, ...workspace } = splitProjectDocument(
    serializeProjectDocumentV3(project)
  ).workspace;
  return workspace;
};

const load = (document: Record<string, unknown>): Project => {
  const loaded = deserializeProjectDocument(document);
  if (loaded.status !== 'loaded') {
    throw new Error(`Expected the document to load, got ${loaded.status}.`);
  }
  return loaded.project;
};

const join = (halves: { content: object; workspace: object | null }, defaults = defaultsOf(workedOnProject())) =>
  joinProjectDocument(
    halves.content as Record<string, unknown>,
    halves.workspace as Record<string, unknown> | null,
    defaults
  );

describe('splitting a project into content and workspace', () => {
  it('rejoins to the same whole document', () => {
    const project = workedOnProject();
    const halves = splitProjectDocument(serializeProjectDocumentV3(project));

    const joined = join({ content: JSON.parse(halves.contentJson), workspace: JSON.parse(halves.workspaceJson) });

    expect(joined.status).toBe('migrated');
    const rejoined = load((joined as { document: Record<string, unknown> }).document);
    // Against the project as loading leaves it: the canonical form sync compares.
    expect(serializeProjectDocumentV3Json(rejoined).documentJson).toBe(
      serializeProjectDocumentV3Json(load(serializeProjectDocumentV3(project))).documentJson
    );
  });

  it('carries every field of the document in one half or the other', () => {
    const project = workedOnProject();
    const geometry = { height: 300, width: 400, x: 10, y: 20 };
    const optional = {
      ...project,
      floatingWidgets: { gallery: { ...geometry, zIndex: 1 } },
      lastFloatingGeometry: { gallery: geometry },
      presetWorkingLayouts: [{ presetId: project.layout.presetId, layout: project.layout }],
      editTree: editedTree(),
    } as unknown as Project;
    const document = serializeProjectDocumentV3(optional);
    const { content, workspace } = splitProjectDocument(document);

    for (const [field, half] of Object.entries(PROJECT_DOCUMENT_HALVES)) {
      const holders = [field in content && 'content', field in workspace && 'workspace'].filter(Boolean);
      expect(holders, field).toEqual(half === 'both' ? ['content', 'workspace'] : [half]);
    }
    expect(Object.keys(document).sort()).toEqual(Object.keys(PROJECT_DOCUMENT_HALVES).sort());
  });

  it('keeps how one member works out of the shared content', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(project));

    expect(Object.keys(content).sort()).toEqual(['canvas', 'documentSchemaVersion', 'id', 'name', 'workflows']);
    expect('stagingArea' in content.canvas).toBe(false);
    expect(content.canvas.document.selectedLayerId).toBeNull();
    expect(content.workflows.entries.some((entry) => 'lastRun' in entry)).toBe(false);
    expect(Object.values(workspace.workflows.lastRuns)).toEqual([RUN]);
    expect(workspace.canvas.stagingArea.selectedImageIndex).toBe(2);
  });

  it('puts the Edit tree nodes in the content and the shown child of each branch in the workspace', () => {
    const tree = editedTree();
    const { content, workspace } = splitProjectDocument(
      serializeProjectDocumentV3({ ...workedOnProject(), editTree: tree })
    );

    expect(content.editTree).toEqual({ nodes: tree.nodes, rootNodeId: 'root' });
    expect(workspace.editTree).toEqual({ selection: { b: 'c', root: 'b' } });
  });

  it('rejoins an Edit tree, and a project without one stays without', () => {
    const rejoin = (project: Project) => {
      const halves = splitProjectDocument(serializeProjectDocumentV3(project));
      const joined = join({ content: JSON.parse(halves.contentJson), workspace: JSON.parse(halves.workspaceJson) });

      return load((joined as { document: Record<string, unknown> }).document);
    };

    expect(rejoin({ ...workedOnProject(), editTree: editedTree() }).editTree).toEqual(editedTree());
    expect(rejoin(workedOnProject())).not.toHaveProperty('editTree');
    expect(Object.keys(splitProjectDocument(serializeProjectDocumentV3(workedOnProject())).content)).not.toContain(
      'editTree'
    );
  });

  it('drops workspace selections that name no child of their branch', () => {
    const halves = splitProjectDocument(serializeProjectDocumentV3({ ...workedOnProject(), editTree: editedTree() }));
    const joined = join({
      content: JSON.parse(halves.contentJson),
      workspace: { ...JSON.parse(halves.workspaceJson), editTree: { selection: { root: 'c', gone: 'a' } } },
    });

    const loaded = load((joined as { document: Record<string, unknown> }).document).editTree;

    expect(getEditableEditTree(loaded)?.selection).toEqual({});
  });

  describe('a stored Edit tree this build does not fully read', () => {
    const base = () => serializeProjectDocumentV3({ ...workedOnProject(), editTree: editedTree() });
    const treeWith = (change: (tree: Record<string, any>) => void): Record<string, unknown> => {
      const tree = JSON.parse(JSON.stringify(editedTree()));
      change(tree);

      return { ...base(), editTree: tree };
    };
    /** The Edit tree in the halves a save of the loaded project writes, beside those of the stored document. */
    const saved = (document: Record<string, unknown>) => {
      const before = splitProjectDocument(document as never);
      const after = splitProjectDocument(serializeProjectDocumentV3(load(document)));

      const editTreeOf = (halves: typeof before) => [
        JSON.stringify(halves.content.editTree),
        JSON.stringify(halves.workspace.editTree),
      ];

      return { after: editTreeOf(after), before: editTreeOf(before) };
    };

    it('is left as it is when the loaded project is normalized again in memory', () => {
      const unreadable = load(
        treeWith((tree) => {
          tree.nodes[2].status = 'rendering';
        })
      );
      const readable = load(base());
      const again = normalizeWorkbenchProject(unreadable);

      expect(again.editTree).toBe(unreadable.editTree);
      expect(normalizeWorkbenchProject(readable).editTree).toEqual(readable.editTree);
      expect(JSON.stringify(serializeProjectDocumentV3(again).editTree)).toBe(
        JSON.stringify(serializeProjectDocumentV3(unreadable).editTree)
      );
    });

    it('writes a tree with a status of a later build back byte for byte, and offers it no edits', () => {
      const document = treeWith((tree) => {
        tree.nodes[2].status = 'rendering';
      });
      const { after, before } = saved(document);

      expect(after).toEqual(before);
      expect(getEditableEditTree(load(document).editTree)).toBeNull();
    });

    it('writes a damaged tree back byte for byte instead of dropping it', () => {
      const { after, before } = saved({ ...base(), editTree: { rootNodeId: 'x', nodes: [{}], selection: { x: 'y' } } });

      expect(after).toEqual(before);
      expect(before[0]).toContain('"rootNodeId":"x"');
    });

    it('keeps the fields a later build added to the tree, its nodes and its references', () => {
      const document = treeWith((tree) => {
        tree.futureTreeField = { flag: true };
        tree.nodes[1].futureNodeField = 7;
        tree.nodes[3].referenceImages[0].crop = [0, 0, 1, 1];
      });
      const { after, before } = saved(document);

      expect(after).toEqual(before);
      expect(getEditableEditTree(load(document).editTree)).not.toBeNull();
    });
  });

  it.each(['__proto__', 'constructor', 'toString'])('round-trips the last run of a workflow with id %s', (id) => {
    const project = createInitialWorkbenchState().projects[0]!;
    const workflow = { ...createBlankWorkflowDocument(), id };
    const ran = recordProjectWorkflowRun(addProjectWorkflow(project, workflow), id, RUN);
    const halves = splitProjectDocument(serializeProjectDocumentV3(ran));

    const joined = join({ content: JSON.parse(halves.contentJson), workspace: JSON.parse(halves.workspaceJson) });

    expect(Object.hasOwn(halves.workspace.workflows.lastRuns, id)).toBe(true);
    const rejoined = load((joined as { document: Record<string, unknown> }).document);
    expect(rejoined.workflows.entries.find((entry) => entry.document.id === id)?.lastRun).toEqual(RUN);
    expect(rejoined.workflows.entries.filter((entry) => entry.lastRun)).toHaveLength(1);
  });

  it('changes only the half an edit belongs to', () => {
    const project = workedOnProject();
    const before = splitProjectDocument(serializeProjectDocumentV3(project));

    const relaid = splitProjectDocument(
      serializeProjectDocumentV3({ ...project, layout: { ...project.layout, centerViewId: 'gallery' } })
    );
    const renamed = splitProjectDocument(serializeProjectDocumentV3({ ...project, name: 'Renamed' }));

    expect(relaid.contentJson).toBe(before.contentJson);
    expect(relaid.workspaceJson).not.toBe(before.workspaceJson);
    expect(renamed.workspaceJson).toBe(before.workspaceJson);
    expect(renamed.contentJson).not.toBe(before.contentJson);
  });
});

describe('joining a project for a member', () => {
  it('starts a member with no workspace on the defaults, on the first workflow', () => {
    const project = workedOnProject();
    const { content } = splitProjectDocument(serializeProjectDocumentV3(project));

    const joined = join({ content, workspace: null });

    const loaded = load((joined as { document: Record<string, unknown> }).document);
    expect(loaded.workflows.activeWorkflowId).toBe(project.workflows.entries[0]!.document.id);
    expect(loaded.workflows.entries.some((entry) => entry.lastRun)).toBe(false);
    expect(loaded.name).toBe(project.name);
    // A member's first open is a new workspace: Auto-switch starts In progress, not a stored project's Off.
    expect(loaded.settings.viewerAutoSwitch).toBe('progress');
  });

  it('keeps the Auto-switch mode a workspace saved, across a split and a join', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(
      serializeProjectDocumentV3({ ...project, settings: { ...project.settings, viewerAutoSwitch: 'latest' } })
    );

    const joined = join({ content, workspace });

    expect(load((joined as { document: Record<string, unknown> }).document).settings.viewerAutoSwitch).toBe('latest');
  });

  it('falls back to the first workflow when the open one was deleted meanwhile', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(project));

    const joined = join({
      content,
      workspace: { ...workspace, workflows: { activeWorkflowId: 'gone', lastRuns: {} } },
    });

    expect(load((joined as { document: Record<string, unknown> }).document).workflows.activeWorkflowId).toBe(
      project.workflows.entries[0]!.document.id
    );
  });

  it('refuses a server’s content that repeats an item id within a workflow, which a lock could not tell apart', () => {
    const project = workedOnProject();
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(project));
    const [first, ...rest] = content.workflows.entries;
    const node = { id: 'node-1', type: 'notes' };
    const repeated = { ...first!, document: { ...first!.document, nodes: [node, node] } };
    const elsewhere = rest.map((entry) => ({ ...entry, document: { ...entry.document, nodes: [node] } }));
    const record = (entries: unknown[]) =>
      ({
        data: { ...content, workflows: { entries } },
        workspace: { data: workspace, revision: 1, updated_at: '' },
      }) as unknown as ProjectWithWorkspaceDTO;

    expect(joinProjectRecord(record([repeated, ...rest])).status).toBe('malformed');
    // An archive's joins: importing creates it whole.
    expect(join({ content: { ...content, workflows: { entries: [repeated, ...rest] } }, workspace }).status).toBe(
      'migrated'
    );
    // Workflow duplication keeps node ids, so two workflows may share one.
    expect(
      joinProjectRecord(record([{ ...first!, document: { ...first!.document, nodes: [node] } }, ...elsewhere])).status
    ).toBe('migrated');
  });

  it('finds repeated edge and form element ids too', () => {
    const { content } = splitProjectDocument(serializeProjectDocumentV3(workedOnProject()));
    const [first, ...rest] = content.workflows.entries;
    const edge = { id: 'edge-1', source: 'a', sourceHandle: 'value', target: 'b', targetHandle: 'value' };
    const element = { data: { children: [], layout: 'column' }, id: 'container-1', type: 'container' };
    const repeating = (document: object) => [{ ...first!, document: { ...first!.document, ...document } }, ...rest];

    expect(findRepeatedWorkflowItem(repeating({ edges: [edge, edge] }))).toBe(first!.document.id);
    expect(
      findRepeatedWorkflowItem(repeating({ form: { elements: { a: element, b: element }, rootElementId: 'a' } }))
    ).toBe(first!.document.id);
    expect(findRepeatedWorkflowItem(content.workflows.entries)).toBeNull();
  });

  it('refuses a workspace of another document schema', () => {
    const { content, workspace } = splitProjectDocument(serializeProjectDocumentV3(workedOnProject()));

    expect(join({ content, workspace: { ...workspace, documentSchemaVersion: 5 } }).status).toBe('malformed');
  });
});
