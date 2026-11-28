import type { CompiledGenerateGraph, ImageWithDims, MainModelConfig } from '@features/generation/contracts';
import type { InstructTreeView, InstructWidgetValues } from '@features/instruct';
import type { ModelConfig } from '@features/models';
import type { InvocationRuntimeInputs } from '@workbench/invocationContracts';

import {
  getInstructStepSettings,
  getInstructTip,
  getInstructValidationReasons,
  normalizeInstructValues,
  resolveEditOutputSize,
  syncInstructValuesWithModels,
} from '@features/instruct';
import { planSeedSubmission, SEED_MAX, wrapSeed, type SeedStep, type SeedMode } from '@platform/core/seed';

import type { Project } from './projectContracts';
import type { EditStepInput, ProjectEditTree } from './projectEditTree';
import type { ScopedDefaults } from './settings/contracts';

import { getBrowserId } from './browserId';
import {
  createEditNodeId,
  createEditStepId,
  getEditableEditTree,
  getEditActivePath,
  getEditNode,
} from './projectEditTree';
import { getEffectiveProjectSettings, PROJECT_SETTINGS_OWNER } from './settings/projectSettings';
import { getProjectWidgetValues } from './widgetState';

/** What a submission carries for an Edit step, compiled before the reducer runs (the edit graph compiler loads lazily). */
/** A step built from a node directly (Run again), whatever the composer holds. */
export interface EditRun {
  instruction: string;
  parentId: string;
  references: ImageWithDims[];
}

export interface EditStepPlan {
  compiled: CompiledGenerateGraph;
  /** The step was built from a node, not the composer, so the composer's draft stays. */
  isRun?: boolean;
  instruction: string;
  /** The seed the editable seed becomes once the step is queued; null when the mode leaves it alone. */
  nextSeed: number | null;
  parentId: string;
  rootNodeId: string;
  seed: number;
  seedStep: SeedStep;
  step: EditStepInput;
}

/** The project as a step reads it when it runs from `run` instead of the composer: a copy, never stored. */
export const withEditRun = (project: Project, run: EditRun): Project => {
  const instance = Object.values(project.widgetInstances).find((candidate) => candidate.typeId === 'instruct');

  return instance
    ? {
        ...project,
        widgetInstances: {
          ...project.widgetInstances,
          [instance.id]: {
            ...instance,
            state: {
              ...instance.state,
              values: {
                ...instance.state.values,
                continueFromNodeId: run.parentId,
                instruction: run.instruction,
                references: run.references,
              },
            },
          },
        },
      }
    : project;
};

/** The tree view the widget reads: nodes, the shown branch, selections. */
export const toInstructTreeView = (tree: ProjectEditTree): InstructTreeView => ({
  nodes: tree.nodes,
  path: getEditActivePath(tree),
  rootNodeId: tree.rootNodeId,
  selection: tree.selection,
});

/** The widget's values the way a step reads them: normalized, and following the installed catalog when it is known. */
export const readInstructValues = (
  project: Pick<Project, 'widgetInstances'>,
  models: readonly ModelConfig[] | undefined
): InstructWidgetValues => {
  const values = normalizeInstructValues(getProjectWidgetValues(project, 'instruct'));

  return models ? syncInstructValuesWithModels(values, models) : values;
};

/** Why the Edit route cannot queue a step now; empty when it can. */
export const getEditRouteValidationReasons = (
  values: Record<string, unknown>,
  storedTree: Project['editTree'],
  models: readonly ModelConfig[] | undefined
): string[] => {
  const instruct = normalizeInstructValues(values);
  const shown = models ? syncInstructValuesWithModels(instruct, models) : instruct;
  const tree = getEditableEditTree(storedTree);

  if (!tree) {
    const reasons = getInstructValidationReasons(shown, models);

    reasons.push(storedTree ? 'The edit history was written by a newer version.' : 'Start from an image first.');

    return reasons;
  }

  const tip = getInstructTip(toInstructTreeView(tree), instruct);
  const root = getEditNode(tree, tree.rootNodeId)!;
  // The size the step will take, which the graph is checked at.
  const size = shown.model
    ? resolveEditOutputSize({
        manual: shown.sizeMode === 'manual' ? { height: shown.height, width: shown.width } : null,
        model: shown.model,
        parent: { height: tip.height, isSource: tip.id === root.id, width: tip.width },
        source: { height: root.height, width: root.width },
      })
    : undefined;
  const reasons = getInstructValidationReasons(shown, models, size);

  if (tip.status === 'pending') {
    reasons.push('The previous step is still running.');
  } else if (tip.status !== 'completed') {
    reasons.push('Continue from an image that exists.');
  }

  return reasons;
};

const pickSeedMode = (mode: SeedMode, variants: number): SeedMode =>
  mode === 'fixed' && variants > 1 ? 'increment' : mode;

/**
 * Compiles the next step of the Edit tree: its parent is the node chosen with "Continue from here", else the end of the
 * shown branch. Throws what the graph compiler refuses (a missing component, a reference it cannot take).
 */
export const prepareEditStep = async ({
  defaults,
  models,
  project: stored,
  run,
  runtime,
}: {
  defaults: ScopedDefaults;
  models: readonly ModelConfig[] | undefined;
  project: Project;
  run?: EditRun;
  runtime: Pick<InvocationRuntimeInputs, 'noiseDevice'>;
}): Promise<EditStepPlan> => {
  const project = run ? withEditRun(stored, run) : stored;
  const tree = getEditableEditTree(project.editTree);
  const values = readInstructValues(project, models);

  if (!tree || !values.model || !values.settings) {
    throw new Error('Edit has no image to continue from or no model.');
  }

  const view = toInstructTreeView(tree);
  const parent = getInstructTip(view, values);
  const root = getEditNode(tree, tree.rootNodeId)!;

  if (parent.status !== 'completed' || !parent.imageName) {
    throw new Error('The image to continue from is not available.');
  }

  // An image that is gone or not shown to this user is refused here (the lookup cannot tell which); a lookup that fails
  // is not a refusal, since the run itself would fail on it anyway.
  const found = await import('@features/gallery')
    .then(({ galleryImages }) => galleryImages.resolveMany([parent.imageName!]))
    .catch(() => null);

  if (found && !found.some((entry) => entry.imageName === parent.imageName)) {
    throw new Error('The image to continue from is not available.');
  }

  const { compileEditGraph } = await import('@features/generation/editGraph');
  const model: MainModelConfig = values.model;
  const size = resolveEditOutputSize({
    manual: values.sizeMode === 'manual' ? { height: values.height, width: values.width } : null,
    model,
    parent: { height: parent.height, isSource: parent.id === root.id, width: parent.width },
    source: { height: root.height, width: root.width },
  });
  const projectSettings = getEffectiveProjectSettings(project.settings, defaults[PROJECT_SETTINGS_OWNER]);
  const source: ImageWithDims = { height: parent.height, image_name: parent.imageName, width: parent.width };
  const compiled = compileEditGraph({
    model,
    projectSettings,
    randDevice: projectSettings.useCpuNoise ? 'cpu' : runtime.noiseDevice,
    referenceImages: values.references,
    settings: values.settings,
    size,
    source,
  });
  const mode = pickSeedMode(values.seedMode, values.variants);
  const seed = values.seedMode === 'random' ? Math.floor(Math.random() * (SEED_MAX + 1)) : values.seed;
  const seedPlan = planSeedSubmission({
    batchCount: values.variants,
    promptCount: 1,
    seedBehaviour: 'per-iteration',
    seedMode: mode,
    startSeed: seed,
  });
  const instruction = values.instruction.trim();

  return {
    compiled,
    instruction,
    ...(run ? { isRun: true } : {}),
    nextSeed: values.seedMode === 'increment' || values.seedMode === 'decrement' ? seedPlan.nextSeed : null,
    parentId: parent.id,
    rootNodeId: tree.rootNodeId,
    seed,
    seedStep: seedPlan.step,
    step: {
      createdAt: new Date().toISOString(),
      height: size.height,
      origin: getBrowserId(),
      instruction,
      referenceImages: values.references.map((reference) => ({ imageName: reference.image_name })),
      settings: getInstructStepSettings(values),
      stepId: createEditStepId(),
      variants: Array.from({ length: values.variants }, (_, index) => ({
        id: createEditNodeId(),
        seed: wrapSeed(seed + seedPlan.step * index),
      })),
      width: size.width,
    },
  };
};
