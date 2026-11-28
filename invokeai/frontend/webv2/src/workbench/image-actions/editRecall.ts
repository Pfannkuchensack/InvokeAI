import type { GenerateWidgetValues, ImageWithDims, MainModelConfig } from '@features/generation/contracts';
import type { InstructWidgetValues } from '@features/instruct';
import type { EditEntryTarget } from '@workbench/editEntry';

import { getEffectiveReferenceImage } from '@features/generation/settings';
import {
  addInstructReferences,
  getInstructMaxReferenceImages,
  getInstructTip,
  isInstructEditModel,
  normalizeInstructValues,
  requestInstructEntry,
  resolveEditOutputSize,
} from '@features/instruct';
import { showEditWorkspace } from '@workbench/editEntry';
import { toInstructTreeView } from '@workbench/editSubmission';
import { getEditableEditTree } from '@workbench/projectEditTree';
import { getProjectWidgetValues } from '@workbench/widgetState';

export interface EditRecallInput {
  /** Remix leaves Edit's own seed alone; Recall All takes the recorded one. */
  kind: 'all' | 'remix';
  /** The record as Generate recall read it: the Edit settings, the instruction and the references. */
  values: GenerateWidgetValues;
  /** The image the recalled step ran from (the record's `edit_source_image_name`). */
  sourceName: string;
  /** The source as the gallery still has it; null once it was deleted or could not be looked up. */
  source: ImageWithDims | null;
}

interface SizeBasis {
  height: number;
  /** Whether the image is the history's first, whose results take the model's ideal area. */
  isRoot: boolean;
  width: number;
}

/** `auto` where Edit would pick the recorded size from the image the next step continues from. */
const getSizeMode = async (
  values: GenerateWidgetValues,
  model: MainModelConfig,
  basis: SizeBasis | null
): Promise<InstructWidgetValues['sizeMode']> => {
  if (!basis) {
    return 'manual';
  }

  const { fitToOptimalArea } = await import('@features/generation/editGraph');
  const auto = resolveEditOutputSize({
    fit: (size) => fitToOptimalArea(size, model),
    manual: null,
    parent: { height: basis.height, isSource: basis.isRoot, width: basis.width },
    source: { height: basis.height, width: basis.width },
  });

  return auto.width === values.width && auto.height === values.height ? 'auto' : 'manual';
};

/** What the next Edit step takes from the record. */
const toEditValues = (
  { kind, sourceName, values }: EditRecallInput,
  model: MainModelConfig,
  sizeMode: InstructWidgetValues['sizeMode']
): Partial<InstructWidgetValues> => {
  const extras = values.referenceImages.flatMap((reference) =>
    reference.config.image ? [getEffectiveReferenceImage(reference.config.image)] : []
  );
  // `ref_images[0]` is the source; the rest are the references beside it.
  const references = addInstructReferences(
    [],
    extras.filter((image) => image.image_name !== sourceName),
    { maxReferenceImages: getInstructMaxReferenceImages(model), sourceImageName: sourceName }
  );
  const read = normalizeInstructValues({
    height: values.height,
    instruction: values.positivePrompt,
    model,
    references,
    seed: values.seed,
    seedMode: values.seedMode,
    settings: values,
    sizeMode,
    width: values.width,
  });

  return {
    height: read.height,
    instruction: read.instruction,
    model: read.model,
    references: read.references,
    ...(kind === 'remix' ? {} : { seed: read.seed, seedMode: read.seedMode }),
    settings: read.settings,
    sizeMode: read.sizeMode,
    width: read.width,
  };
};

/**
 * Recalls an Edit step into Edit: its settings and instruction go to the composer, the references beside the source
 * with them. The source becomes the continue point of the history it is in, or is offered as a new start image.
 * `declined` when the record's model is not one Edit can run, so the caller recalls into Generate as for any image;
 * `stale` when the user left the project meanwhile, which changes nothing and has nothing to report.
 */
export type EditRecallOutcome = 'declined' | 'recalled' | 'stale';

export const recallIntoEdit = async (target: EditEntryTarget, input: EditRecallInput): Promise<EditRecallOutcome> => {
  const model = input.values.model;

  if (!isInstructEditModel(model)) {
    return 'declined';
  }

  const project = target.getProject(target.projectId);

  if (!project || !target.isActiveProject(target.projectId)) {
    return 'stale';
  }

  const stored = project.editTree;
  const tree = getEditableEditTree(stored);
  const sourceNode = tree?.nodes.find((node) => node.imageName === input.sourceName);
  const basisOf = (node: { height: number; parentId: string | null; width: number }): SizeBasis => ({
    height: node.height,
    isRoot: node.parentId === null,
    width: node.width,
  });
  const newRoot: SizeBasis | null = input.source ? { ...input.source, isRoot: true } : null;
  const values = toEditValues(
    input,
    model,
    await getSizeMode(input.values, model, sourceNode ? basisOf(sourceNode) : newRoot)
  );

  // The awaits above gave the user time to leave the project; its Edit must not change then.
  if (!target.isActiveProject(target.projectId)) {
    return 'stale';
  }

  // An edit history this build cannot read is never replaced, so it only takes the values; so does a missing source.
  if (input.source && (!stored || tree)) {
    // Keeping the old history, the next step continues from its own tip, whose size rule is what counts.
    const tip =
      tree && !sourceNode
        ? getInstructTip(toInstructTreeView(tree), normalizeInstructValues(getProjectWidgetValues(project, 'instruct')))
        : null;

    requestInstructEntry({
      continueFromImage: true,
      image: input.source,
      projectId: target.projectId,
      values,
      ...(tip ? { valuesIfKept: { ...values, sizeMode: await getSizeMode(input.values, model, basisOf(tip)) } } : {}),
    });
  } else {
    target.commands.widgets.patchValues('instruct', values, target.projectId);
  }

  showEditWorkspace(target);

  return 'recalled';
};
