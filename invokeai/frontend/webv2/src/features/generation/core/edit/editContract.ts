import type { FamilyPipelineSettings } from '@features/generation/core/familyGraph';
import type { GenerationIssue } from '@features/generation/core/generationIssues';
import type {
  GenerateReferenceImage,
  GenerateReferenceImageConfig,
  GenerateSettings,
  ImageWithDims,
  MainModelConfig,
} from '@features/generation/core/types';

import {
  createReferenceImageId,
  type EditBase,
  getDefaultGenerateSettings,
  getGenerationDimensions,
  getGenerationValidationIssues,
  getInstructionEditSupport,
  isEditBase,
} from '@features/generation/core/baseGenerationPolicies';
import { calculateNewSize } from '@features/generation/core/settings';

/**
 * What Edit knows of a family without building its graph. Eager, so the panel can validate and label before the lazy
 * graph chunk loads; `compileEditGraph` adds each family's builder.
 */
interface EditFamilyInfo {
  /** The `generation_mode` the result's metadata records. */
  mode: string;
  /** The reference type Generate records for this family, which a recall restores into Generate's list. */
  refType: GenerateReferenceImageConfig['type'];
  /** What the model calls the image in an instruction, where that is not Edit's "Image N" (one-based). */
  referenceTag?: (imageNumber: number) => string;
}

/** An entry per base in `EDIT_BASES`; a base added there fails to compile until it is described here. */
export const EDIT_FAMILY_INFO: Record<EditBase, EditFamilyInfo> = {
  flux2: { mode: 'flux2_edit', refType: 'flux2_reference_image' },
  // FLUX.3 reads `<ref_image_0>`, `<ref_image_1>`, ... in the order the references are connected.
  flux3: {
    mode: 'flux3_edit',
    refType: 'flux2_reference_image',
    referenceTag: (imageNumber) => `<ref_image_${imageNumber - 1}>`,
  },
  // Only the edit variant (`requires_variant`); Generate records the same reference type for its own Qwen-Edit graph.
  'qwen-image': { mode: 'qwen_image_edit', refType: 'qwen_image_reference_image' },
  'qwen-image-2-1': { mode: 'qwen_image_2_1_edit', refType: 'qwen_image_2_1_reference_image' },
};

/** The settings an edit step reads; prompt and seed enter at submission through the graph's `positive_prompt` and `seed` nodes. */
export type EditGraphSettings = FamilyPipelineSettings &
  Pick<GenerateSettings, 'qwen3VLVisionTowerModel' | 'qwenImageEncoderQuantization' | 'qwenImageShift'>;

/** Every field the edit graph reads; the record type fails to compile until a new pipeline setting is placed here. */
export const EDIT_SETTINGS = {
  cfgRescaleMultiplier: true,
  cfgScale: true,
  clipEmbedModel: true,
  clipSkip: true,
  componentSourceModel: true,
  loras: true,
  mistralEncoderModel: true,
  qwen35EncoderModel: true,
  qwen3EncoderModel: true,
  qwen3VLEncoderModel: true,
  qwen3VLVisionTowerModel: true,
  qwenImageEncoderQuantization: true,
  qwenImageShift: true,
  qwenVLEncoderModel: true,
  scheduler: true,
  steps: true,
  t5EncoderModel: true,
  vae: true,
  vaePrecision: true,
} as const satisfies Record<keyof EditGraphSettings, true>;

/** Only what the edit graph reads, so a wider object (Generate's settings) never reaches the validator or the graph. */
export const pickEditSettings = (settings: EditGraphSettings): EditGraphSettings =>
  Object.fromEntries(
    Object.keys(EDIT_SETTINGS).map((key) => [key, settings[key as keyof EditGraphSettings]])
  ) as unknown as EditGraphSettings;

/** The source and its references as the family's reference images, the source first. */
export const getEditReferenceImages = (base: EditBase, images: readonly ImageWithDims[]): GenerateReferenceImage[] =>
  images.map((image) => ({
    config: { image: { original: { image } }, type: EDIT_FAMILY_INFO[base].refType } as GenerateReferenceImageConfig,
    id: createReferenceImageId(),
    isEnabled: true,
  }));

/** Generate's wording of a refusal, reworded for Edit: its own name, and its images counted with the one being edited. */
export const toEditMessage = (message: string): string =>
  message
    .replace(/^Generate /, 'Edit ')
    .replace(
      /^(.+) supports at most (\d+) reference images?\. Turn off or remove the extra ones\.$/,
      '$1 takes at most $2 reference images, the image you edit counted as one. Remove the extra references.'
    );

/**
 * What Generate would refuse for this step: the size, the references and the components the loader needs. One check
 * for the panel before Send and for the compiler, so what Send allows is what compiles. Empty for a model Edit cannot
 * run (the callers name that themselves).
 */
export const getEditValidationIssues = ({
  images,
  model,
  settings,
  size,
}: {
  model: MainModelConfig;
  settings: EditGraphSettings;
  /** The source first, then the references. */
  images: readonly ImageWithDims[];
  /** Without it the model's default size, which is on its grid. */
  size?: { width: number; height: number };
}): GenerationIssue[] => {
  if (!getInstructionEditSupport(model) || !isEditBase(model.base)) {
    return [];
  }

  const defaults = getDefaultGenerateSettings(model);

  return getGenerationValidationIssues(model, {
    ...defaults,
    ...pickEditSettings(settings),
    ...(size ?? { height: defaults.height, width: defaults.width }),
    referenceImages: getEditReferenceImages(model.base, images),
  }).map((issue) => ({ ...issue, message: toEditMessage(issue.message) }));
};

/**
 * The size an edit result takes from a source image: its aspect ratio at the model's ideal area on the model's grid,
 * whether the source is larger or smaller.
 */
export const fitToOptimalArea = (
  source: { width: number; height: number },
  model: Pick<MainModelConfig, 'base' | 'type' | 'variant'>
): { width: number; height: number } => {
  const { grid, optimal } = getGenerationDimensions(model);

  return calculateNewSize(source.width / source.height, optimal * optimal, grid);
};
