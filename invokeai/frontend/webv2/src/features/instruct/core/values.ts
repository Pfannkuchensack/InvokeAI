import type { GenerateSettings, ImageWithDims, MainModelConfig } from '@features/generation/contracts';
import type { EditGraphSettings } from '@features/generation/editGraph';

import {
  getComponentValidationIssues,
  getDefaultGenerateSettings,
  getGenerateModelSelectionResult,
  getInstructionEditSupport,
  isMainModelConfig,
  normalizeGenerateSettings,
} from '@features/generation/settings';
import { isSeedMode, SEED_MAX } from '@platform/core/seed';

import type { InstructModelCatalogItem, InstructWidgetValues } from './types';

import { addInstructReferences } from './references';

/** A model Edit can run: it edits from instructions and starts from noise (see `compileEditGraph`). */
export const isInstructEditModel = (model: InstructModelCatalogItem | null | undefined): model is MainModelConfig => {
  if (!isMainModelConfig(model) || model.base !== 'flux2') {
    return false;
  }

  const support = getInstructionEditSupport(model);

  return support !== null && !support.img2imgStart;
};

export const getInstructMaxReferenceImages = (model: MainModelConfig | null): number =>
  (model && getInstructionEditSupport(model)?.maxReferenceImages) || 1;

/** Every field the edit graph reads, so a key the pipeline gains cannot go unsaved. */
const PIPELINE_KEYS = {
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
  qwenVLEncoderModel: true,
  scheduler: true,
  steps: true,
  t5EncoderModel: true,
  vae: true,
  vaePrecision: true,
} as const satisfies Record<keyof EditGraphSettings, true>;

export const INSTRUCT_PIPELINE_KEYS = Object.keys(PIPELINE_KEYS) as (keyof EditGraphSettings)[];

export const MAX_INSTRUCT_VARIANTS = 4;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const pickPipeline = (settings: unknown): EditGraphSettings =>
  Object.fromEntries(
    INSTRUCT_PIPELINE_KEYS.map((key) => [key, (settings as Record<string, unknown>)[key]])
  ) as unknown as EditGraphSettings;

/** The pipeline settings Generate would start the model with. */
export const getDefaultInstructPipeline = (model: MainModelConfig): EditGraphSettings =>
  pickPipeline(getDefaultGenerateSettings(model));

const isImageWithDims = (value: unknown): value is ImageWithDims =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  value.image_name.length > 0 &&
  typeof value.width === 'number' &&
  typeof value.height === 'number';

const getBoundedInteger = (value: unknown, fallback: number, min: number, max: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;

export const createDefaultInstructValues = (models: readonly InstructModelCatalogItem[] = []): InstructWidgetValues => {
  const candidates = models.filter(isInstructEditModel);
  const model = candidates.find((candidate) => !candidate.is_missing) ?? candidates[0] ?? null;
  const defaults = model ? getDefaultGenerateSettings(model) : null;

  return {
    continueFromNodeId: null,
    height: defaults?.height ?? 1024,
    instruction: '',
    model,
    references: [],
    seed: 0,
    seedMode: 'random',
    settings: model ? getDefaultInstructPipeline(model) : null,
    sizeMode: 'auto',
    variants: 1,
    width: defaults?.width ?? 1024,
  };
};

/** Reads stored widget values; whatever is missing or malformed takes its default. */
export const normalizeInstructValues = (raw: Record<string, unknown>): InstructWidgetValues => {
  const model = isMainModelConfig(raw.model) ? raw.model : null;
  const defaults = model ? getDefaultGenerateSettings(model) : null;
  const stored = isRecord(raw.settings) ? raw.settings : null;
  // Generate's own reader repairs malformed pipeline values and fills what the stored object lacks.
  const read =
    model && defaults ? normalizeGenerateSettings({ ...defaults, ...stored, model, modelKey: model.key }) : null;

  return {
    continueFromNodeId:
      typeof raw.continueFromNodeId === 'string' && raw.continueFromNodeId ? raw.continueFromNodeId : null,
    height: getBoundedInteger(raw.height, defaults?.height ?? 1024, 64, 8192),
    instruction: typeof raw.instruction === 'string' ? raw.instruction : '',
    model,
    references: Array.isArray(raw.references) ? raw.references.filter(isImageWithDims) : [],
    seed: getBoundedInteger(raw.seed, 0, 0, SEED_MAX),
    seedMode: isSeedMode(raw.seedMode) ? raw.seedMode : 'random',
    settings: read ? pickPipeline(read) : null,
    sizeMode: raw.sizeMode === 'manual' ? 'manual' : 'auto',
    variants: getBoundedInteger(raw.variants, 1, 1, MAX_INSTRUCT_VARIANTS),
    width: getBoundedInteger(raw.width, defaults?.width ?? 1024, 64, 8192),
  };
};

/** The values after choosing `model`: its defaults for the pipeline, the user's LoRAs kept where they still fit. */
export const selectInstructModel = (
  values: InstructWidgetValues,
  model: MainModelConfig,
  models: readonly InstructModelCatalogItem[]
): InstructWidgetValues => {
  // The previous model's key makes this a switch, so the new model's default components fill empty slots as in Generate.
  const { settings } = getGenerateModelSelectionResult({
    currentValues: {
      ...getDefaultGenerateSettings(values.model ?? model),
      ...values.settings,
      model,
      modelKey: values.model?.key,
    },
    model,
    models,
  });

  return {
    ...values,
    model,
    references: addInstructReferences([], values.references, {
      maxReferenceImages: getInstructMaxReferenceImages(model),
      sourceImageName: null,
    }),
    settings: pickPipeline(settings),
  };
};

/**
 * Follows the installed catalog: an uninstalled or unsuitable model gives way to the first installed edit model, and
 * stored LoRAs and components that are gone or no longer fit are dropped. Same object when nothing changes.
 */
export const syncInstructValuesWithModels = (
  values: InstructWidgetValues,
  models: readonly InstructModelCatalogItem[]
): InstructWidgetValues => {
  const installed = values.model ? models.find((candidate) => candidate.key === values.model?.key) : undefined;
  const model = isInstructEditModel(installed) ? installed : null;

  if (!model) {
    const fallback = createDefaultInstructValues(models).model;

    return fallback ? selectInstructModel(values, fallback, models) : values;
  }

  if (!values.settings) {
    return selectInstructModel(values, model, models);
  }

  const { settings } = getGenerateModelSelectionResult({
    currentValues: { ...getDefaultGenerateSettings(model), ...values.settings, model, modelKey: model.key },
    model,
    models,
  });
  const next = pickPipeline(settings);
  const isSame = INSTRUCT_PIPELINE_KEYS.every((key) => Object.is(next[key], values.settings?.[key]));

  return isSame && Object.is(model, values.model)
    ? values
    : { ...values, model, settings: isSame ? values.settings : next };
};

/**
 * A function-form patch computed against the values the panel shows: stored values followed to the installed models
 * (a fresh project stores no model yet, and its settings would otherwise be missing from the patch's input).
 */
export const resolveInstructPatch = (
  stored: Record<string, unknown>,
  patch: (current: InstructWidgetValues) => Partial<InstructWidgetValues>,
  models: readonly InstructModelCatalogItem[] | undefined
): Partial<InstructWidgetValues> => {
  const values = normalizeInstructValues(stored);
  const shown = models ? syncInstructValuesWithModels(values, models) : values;
  const result = patch(shown);

  // The stored values read back as the shown ones only with the shown model stored: without a model, stored settings are
  // not read, and the defaults of the model the panel falls back to would replace what the patch set.
  return shown.model && shown.model.key !== values.model?.key
    ? { model: shown.model, settings: shown.settings, ...result }
    : result;
};

export const getInstructValidationReasons = (
  values: InstructWidgetValues,
  models: readonly InstructModelCatalogItem[] | undefined
): string[] => {
  const reasons: string[] = [];
  const installed = values.model ? models?.find((candidate) => candidate.key === values.model?.key) : values.model;

  if (!values.model || !values.settings || (models && !isInstructEditModel(installed))) {
    reasons.push('Choose an edit model.');
  } else if (installed?.is_missing) {
    reasons.push(`${values.model.name} is not installed.`);
  }

  if (values.model && values.settings) {
    // The components the edit graph compile refuses without, worded for this surface.
    const settings = { ...getDefaultGenerateSettings(values.model), ...values.settings } as GenerateSettings;

    reasons.push(
      ...getComponentValidationIssues(values.model, settings).map(({ message }) =>
        message.replace(/^Generate /, 'Edit ')
      )
    );
  }

  if (values.instruction.trim().length === 0) {
    reasons.push('Write an instruction.');
  }

  return reasons;
};

const toIdentifier = (value: unknown): unknown =>
  isRecord(value) ? { base: value.base, hash: value.hash, key: value.key, name: value.name, type: value.type } : value;

/**
 * What a node's snapshot keeps: only the fields an edit step reads, models by identifier, so the tree stays small (a
 * full Generate snapshot is about four times the size).
 */
export const getInstructStepSettings = (values: InstructWidgetValues): Record<string, unknown> => {
  const settings = values.settings;

  return JSON.parse(
    JSON.stringify({
      cfgScale: settings?.cfgScale,
      loras: settings?.loras.map((lora) => ({ ...lora, model: toIdentifier(lora.model) })),
      mistralEncoderModel: toIdentifier(settings?.mistralEncoderModel),
      model: toIdentifier(values.model),
      scheduler: settings?.scheduler,
      steps: settings?.steps,
      vae: toIdentifier(settings?.vae),
    })
  ) as Record<string, unknown>;
};
