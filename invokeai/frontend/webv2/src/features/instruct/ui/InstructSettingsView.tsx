/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop -- one panel per project; sections render only the values they read. */
import type {
  GenerateLora,
  GenerateSettings,
  GenerationModelCatalogItem,
  ImageWithDims,
} from '@features/generation/contracts';
import type { ModelTaxonomyType } from '@features/models';
import type { SeedInputPatch } from '@platform/ui/SeedInput';

import { Box, HStack, Icon, Image, Stack, Switch, Text } from '@chakra-ui/react';
import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import { galleryImageUrls } from '@features/gallery/utility';
import {
  addConcept,
  ConceptList,
  type ConceptModelPort,
  ConceptRow,
  type ConceptUpdate,
  GenerationSettingsSection,
  removeConcept,
  SeedField,
  updateConcept,
} from '@features/generation/components';
import { fitToOptimalArea } from '@features/generation/editGraph';
import {
  type ComponentSlotPolicy,
  getComponentPolicyContext,
  getComponentSectionPolicy,
  getComponentSlotIssue,
  getDefaultGenerateSettings,
  getDimensionGrid,
  getGenerationUiPolicy,
  getGuidanceBoundReason,
  getGuidanceInputMax,
  getGuidanceSliderMax,
  getSchedulerOptions,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
} from '@features/generation/settings';
import { getInstructTip } from '@features/instruct/core/history';
import {
  addInstructReferences,
  getReferenceNumber,
  getReferenceRoom,
  removeInstructReference,
} from '@features/instruct/core/references';
import {
  MAX_EDIT_DIMENSION,
  MIN_EDIT_DIMENSION,
  resolveEditOutputSize,
  snapEditDimension,
} from '@features/instruct/core/size';
import {
  getInstructMaxReferenceImages,
  isInstructEditModel,
  MAX_INSTRUCT_VARIANTS,
  selectInstructModel,
} from '@features/instruct/core/values';
import { useInstructUi } from '@features/instruct/ui/InstructUiContext';
import {
  getModelBaseColorPalette,
  getModelBaseLabel,
  getModelImageUrl,
  useModelsSelector,
  useOpenModelInManager,
} from '@features/models';
import { ModelSelect } from '@features/models/react';
import { IconButton } from '@platform/ui/Button';
import { Combobox } from '@platform/ui/Combobox';
import { Field } from '@platform/ui/Field';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { ImageOffIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useEnsureInstructCatalog, useInstructValues } from './useInstructValues';

const MAIN_MODEL_TYPES: readonly ModelTaxonomyType[] = ['main'];
const LORA_MODEL_TYPES: readonly ModelTaxonomyType[] = ['lora'];
const REFERENCE_ACCEPT = ['image'] as const;
const STEPS_SLIDER_MAX = 100;
const NO_LORAS: GenerateLora[] = [];

const SectionIds = {
  components: 'instruct-components',
  concepts: 'instruct-concepts',
  model: 'instruct-model',
  output: 'instruct-output',
  references: 'instruct-references',
  render: 'instruct-render',
} as const;

const ModelSection = () => {
  const { t } = useTranslation();
  const { patchValues } = useInstructUi();
  const values = useInstructValues();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const select = useCallback(
    (candidate: GenerationModelCatalogItem | null) => {
      if (isInstructEditModel(candidate)) {
        patchValues((current) => {
          const next = selectInstructModel(current, candidate, models);

          return { model: next.model, references: next.references, settings: next.settings };
        });
      }
    },
    [models, patchValues]
  );

  return (
    <GenerationSettingsSection label={t('widgets.instruct.model')} sectionId={SectionIds.model} defaultOpen>
      <Field
        error={values.model ? undefined : t('widgets.instruct.modelRequired')}
        label={t('widgets.instruct.model')}
        labelPlacement="inside"
      >
        <ModelSelect
          filter={isInstructEditModel}
          installAction
          invalid={!values.model}
          modelTypes={MAIN_MODEL_TYPES}
          placeholder={t('widgets.instruct.selectModel')}
          value={values.model?.key ?? null}
          onChange={select}
        />
      </Field>
    </GenerationSettingsSection>
  );
};

const ConceptsSection = () => {
  const { t } = useTranslation();
  const { patchValues, projectId } = useInstructUi();
  const { model, settings } = useInstructValues();
  const loras = settings?.loras ?? NO_LORAS;
  const openInModelManager = useOpenModelInManager();
  const conceptModels = useMemo<ConceptModelPort>(
    () => ({
      getBaseColorPalette: getModelBaseColorPalette,
      getBaseLabel: getModelBaseLabel,
      getImageUrl: getModelImageUrl,
      openInModelManager: openInModelManager ?? undefined,
    }),
    [openInModelManager]
  );
  const selectedKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);
  const loraFilter = useCallback(
    (candidate: GenerationModelCatalogItem) =>
      Boolean(model && isLoraModelConfig(candidate) && isLoraCompatibleWithModel(candidate, model)),
    [model]
  );
  const change = useCallback(
    (update: (current: typeof loras) => typeof loras) =>
      patchValues((current) =>
        current.settings ? { settings: { ...current.settings, loras: update(current.settings.loras) } } : {}
      ),
    [patchValues]
  );

  return (
    <GenerationSettingsSection label={t('widgets.instruct.concepts')} sectionId={SectionIds.concepts} defaultOpen>
      <Stack gap="2.5">
        <ModelSelect
          disabled={!model}
          excludeKeys={selectedKeys}
          filter={loraFilter}
          modelTypes={LORA_MODEL_TYPES}
          placeholder={t('widgets.instruct.addLora')}
          relatedTo={model?.key ?? null}
          scopeLabel={t('models.scopeConcepts')}
          value={null}
          onChange={(candidate) => {
            if (isLoraModelConfig(candidate)) {
              change((current) => addConcept(current, candidate, model));
            }
          }}
        />
        {loras.length > 0 ? (
          <ConceptList label={t('widgets.instruct.concepts')} projectId={projectId}>
            {loras.map((lora) => (
              <ConceptRow
                key={lora.model.key}
                isCompatible={Boolean(model && isLoraCompatibleWithModel(lora.model, model))}
                lora={lora}
                models={conceptModels}
                onRemove={(key: string) => change((current) => removeConcept(current, key))}
                onUpdate={(key: string, update: ConceptUpdate) =>
                  change((current) => updateConcept(current, key, update))
                }
              />
            ))}
          </ConceptList>
        ) : (
          <Text color="fg.muted" fontSize="xs">
            {t('widgets.instruct.noLoras')}
          </Text>
        )}
      </Stack>
    </GenerationSettingsSection>
  );
};

const RenderSection = () => {
  const { t } = useTranslation();
  const { patchValues } = useInstructUi();
  const { model, seed, seedMode, settings, variants } = useInstructValues();
  const policy = getGenerationUiPolicy(model ?? undefined);
  const defaults = model ? getDefaultGenerateSettings(model) : null;
  const patchSettings = useCallback(
    (patch: Partial<GenerateSettings>) =>
      patchValues((current) => (current.settings ? { settings: { ...current.settings, ...patch } } : {})),
    [patchValues]
  );

  if (!model || !settings) {
    return null;
  }

  return (
    <GenerationSettingsSection label={t('widgets.instruct.render')} sectionId={SectionIds.render} defaultOpen>
      <Stack gap="2.5">
        <ScrubberField
          defaultValue={defaults?.steps}
          inputMax={Number.MAX_SAFE_INTEGER}
          label={t('widgets.generate.steps')}
          marks={defaults ? [defaults.steps] : undefined}
          max={STEPS_SLIDER_MAX}
          min={1}
          step={1}
          value={settings.steps}
          onChange={(steps) => Number.isFinite(steps) && patchSettings({ steps })}
        />
        <ScrubberField
          defaultValue={defaults?.cfgScale}
          error={getGuidanceBoundReason(model, settings.cfgScale)}
          inputMax={getGuidanceInputMax(policy.guidanceMax)}
          label={policy.guidanceLabel}
          marks={defaults ? [defaults.cfgScale] : undefined}
          max={getGuidanceSliderMax(policy.guidanceMin, policy.guidanceMax)}
          min={policy.guidanceMin}
          step={0.5}
          value={settings.cfgScale}
          onChange={(cfgScale) => Number.isFinite(cfgScale) && patchSettings({ cfgScale })}
        />
        {policy.schedulerVisible ? (
          <Field label={t('widgets.generate.scheduler')} labelPlacement="inside">
            <Combobox
              aria-label={t('widgets.generate.scheduler')}
              options={getSchedulerOptions(model, settings.scheduler)}
              value={settings.scheduler}
              onValueChange={(scheduler) => patchSettings({ scheduler })}
            />
          </Field>
        ) : null}
        <SeedField
          batchCount={variants}
          label={t('common.seed')}
          seed={seed}
          seedMode={seedMode}
          onCommit={(patch: SeedInputPatch) => patchValues(patch)}
        />
        <ScrubberField
          defaultValue={1}
          inputMax={MAX_INSTRUCT_VARIANTS}
          label={t('widgets.instruct.variants')}
          max={MAX_INSTRUCT_VARIANTS}
          min={1}
          step={1}
          value={variants}
          onChange={(next) => Number.isFinite(next) && patchValues({ variants: Math.round(next) })}
        />
      </Stack>
    </GenerationSettingsSection>
  );
};

const ComponentRow = ({
  onChange,
  settings,
  slot,
  model,
}: {
  model: NonNullable<ReturnType<typeof useInstructValues>['model']>;
  onChange: (slot: ComponentSlotPolicy, candidate: GenerationModelCatalogItem | null) => void;
  settings: GenerateSettings;
  slot: ComponentSlotPolicy;
}) => {
  const { t } = useTranslation();
  const context = getComponentPolicyContext(model, settings);
  const value = context.selectedComponents[slot.key];
  const filter = useCallback(
    (candidate: GenerationModelCatalogItem) => slot.filter?.(candidate, context) ?? false,
    [context, slot]
  );
  const error = getComponentSlotIssue(slot, context) ?? undefined;

  return (
    <Field error={error} helpText={slot.helpText} label={slot.label} labelPlacement="inside">
      <ModelSelect
        filter={slot.filter ? filter : undefined}
        invalid={Boolean(error)}
        isClearable
        modelTypes={slot.modelTypes}
        placeholder={
          slot.required?.(context)
            ? t('widgets.generate.selectComponent')
            : (slot.placeholder ?? t('widgets.generate.modelDefault'))
        }
        relatedTo={model.key}
        value={value?.key ?? null}
        onChange={(candidate) => onChange(slot, candidate)}
      />
    </Field>
  );
};

/** The encoder and VAE slots Generate offers FLUX.2, for the fields the edit graph reads. */
const ComponentsSection = () => {
  const { t } = useTranslation();
  const { patchValues } = useInstructUi();
  const { model, settings } = useInstructValues();
  const full = useMemo(
    () => (model && settings ? ({ ...getDefaultGenerateSettings(model), ...settings } as GenerateSettings) : null),
    [model, settings]
  );
  const commit = useCallback(
    (slot: ComponentSlotPolicy, candidate: GenerationModelCatalogItem | null) => {
      const next =
        slot.valueKind === 'main'
          ? isMainModelConfig(candidate)
            ? candidate
            : null
          : slot.valueKind === 'vae'
            ? isVaeModelConfig(candidate)
              ? candidate
              : null
            : isModelIdentifierConfig(candidate)
              ? candidate
              : null;

      patchValues((current) => (current.settings ? { settings: { ...current.settings, [slot.key]: next } } : {}));
    },
    [patchValues]
  );

  if (!model || !full) {
    return null;
  }

  const slots = getComponentSectionPolicy(model, full).slots.filter((slot) => slot.key in settings!);

  return slots.length === 0 ? null : (
    <GenerationSettingsSection label={t('widgets.instruct.components')} sectionId={SectionIds.components}>
      <Stack gap="2.5">
        {slots.map((slot) => (
          <ComponentRow key={slot.key} model={model} settings={full} slot={slot} onChange={commit} />
        ))}
      </Stack>
    </GenerationSettingsSection>
  );
};

const ReferenceChip = ({
  imageName,
  label,
  onRemove,
  removeLabel,
}: {
  imageName: string;
  label: string;
  onRemove?: () => void;
  removeLabel?: string;
}) => {
  const { t } = useTranslation();
  const { missingImages } = useInstructUi();
  const isMissing = missingImages.has(imageName);

  return (
    <HStack bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" gap="2" p="1.5" pe="2" rounded="md">
      {isMissing ? (
        <Icon
          aria-label={t('widgets.instruct.referenceMissing')}
          as={ImageOffIcon}
          boxSize="10"
          color="fg.muted"
          p="2"
        />
      ) : (
        <Image
          alt=""
          boxSize="10"
          flexShrink={0}
          objectFit="cover"
          rounded="sm"
          src={galleryImageUrls.thumbnail(imageName)}
        />
      )}
      <Text flex="1" fontSize="sm" fontWeight="500" minW="0">
        {isMissing ? `${label} · ${t('widgets.instruct.unavailable')}` : label}
      </Text>
      {onRemove ? (
        <IconButton aria-label={removeLabel ?? ''} size="xs" tooltip={removeLabel} variant="ghost" onClick={onRemove}>
          <XIcon />
        </IconButton>
      ) : null}
    </HStack>
  );
};

const ReferencesSection = () => {
  const { t } = useTranslation();
  const { patchValues, tree } = useInstructUi();
  const values = useInstructValues();
  const tip = tree ? getInstructTip(tree, values) : null;
  const max = getInstructMaxReferenceImages(values.model);
  const room = getReferenceRoom(max, values.references);
  const sourceImageName = tip?.imageName ?? null;
  const add = useCallback(
    (item: { kind: string; name: string; width?: number; height?: number } | null) => {
      if (item?.kind === 'image' && item.width && item.height) {
        const image: ImageWithDims = { height: item.height, image_name: item.name, width: item.width };

        patchValues((current) => ({
          references: addInstructReferences(current.references, [image], {
            maxReferenceImages: getInstructMaxReferenceImages(current.model),
            sourceImageName,
          }),
        }));
      }
    },
    [patchValues, sourceImageName]
  );

  return (
    <GenerationSettingsSection label={t('widgets.instruct.references')} sectionId={SectionIds.references} defaultOpen>
      <Stack gap="2">
        {tip?.imageName ? (
          <ReferenceChip imageName={tip.imageName} label={t('widgets.instruct.imageN', { number: 1 })} />
        ) : null}
        {values.references.map((reference, index) => (
          <ReferenceChip
            key={reference.image_name}
            imageName={reference.image_name}
            label={t('widgets.instruct.imageN', { number: getReferenceNumber(index) })}
            removeLabel={t('widgets.instruct.removeReference', { number: getReferenceNumber(index) })}
            onRemove={() =>
              patchValues((current) => ({
                references: removeInstructReference(current.references, reference.image_name),
              }))
            }
          />
        ))}
        {room > 0 ? (
          <GalleryMediaSlot
            accept={REFERENCE_ACCEPT}
            dropId="instruct-reference-slot"
            labels={{ choose: t('widgets.instruct.addReference'), drop: t('widgets.instruct.dropReference') }}
            value={null}
            onChange={add}
          />
        ) : null}
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.instruct.referencesHint', { count: max - 1 })}
        </Text>
      </Stack>
    </GenerationSettingsSection>
  );
};

const OutputSection = () => {
  const { t } = useTranslation();
  const { patchValues, tree } = useInstructUi();
  const values = useInstructValues();
  const tip = tree ? getInstructTip(tree, values) : null;
  const root = tree?.nodes.find((node) => node.id === tree.rootNodeId);
  const grid = (values.model ? getDimensionGrid(values.model.base, values.model.variant) : null) ?? 16;
  const auto =
    values.model && tip && root
      ? resolveEditOutputSize({
          fit: (source) => fitToOptimalArea(source, values.model!),
          manual: null,
          parent: { height: tip.height, isSource: tip.id === root.id, width: tip.width },
          source: { height: root.height, width: root.width },
        })
      : null;
  const isManual = values.sizeMode === 'manual';
  const startManual = (checked: boolean) =>
    patchValues(
      checked
        ? { height: auto?.height ?? values.height, sizeMode: 'manual', width: auto?.width ?? values.width }
        : { sizeMode: 'auto' }
    );

  return (
    <GenerationSettingsSection label={t('widgets.instruct.output')} sectionId={SectionIds.output}>
      <Stack gap="2.5">
        <Switch.Root checked={isManual} onCheckedChange={({ checked }) => startManual(checked)}>
          <Switch.HiddenInput />
          <Switch.Control />
          <Switch.Label>{t('widgets.instruct.manualSize')}</Switch.Label>
        </Switch.Root>
        {isManual ? (
          <>
            <ScrubberField
              inputMax={MAX_EDIT_DIMENSION}
              inputMin={MIN_EDIT_DIMENSION}
              label={t('widgets.generate.width')}
              max={4096}
              min={MIN_EDIT_DIMENSION}
              step={grid}
              value={values.width}
              onChange={(width) => Number.isFinite(width) && patchValues({ width: snapEditDimension(width, grid) })}
            />
            <ScrubberField
              inputMax={MAX_EDIT_DIMENSION}
              inputMin={MIN_EDIT_DIMENSION}
              label={t('widgets.generate.height')}
              max={4096}
              min={MIN_EDIT_DIMENSION}
              step={grid}
              value={values.height}
              onChange={(height) => Number.isFinite(height) && patchValues({ height: snapEditDimension(height, grid) })}
            />
          </>
        ) : (
          <Text color="fg.muted" fontSize="sm">
            {auto
              ? t('widgets.instruct.autoSize', { height: auto.height, width: auto.width })
              : t('widgets.instruct.autoSizeNone')}
          </Text>
        )}
      </Stack>
    </GenerationSettingsSection>
  );
};

/** The Edit widget's left panel: what each step is made with. The instruction itself is written in the chat. */
export const InstructSettingsView = () => {
  useEnsureInstructCatalog();

  return (
    <Box minH="0" overflowY="auto" w="full">
      <ModelSection />
      <ReferencesSection />
      <ConceptsSection />
      <RenderSection />
      <OutputSection />
      <ComponentsSection />
    </Box>
  );
};
