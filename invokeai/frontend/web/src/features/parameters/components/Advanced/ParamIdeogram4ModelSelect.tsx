import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  ideogram4Qwen3EncoderModelSelected,
  ideogram4UnconditionalTransformerModelSelected,
  ideogram4VaeModelSelected,
  selectIdeogram4Qwen3EncoderModel,
  selectIdeogram4UnconditionalTransformerModel,
  selectIdeogram4VaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useFlux2VAEModels,
  useIdeogram4DiffusersModels,
  useIdeogram4GGUFUnconditionalModels,
  useQwen3VLEncoderModels,
} from 'services/api/hooks/modelsByType';
import type { MainModelConfig, Qwen3VLEncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * Ideogram 4 Unconditional Transformer Select
 *
 * A GGUF Ideogram 4 model is only the conditional CFG branch. Both branches run on every denoise
 * step and are separately trained weights, so the unconditional half is mandatory. It is chosen
 * explicitly rather than inferred: the two files differ only in name, so auto-pairing could quietly
 * combine mismatched quantization levels. The list is filtered on the `branch` recorded at install
 * time, which makes selecting a second conditional branch impossible here.
 */
const ParamIdeogram4UnconditionalTransformerSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModel = useAppSelector(selectIdeogram4UnconditionalTransformerModel);
  const [modelConfigs, { isLoading }] = useIdeogram4GGUFUnconditionalModels();

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      dispatch(ideogram4UnconditionalTransformerModelSelected(model ? zModelIdentifierField.parse(model) : null));
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.ideogram4UnconditionalTransformer')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.ideogram4UnconditionalTransformerPlaceholder')}
      />
    </FormControl>
  );
});

ParamIdeogram4UnconditionalTransformerSelect.displayName = 'ParamIdeogram4UnconditionalTransformerSelect';

/**
 * Ideogram 4 VAE Select
 *
 * Ideogram 4's VAE is bit-identical to the FLUX.2 VAE, so an already installed FLUX.2 VAE works and
 * nothing extra needs downloading. Optional: without one, the VAE is taken from an installed
 * Diffusers Ideogram 4 model.
 */
const ParamIdeogram4VaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModel = useAppSelector(selectIdeogram4VaeModel);
  const [modelConfigs, { isLoading }] = useFlux2VAEModels();
  const [diffusersModels] = useIdeogram4DiffusersModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      dispatch(ideogram4VaeModelSelected(model ? zModelIdentifierField.parse(model) : null));
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    isLoading,
  });

  const placeholder =
    diffusersModels.length > 0
      ? t('modelManager.ideogram4VaePlaceholder')
      : t('modelManager.ideogram4VaeNoModelPlaceholder');

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.ideogram4Vae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={placeholder}
      />
    </FormControl>
  );
});

ParamIdeogram4VaeModelSelect.displayName = 'ParamIdeogram4VaeModelSelect';

/**
 * Ideogram 4 Qwen3-VL Encoder Select
 *
 * Optional: without one, the encoder is taken from an installed Diffusers Ideogram 4 model.
 */
const ParamIdeogram4Qwen3EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModel = useAppSelector(selectIdeogram4Qwen3EncoderModel);
  const [allModelConfigs, { isLoading }] = useQwen3VLEncoderModels();
  const [diffusersModels] = useIdeogram4DiffusersModels();

  // Ideogram 4 conditions on Qwen3-VL 8B; Krea-2's 4B has a 2560-wide hidden state and would fail
  // with a shape error deep inside inference, so it must not be offerable here.
  const modelConfigs = useMemo(
    () => allModelConfigs.filter((config) => config.variant === 'qwen3_vl_8b'),
    [allModelConfigs]
  );

  const _onChange = useCallback(
    (model: Qwen3VLEncoderModelConfig | null) => {
      dispatch(ideogram4Qwen3EncoderModelSelected(model ? zModelIdentifierField.parse(model) : null));
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    isLoading,
  });

  const placeholder =
    diffusersModels.length > 0
      ? t('modelManager.ideogram4Qwen3EncoderPlaceholder')
      : t('modelManager.ideogram4Qwen3EncoderNoModelPlaceholder');

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.ideogram4Qwen3Encoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={placeholder}
      />
    </FormControl>
  );
});

ParamIdeogram4Qwen3EncoderModelSelect.displayName = 'ParamIdeogram4Qwen3EncoderModelSelect';

/**
 * Combined component for Ideogram 4 GGUF component selection.
 */
const ParamIdeogram4ModelSelects = () => {
  return (
    <>
      <ParamIdeogram4UnconditionalTransformerSelect />
      <ParamIdeogram4VaeModelSelect />
      <ParamIdeogram4Qwen3EncoderModelSelect />
    </>
  );
};

export default memo(ParamIdeogram4ModelSelects);
