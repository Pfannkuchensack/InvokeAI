import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  devMistralEncoderModelSelected,
  devVaeModelSelected,
  selectDevMistralEncoderModel,
  selectDevVaeModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useFlux2VAEModels, useMistralEncoderModels } from 'services/api/hooks/modelsByType';
import type { MistralEncoderModelConfig, VAEModelConfig } from 'services/api/types';

/**
 * FLUX.2 Dev VAE Model Select
 * Selects a FLUX VAE model for FLUX.2 Dev
 * Dev can use the same 16-channel VAE as FLUX.1
 */
const ParamFlux2DevVaeModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const devVaeModel = useAppSelector(selectDevVaeModel);
  const [modelConfigs, { isLoading }] = useFlux2VAEModels();

  const _onChange = useCallback(
    (model: VAEModelConfig | null) => {
      if (model) {
        dispatch(devVaeModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(devVaeModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: devVaeModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2DevVae')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.flux2DevVaePlaceholder')}
      />
    </FormControl>
  );
});

ParamFlux2DevVaeModelSelect.displayName = 'ParamFlux2DevVaeModelSelect';

/**
 * FLUX.2 Dev Mistral Encoder Model Select
 * Selects a Mistral Small 3.1 text encoder model for FLUX.2 Dev
 */
const ParamFlux2DevMistralEncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const devMistralEncoderModel = useAppSelector(selectDevMistralEncoderModel);
  const [modelConfigs, { isLoading }] = useMistralEncoderModels();

  const _onChange = useCallback(
    (model: MistralEncoderModelConfig | null) => {
      if (model) {
        dispatch(devMistralEncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(devMistralEncoderModelSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: devMistralEncoderModel,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.flux2DevMistralEncoder')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.flux2DevMistralEncoderPlaceholder')}
      />
    </FormControl>
  );
});

ParamFlux2DevMistralEncoderModelSelect.displayName = 'ParamFlux2DevMistralEncoderModelSelect';

/**
 * Combined component for FLUX.2 Dev model selection
 */
const ParamFlux2DevModelSelects = () => {
  return (
    <>
      <ParamFlux2DevVaeModelSelect />
      <ParamFlux2DevMistralEncoderModelSelect />
    </>
  );
};

export default memo(ParamFlux2DevModelSelects);
