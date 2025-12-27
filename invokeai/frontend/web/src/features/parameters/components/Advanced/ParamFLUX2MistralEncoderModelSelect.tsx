import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { flux2MistralEncoderSelected, selectFLUX2MistralEncoder } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useMistralEncoderModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';

const ParamFLUX2MistralEncoderModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const mistralEncoder = useAppSelector(selectFLUX2MistralEncoder);
  const [modelConfigs, { isLoading }] = useMistralEncoderModels();

  const _onChange = useCallback(
    (model: AnyModelConfig | null) => {
      if (model) {
        dispatch(flux2MistralEncoderSelected(zModelIdentifierField.parse(model)));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: mistralEncoder,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.mistralEncoder')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
};

export default memo(ParamFLUX2MistralEncoderModelSelect);
