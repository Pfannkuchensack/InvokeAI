import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { flux2VAESelected, selectFLUX2VAE } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

const ParamFLUX2VAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const vae = useAppSelector(selectFLUX2VAE);
  const [allVaeModels, { isLoading }] = useVAEModels();

  // Filter to only FLUX.2 VAE models (base === 'flux2')
  const modelConfigs = useMemo(() => {
    return allVaeModels.filter((model) => model.base === 'flux2');
  }, [allVaeModels]);

  const _onChange = useCallback(
    (vae: VAEModelConfig | null) => {
      if (vae) {
        dispatch(flux2VAESelected(zModelIdentifierField.parse(vae)));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: vae,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1} gap={2}>
      <InformationalPopover feature="paramVAE">
        <FormLabel m={0}>{t('modelManager.vae')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
};

export default memo(ParamFLUX2VAEModelSelect);
