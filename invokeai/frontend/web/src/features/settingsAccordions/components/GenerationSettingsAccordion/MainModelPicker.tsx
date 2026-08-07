import { Flex, FormLabel, Icon } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { isExternalModelUnsupportedForTab } from 'features/parameters/components/MainModel/mainModelPickerUtils';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { modelSelected } from 'features/parameters/store/actions';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';
import { useMainModels } from 'services/api/hooks/modelsByType';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import { type AnyModelConfigWithExternal, isNonCommercialMainModelConfig } from 'services/api/types';

export const MainModelPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const activeTab = useAppSelector(selectActiveTab);
  const [allModelConfigs] = useMainModels();
  // Some GGUF models are only one half of a pair and belong in a dedicated slot in the advanced
  // section, not as a primary main: Wan's low-noise expert, and Ideogram 4's unconditional CFG
  // branch. Filter them out here so they can't be wired backwards.
  const modelConfigs = useMemo(
    () =>
      allModelConfigs.filter((c) => {
        if (
          c.type === 'main' &&
          c.base === 'wan' &&
          c.format === 'gguf_quantized' &&
          'expert' in c &&
          c.expert === 'low'
        ) {
          return false;
        }
        if (
          c.type === 'main' &&
          c.base === 'ideogram-4' &&
          c.format === 'gguf_quantized' &&
          'branch' in c &&
          c.branch === 'unconditional'
        ) {
          return false;
        }
        return true;
      }),
    [allModelConfigs]
  );
  const selectedModelConfig = useSelectedModelConfig();
  const onChange = useCallback(
    (modelConfig: AnyModelConfigWithExternal) => {
      dispatch(modelSelected(modelConfig));
    },
    [dispatch]
  );

  const isNonCommercialSelected = useMemo(
    () => selectedModelConfig && isNonCommercialMainModelConfig(selectedModelConfig),
    [selectedModelConfig]
  );

  const getIsOptionDisabled = useCallback(
    (modelConfig: AnyModelConfigWithExternal) => isExternalModelUnsupportedForTab(modelConfig, activeTab),
    [activeTab]
  );

  return (
    <Flex alignItems="center" gap={2}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
      {isNonCommercialSelected && (
        <InformationalPopover feature="fluxDevLicense" hideDisable={true}>
          <Flex justifyContent="flex-start">
            <Icon as={MdMoneyOff} />
          </Flex>
        </InformationalPopover>
      )}
      <ModelPicker
        pickerId="main-model"
        modelConfigs={modelConfigs}
        selectedModelConfig={selectedModelConfig}
        onChange={onChange}
        grouped
        getIsOptionDisabled={getIsOptionDisabled}
      />
      <UseDefaultSettingsButton />
    </Flex>
  );
});
MainModelPicker.displayName = 'MainModelPicker';
