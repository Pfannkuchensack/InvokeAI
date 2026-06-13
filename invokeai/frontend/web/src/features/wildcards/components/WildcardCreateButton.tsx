import { IconButton } from '@invoke-ai/ui-library';
import { $wildcardModalState } from 'features/wildcards/store/wildcardModal';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const WildcardCreateButton = () => {
  const { t } = useTranslation();

  const handleClickAddNew = useCallback(() => {
    $wildcardModalState.set({
      prefilledFormData: null,
      updatingWildcardId: null,
      isModalOpen: true,
    });
  }, []);

  return (
    <IconButton
      size="sm"
      variant="link"
      alignSelf="stretch"
      icon={<PiPlusBold />}
      tooltip={t('wildcards.addWildcard')}
      aria-label={t('wildcards.addWildcard')}
      onClick={handleClickAddNew}
    />
  );
};
