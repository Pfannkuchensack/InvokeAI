import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { WildcardRecordDTO } from 'services/api/endpoints/wildcards';
import { useDeleteWildcardMutation } from 'services/api/endpoints/wildcards';

const $wildcardToDelete = atom<WildcardRecordDTO | null>(null);
const clearWildcardToDelete = () => $wildcardToDelete.set(null);

export const useDeleteWildcard = () => {
  const deleteWildcard = useCallback((wildcard: WildcardRecordDTO) => {
    $wildcardToDelete.set(wildcard);
  }, []);

  return deleteWildcard;
};

export const DeleteWildcardDialog = memo(() => {
  useAssertSingleton('DeleteWildcardDialog');
  const { t } = useTranslation();
  const wildcardToDelete = useStore($wildcardToDelete);
  const [_deleteWildcard] = useDeleteWildcardMutation();

  const deleteWildcard = useCallback(async () => {
    if (!wildcardToDelete) {
      return;
    }
    try {
      await _deleteWildcard(wildcardToDelete.id).unwrap();
      toast({
        status: 'success',
        title: t('wildcards.wildcardDeleted'),
      });
    } catch {
      toast({
        status: 'error',
        title: t('wildcards.unableToDeleteWildcard'),
      });
    }
  }, [wildcardToDelete, _deleteWildcard, t]);

  return (
    <ConfirmationAlertDialog
      isOpen={wildcardToDelete !== null}
      onClose={clearWildcardToDelete}
      title={t('wildcards.deleteWildcard')}
      acceptCallback={deleteWildcard}
      acceptButtonText={t('common.delete')}
      cancelButtonText={t('common.cancel')}
      useInert={false}
    >
      <Text>{t('wildcards.deleteWildcard2')}</Text>
    </ConfirmationAlertDialog>
  );
});

DeleteWildcardDialog.displayName = 'DeleteWildcardDialog';
