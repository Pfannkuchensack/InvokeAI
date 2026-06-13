import {
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { $wildcardModalState } from 'features/wildcards/store/wildcardModal';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { WildcardFormData } from './WildcardForm';
import { WildcardForm } from './WildcardForm';

export const WildcardModal = () => {
  useAssertSingleton('WildcardModal');
  const { t } = useTranslation();
  const wildcardModalState = useStore($wildcardModalState);

  const modalTitle = useMemo(() => {
    return wildcardModalState.updatingWildcardId ? t('wildcards.editWildcard') : t('wildcards.addWildcard');
  }, [wildcardModalState.updatingWildcardId, t]);

  const formData = useMemo<WildcardFormData | null>(() => {
    if (!wildcardModalState.prefilledFormData) {
      return null;
    }
    const { name, values, is_public } = wildcardModalState.prefilledFormData;
    return {
      name,
      values: values.join('\n'),
      is_public,
    };
  }, [wildcardModalState.prefilledFormData]);

  const handleCloseModal = useCallback(() => {
    $wildcardModalState.set({
      prefilledFormData: null,
      updatingWildcardId: null,
      isModalOpen: false,
    });
  }, []);

  return (
    <Modal isOpen={wildcardModalState.isModalOpen} onClose={handleCloseModal} isCentered size="2xl" useInert={false}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{modalTitle}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          {wildcardModalState.isModalOpen && (
            <WildcardForm
              key={wildcardModalState.updatingWildcardId ?? 'new'}
              updatingWildcardId={wildcardModalState.updatingWildcardId}
              formData={formData}
            />
          )}
        </ModalBody>
        <ModalFooter p={2} />
      </ModalContent>
    </Modal>
  );
};
