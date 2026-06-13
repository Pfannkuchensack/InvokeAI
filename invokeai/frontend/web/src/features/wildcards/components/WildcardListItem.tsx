import { Badge, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useDeleteWildcard } from 'features/wildcards/components/DeleteWildcardDialog';
import { $wildcardModalState } from 'features/wildcards/store/wildcardModal';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilBold, PiTrashBold } from 'react-icons/pi';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import type { WildcardRecordDTO } from 'services/api/endpoints/wildcards';

export const WildcardListItem = ({ wildcard }: { wildcard: WildcardRecordDTO }) => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const isMultiUser = setupStatus?.multiuser_enabled ?? false;
  // On a single-user instance there is no logged-in frontend user, so ownership can't be derived
  // from the auth store — the local user owns everything they can see and may edit it.
  const isOwner = !isMultiUser || currentUser?.user_id === wildcard.user_id;
  const deleteWildcard = useDeleteWildcard();

  const handleClickEdit = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      $wildcardModalState.set({
        prefilledFormData: {
          name: wildcard.name,
          values: wildcard.values,
          is_public: wildcard.is_public ?? false,
        },
        updatingWildcardId: wildcard.id,
        isModalOpen: true,
      });
    },
    [wildcard]
  );

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      deleteWildcard(wildcard);
    },
    [deleteWildcard, wildcard]
  );

  return (
    <Flex gap={3} p={3} w="full" alignItems="center" minW={0}>
      <Flex flexDir="column" w="full" alignItems="flex-start" flexGrow={1} minW={0} gap={1}>
        <Text fontSize="md" noOfLines={1} fontWeight="semibold" color="base.100">
          {wildcard.name}
        </Text>
        <Text fontSize="sm" color="base.400">
          {t('wildcards.valueCount', { count: wildcard.values.length })}
        </Text>
      </Flex>
      {wildcard.is_public && (
        <Badge color="invokeBlue.400" borderColor="invokeBlue.700" borderWidth={1} bg="transparent" flexShrink={0}>
          {t('wildcards.shared')}
        </Badge>
      )}
      <Spacer />
      {isOwner && (
        <>
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            aria-label={t('wildcards.editWildcard')}
            onClick={handleClickEdit}
            icon={<PiPencilBold />}
          />
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            aria-label={t('wildcards.deleteWildcard')}
            onClick={handleClickDelete}
            colorScheme="error"
            icon={<PiTrashBold />}
          />
        </>
      )}
    </Flex>
  );
};
