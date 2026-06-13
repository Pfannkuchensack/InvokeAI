import { Flex } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { selectWildcardSearchTerm } from 'features/wildcards/store/wildcardSlice';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import type { WildcardRecordDTO } from 'services/api/endpoints/wildcards';
import { useListWildcardsQuery } from 'services/api/endpoints/wildcards';

import { WildcardCreateButton } from './WildcardCreateButton';
import { WildcardImportButton } from './WildcardImportButton';
import { WildcardList } from './WildcardList';
import WildcardSearch from './WildcardSearch';

export const WildcardMenu = () => {
  const { t } = useTranslation();
  const searchTerm = useAppSelector(selectWildcardSearchTerm);
  const currentUser = useAppSelector(selectCurrentUser);
  const currentUserId = currentUser?.user_id;
  const { data: setupStatus } = useGetSetupStatusQuery();
  const isMultiUser = setupStatus?.multiuser_enabled ?? false;

  const { data } = useListWildcardsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const filteredData =
        data?.filter((wildcard) => wildcard.name.toLowerCase().includes(searchTerm.toLowerCase())) || EMPTY_ARRAY;

      const groupedData = filteredData.reduce(
        (
          acc: {
            myWildcards: WildcardRecordDTO[];
            sharedWildcards: WildcardRecordDTO[];
          },
          wildcard
        ) => {
          // Single-user: everything is the local user's. Multi-user: split own vs others' public.
          if (!isMultiUser || wildcard.user_id === currentUserId) {
            acc.myWildcards.push(wildcard);
          } else {
            acc.sharedWildcards.push(wildcard);
          }
          return acc;
        },
        { myWildcards: [], sharedWildcards: [] }
      );

      return {
        data: groupedData,
      };
    },
  });

  return (
    <Flex flexDir="column" gap={2} padding={3} layerStyle="second" borderRadius="base">
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <WildcardSearch />
        <Flex alignItems="center" gap={1}>
          <WildcardCreateButton />
          <WildcardImportButton />
        </Flex>
      </Flex>

      <WildcardList title={t('wildcards.myWildcards')} data={data.myWildcards} />
      {isMultiUser && <WildcardList title={t('wildcards.sharedWildcards')} data={data.sharedWildcards} />}
    </Flex>
  );
};
