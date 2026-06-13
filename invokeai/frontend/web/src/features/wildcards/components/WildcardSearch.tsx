import { IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWildcardSearchTerm, wildcardSearchTermChanged } from 'features/wildcards/store/wildcardSlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const WildcardSearch = () => {
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectWildcardSearchTerm);
  const { t } = useTranslation();

  const handleSearch = useCallback(
    (newSearchTerm: string) => {
      dispatch(wildcardSearchTermChanged(newSearchTerm));
    },
    [dispatch]
  );

  const clearSearch = useCallback(() => {
    dispatch(wildcardSearchTermChanged(''));
  }, [dispatch]);

  const handleKeydown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      // exit search mode on escape
      if (e.key === 'Escape') {
        clearSearch();
      }
    },
    [clearSearch]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      handleSearch(e.target.value);
    },
    [handleSearch]
  );

  return (
    <InputGroup>
      <Input
        placeholder={t('wildcards.searchByName')}
        value={searchTerm}
        onKeyDown={handleKeydown}
        onChange={handleChange}
      />
      {searchTerm && searchTerm.length && (
        <InputRightElement h="full" pe={2}>
          <IconButton
            onClick={clearSearch}
            size="sm"
            variant="link"
            aria-label={t('boards.clearSearch')}
            icon={<PiXBold />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
};

export default memo(WildcardSearch);
