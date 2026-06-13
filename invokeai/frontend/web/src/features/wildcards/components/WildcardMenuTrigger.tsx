import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $isWildcardsMenuOpen } from 'features/wildcards/store/wildcardSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

const _hover: SystemStyleObject = {
  bg: 'base.750',
};

export const WildcardMenuTrigger = () => {
  const isMenuOpen = useStore($isWildcardsMenuOpen);
  const { t } = useTranslation();

  const handleToggle = useCallback(() => {
    $isWildcardsMenuOpen.set(!isMenuOpen);
  }, [isMenuOpen]);

  return (
    <Flex
      onClick={handleToggle}
      backgroundColor="base.800"
      justifyContent="space-between"
      alignItems="center"
      py={2}
      px={3}
      borderRadius="base"
      gap={2}
      role="button"
      _hover={_hover}
      transitionProperty="background-color"
      transitionDuration="normal"
      w="full"
    >
      <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
        {t('wildcards.wildcards')}
      </Text>
      <IconButton aria-label={t('wildcards.viewList')} variant="ghost" icon={<PiCaretDownBold />} size="sm" />
    </Flex>
  );
};
