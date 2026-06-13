import { Button, Collapse, Flex, Icon, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { fixTooltipCloseOnScrollStyles } from 'common/util/fixTooltipCloseOnScrollStyles';
import { selectWildcardSearchTerm } from 'features/wildcards/store/wildcardSlice';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import type { WildcardRecordDTO } from 'services/api/endpoints/wildcards';

import { WildcardListItem } from './WildcardListItem';

export const WildcardList = ({ title, data }: { title: string; data: WildcardRecordDTO[] }) => {
  const { t } = useTranslation();
  const { onToggle, isOpen } = useDisclosure({ defaultIsOpen: true });
  const searchTerm = useAppSelector(selectWildcardSearchTerm);

  return (
    <Flex flexDir="column">
      <Button variant="unstyled" onClick={onToggle}>
        <Flex gap={2} alignItems="center">
          <Icon boxSize={4} as={PiCaretDownBold} transform={isOpen ? undefined : 'rotate(-90deg)'} fill="base.500" />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {title}
          </Text>
        </Flex>
      </Button>
      <Collapse in={isOpen} style={fixTooltipCloseOnScrollStyles}>
        {data.length ? (
          data.map((wildcard) => <WildcardListItem wildcard={wildcard} key={wildcard.id} />)
        ) : (
          <IAINoContentFallback
            fontSize="sm"
            py={4}
            label={searchTerm ? t('wildcards.noMatchingWildcards') : t('wildcards.noWildcards')}
            icon={null}
          />
        )}
      </Collapse>
    </Flex>
  );
};
