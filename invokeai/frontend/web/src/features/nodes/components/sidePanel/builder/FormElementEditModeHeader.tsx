import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, forwardRef, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { ContainerElementSettings } from 'features/nodes/components/sidePanel/builder/ContainerElementSettings';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { NodeFieldElementSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementSettings';
import { formElementRemoved } from 'features/nodes/store/workflowSlice';
import { type FormElement, isContainerElement, isNodeFieldElement } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import { useIsRootElement } from './dnd-hooks';

const sx: SystemStyleObject = {
  w: 'full',
  ps: 2,
  h: 8,
  minH: 8,
  maxH: 8,
  borderTopRadius: 'base',
  alignItems: 'center',
  color: 'base.300',
  bg: 'baseAlpha.250',
  '&[data-depth="0"]': { bg: 'baseAlpha.100' },
  '&[data-depth="1"]': { bg: 'baseAlpha.150' },
  '&[data-depth="2"]': { bg: 'baseAlpha.200' },
  '&[data-is-root="false"]': { cursor: 'grab' },
};

export const FormElementEditModeHeader = memo(
  forwardRef(({ element }: { element: FormElement }, ref) => {
    const { t } = useTranslation();
    const depth = useDepthContext();
    const dispatch = useAppDispatch();
    const isRootElement = useIsRootElement(element.id);
    const removeElement = useCallback(() => {
      if (isRootElement) {
        return;
      }
      dispatch(formElementRemoved({ id: element.id }));
    }, [dispatch, element.id, isRootElement]);
    const label = useMemo(() => {
      if (isRootElement) {
        return 'Root Container';
      }
      if (isContainerElement(element) && element.data.layout === 'column') {
        return `Container (column layout)`;
      }
      if (isContainerElement(element) && element.data.layout === 'row') {
        return `Container (row layout)`;
      }
      return startCase(element.type);
    }, [element, isRootElement]);

    return (
      <Flex ref={ref} sx={sx} data-depth={depth} data-is-root={isRootElement}>
        <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
          {label}
        </Text>
        <Spacer />
        {isContainerElement(element) && !isRootElement && <ContainerElementSettings element={element} />}
        {isNodeFieldElement(element) && <NodeFieldElementSettings element={element} />}
        {!isRootElement && (
          <IconButton
            tooltip={t('common.delete')}
            aria-label={t('common.delete')}
            onClick={removeElement}
            icon={<PiXBold />}
            variant="link"
            size="sm"
            alignSelf="stretch"
            colorScheme="error"
          />
        )}
      </Flex>
    );
  })
);
FormElementEditModeHeader.displayName = 'FormElementEditModeHeader';
