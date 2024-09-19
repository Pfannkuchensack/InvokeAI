import {
  Box,
  Flex,
  IconButton,
  Image,
  Popover,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverHeader,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import InvokeSymbol from 'public/assets/images/invoke-favicon.png';
import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from '../../../app/store/storeHooks';
import { PiLightbulbFilamentBold } from 'react-icons/pi';
import { CanvasV2Announcement } from './canvasV2Announcement';
import { useCallback } from 'react';
import { shouldShowNotificationIndicatorChanged } from '../store/uiSlice';

export const Notifications = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const shouldShowNotificationIndicator = useAppSelector((s) => s.ui.shouldShowNotificationIndicator);
  const resetIndicator = useCallback(() => {
    dispatch(shouldShowNotificationIndicatorChanged(false));
  }, [dispatch]);

  return (
    <Popover onOpen={resetIndicator} placement="top-start">
      <PopoverTrigger>
        <Flex pos="relative">
          <IconButton
            aria-label="Notifications"
            variant="link"
            icon={<PiLightbulbFilamentBold fontSize={20} />}
            boxSize={8}
          />
          {shouldShowNotificationIndicator && (
            <Box
              pos="absolute"
              top={0}
              right={'2px'}
              w={2}
              h={2}
              backgroundColor="invokeYellow.500"
              borderRadius="100%"
            />
          )}
        </Flex>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverCloseButton />
        <PopoverHeader fontSize="md" fontWeight="semibold">
          <Flex alignItems="center" gap={3}>
            <Image src={InvokeSymbol} boxSize={6} />
            {t('whatsNew.whatsNewInInvoke')}
          </Flex>
        </PopoverHeader>
        <PopoverBody p={2}>
          <CanvasV2Announcement />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};