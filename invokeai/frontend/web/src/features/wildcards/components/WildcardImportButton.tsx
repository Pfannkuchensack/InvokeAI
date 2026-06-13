import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, ListItem, spinAnimation, Text, UnorderedList } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiSpinner, PiUploadSimpleBold } from 'react-icons/pi';
import { useImportWildcardsMutation } from 'services/api/endpoints/wildcards';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const WildcardImportButton = () => {
  const [importWildcards, { isLoading }] = useImportWildcardsMutation();
  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      if (!files.length) {
        return;
      }
      importWildcards(files)
        .unwrap()
        .then(() => {
          toast({
            status: 'success',
            title: t('toast.importSuccessful'),
          });
        })
        .catch((error) => {
          toast({
            status: 'error',
            title: t('toast.importFailed'),
            description: error ? `${error.data?.detail}` : undefined,
          });
        });
    },
    [importWildcards, t]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'text/plain': ['.txt'], 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: true,
  });

  return (
    <>
      <IconButton
        size="sm"
        variant="link"
        alignSelf="stretch"
        icon={!isLoading ? <PiUploadSimpleBold /> : <PiSpinner />}
        tooltip={<TooltipContent />}
        aria-label={t('wildcards.importWildcards')}
        sx={isLoading ? loadingStyles : undefined}
        isDisabled={isLoading}
        {...getRootProps()}
      />
      <input {...getInputProps()} />
    </>
  );
};

const TooltipContent = () => {
  const { t } = useTranslation();
  return (
    <Flex flexDir="column">
      <Text pb={1} fontWeight="semibold">
        {t('wildcards.importWildcards')}
      </Text>
      <Text>{t('wildcards.importDesc')}</Text>
      <UnorderedList>
        <ListItem>{t('wildcards.importTxtDesc')}</ListItem>
        <ListItem>{t('wildcards.importJsonDesc')}</ListItem>
      </UnorderedList>
    </Flex>
  );
};
