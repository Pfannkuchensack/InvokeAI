import { Button, Flex, FormControl, FormLabel, Input, Spacer, Switch, Textarea } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { $wildcardModalState } from 'features/wildcards/store/wildcardModal';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';
import type { Control, SubmitHandler } from 'react-hook-form';
import { useController, useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useCreateWildcardMutation, useUpdateWildcardMutation } from 'services/api/endpoints/wildcards';

export type WildcardFormData = {
  name: string;
  values: string;
  is_public: boolean;
};

const splitValues = (values: string): string[] =>
  values
    .split('\n')
    .map((value) => value.trim())
    .filter((value) => value.length > 0);

export const WildcardForm = ({
  updatingWildcardId,
  formData,
}: {
  updatingWildcardId: string | null;
  formData: WildcardFormData | null;
}) => {
  const [createWildcard, { isLoading: isCreating }] = useCreateWildcardMutation();
  const [updateWildcard, { isLoading: isUpdating }] = useUpdateWildcardMutation();
  const { t } = useTranslation();

  // Sharing only makes sense on a multi-user instance, so hide the toggle otherwise.
  const { data: setupStatus } = useGetSetupStatusQuery();
  const isMultiUser = setupStatus?.multiuser_enabled ?? false;

  const { handleSubmit, control, register, formState } = useForm<WildcardFormData>({
    defaultValues: formData || {
      name: '',
      values: '',
      is_public: false,
    },
    mode: 'onChange',
  });

  const handleClickSave = useCallback<SubmitHandler<WildcardFormData>>(
    async (data) => {
      const values = splitValues(data.values);

      try {
        if (updatingWildcardId) {
          await updateWildcard({
            id: updatingWildcardId,
            changes: {
              name: data.name,
              values,
              is_public: data.is_public,
            },
          }).unwrap();
        } else {
          await createWildcard({
            name: data.name,
            values,
            is_public: data.is_public,
          }).unwrap();
        }
      } catch (error) {
        const detail =
          error && typeof error === 'object' && 'data' in error
            ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
              `${(error as any).data?.detail ?? ''}`
            : '';
        toast({
          status: 'error',
          title: t('wildcards.unableToSaveWildcard'),
          description: detail || undefined,
        });
        return;
      }

      $wildcardModalState.set({
        prefilledFormData: null,
        updatingWildcardId: null,
        isModalOpen: false,
      });
    },
    [updatingWildcardId, updateWildcard, createWildcard, t]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl orientation="vertical">
        <FormLabel>{t('wildcards.name')}</FormLabel>
        <Input size="md" {...register('name', { required: true, minLength: 1 })} />
      </FormControl>

      <FormControl orientation="vertical">
        <FormLabel>{t('wildcards.values')}</FormLabel>
        <Textarea
          rows={10}
          placeholder={t('wildcards.oneValuePerLine')}
          {...register('values', { validate: (value) => splitValues(value).length > 0 })}
        />
      </FormControl>

      {isMultiUser && <WildcardIsPublicField control={control} />}

      <Flex justifyContent="space-between" alignItems="flex-end" gap={10}>
        <Spacer />
        <Button
          onClick={handleSubmit(handleClickSave)}
          isDisabled={!formState.isValid}
          isLoading={isCreating || isUpdating}
        >
          {t('common.save')}
        </Button>
      </Flex>
    </Flex>
  );
};

const WildcardIsPublicField = ({ control }: { control: Control<WildcardFormData> }) => {
  const { t } = useTranslation();
  const { field } = useController({ control, name: 'is_public' });

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      field.onChange(e.target.checked);
    },
    [field]
  );

  return (
    <FormControl>
      <FormLabel>{t('wildcards.makePublic')}</FormLabel>
      <Switch isChecked={field.value} onChange={onChange} />
    </FormControl>
  );
};
