/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop -- a field in one panel section; props change on user edits only. */
import type { QwenImageEncoderQuantization } from '@features/generation/core/types';

import { createListCollection } from '@chakra-ui/react';
import { isQwenImageEncoderQuantization, QWEN_IMAGE_ENCODER_QUANTIZATIONS } from '@features/generation/core/settings';
import { Field, Select } from '@platform/ui';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ENCODER_QUANTIZATION_LABEL_KEYS: Record<QwenImageEncoderQuantization, string> = {
  int8: 'widgets.generate.encoderQuantizationInt8',
  nf4: 'widgets.generate.encoderQuantizationNf4',
  none: 'widgets.generate.encoderQuantizationNone',
};

/** The Qwen VL encoder's load-time quantization choice, as Generate's Advanced section and Edit offer it. */
export const QwenImageEncoderQuantizationField = ({
  onChange,
  value,
}: {
  value: QwenImageEncoderQuantization;
  onChange: (quantization: QwenImageEncoderQuantization) => void;
}) => {
  const { t } = useTranslation();
  const collection = useMemo(
    () =>
      createListCollection({
        items: QWEN_IMAGE_ENCODER_QUANTIZATIONS.map((item) => ({
          label: t(ENCODER_QUANTIZATION_LABEL_KEYS[item]),
          value: item,
        })),
      }),
    [t]
  );

  return (
    <Field
      hint="encoderQuantization"
      label={t('widgets.generate.encoderQuantization')}
      labelPlacement="inside"
      helpText={t('widgets.generate.encoderQuantizationHelp')}
    >
      <Select
        aria-label={t('widgets.generate.encoderQuantization')}
        collection={collection}
        value={[value]}
        onValueChange={({ value: picked }) => {
          const quantization = picked[0];

          if (isQwenImageEncoderQuantization(quantization)) {
            onChange(quantization);
          }
        }}
      />
    </Field>
  );
};
