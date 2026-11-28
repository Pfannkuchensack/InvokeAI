/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop -- a field in one panel section; props change on user edits only. */
import { DEFAULT_TIMESTEP_SHIFT, MAX_TIMESTEP_SHIFT, MIN_TIMESTEP_SHIFT } from '@features/generation/core/settings';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { useTranslation } from 'react-i18next';

import { GenerateToggleSwitch } from './GenerateToggleSwitch';

/** The useful shift range; the input accepts the node's full `ge=0` span up to 10. */
const TIMESTEP_SHIFT_SLIDER_MIN = 1;
const TIMESTEP_SHIFT_SLIDER_MAX = 7;
const TIMESTEP_SHIFT_MARKS = [{ label: String(DEFAULT_TIMESTEP_SHIFT), value: DEFAULT_TIMESTEP_SHIFT }];

/** Null lets the denoise node derive the shift from the image size; the override starts at the Lightning value. */
export const TimestepShiftField = ({
  helpText,
  onChange,
  shift,
}: {
  helpText: string;
  shift: number | null;
  onChange: (shift: number | null) => void;
}) => {
  const { t } = useTranslation();

  return (
    <GenerateToggleSwitch
      checked={shift !== null}
      helpText={helpText}
      hint="timestepShift"
      label={t('widgets.generate.timestepShift')}
      onCheckedChange={(checked) => onChange(checked ? DEFAULT_TIMESTEP_SHIFT : null)}
    >
      {shift !== null ? (
        <SliderNumberField
          ariaLabel={t('widgets.generate.timestepShift')}
          marks={TIMESTEP_SHIFT_MARKS}
          max={TIMESTEP_SHIFT_SLIDER_MAX}
          min={TIMESTEP_SHIFT_SLIDER_MIN}
          numberInputMax={MAX_TIMESTEP_SHIFT}
          numberInputMin={MIN_TIMESTEP_SHIFT}
          step={0.1}
          value={shift}
          // The input reports each keystroke, so typing 0.5 passes through 0. Settings read a shift outside its bounds
          // as auto, which would drop the override mid-edit; the input clamps it on blur instead.
          onChange={(next) => {
            if (next >= MIN_TIMESTEP_SHIFT && next <= MAX_TIMESTEP_SHIFT) {
              onChange(next);
            }
          }}
        />
      ) : null}
    </GenerateToggleSwitch>
  );
};
