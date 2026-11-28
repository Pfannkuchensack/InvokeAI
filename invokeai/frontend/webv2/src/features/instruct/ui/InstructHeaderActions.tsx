/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon } from '@chakra-ui/react';
import { useInstructActions, useInstructUi } from '@features/instruct/ui/InstructUiContext';
import { IconButton, Tooltip } from '@platform/ui';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { RotateCcwIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/** The chat's reset; the settings panel has no header actions of its own. */
export const InstructHeaderActions = ({ region }: { region?: string }) => {
  const { t } = useTranslation();
  const { canReset, hasUnreadableHistory, stepRuns } = useInstructUi();
  const { reset } = useInstructActions();
  const [isOpen, setIsOpen] = useState(false);

  if (region !== 'center') {
    return null;
  }

  const label = t('widgets.instruct.resetTitle');
  const hasRunningSteps = Object.keys(stepRuns).length > 0;
  const close = () => setIsOpen(false);

  return (
    <>
      <Tooltip content={label}>
        <IconButton
          aria-label={label}
          disabled={!canReset}
          size="sm"
          variant="ghost"
          // An unreadable history is never reset: skip the question and let the reset say so.
          onClick={hasUnreadableHistory ? reset : () => setIsOpen(true)}
        >
          <Icon as={RotateCcwIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
      <ConfirmDialog
        body={t(hasRunningSteps ? 'widgets.instruct.resetBodyRunning' : 'widgets.instruct.resetBody')}
        confirmLabel={t('widgets.instruct.resetConfirm')}
        isOpen={isOpen}
        title={label}
        onClose={close}
        onConfirm={() => {
          reset();
          close();
        }}
      />
    </>
  );
};
