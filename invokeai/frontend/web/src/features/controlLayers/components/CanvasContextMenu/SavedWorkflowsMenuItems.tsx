import { MenuItem, Spinner } from '@invoke-ai/ui-library';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlayBold } from 'react-icons/pi';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';

type SavedWorkflowsMenuItemsProps = {
  onSelect: (workflowId: string) => void;
  isBusy: boolean;
};

export const SavedWorkflowsMenuItems = memo(({ onSelect, isBusy }: SavedWorkflowsMenuItemsProps) => {
  const { t } = useTranslation();

  const queryArg = useMemo(
    () => ({
      page: 0,
      per_page: 20,
      order_by: 'opened_at' as const,
      direction: 'DESC' as const,
      categories: ['user'] as ('user' | 'default')[],
    }),
    []
  );

  const { data, isLoading } = useListWorkflowsInfiniteInfiniteQuery({ queryArg, pageParam: 0 });

  const workflows = useMemo(() => {
    if (!data?.pages) {
      return [];
    }
    return data.pages.flatMap((page) => page.items);
  }, [data]);

  if (isLoading) {
    return (
      <MenuItem isDisabled icon={<Spinner size="xs" />}>
        Loading...
      </MenuItem>
    );
  }

  if (workflows.length === 0) {
    return (
      <MenuItem isDisabled>
        {t('controlLayers.canvasContextMenu.noSavedWorkflows')}
      </MenuItem>
    );
  }

  return (
    <>
      {workflows.map((workflow) => (
        <MenuItem
          key={workflow.workflow_id}
          icon={<PiPlayBold />}
          isDisabled={isBusy}
          onClick={() => onSelect(workflow.workflow_id)}
        >
          {workflow.name || workflow.workflow_id}
        </MenuItem>
      ))}
    </>
  );
});

SavedWorkflowsMenuItems.displayName = 'SavedWorkflowsMenuItems';
