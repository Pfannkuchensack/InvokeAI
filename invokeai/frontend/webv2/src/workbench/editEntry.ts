import type { ImageWithDims } from '@features/generation/contracts';

import { requestInstructEntry } from '@features/instruct';

import type { Project } from './projectContracts';
import type { WorkbenchCommands } from './workbenchStore';

import { getEditableEditTree } from './projectEditTree';

export interface EditEntryTarget {
  commands: Pick<WorkbenchCommands, 'layout' | 'widgets'>;
  getProject: (projectId: string) => Project | null | undefined;
  /** An entry made after an await must not land on a project the user has since left. */
  isActiveProject: (projectId: string) => boolean;
  projectId: string;
}

/**
 * Makes the Edit chat the view on screen without touching the arrangement the user made. A hidden chat view runs no
 * effects, so a request would otherwise wait for them.
 * - A chat that is a center view comes forward (any one: the preset's, or a custom preset's); one already in front stays.
 * - A floating instance is never the chat: it renders the settings view, so it does not count.
 * - On the Edit preset without a chat (closed), it is opened in the center: applying the preset again would revert the
 *   user's unsaved layout changes.
 * - Any other layout switches to the Edit preset.
 */
export const showEditWorkspace = ({ commands, getProject, isActiveProject, projectId }: EditEntryTarget): void => {
  const project = getProject(projectId);

  if (!project || !isActiveProject(projectId)) {
    return;
  }

  const isChat = (instanceId: string | null | undefined) =>
    instanceId !== null && instanceId !== undefined && project.widgetInstances[instanceId]?.typeId === 'instruct';
  const center = project.widgetRegions.center;
  const centerChat = center.instanceIds.find(isChat);

  if (isChat(center.activeInstanceId)) {
    return;
  }

  if (centerChat) {
    commands.widgets.select({ projectId, region: 'center', widgetId: centerChat });
  } else if (project.layout.presetId === 'instruct') {
    // A second instance beside the settings panel's: the chat view is the center region's rendering of the widget.
    commands.widgets.open({ createNew: true, projectId, region: 'center', widgetId: 'instruct' });
  } else {
    void commands.layout.activatePreset('instruct');
  }
};

/** A stored history this build cannot read is never replaced, so nothing can start a new one in its place. */
export const hasUnreadableEditHistory = (project: Pick<Project, 'editTree'> | null | undefined): boolean =>
  project?.editTree !== undefined && getEditableEditTree(project.editTree) === null;

/** Sends an image to Edit. The view starts the history from it, jumps to it, or asks before replacing the history. */
export const openImageInEdit = (target: EditEntryTarget, image: ImageWithDims): void => {
  if (!target.isActiveProject(target.projectId)) {
    return;
  }

  requestInstructEntry({ image, projectId: target.projectId });
  showEditWorkspace(target);
};
