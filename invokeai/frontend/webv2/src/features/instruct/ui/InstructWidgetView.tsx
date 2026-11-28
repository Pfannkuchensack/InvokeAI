import { InstructChatView } from './InstructChatView';
import { InstructSettingsView } from './InstructSettingsView';

/** One widget type with two faces, as Gallery has: the chat is the center view, the settings fill the side panels. */
export const InstructWidgetView = ({ region }: { region: string }) =>
  region === 'center' ? <InstructChatView /> : <InstructSettingsView />;
