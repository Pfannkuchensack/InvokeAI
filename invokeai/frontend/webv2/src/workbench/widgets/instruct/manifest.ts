import type { WidgetManifest } from '@workbench/widgetContracts';

import { WandSparklesIcon } from 'lucide-react';

export const instructWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'center'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'instruct-graph', sourceId: 'instruct', surfaces: ['left'] },
  icon: WandSparklesIcon,
  id: 'instruct',
  label: (t) => t('widgets.labels.instruct'),
  load: () => import('@features/instruct/widget').then((module) => module.widgetImplementation),
  version: 1,
};
