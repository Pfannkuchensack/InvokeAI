import { describe, expect, it } from 'vitest';

import { getWidgetTypeIdForSourceId } from './graphWidgets';
import { getLayoutPresetLabel, layoutPresets } from './layoutPresets';

describe('built-in layout preset descriptors', () => {
  it('has a widget instance for every instance id a preset places', () => {
    for (const preset of layoutPresets) {
      for (const region of Object.values(preset.snapshot.widgetRegions)) {
        for (const instanceId of region.instanceIds) {
          expect(preset.snapshot.widgetInstances[instanceId], `${preset.id} ${instanceId}`).toBeDefined();
        }
      }
    }
  });

  it('names each built-in preset through the locale, Automate as Workflow', () => {
    const t = (key: string) => `t:${key}`;

    expect(layoutPresets.map((preset) => getLayoutPresetLabel(preset, t))).toEqual([
      't:topbar.presets.builtInLabels.compose',
      't:topbar.presets.builtInLabels.edit',
      't:topbar.presets.builtInLabels.video',
      't:topbar.presets.builtInLabels.automate',
      't:topbar.presets.builtInLabels.instruct',
    ]);
    expect(getLayoutPresetLabel({ label: 'My layout' }, t)).toBe('My layout');
  });

  it('places the widget of its default route in every preset', () => {
    for (const preset of layoutPresets) {
      const placed = Object.values(preset.snapshot.widgetInstances).map((instance) => instance.typeId);

      expect(placed, preset.id).toContain(getWidgetTypeIdForSourceId(preset.defaultRoute.sourceId));
    }
  });
});
