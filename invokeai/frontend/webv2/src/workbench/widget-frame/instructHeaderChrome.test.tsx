import { instructWidgetManifest } from '@workbench/widgets/instruct/manifest';
import { describe, expect, it } from 'vitest';

import { hasWidgetHeaderActions } from './WidgetFrames';

describe('Edit header chrome', () => {
  it('shows an island in the center only because the chat contributes actions', () => {
    expect(hasWidgetHeaderActions({ actions: null, manifest: instructWidgetManifest, region: 'center' })).toBe(false);
    expect(hasWidgetHeaderActions({ actions: 'reset', manifest: instructWidgetManifest, region: 'center' })).toBe(true);
  });

  it('keeps the settings panel header as it was', () => {
    expect(hasWidgetHeaderActions({ actions: null, manifest: instructWidgetManifest, region: 'left' })).toBe(true);
  });
});
