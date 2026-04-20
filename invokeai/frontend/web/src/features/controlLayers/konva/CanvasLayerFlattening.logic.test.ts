import {
  computeFlatHash,
  computeFlatRect,
  type FlattenableAdapter,
  getStageActiveAdapter,
  getVisibleAdaptersForFlattening,
  MAX_FLAT_CANVAS_DIMENSION,
  splitAdapters,
} from 'features/controlLayers/konva/CanvasLayerFlattening.logic';
import type { Rect } from 'features/controlLayers/store/types';
import type { JsonObject } from 'type-fest';
import { describe, expect, it } from 'vitest';

type MakeAdapterArg = Partial<{
  id: string;
  type: string;
  hasObjects: boolean;
  rect: Rect;
  silentTransform: boolean;
  isDisabled: boolean;
  isEmpty: boolean;
  isOnScreen: boolean;
  hashableState: JsonObject;
}>;

const makeAdapter = (arg: MakeAdapterArg = {}): FlattenableAdapter => {
  const {
    id = 'a',
    type = 'raster_layer',
    hasObjects = true,
    rect = { x: 0, y: 0, width: 100, height: 100 },
    silentTransform = false,
    isDisabled = false,
    isEmpty = false,
    isOnScreen = true,
    hashableState = { id },
  } = arg;
  return {
    id,
    entityIdentifier: { type },
    renderer: { hasObjects: () => hasObjects },
    transformer: {
      getRelativeRect: () => rect,
      $silentTransform: { get: () => silentTransform },
    },
    $isDisabled: { get: () => isDisabled },
    $isEmpty: { get: () => isEmpty },
    $isOnScreen: { get: () => isOnScreen },
    getHashableState: () => hashableState,
  };
};

const STAGE_RECT: Rect = { x: 0, y: 0, width: 1920, height: 1080 };

describe('CanvasLayerFlattening.logic', () => {
  describe('getStageActiveAdapter', () => {
    const A = makeAdapter({ id: 'A' });
    const B = makeAdapter({ id: 'B' });
    const C = makeAdapter({ id: 'C' });
    const findById = (id: string) => [A, B, C].find((a) => a.id === id) ?? null;

    it('returns filtering adapter over selected', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: B,
        transformingAdapter: null,
        segmentingAdapter: null,
        selectedId: 'A',
        findAdapterById: findById,
      });
      expect(result).toBe(B);
    });

    it('returns non-silent transforming adapter over selected', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: null,
        transformingAdapter: B,
        segmentingAdapter: null,
        selectedId: 'A',
        findAdapterById: findById,
      });
      expect(result).toBe(B);
    });

    it('ignores silent transforming adapter and falls through to selected', () => {
      const silentB = makeAdapter({ id: 'B', silentTransform: true });
      const result = getStageActiveAdapter({
        filteringAdapter: null,
        transformingAdapter: silentB,
        segmentingAdapter: null,
        selectedId: 'A',
        findAdapterById: (id) => (id === 'A' ? A : null),
      });
      expect(result).toBe(A);
    });

    it('returns segmenting adapter over selected', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: null,
        transformingAdapter: null,
        segmentingAdapter: C,
        selectedId: 'A',
        findAdapterById: findById,
      });
      expect(result).toBe(C);
    });

    it('filter wins over transform + segment + selected', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: A,
        transformingAdapter: B,
        segmentingAdapter: C,
        selectedId: 'B',
        findAdapterById: findById,
      });
      expect(result).toBe(A);
    });

    it('returns selected adapter when no preview mode is active', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: null,
        transformingAdapter: null,
        segmentingAdapter: null,
        selectedId: 'A',
        findAdapterById: findById,
      });
      expect(result).toBe(A);
    });

    it('returns null when nothing is selected or preview-active', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: null,
        transformingAdapter: null,
        segmentingAdapter: null,
        selectedId: null,
        findAdapterById: findById,
      });
      expect(result).toBeNull();
    });

    it('returns null when selected id does not resolve', () => {
      const result = getStageActiveAdapter({
        filteringAdapter: null,
        transformingAdapter: null,
        segmentingAdapter: null,
        selectedId: 'ghost',
        findAdapterById: () => null,
      });
      expect(result).toBeNull();
    });
  });

  describe('splitAdapters', () => {
    const A = makeAdapter({ id: 'A' });
    const B = makeAdapter({ id: 'B' });
    const C = makeAdapter({ id: 'C' });
    const D = makeAdapter({ id: 'D' });
    const all = [A, B, C, D];

    it('splits around the active adapter', () => {
      const { behind, active, ahead } = splitAdapters(all, C);
      expect(behind).toEqual([A, B]);
      expect(active).toBe(C);
      expect(ahead).toEqual([D]);
    });

    it('active at head yields empty behind', () => {
      const { behind, active, ahead } = splitAdapters(all, A);
      expect(behind).toEqual([]);
      expect(active).toBe(A);
      expect(ahead).toEqual([B, C, D]);
    });

    it('active at tail yields empty ahead', () => {
      const { behind, active, ahead } = splitAdapters(all, D);
      expect(behind).toEqual([A, B, C]);
      expect(active).toBe(D);
      expect(ahead).toEqual([]);
    });

    it('null active puts everything behind', () => {
      const { behind, active, ahead } = splitAdapters(all, null);
      expect(behind).toBe(all);
      expect(active).toBeNull();
      expect(ahead).toEqual([]);
    });

    it('active not in list falls back to "everything behind"', () => {
      const orphan = makeAdapter({ id: 'orphan' });
      const { behind, active, ahead } = splitAdapters(all, orphan);
      expect(behind).toBe(all);
      expect(active).toBeNull();
      expect(ahead).toEqual([]);
    });
  });

  describe('computeFlatRect', () => {
    it('returns null for empty adapter list', () => {
      expect(computeFlatRect([], STAGE_RECT)).toBeNull();
    });

    it('returns null when no adapter has objects', () => {
      const empty = makeAdapter({ hasObjects: false });
      expect(computeFlatRect([empty], STAGE_RECT)).toBeNull();
    });

    it('returns the union rect when within limits', () => {
      const a = makeAdapter({ id: 'A', rect: { x: 0, y: 0, width: 100, height: 100 } });
      const b = makeAdapter({ id: 'B', rect: { x: 50, y: 50, width: 100, height: 100 } });
      const rect = computeFlatRect([a, b], STAGE_RECT);
      expect(rect).toEqual({ x: 0, y: 0, width: 150, height: 150 });
    });

    it('ignores adapters without objects when computing union', () => {
      const a = makeAdapter({ id: 'A', rect: { x: 0, y: 0, width: 100, height: 100 } });
      const b = makeAdapter({ id: 'B', hasObjects: false, rect: { x: 9999, y: 9999, width: 100, height: 100 } });
      const rect = computeFlatRect([a, b], STAGE_RECT);
      expect(rect).toEqual({ x: 0, y: 0, width: 100, height: 100 });
    });

    it('clips to stage rect when union would exceed browser canvas limit (H3)', () => {
      const a = makeAdapter({ id: 'A', rect: { x: 0, y: 0, width: 100, height: 100 } });
      const b = makeAdapter({ id: 'B', rect: { x: 20000, y: 20000, width: 100, height: 100 } });
      const rect = computeFlatRect([a, b], STAGE_RECT);
      expect(rect).not.toBeNull();
      expect(rect!.width).toBeLessThanOrEqual(MAX_FLAT_CANVAS_DIMENSION);
      expect(rect!.height).toBeLessThanOrEqual(MAX_FLAT_CANVAS_DIMENSION);
      // Clipped to stage rect
      expect(rect).toEqual(STAGE_RECT);
    });

    it('returns null when clipped intersection has zero area', () => {
      // Adapters are far from stage rect — union is huge, intersection with stage is empty
      const a = makeAdapter({ id: 'A', rect: { x: 50000, y: 50000, width: 100, height: 100 } });
      const b = makeAdapter({ id: 'B', rect: { x: 100000, y: 100000, width: 100, height: 100 } });
      const rect = computeFlatRect([a, b], STAGE_RECT);
      expect(rect).toBeNull();
    });

    it('final clamp when stage rect itself exceeds max dim (extreme zoom-out)', () => {
      const a = makeAdapter({ id: 'A', rect: { x: 0, y: 0, width: 100, height: 100 } });
      const b = makeAdapter({ id: 'B', rect: { x: 20000, y: 20000, width: 100, height: 100 } });
      const hugeStage: Rect = { x: -50000, y: -50000, width: 200000, height: 200000 };
      const rect = computeFlatRect([a, b], hugeStage);
      expect(rect).not.toBeNull();
      expect(rect!.width).toBeLessThanOrEqual(MAX_FLAT_CANVAS_DIMENSION);
      expect(rect!.height).toBeLessThanOrEqual(MAX_FLAT_CANVAS_DIMENSION);
    });

    it('respects custom maxDim parameter', () => {
      const a = makeAdapter({ id: 'A', rect: { x: 0, y: 0, width: 100, height: 100 } });
      const b = makeAdapter({ id: 'B', rect: { x: 5000, y: 5000, width: 100, height: 100 } });
      // Union is ~5100 — exceeds a 4096 cap, clips to stage.
      const rect = computeFlatRect([a, b], STAGE_RECT, 4096);
      expect(rect).toEqual(STAGE_RECT);
    });
  });

  describe('computeFlatHash', () => {
    it('returns stable hash for equal inputs', () => {
      const a = makeAdapter({ id: 'A', hashableState: { foo: 1 } });
      const rect: Rect = { x: 0, y: 0, width: 100, height: 100 };
      expect(computeFlatHash([a], rect)).toBe(computeFlatHash([a], rect));
    });

    it('changes hash when adapter state changes', () => {
      const a1 = makeAdapter({ id: 'A', hashableState: { foo: 1 } });
      const a2 = makeAdapter({ id: 'A', hashableState: { foo: 2 } });
      const rect: Rect = { x: 0, y: 0, width: 100, height: 100 };
      expect(computeFlatHash([a1], rect)).not.toBe(computeFlatHash([a2], rect));
    });

    it('changes hash when rect changes (H3: pan/zoom invalidates cache under clipping)', () => {
      const a = makeAdapter({ id: 'A', hashableState: { foo: 1 } });
      const r1: Rect = { x: 0, y: 0, width: 100, height: 100 };
      const r2: Rect = { x: 10, y: 10, width: 100, height: 100 };
      expect(computeFlatHash([a], r1)).not.toBe(computeFlatHash([a], r2));
    });

    it('changes hash when adapter list changes', () => {
      const a = makeAdapter({ id: 'A', hashableState: { foo: 1 } });
      const b = makeAdapter({ id: 'B', hashableState: { foo: 2 } });
      const rect: Rect = { x: 0, y: 0, width: 100, height: 100 };
      expect(computeFlatHash([a], rect)).not.toBe(computeFlatHash([a, b], rect));
    });

    it('handles null rect', () => {
      const a = makeAdapter({ id: 'A' });
      expect(() => computeFlatHash([a], null)).not.toThrow();
      expect(computeFlatHash([a], null)).toBe(computeFlatHash([a], null));
    });

    it('order matters (draw order is part of the state)', () => {
      const a = makeAdapter({ id: 'A' });
      const b = makeAdapter({ id: 'B' });
      const rect: Rect = { x: 0, y: 0, width: 100, height: 100 };
      expect(computeFlatHash([a, b], rect)).not.toBe(computeFlatHash([b, a], rect));
    });
  });

  describe('getVisibleAdaptersForFlattening', () => {
    const neverTypeHidden = () => false;

    it('excludes disabled adapters', () => {
      const a = makeAdapter({ id: 'A' });
      const b = makeAdapter({ id: 'B', isDisabled: true });
      const result = getVisibleAdaptersForFlattening({
        adapters: [a, b],
        isIsolatedStaging: false,
        isTypeHidden: neverTypeHidden,
      });
      expect(result).toEqual([a]);
    });

    it('excludes empty adapters', () => {
      const a = makeAdapter({ id: 'A' });
      const b = makeAdapter({ id: 'B', isEmpty: true });
      const result = getVisibleAdaptersForFlattening({
        adapters: [a, b],
        isIsolatedStaging: false,
        isTypeHidden: neverTypeHidden,
      });
      expect(result).toEqual([a]);
    });

    it('excludes type-hidden adapters', () => {
      const a = makeAdapter({ id: 'A', type: 'raster_layer' });
      const b = makeAdapter({ id: 'B', type: 'inpaint_mask' });
      const result = getVisibleAdaptersForFlattening({
        adapters: [a, b],
        isIsolatedStaging: false,
        isTypeHidden: (adapter) => adapter.entityIdentifier.type === 'inpaint_mask',
      });
      expect(result).toEqual([a]);
    });

    it('excludes off-screen adapters (M2)', () => {
      const a = makeAdapter({ id: 'A', isOnScreen: true });
      const b = makeAdapter({ id: 'B', isOnScreen: false });
      const result = getVisibleAdaptersForFlattening({
        adapters: [a, b],
        isIsolatedStaging: false,
        isTypeHidden: neverTypeHidden,
      });
      expect(result).toEqual([a]);
    });

    it('under isolated staging, excludes non-raster-layer types', () => {
      const raster = makeAdapter({ id: 'R', type: 'raster_layer' });
      const mask = makeAdapter({ id: 'M', type: 'inpaint_mask' });
      const control = makeAdapter({ id: 'C', type: 'control_layer' });
      const result = getVisibleAdaptersForFlattening({
        adapters: [raster, mask, control],
        isIsolatedStaging: true,
        isTypeHidden: neverTypeHidden,
      });
      expect(result).toEqual([raster]);
    });

    it('includes everything when nothing excludes', () => {
      const a = makeAdapter({ id: 'A' });
      const b = makeAdapter({ id: 'B' });
      const result = getVisibleAdaptersForFlattening({
        adapters: [a, b],
        isIsolatedStaging: false,
        isTypeHidden: neverTypeHidden,
      });
      expect(result).toEqual([a, b]);
    });
  });
});
