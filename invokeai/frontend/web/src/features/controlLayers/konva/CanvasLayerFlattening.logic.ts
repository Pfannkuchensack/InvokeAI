import { getRectIntersection, getRectUnion } from 'features/controlLayers/konva/util';
import type { Rect } from 'features/controlLayers/store/types';
import stableHash from 'stable-hash';
import type { JsonObject } from 'type-fest';

/**
 * Structural shape the flattening logic needs from an adapter. The real `CanvasEntityAdapter` satisfies this
 * shape with extra fields; tests can satisfy it with lightweight stubs.
 */
export interface FlattenableAdapter {
  id: string;
  entityIdentifier: { type: string };
  renderer: { hasObjects: () => boolean };
  transformer: {
    getRelativeRect: () => Rect;
    $silentTransform: { get: () => boolean };
  };
  $isDisabled: { get: () => boolean };
  $isEmpty: { get: () => boolean };
  $isOnScreen: { get: () => boolean };
  getHashableState: () => JsonObject;
}

/**
 * Maximum dimension (in world pixels) for a flattened composite canvas. Set below every mainstream browser
 * limit so `canvas.getContext('2d')` is guaranteed to succeed: Safari iOS ≤ 4096², desktop Safari ≤ ~16M total
 * pixels, Firefox ≤ 11180² (≈125M px), Chromium ≤ 16384². 8192² sits safely below all of them.
 */
export const MAX_FLAT_CANVAS_DIMENSION = 8192;

/**
 * Resolves which adapter's layer should be on the stage. Filter / non-silent transform / segment targets take
 * priority over the selected entity, because their previews live inside the target adapter's own Konva layer —
 * if that layer is off-stage, the preview is invisible. Falls back to the selected entity.
 */
export const getStageActiveAdapter = <A extends FlattenableAdapter>(args: {
  filteringAdapter: A | null;
  transformingAdapter: A | null;
  segmentingAdapter: A | null;
  selectedId: string | null;
  findAdapterById: (id: string) => A | null;
}): A | null => {
  const { filteringAdapter, transformingAdapter, segmentingAdapter, selectedId, findAdapterById } = args;
  if (filteringAdapter) {
    return filteringAdapter;
  }
  if (transformingAdapter && !transformingAdapter.transformer.$silentTransform.get()) {
    return transformingAdapter;
  }
  if (segmentingAdapter) {
    return segmentingAdapter;
  }
  if (!selectedId) {
    return null;
  }
  return findAdapterById(selectedId);
};

/**
 * Splits adapters (in draw order) into behind/active/ahead around the given active adapter.
 */
export const splitAdapters = <A extends FlattenableAdapter>(
  allAdapters: A[],
  active: A | null
): { behind: A[]; active: A | null; ahead: A[] } => {
  if (!active) {
    return { behind: allAdapters, active: null, ahead: [] };
  }
  const idx = allAdapters.findIndex((a) => a.id === active.id);
  if (idx === -1) {
    return { behind: allAdapters, active: null, ahead: [] };
  }
  return {
    behind: allAdapters.slice(0, idx),
    active: allAdapters[idx]!,
    ahead: allAdapters.slice(idx + 1),
  };
};

/**
 * Computes the rect to rasterize for the given adapters. Normally this is the union of the adapters' rects, but
 * if that union exceeds `maxDim` (e.g. adapters scattered very far apart in world space), we clip to the stage-
 * visible rect so only what the user can actually see is rasterized. Returns null if there is nothing to render
 * or the clipped rect has no area.
 */
export const computeFlatRect = <A extends FlattenableAdapter>(
  adapters: A[],
  stageRect: Rect,
  maxDim: number = MAX_FLAT_CANVAS_DIMENSION
): Rect | null => {
  const rects: Rect[] = [];
  for (const adapter of adapters) {
    if (adapter.renderer.hasObjects()) {
      rects.push(adapter.transformer.getRelativeRect());
    }
  }
  if (rects.length === 0) {
    return null;
  }

  let rect = getRectUnion(...rects);

  if (rect.width > maxDim || rect.height > maxDim) {
    const intersection = getRectIntersection(rect, stageRect);
    if (intersection.width <= 0 || intersection.height <= 0) {
      return null;
    }
    rect = intersection;
  }

  // Final safety clamp — stage rect can itself exceed maxDim at extreme zoom-out.
  if (rect.width > maxDim || rect.height > maxDim) {
    rect = {
      x: rect.x,
      y: rect.y,
      width: Math.min(rect.width, maxDim),
      height: Math.min(rect.height, maxDim),
    };
  }

  return rect;
};

/**
 * Computes a hash for a set of adapters + the resolved render rect. The rect is part of the hash so that when
 * the union is clipped to the stage-visible rect (see `computeFlatRect`), panning/zooming invalidates the cache.
 * When no clipping applies, the rect is the union of adapter rects and only changes when adapter state changes.
 */
export const computeFlatHash = <A extends FlattenableAdapter>(adapters: A[], rect: Rect | null): string => {
  const adapterHashes: JsonObject[] = [];
  for (const adapter of adapters) {
    adapterHashes.push(adapter.getHashableState());
  }
  return stableHash({ adapterHashes, rect });
};

/**
 * Filters adapters to only those that should appear in a flattened composite. Excludes disabled, empty,
 * type-hidden, and off-screen entities. Under isolated-staging preview only raster layers pass.
 */
export const getVisibleAdaptersForFlattening = <A extends FlattenableAdapter>(args: {
  adapters: A[];
  isIsolatedStaging: boolean;
  isTypeHidden: (adapter: A) => boolean;
}): A[] => {
  const { adapters, isIsolatedStaging, isTypeHidden } = args;
  return adapters.filter((adapter) => {
    if (adapter.$isDisabled.get()) {
      return false;
    }
    if (adapter.$isEmpty.get()) {
      return false;
    }
    if (isTypeHidden(adapter)) {
      return false;
    }
    if (!adapter.$isOnScreen.get()) {
      return false;
    }
    if (isIsolatedStaging && adapter.entityIdentifier.type !== 'raster_layer') {
      return false;
    }
    return true;
  });
};
