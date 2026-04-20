import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import {
  computeFlatHash as _computeFlatHash,
  computeFlatRect as _computeFlatRect,
  getStageActiveAdapter as _getStageActiveAdapter,
  getVisibleAdaptersForFlattening as _getVisibleAdaptersForFlattening,
  splitAdapters as _splitAdapters,
} from 'features/controlLayers/konva/CanvasLayerFlattening.logic';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectIsolatedLayerPreview,
  selectIsolatedStagingPreview,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasEntityState, Rect } from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import Konva from 'konva';
import rafThrottle from 'raf-throttle';
import type { Logger } from 'roarr';
import { assert } from 'tsafe';

/**
 * Reduces the number of active HTML canvas elements from N (one per entity) to 3 (constant).
 *
 * Architecture:
 * - behindLayer: flattened composite of all entities below the active (selected) entity
 * - active entity layer: the selected entity's own Konva.Layer, fully interactive
 * - aheadLayer: flattened composite of all entities above the active entity
 *
 * Entity adapters still create their own Konva.Layer for off-screen rendering (getCanvas()),
 * but only the active entity's layer is added to the Konva stage.
 */
export class CanvasLayerFlatteningModule extends CanvasModuleBase {
  readonly type = 'layer_flattening';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  subscriptions = new Set<() => void>();

  konva: {
    behindLayer: Konva.Layer;
    aheadLayer: Konva.Layer;
    behindImage: Konva.Image | null;
    aheadImage: Konva.Image | null;
  };

  /**
   * Hash of the behind composite, for change detection.
   */
  private _behindHash: string | null = null;

  /**
   * Hash of the ahead composite, for change detection.
   */
  private _aheadHash: string | null = null;

  /**
   * The ID of the currently active (on-stage) entity adapter.
   */
  private _activeAdapterId: string | null = null;

  /**
   * Persistent canvas buffers for the behind/ahead composites. Reused across re-flattens to avoid GC pressure
   * from per-frame `document.createElement('canvas')` during e.g. opacity-slider drags. Each `_update` resizes
   * and redraws the same canvas; we then force a `batchDraw()` since Konva doesn't detect mutations of an image
   * whose attribute reference is unchanged.
   */
  private _behindBuffer: HTMLCanvasElement | null = null;
  private _aheadBuffer: HTMLCanvasElement | null = null;

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('layer_flattening');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = {
      behindLayer: new Konva.Layer({
        name: 'layer_flattening:behind',
        listening: false,
        imageSmoothingEnabled: false,
      }),
      aheadLayer: new Konva.Layer({
        name: 'layer_flattening:ahead',
        listening: false,
        imageSmoothingEnabled: false,
      }),
      behindImage: null,
      aheadImage: null,
    };
  }

  initialize = () => {
    this.log.debug('Initializing module');

    // Subscribe to canvas state changes (entity list, selection, entity state)
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSlice, this.requestUpdate));

    // Subscribe to isolated preview modes
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsolatedLayerPreview, this.requestUpdate)
    );
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsolatedStagingPreview, this.requestUpdate)
    );

    // Subscribe to filtering/transforming/segmenting adapters
    this.subscriptions.add(this.manager.stateApi.$filteringAdapter.listen(this.requestUpdate));
    this.subscriptions.add(this.manager.stateApi.$transformingAdapter.listen(this.requestUpdate));
    this.subscriptions.add(this.manager.stateApi.$segmentingAdapter.listen(this.requestUpdate));

    // Stage pan/zoom can change the clipped flat rect when the adapter union exceeds the browser canvas
    // size limit (see computeFlatRect). The hash encodes the resolved rect, so when no clipping applies
    // this listener no-ops; when clipping is active it keeps the composite aligned with the viewport.
    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.requestUpdate));

    // Subscribe to staging state
    this.subscriptions.add(this.manager.stagingArea.$isStaging.listen(this.requestUpdate));

    // Do initial update
    this._update();
  };

  /**
   * Returns all entity adapters in draw order: rasterLayers → controlLayers → regions → inpaintMasks.
   * This mirrors the order used by CanvasEntityRendererModule.arrangeEntities.
   */
  getAdaptersInDrawOrder = (): CanvasEntityAdapter[] => {
    const adapters: CanvasEntityAdapter[] = [];
    const stateApi = this.manager.stateApi;

    const pushAdapters = (entities: CanvasEntityState[]) => {
      for (const entity of entities) {
        const adapter = this.manager.getAdapter(getEntityIdentifier(entity));
        if (adapter) {
          adapters.push(adapter);
        }
      }
    };

    pushAdapters(stateApi.getRasterLayersState().entities);
    pushAdapters(stateApi.getControlLayersState().entities);
    pushAdapters(stateApi.getRegionsState().entities);
    pushAdapters(stateApi.getInpaintMasksState().entities);

    return adapters;
  };

  getStageActiveAdapter = (): CanvasEntityAdapter | null =>
    _getStageActiveAdapter<CanvasEntityAdapter>({
      filteringAdapter: this.manager.stateApi.$filteringAdapter.get(),
      transformingAdapter: this.manager.stateApi.$transformingAdapter.get(),
      segmentingAdapter: this.manager.stateApi.$segmentingAdapter.get(),
      selectedId: this.manager.stateApi.getCanvasState().selectedEntityIdentifier?.id ?? null,
      findAdapterById: this.findAdapterById,
    });

  splitAdapters = (): {
    behind: CanvasEntityAdapter[];
    active: CanvasEntityAdapter | null;
    ahead: CanvasEntityAdapter[];
  } => _splitAdapters(this.getAdaptersInDrawOrder(), this.getStageActiveAdapter());

  getVisibleAdaptersForFlattening = (adapters: CanvasEntityAdapter[]): CanvasEntityAdapter[] => {
    const isIsolatedStaging =
      this.manager.stateApi.runSelector(selectIsolatedStagingPreview) && this.manager.stagingArea.$isStaging.get();
    const stateApi = this.manager.stateApi;
    return _getVisibleAdaptersForFlattening({
      adapters,
      isIsolatedStaging,
      isTypeHidden: (adapter) => stateApi.runSelector(adapter.selectIsTypeHidden),
    });
  };

  computeFlatHash = (adapters: CanvasEntityAdapter[], rect: Rect | null): string => _computeFlatHash(adapters, rect);

  computeFlatRect = (adapters: CanvasEntityAdapter[]): Rect | null =>
    _computeFlatRect(adapters, this.manager.stage.getScaledStageRect());

  /**
   * Renders a set of adapters into a single flat canvas at the given rect.
   * Reuses the same compositing pattern as CanvasCompositorModule.getCompositeCanvas.
   *
   * If `buffer` is provided, it's resized and drawn into in-place (avoids allocating a new canvas per frame
   * during e.g. opacity-slider drags). Setting `canvas.width`/`height` resets pixel contents per HTML spec.
   *
   * @returns The flat canvas, or null if no adapters to render.
   */
  flattenToCanvas = (
    adapters: CanvasEntityAdapter[],
    rect: Rect,
    buffer?: HTMLCanvasElement
  ): { canvas: HTMLCanvasElement; rect: Rect } | null => {
    if (adapters.length === 0) {
      return null;
    }

    const canvas = buffer ?? document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null, 'Canvas 2D context is null');
    ctx.imageSmoothingEnabled = false;
    // Defensive clear: setting width/height to the same value does not reliably reset pixel contents in all
    // browsers when the canvas is being reused.
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const adapter of adapters) {
      // Apply globalCompositeOperation for raster/control layers (same as CanvasCompositorModule)
      const layerCompositeOp =
        adapter.state.type === 'raster_layer' || adapter.state.type === 'control_layer'
          ? (adapter.state as { globalCompositeOperation?: GlobalCompositeOperation }).globalCompositeOperation
          : undefined;
      ctx.globalCompositeOperation = layerCompositeOp || 'source-over';

      const adapterCanvas = adapter.getDisplayCanvas(rect);
      ctx.drawImage(adapterCanvas, 0, 0);
    }

    return { canvas, rect };
  };

  /**
   * Updates a composite layer's Konva.Image with a flattened canvas result.
   */
  private updateFlatImage = (
    layer: Konva.Layer,
    imageKey: 'behindImage' | 'aheadImage',
    result: { canvas: HTMLCanvasElement; rect: Rect } | null
  ): void => {
    if (!result) {
      // No content — hide the layer and remove any existing image
      layer.visible(false);
      if (this.konva[imageKey]) {
        this.konva[imageKey].destroy();
        this.konva[imageKey] = null;
      }
      return;
    }

    const { canvas, rect } = result;

    if (this.konva[imageKey]) {
      // Update existing image
      this.konva[imageKey].image(canvas);
      this.konva[imageKey].setAttrs({
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      });
    } else {
      // Create new Konva.Image
      const image = new Konva.Image({
        name: `layer_flattening:${imageKey}`,
        image: canvas,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        listening: false,
        imageSmoothingEnabled: false,
      });
      layer.add(image);
      this.konva[imageKey] = image;
    }

    layer.visible(true);
    // With the canvas-buffer pool, the Konva.Image's `image` attribute keeps the same HTMLCanvasElement
    // reference across re-flattens. Konva treats that as unchanged and won't auto-redraw, so force it.
    layer.batchDraw();
  };

  /**
   * Arranges z-indices of all layers on the stage:
   * background(0) → behind(1) → active(2) → ahead(3) → preview(4)
   */
  private arrangeZIndices = (): void => {
    let zIndex = 0;
    this.manager.background.konva.layer.zIndex(zIndex++);
    this.konva.behindLayer.zIndex(zIndex++);

    // The active entity's layer is on the stage — set its z-index
    if (this._activeAdapterId) {
      const activeAdapter = this.findAdapterById(this._activeAdapterId);
      if (activeAdapter && activeAdapter.konva.layer.getStage()) {
        activeAdapter.konva.layer.zIndex(zIndex++);
      }
    }

    this.konva.aheadLayer.zIndex(zIndex++);
    this.manager.konva.previewLayer.zIndex(zIndex++);
  };

  /**
   * Checks if we are in an isolated preview mode (filtering, transforming, or segmenting
   * with isolatedLayerPreview enabled).
   */
  private isInIsolatedPreview = (): boolean => {
    if (!this.manager.stateApi.runSelector(selectIsolatedLayerPreview)) {
      return false;
    }

    const filteringAdapter = this.manager.stateApi.$filteringAdapter.get();
    if (filteringAdapter) {
      return true;
    }

    const transformingAdapter = this.manager.stateApi.$transformingAdapter.get();
    if (transformingAdapter && !transformingAdapter.transformer.$silentTransform.get()) {
      return true;
    }

    const segmentingAdapter = this.manager.stateApi.$segmentingAdapter.get();
    if (segmentingAdapter) {
      return true;
    }

    return false;
  };

  /**
   * The core update method. Recomputes behind/ahead composites and manages
   * which entity layer is on stage.
   */
  private _update = (): void => {
    const { behind, active, ahead } = this.splitAdapters();

    // Handle isolated preview modes — hide composites, only the isolated entity is visible
    if (this.isInIsolatedPreview()) {
      this.konva.behindLayer.visible(false);
      this.konva.aheadLayer.visible(false);

      // Ensure the active entity's layer is on stage (the isolated entity manages its own visibility)
      if (active && this._activeAdapterId !== active.id) {
        this.swapActiveLayer(active);
      }

      this.arrangeZIndices();
      return;
    }

    // Swap the active entity's layer on/off stage if selection changed
    if (active) {
      if (this._activeAdapterId !== active.id) {
        this.swapActiveLayer(active);
      }
    } else {
      // No active entity — remove any existing active layer from stage
      if (this._activeAdapterId) {
        this.removeActiveLayerFromStage();
      }
    }

    // Flatten the behind composite
    const visibleBehind = this.getVisibleAdaptersForFlattening(behind);
    const behindRect = this.computeFlatRect(visibleBehind);
    const behindHash = this.computeFlatHash(visibleBehind, behindRect);
    if (behindHash !== this._behindHash) {
      this.log.trace('Re-flattening behind composite');
      const behindResult = behindRect
        ? this.flattenToCanvas(visibleBehind, behindRect, this._behindBuffer ?? undefined)
        : null;
      if (behindResult) {
        this._behindBuffer = behindResult.canvas;
      }
      this.updateFlatImage(this.konva.behindLayer, 'behindImage', behindResult);
      this._behindHash = behindHash;
    }

    // Flatten the ahead composite
    const visibleAhead = this.getVisibleAdaptersForFlattening(ahead);
    const aheadRect = this.computeFlatRect(visibleAhead);
    const aheadHash = this.computeFlatHash(visibleAhead, aheadRect);
    if (aheadHash !== this._aheadHash) {
      this.log.trace('Re-flattening ahead composite');
      const aheadResult = aheadRect
        ? this.flattenToCanvas(visibleAhead, aheadRect, this._aheadBuffer ?? undefined)
        : null;
      if (aheadResult) {
        this._aheadBuffer = aheadResult.canvas;
      }
      this.updateFlatImage(this.konva.aheadLayer, 'aheadImage', aheadResult);
      this._aheadHash = aheadHash;
    }

    this.arrangeZIndices();
  };

  /**
   * RAF-throttled update method. This is the public entry point.
   */
  requestUpdate = rafThrottle(() => {
    this._update();
  });

  /**
   * Swaps the active entity layer on stage: removes the old one, adds the new one.
   */
  private swapActiveLayer = (newActive: CanvasEntityAdapter): void => {
    // Remove old active layer from stage
    this.removeActiveLayerFromStage();

    // Add new active layer to stage
    this.manager.stage.addLayer(newActive.konva.layer);
    this._activeAdapterId = newActive.id;

    // Invalidate both hashes to force re-flatten (the active entity is no longer in either composite)
    this._behindHash = null;
    this._aheadHash = null;

    this.log.trace({ activeAdapterId: this._activeAdapterId }, 'Swapped active layer');
  };

  /**
   * Removes the currently active adapter's layer from the stage.
   */
  private removeActiveLayerFromStage = (): void => {
    if (!this._activeAdapterId) {
      return;
    }

    // Find the old active adapter across all adapter maps
    const oldAdapter = this.findAdapterById(this._activeAdapterId);
    if (oldAdapter && oldAdapter.konva.layer.getStage()) {
      this.manager.stage.removeLayer(oldAdapter.konva.layer);
    }

    this._activeAdapterId = null;
  };

  /**
   * Finds an adapter by ID across all adapter maps.
   */
  private findAdapterById = (id: string): CanvasEntityAdapter | null => {
    return (
      this.manager.adapters.rasterLayers.get(id) ??
      this.manager.adapters.controlLayers.get(id) ??
      this.manager.adapters.regionMasks.get(id) ??
      this.manager.adapters.inpaintMasks.get(id) ??
      null
    );
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.behindImage?.destroy();
    this.konva.aheadImage?.destroy();
    this.konva.behindLayer.destroy();
    this.konva.aheadLayer.destroy();
    this._behindBuffer = null;
    this._aheadBuffer = null;
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      activeAdapterId: this._activeAdapterId,
      behindHash: this._behindHash,
      aheadHash: this._aheadHash,
    };
  };
}
