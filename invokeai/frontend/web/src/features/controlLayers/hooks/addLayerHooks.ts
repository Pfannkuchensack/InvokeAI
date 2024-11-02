import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  rasterLayerAdded,
  referenceImageAdded,
  rgAdded,
  rgIPAdapterAdded,
  rgNegativePromptChanged,
  rgPositivePromptChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CanvasRegionalGuidanceState,
  ControlNetConfig,
  IPAdapterConfig,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import { initialControlNet, initialIPAdapter, initialT2IAdapter } from 'features/controlLayers/store/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import type { ControlNetModelConfig, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { isControlNetOrT2IAdapterModelConfig, isIPAdapterModelConfig } from 'services/api/types';

/** @knipignore */
export const selectDefaultControlAdapter = createSelector(
  selectModelConfigsQuery,
  selectBase,
  (query, base): ControlNetConfig | T2IAdapterConfig => {
    const { data } = query;
    let model: ControlNetModelConfig | T2IAdapterModelConfig | null = null;
    if (data) {
      const modelConfigs = modelConfigsAdapterSelectors
        .selectAll(data)
        .filter(isControlNetOrT2IAdapterModelConfig)
        .sort((a) => (a.type === 'controlnet' ? -1 : 1)); // Prefer ControlNet models
      const compatibleModels = modelConfigs.filter((m) => (base ? m.base === base : true));
      model = compatibleModels[0] ?? modelConfigs[0] ?? null;
    }
    const controlAdapter = model?.type === 't2i_adapter' ? deepClone(initialT2IAdapter) : deepClone(initialControlNet);
    if (model) {
      controlAdapter.model = zModelIdentifierField.parse(model);
    }
    return controlAdapter;
  }
);

export const selectDefaultIPAdapter = createSelector(
  selectModelConfigsQuery,
  selectBase,
  (query, base): IPAdapterConfig => {
    const { data } = query;
    let model: IPAdapterModelConfig | null = null;
    if (data) {
      const modelConfigs = modelConfigsAdapterSelectors.selectAll(data).filter(isIPAdapterModelConfig);
      const compatibleModels = modelConfigs.filter((m) => (base ? m.base === base : true));
      model = compatibleModels[0] ?? modelConfigs[0] ?? null;
    }
    const ipAdapter = deepClone(initialIPAdapter);
    if (model) {
      ipAdapter.model = zModelIdentifierField.parse(model);
    }
    return ipAdapter;
  }
);

export const useAddControlLayer = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    const overrides = { controlAdapter: deepClone(initialControlNet) };
    dispatch(controlLayerAdded({ isSelected: true, overrides }));
  }, [dispatch]);

  return func;
};

export const useAddRasterLayer = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rasterLayerAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useAddInpaintMask = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(inpaintMaskAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useAddRegionalGuidance = () => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rgAdded({ isSelected: true }));
  }, [dispatch]);

  return func;
};

export const useAddRegionalReferenceImage = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);

  const func = useCallback(() => {
    const overrides: Partial<CanvasRegionalGuidanceState> = {
      referenceImages: [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter: defaultIPAdapter }],
    };
    dispatch(rgAdded({ isSelected: true, overrides }));
  }, [defaultIPAdapter, dispatch]);

  return func;
};

export const useAddGlobalReferenceImage = () => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);
  const func = useCallback(() => {
    const overrides = { ipAdapter: defaultIPAdapter };
    dispatch(referenceImageAdded({ isSelected: true, overrides }));
  }, [defaultIPAdapter, dispatch]);

  return func;
};

export const useAddRegionalGuidanceIPAdapter = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const defaultIPAdapter = useAppSelector(selectDefaultIPAdapter);
  const func = useCallback(() => {
    dispatch(rgIPAdapterAdded({ entityIdentifier, overrides: { ipAdapter: defaultIPAdapter } }));
  }, [defaultIPAdapter, dispatch, entityIdentifier]);

  return func;
};

export const useAddRegionalGuidancePositivePrompt = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(rgPositivePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);

  return func;
};

export const useAddRegionalGuidanceNegativePrompt = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) => {
  const dispatch = useAppDispatch();
  const runc = useCallback(() => {
    dispatch(rgNegativePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);

  return runc;
};

export const buildSelectValidRegionalGuidanceActions = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>
) => {
  return createMemoizedSelector(selectCanvasSlice, (canvas) => {
    const entity = selectEntityOrThrow(canvas, entityIdentifier);
    return {
      canAddPositivePrompt: entity?.positivePrompt === null,
      canAddNegativePrompt: entity?.negativePrompt === null,
    };
  });
};
