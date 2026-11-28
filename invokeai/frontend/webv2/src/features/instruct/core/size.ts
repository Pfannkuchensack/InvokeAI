import type { MainModelConfig } from '@features/generation/contracts';

import { fitToOptimalArea, getGenerationDimensions } from '@features/generation/settings';

/** The nearest multiple of the grid, within the model's limits. */
export const snapEditDimension = (
  value: number,
  { grid, max, min }: { grid: number; max: number; min: number }
): number => Math.min(max, Math.max(min, Math.round(value / grid) * grid));

/**
 * The size of the next step's result: the manual size when set; else the model's ideal area at the source's aspect
 * ratio for the first step, and the parent's size for every later one, so a chain never drifts. A manual or inherited
 * size from another model is snapped to this model's grid (to the nearest multiple, halves rounding up) and clamped.
 */
export const resolveEditOutputSize = ({
  manual,
  model,
  parent,
  source,
}: {
  model: Pick<MainModelConfig, 'base' | 'type' | 'variant'>;
  manual: { width: number; height: number } | null;
  parent: { isSource: boolean; width: number; height: number };
  source: { width: number; height: number };
}): { width: number; height: number } => {
  if (!manual && parent.isSource) {
    return fitToOptimalArea(source, model);
  }

  const limits = getGenerationDimensions(model);
  const size = manual ?? parent;

  return { height: snapEditDimension(size.height, limits), width: snapEditDimension(size.width, limits) };
};
