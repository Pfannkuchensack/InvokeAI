/** The grid the manual size snaps to, within the backend's limits. */
export const MIN_EDIT_DIMENSION = 64;
export const MAX_EDIT_DIMENSION = 8192;

export const snapEditDimension = (value: number, grid: number): number =>
  Math.min(MAX_EDIT_DIMENSION, Math.max(MIN_EDIT_DIMENSION, Math.round(value / grid) * grid));

/**
 * The size of the next step's result: the manual size when set; else the model's ideal area at the source's aspect
 * ratio for the first step, and the parent's size for every later one, so a chain never drifts.
 */
export const resolveEditOutputSize = ({
  fit,
  manual,
  parent,
  source,
}: {
  fit: (source: { width: number; height: number }) => { width: number; height: number };
  manual: { width: number; height: number } | null;
  parent: { isSource: boolean; width: number; height: number };
  source: { width: number; height: number };
}): { width: number; height: number } =>
  manual
    ? { height: manual.height, width: manual.width }
    : parent.isSource
      ? fit(source)
      : { height: parent.height, width: parent.width };
