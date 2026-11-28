import type { ImageWithDims } from '@features/generation/contracts';

/** The number the model and the instruction call a reference: the source is Image 1, so the first extra is Image 2. */
export const getReferenceNumber = (index: number): number => index + 2;

/** How many references can still be added beside the source. */
export const getReferenceRoom = (maxReferenceImages: number, references: readonly ImageWithDims[]): number =>
  Math.max(0, maxReferenceImages - 1 - references.length);

/** Appends what fits and skips images already listed or the source itself. */
export const addInstructReferences = (
  references: readonly ImageWithDims[],
  additions: readonly ImageWithDims[],
  { maxReferenceImages, sourceImageName }: { maxReferenceImages: number; sourceImageName: string | null }
): ImageWithDims[] => {
  const next = [...references];

  for (const image of additions) {
    if (
      next.length >= maxReferenceImages - 1 ||
      image.image_name === sourceImageName ||
      next.some((reference) => reference.image_name === image.image_name)
    ) {
      continue;
    }
    next.push(image);
  }

  return next.length === references.length ? [...references] : next;
};

export const removeInstructReference = (references: readonly ImageWithDims[], imageName: string): ImageWithDims[] =>
  references.filter((reference) => reference.image_name !== imageName);
