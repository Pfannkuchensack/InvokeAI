import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';
import type { ImageWithDims } from '@features/generation/core/types';

import { addEdge, addNode, createId } from '@features/generation/core/graphBuilder';

/** The area diffusers' QwenImageEditPlusPipeline encodes every reference at (its `VAE_IMAGE_SIZE`). */
const REFERENCE_AREA = 1024 * 1024;
const REFERENCE_GRID = 32;

/** Python's `round` for the grid: halves go to the even neighbour, which is what diffusers' sizes come from. */
const roundHalfEven = (value: number): number => {
  const floor = Math.floor(value);
  const diff = value - floor;

  if (diff !== 0.5) {
    return Math.round(value);
  }

  return floor % 2 === 0 ? floor : floor + 1;
};

/**
 * diffusers' `calculate_dimensions(1024 * 1024, width / height)`: the image's aspect ratio at about one megapixel, each
 * side rounded to a multiple of 32 (so up to ~10% above the area for a wide image: 16:9 gives 1376x768).
 */
export const getQwenEditReferenceSize = (image: {
  width: number;
  height: number;
}): { width: number; height: number } => {
  const ratio = image.width / image.height;
  const width = Math.sqrt(REFERENCE_AREA * ratio);
  const height = width / ratio;

  return {
    height: roundHalfEven(height / REFERENCE_GRID) * REFERENCE_GRID,
    width: roundHalfEven(width / REFERENCE_GRID) * REFERENCE_GRID,
  };
};

/**
 * Qwen-Image-Edit's references as the multi-image Edit Plus path takes them, which Generate's Qwen-Edit graph does not:
 * each image is VAE-encoded at its own one-megapixel size and the latents reach the denoise as an ordered list, and
 * both text encoders see the same images in the same order (the caller sets their `reference_template` to `edit_plus`,
 * which labels them "Picture N" and fits each to the 384-pixel condition size). The order is the images' order, the
 * source first; chained collect nodes keep it (one collect with several item edges has no defined order). The negative
 * encoder takes the images too, as diffusers conditions its negative prompt on them.
 */
export const addQwenImageEditReferences = (
  graph: BackendGraphContract,
  nodes: {
    denoise: BackendInvocationContract;
    modelLoader: BackendInvocationContract;
    negativeEncoder: BackendInvocationContract | null;
    positiveEncoder: BackendInvocationContract;
  },
  images: readonly ImageWithDims[]
): void => {
  let imageCollect: BackendInvocationContract | null = null;
  let latentsCollect: BackendInvocationContract | null = null;

  for (const image of images) {
    const imageNode = addNode(graph, {
      id: createId('qwen_edit_ref_img'),
      image: { image_name: image.image_name },
      type: 'image',
    });
    const i2l = addNode(graph, {
      id: createId('qwen_edit_ref_i2l'),
      ...getQwenEditReferenceSize(image),
      type: 'qwen_image_i2l',
    });
    addEdge(graph, imageNode, 'image', i2l, 'image');
    addEdge(graph, nodes.modelLoader, 'vae', i2l, 'vae');

    const nextImageCollect = addNode(graph, { id: createId('qwen_edit_ref_img_collect'), type: 'collect' });
    addEdge(graph, imageNode, 'image', nextImageCollect, 'item');
    if (imageCollect) {
      addEdge(graph, imageCollect, 'collection', nextImageCollect, 'collection');
    }
    imageCollect = nextImageCollect;

    const nextLatentsCollect = addNode(graph, { id: createId('qwen_edit_ref_latents_collect'), type: 'collect' });
    addEdge(graph, i2l, 'latents', nextLatentsCollect, 'item');
    if (latentsCollect) {
      addEdge(graph, latentsCollect, 'collection', nextLatentsCollect, 'collection');
    }
    latentsCollect = nextLatentsCollect;
  }

  if (!imageCollect || !latentsCollect) {
    return;
  }

  for (const encoder of [nodes.positiveEncoder, nodes.negativeEncoder]) {
    if (encoder) {
      addEdge(graph, imageCollect, 'collection', encoder, 'reference_images');
    }
  }
  addEdge(graph, latentsCollect, 'collection', nodes.denoise, 'reference_latents');
};
