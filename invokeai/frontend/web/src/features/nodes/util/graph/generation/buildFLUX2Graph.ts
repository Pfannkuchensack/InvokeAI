import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { assert } from 'tsafe';

const log = logger('system');

/**
 * Builds the generation graph for FLUX.2 models.
 *
 * FLUX.2 uses:
 * - Single Mistral Small 3.1 text encoder (instead of CLIP + T5)
 * - 32-channel VAE (instead of 16-channel)
 * - Different transformer architecture (8 double + 48 single blocks)
 * - Multi-image reference support (up to 10 images)
 */
export const buildFLUX2Graph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;
  log.debug({ generationMode, manager: manager?.id }, 'Building FLUX.2 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'flux2', 'Selected model is not a FLUX.2 model');

  const params = selectParamsSlice(state);
  // TODO: Use these for multi-image reference support
  const _canvas = selectCanvasSlice(state);
  const _refImages = selectRefImagesSlice(state);

  const { guidance, steps, flux2MistralEncoder, flux2VAE } = params;

  assert(flux2MistralEncoder, 'No Mistral Encoder model found in state');
  assert(flux2VAE, 'No FLUX.2 VAE model found in state');

  const g = new Graph(getPrefixedId('flux2_graph'));

  // Model loader
  const modelLoader = g.addNode({
    type: 'flux2_model_loader',
    id: getPrefixedId('flux2_model_loader'),
    model,
    mistral_encoder_model: flux2MistralEncoder,
    vae_model: flux2VAE,
  });

  // Text encoding
  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'flux2_text_encoder',
    id: getPrefixedId('flux2_text_encoder'),
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });

  // Seed and denoise
  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'flux2_denoise',
    id: getPrefixedId('flux2_denoise'),
    guidance,
    num_steps: steps,
    cfg_scale: params.cfgScale ?? 1.0,
  });

  // VAE decode
  const l2i = g.addNode({
    type: 'flux2_vae_decode',
    id: getPrefixedId('flux2_vae_decode'),
  });

  // Connect model outputs
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  // Connect Mistral encoder
  g.addEdge(modelLoader, 'mistral_encoder', posCond, 'mistral_encoder');
  g.addEdge(modelLoader, 'max_seq_len', posCond, 'max_seq_len');

  // Connect conditioning
  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_text_conditioning');

  // Connect seed and latents
  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Add metadata
  g.upsertMetadata({
    guidance,
    model: Graph.getModelMetadataField(model),
    steps,
    vae: flux2VAE,
    mistral_encoder: flux2MistralEncoder,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  // Handle different generation modes
  if (generationMode === 'txt2img') {
    addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'flux_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux2_vae_encode',
      id: getPrefixedId('flux2_vae_encode'),
    });
    g.addEdge(modelLoader, 'vae', i2l, 'vae');
    // TODO: Complete img2img implementation
    g.upsertMetadata({ generation_mode: 'flux_img2img' });
  } else {
    // TODO: Add inpaint/outpaint support for FLUX.2
    throw new Error('FLUX.2 currently only supports txt2img and img2img');
  }

  return {
    g,
    seed,
    positivePrompt,
  };
};
