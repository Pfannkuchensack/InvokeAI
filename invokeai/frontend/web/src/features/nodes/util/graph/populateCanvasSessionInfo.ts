import type { RootState } from 'app/store/store';
import type { Graph } from 'services/api/types';

/**
 * Finds all `canvas_session_info` nodes in a graph and populates
 * their input values from the current canvas/params Redux state.
 *
 * This is called when a workflow is executed from the canvas context menu,
 * so the canvas session info nodes automatically receive the current canvas state.
 */
export const populateCanvasSessionInfoNodes = (graph: Required<Graph>, state: RootState): void => {
  const { bbox } = state.canvas.present;
  const params = state.params;

  for (const nodeId of Object.keys(graph.nodes)) {
    const node = graph.nodes[nodeId];
    if (!node || node.type !== 'canvas_session_info') {
      continue;
    }

    // Cast to Record for dynamic property assignment
    const n = node as Record<string, unknown>;

    // Populate bbox fields
    n.bbox_x = bbox.rect.x;
    n.bbox_y = bbox.rect.y;
    n.bbox_width = bbox.rect.width;
    n.bbox_height = bbox.rect.height;
    n.scaled_width = bbox.scaledSize.width;
    n.scaled_height = bbox.scaledSize.height;

    // Populate model fields
    if (params.model) {
      n.model_name = params.model.name ?? '';
      n.model_base = params.model.base ?? '';
      n.model_key = params.model.key ?? '';
    }

    // Populate prompts
    n.positive_prompt = params.positivePrompt;
    n.negative_prompt = params.negativePrompt ?? '';

    // Populate generation params
    n.steps = params.steps;
    n.cfg_scale = params.cfgScale;
    n.guidance = params.guidance;
    n.seed = params.seed;
    n.scheduler = params.scheduler;
    n.img2img_strength = params.img2imgStrength;
  }
};
