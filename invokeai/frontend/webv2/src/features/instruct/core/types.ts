import type { GenerationModelCatalogItem, ImageWithDims, MainModelConfig } from '@features/generation/contracts';
import type { EditGraphSettings } from '@features/generation/editGraph';
import type { SeedMode } from '@platform/core/seed';

export type InstructModelCatalogItem = GenerationModelCatalogItem;

/** What the Edit widget keeps in its instance values. */
export interface InstructWidgetValues {
  /** The edit model; null until one is installed and chosen. */
  model: MainModelConfig | null;
  /** The pipeline settings an edit step reads (null while there is no model). */
  settings: EditGraphSettings | null;
  seed: number;
  seedMode: SeedMode;
  /** Images one step makes (1-4). */
  variants: number;
  /** The composer's text, the step's positive prompt. */
  instruction: string;
  /** References after the source (Image 2, 3 ...); they stay for later steps until removed. */
  references: ImageWithDims[];
  /** `auto` follows the source or the parent step; `manual` uses `width` and `height`. */
  sizeMode: 'auto' | 'manual';
  width: number;
  height: number;
  /** The node the next step continues from; null continues from the end of the shown path. */
  continueFromNodeId: string | null;
}

/** The part of an edit tree node the UI reads; the workbench's node type satisfies it. */
export interface InstructNode {
  id: string;
  parentId: string | null;
  stepId: string;
  imageName: string | null;
  status: 'pending' | 'completed' | 'failed' | 'canceled' | 'deleted';
  instruction: string;
  referenceImages: readonly { imageName: string }[];
  seed: number;
  width: number;
  height: number;
  error?: string;
}

/** The tree as the UI reads it. */
export interface InstructTreeView {
  nodes: readonly InstructNode[];
  /** The shown branch, root first. */
  path: readonly InstructNode[];
  rootNodeId: string;
  selection: Readonly<Record<string, string>>;
}
