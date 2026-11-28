import type { InstructWidgetValues } from '@features/instruct/core/types';

import { useInstructUi } from '@features/instruct/ui/InstructUiContext';
import { ensureModelsLoaded } from '@features/models';
import { ensureArchitectureCapabilitiesLoaded } from '@features/models/architecturesStore';
import { useMountEffect } from '@platform/react/useMountEffect';

/** Asks for the catalog and the architecture table the shown values are judged against. */
export const useEnsureInstructCatalog = (): void =>
  useMountEffect(() => {
    void ensureModelsLoaded();
    void ensureArchitectureCapabilitiesLoaded();
  });

/** The widget's values as the catalog allows them; the workbench computes them once for every reader. */
export const useInstructValues = (): InstructWidgetValues => useInstructUi().values;
