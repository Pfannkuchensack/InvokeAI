import type { ReactNode } from 'react';

import { Text } from '@chakra-ui/react';

/** Out of the layout while empty, so a note that appears is announced without leaving a gap before it. */
const LIVE_REGION_CSS = { '&:empty': { position: 'absolute' } } as const;

/** A short advisory under a control. The region stays mounted, so its text is announced when it fills. */
export const LiveNote = ({ children }: { children?: ReactNode }) => (
  <Text aria-live="polite" color="fg.warning" css={LIVE_REGION_CSS} fontSize="xs" role="status" textWrap="pretty">
    {children}
  </Text>
);
