import { describe, expect, it } from 'vitest';

import { getBatchCount } from './useInvocationState';

describe('getBatchCount', () => {
  it('counts the variants an Edit step makes, not the Generate batch', () => {
    expect(getBatchCount('instruct', { batchCount: 5, variants: 3 })).toBe(3);
    expect(getBatchCount('generate', { batchCount: 5, variants: 3 })).toBe(5);
  });
});
