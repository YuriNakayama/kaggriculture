/**
 * CPython bit-compatibility of PyRandom. Reference values generated with:
 *   python3 -c "import random; r=random.Random(123456789012345); print([r.random() for _ in range(3)])"
 *   python3 -c "import random; r=random.Random(42); print([r.choice(list(range(7))) for _ in range(10)])"
 */

import { describe, expect, it } from 'vitest';
import { PyRandom } from '../rng';

describe('PyRandom vs CPython random.Random', () => {
  it('random() matches for a large (>32-bit) seed', () => {
    const r = new PyRandom(123456789012345n);
    expect(r.random()).toBeCloseTo(0.6744750084019356, 15);
    expect(r.random()).toBeCloseTo(0.2923275784421222, 15);
    expect(r.random()).toBeCloseTo(0.22484093741322975, 15);
  });

  it('choice() matches (randbelow/getrandbits path)', () => {
    const r = new PyRandom(42);
    const items = [0, 1, 2, 3, 4, 5, 6];
    const picks = Array.from({ length: 10 }, () => r.choice(items));
    expect(picks).toEqual([5, 0, 0, 5, 2, 1, 1, 1, 5, 0]);
  });
});
