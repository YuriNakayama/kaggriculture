/**
 * Mersenne Twister (MT19937) PRNG, bit-compatible with CPython's
 * `random.Random`:
 *
 * - integer seeding goes through the same pipeline CPython uses
 *   (abs(seed) split into 32-bit words → init_by_array), and
 * - `choice` reproduces `Random.choice` exactly (`_randbelow` via
 *   `getrandbits` with rejection sampling), NOT `floor(random() * n)`.
 *
 * This makes stochastic engine events (weed spawn, end-of-day shop unlock)
 * identical to Python rollouts for the same seed — verified by
 * `__tests__/parity.spec.ts` against a real Python-engine trace.
 * (The upstream vendored version was deterministic but not CPython-exact;
 * rewritten locally, see frontend/UPSTREAM.md.)
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

/** abs(seed) split into little-endian 32-bit words — CPython's int→key step. */
function seedToKey(seed: number | bigint): number[] {
  let n = typeof seed === 'bigint' ? seed : BigInt(Math.trunc(seed));
  if (n < 0n) n = -n;
  const key: number[] = [];
  while (n > 0n) {
    key.push(Number(n & 0xffffffffn));
    n >>= 32n;
  }
  if (key.length === 0) key.push(0);
  return key;
}

export class PyRandom {
  private mt: Uint32Array;
  private index: number;

  constructor(seed: number | bigint) {
    this.mt = new Uint32Array(N);
    this.index = N + 1;
    this.initByArray(seedToKey(seed));
  }

  private initGenrand(s: number): void {
    this.mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = this.mt[i - 1];
      const x = Math.imul(1812433253, prev ^ (prev >>> 30)) + i;
      this.mt[i] = x >>> 0;
    }
    this.index = N;
  }

  private initByArray(key: number[]): void {
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(N, key.length);
    for (; k; k--) {
      const prev = this.mt[i - 1];
      this.mt[i] = (((this.mt[i] ^ Math.imul(prev ^ (prev >>> 30), 1664525)) >>> 0) + key[j] + j) >>> 0;
      i++;
      j++;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (k = N - 1; k; k--) {
      const prev = this.mt[i - 1];
      this.mt[i] = (((this.mt[i] ^ Math.imul(prev ^ (prev >>> 30), 1566083941)) >>> 0) - i) >>> 0;
      i++;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
    this.index = N;
  }

  private generate(): void {
    for (let i = 0; i < N; i++) {
      const y = ((this.mt[i] & UPPER_MASK) | (this.mt[(i + 1) % N] & LOWER_MASK)) >>> 0;
      const next = (this.mt[(i + M) % N] ^ (y >>> 1)) >>> 0;
      this.mt[i] = (y & 1) === 0 ? next : (next ^ MATRIX_A) >>> 0;
    }
    this.index = 0;
  }

  nextUint32(): number {
    if (this.index >= N) this.generate();
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** [0, 1) with 53 bits of randomness, matching Python's random.random(). */
  random(): number {
    const a = this.nextUint32() >>> 5; // 27 bits
    const b = this.nextUint32() >>> 6; // 26 bits
    return (a * 67108864 + b) / 9007199254740992;
  }

  /** Inclusive integer in [a, b]. */
  randint(a: number, b: number): number {
    return a + Math.floor(this.random() * (b - a + 1));
  }

  uniform(a: number, b: number): number {
    return a + (b - a) * this.random();
  }

  /** CPython `getrandbits(k)` for k in [1, 32]. */
  getrandbits(k: number): number {
    if (k <= 0 || k > 32) throw new Error('getrandbits: k out of range');
    return this.nextUint32() >>> (32 - k);
  }

  /** CPython `Random._randbelow` — rejection sampling over getrandbits. */
  private randbelow(n: number): number {
    if (n <= 0) throw new Error('randbelow: n must be positive');
    const k = 32 - Math.clz32(n); // n.bit_length()
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** Bit-exact `random.Random.choice` — uniform pick from a non-empty list. */
  choice<T>(items: readonly T[]): T {
    if (items.length === 0) throw new Error('rng.choice on empty array');
    return items[this.randbelow(items.length)];
  }
}

/**
 * Mirrors the per-day RNG seeding used by `_end_of_day`:
 *   random.Random((seed * 1_000_003) ^ day)
 *
 * Done with BigInt because (seed * 1_000_003) can exceed Number's safe
 * integer range for large seeds. Passed to PyRandom at full width — the
 * CPython-style seeding splits it into 32-bit words.
 */
export function endOfDaySeed(seed: number, day: number): bigint {
  // Full-width like Python — PyRandom's seeding splits any size into words.
  return (BigInt(seed) * 1_000_003n) ^ BigInt(day);
}
