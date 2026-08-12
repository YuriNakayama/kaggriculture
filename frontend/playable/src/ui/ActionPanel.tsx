/**
 * Composes the human player's PlayerAction: farmer op + one op per hand +
 * a list of market orders. Submit bundles them into a PlayerAction and the
 * panel resets to PASS / no orders.
 */

import { useId, useState } from 'react';
import { ANIMALS, CROPS, PRODUCTS } from '../engine/constants';
import { auditAction, legalMarket, legalUnitOps, type UnitOpName } from '../engine/legality';
import { marketPrice } from '../engine/market';
import type { AnimalId, CropId, GameState, PlayerAction, ProductId, ShedItemId } from '../engine/types';
import { MARKET_HELP, OP_HELP } from './opHelp';
import { commandTargets } from './smartTasks';
import { ROLE_LABELS, type HandRole } from './useHandRoles';
import { defaultMarket, type MarketDraft, type TurnDraft, type UnitDraft, type UnitOp } from './useTurnDraft';

/**
 * 売却の概算: エンジンの逐次コミット (1個ずつ現在庫で値付け→在庫+1) を
 * なぞる。相手の同時売買までは読めないため「概算」。
 */
function estimateSell(state: GameState, item: ProductId, qty: number): { total: number; after: number } {
  let inv = state.market.inventory[item];
  let total = 0;
  for (let i = 0; i < qty; i++) {
    const p = marketPrice(item, inv, state.market.params);
    total += p;
    if (p > 1) inv += 1;
  }
  return { total, after: marketPrice(item, inv, state.market.params) };
}

const MOVE_OPS = ['NORTH', 'SOUTH', 'EAST', 'WEST'] as const;

const UNIT_OPS: UnitOp[] = [
  'PASS',
  'NORTH',
  'SOUTH',
  'EAST',
  'WEST',
  'WATER',
  'HARVEST',
  'FERTILIZE',
  'DIG',
  'BUILD_COOP',
  'BUILD_PASTURE',
  'FEED',
  'COLLECT_FERTILIZER',
  'CARE',
  'PLANT',
  'PICKUP',
  'PLACE',
];

const CROP_IDS = Object.keys(CROPS) as CropId[];
const ANIMAL_IDS = Object.keys(ANIMALS) as AnimalId[];
const SHED_ITEMS: ShedItemId[] = [...PRODUCTS, ...ANIMAL_IDS];

const MARKET_KINDS = ['HIRE', 'BUY_LAND', 'BUY_SEED', 'BUY_PRODUCT', 'BUY_ANIMAL', 'SELL'] as const;
type MarketKind = (typeof MARKET_KINDS)[number];

interface Props {
  state: GameState;
  player: number;
  busy: boolean;
  draft: TurnDraft;
  roles: Record<number, HandRole>;
  onRoleChange(unit: number, role: HandRole): void;
  omakase: boolean;
  onOmakaseChange(on: boolean): void;
  onSubmit(action: PlayerAction): void;
}

export function ActionPanel({ state, player, busy, draft, roles, onRoleChange, omakase, onOmakaseChange, onSubmit }: Props) {
  const { farmer, hands, orders, setFarmer, setHand, setOrders } = draft;
  const [noopNotes, setNoopNotes] = useState<string[]>([]);
  const idPrefix = useId();
  const marketLegal = legalMarket(state, player);

  const submit = () => {
    const action = draft.buildAction();
    // Audit BEFORE stepping: anything flagged here the engine will silently
    // discard — surfacing that is the whole point of playing by hand.
    setNoopNotes(auditAction(state, player, action));
    onSubmit(action);
    draft.afterSubmit(action);
  };

  const renderUnit = (label: string, unitIdx: number, draft: UnitDraft, onChange: (d: UnitDraft) => void) => {
    const opId = `${idPrefix}-${label}-op`;
    const legal = legalUnitOps(state, player, unitIdx);
    return (
      <div className="action-row" key={label}>
        <label className="action-label" htmlFor={opId}>
          {label}
        </label>
        <select
          id={opId}
          value={draft.op}
          title={OP_HELP[draft.op]}
          onChange={(e) => onChange({ ...draft, op: e.target.value as UnitOp })}
        >
          {UNIT_OPS.map((op) => (
            <option key={op} value={op} disabled={!legal[op as UnitOpName]} title={OP_HELP[op]}>
              {legal[op as UnitOpName] ? op : `${op} ✕`}
            </option>
          ))}
        </select>
        <details className="op-help">
          <summary title="この操作の説明">ⓘ</summary>
          <p>{OP_HELP[draft.op]}</p>
        </details>
        {draft.op === 'PLANT' && (
          <select value={draft.crop} onChange={(e) => onChange({ ...draft, crop: e.target.value as CropId })}>
            {CROP_IDS.map((c) => {
              const n = state.privates[player].seeds[c] ?? 0;
              return (
                <option key={c} value={c} disabled={n === 0}>
                  {c} (種x{n})
                </option>
              );
            })}
          </select>
        )}
        {(draft.op === 'PICKUP' || draft.op === 'PLACE') && (
          <>
            <select value={draft.item} onChange={(e) => onChange({ ...draft, item: e.target.value as ShedItemId })}>
              {SHED_ITEMS.map((it) => {
                // PICKUP は倉庫在庫、PLACE はそのユニットの所持数で絞り込む。
                const n =
                  draft.op === 'PICKUP'
                    ? (state.privates[player].shed[it] ?? 0)
                    : (state.privates[player].inventories[unitIdx]?.[it] ?? 0);
                return (
                  <option key={it} value={it} disabled={n === 0}>
                    {it} (x{n})
                  </option>
                );
              })}
            </select>
            <input
              type="number"
              min={1}
              value={draft.qty}
              onChange={(e) => onChange({ ...draft, qty: Number(e.target.value) || 1 })}
              style={{ width: 60 }}
            />
          </>
        )}
      </div>
    );
  };

  const renderOrder = (draft: MarketDraft, idx: number) => (
    <div className="action-row" key={idx}>
      <select
        value={draft.kind}
        onChange={(e) => {
          const next = [...orders];
          next[idx] = { ...draft, kind: e.target.value as MarketKind };
          setOrders(next);
        }}
      >
        {MARKET_KINDS.map((k) => {
          const legalKind =
            k === 'HIRE'
              ? marketLegal.hire.legal
              : k === 'BUY_LAND'
                ? marketLegal.buyLand.legal
                : k === 'SELL'
                  ? Object.values(marketLegal.sell).some((s) => s?.legal)
                  : true;
          const suffix =
            k === 'HIRE'
              ? ` ($${marketLegal.hire.cost})`
              : k === 'BUY_LAND' && marketLegal.buyLand.cost !== null
                ? ` (${marketLegal.buyLand.quadrant} $${marketLegal.buyLand.cost})`
                : '';
          return (
            <option key={k} value={k} disabled={!legalKind}>
              {k}
              {suffix}
              {legalKind ? '' : ' ✕'}
            </option>
          );
        })}
      </select>
      {draft.kind === 'BUY_SEED' && (
        <select
          value={draft.crop}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, crop: e.target.value as CropId };
            setOrders(next);
          }}
        >
          {CROP_IDS.map((c) => {
            const info = marketLegal.buySeed[c];
            return (
              <option key={c} value={c} disabled={!info?.legal}>
                {c} (${info?.cost ?? '?'}){info?.legal ? '' : ' ✕資金不足'}
              </option>
            );
          })}
        </select>
      )}
      {draft.kind === 'BUY_PRODUCT' && (
        <select
          value={draft.product}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, product: e.target.value as ShedItemId };
            setOrders(next);
          }}
        >
          {(['WHEAT', 'FERTILIZER'] as const).map((p) => {
            const info = marketLegal.buyProduct[p];
            return (
              <option key={p} value={p} disabled={!info?.legal}>
                {p} (${info?.cost ?? '?'}){info?.legal ? '' : ' ✕資金不足'}
              </option>
            );
          })}
        </select>
      )}
      {draft.kind === 'BUY_ANIMAL' && (
        <select
          value={draft.animal}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, animal: e.target.value as AnimalId };
            setOrders(next);
          }}
        >
          {ANIMAL_IDS.map((a) => {
            const info = marketLegal.buyAnimal[a];
            return (
              <option key={a} value={a} disabled={!info?.legal}>
                {a} (${info?.cost ?? '?'}){info?.legal ? '' : ' ✕資金不足'}
              </option>
            );
          })}
        </select>
      )}
      {draft.kind === 'SELL' && (
        <select
          value={draft.product}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, product: e.target.value as ShedItemId };
            setOrders(next);
          }}
        >
          {PRODUCTS.map((p) => {
            const info = marketLegal.sell[p];
            return (
              <option key={p} value={p} disabled={!info?.legal}>
                {p} (x{info?.stock ?? 0} @ ${info?.price ?? '?'}){info?.legal ? '' : ' ✕'}
              </option>
            );
          })}
        </select>
      )}
      <details className="op-help">
        <summary title="この注文の説明">ⓘ</summary>
        <p>{MARKET_HELP[draft.kind]}</p>
      </details>
      {draft.kind !== 'HIRE' && draft.kind !== 'BUY_LAND' && (
        <input
          type="number"
          min={1}
          value={draft.qty}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, qty: Number(e.target.value) || 1 };
            setOrders(next);
          }}
          style={{ width: 60 }}
        />
      )}
      <button type="button" onClick={() => setOrders(orders.filter((_, i) => i !== idx))}>
        ×
      </button>
      {draft.kind === 'SELL' &&
        draft.product !== 'GOOSE' &&
        draft.product !== 'COW' &&
        draft.product !== 'SHEEP' &&
        (() => {
          const est = estimateSell(state, draft.product as ProductId, Math.max(1, Math.floor(draft.qty)));
          return (
            <span className="sell-estimate" title="1個ずつ売るたびに市場在庫が増えて値が下がる (相手の売買は含まない概算)">
              ≈ ${est.total.toLocaleString()} / 売却後 ${est.after}
            </span>
          );
        })()}
    </div>
  );

  return (
    <div className="action-panel">
      <h3>Your turn — Player {player + 1}</h3>
      <div className="facts-row" title="今できることの件数 (スマートコマンドで一括実行できます)">
        <span>💧 {commandTargets(state, player, 'WATER_ALL').length}</span>
        <span>🌾 {commandTargets(state, player, 'HARVEST_ALL').length}</span>
        <span>🐄 {commandTargets(state, player, 'TEND_ANIMALS').length}</span>
        <span>🧹 {commandTargets(state, player, 'CLEAR_WEEDS').length}</span>
        <span>
          🌱{' '}
          {CROP_IDS.filter((c) => (state.privates[player].seeds[c] ?? 0) > 0)
            .map((c) => `${c.slice(0, 3)}x${state.privates[player].seeds[c]}`)
            .join(' ') || '種なし'}
        </span>
      </div>
      <label className="omakase-toggle" title="農作業を内蔵 starter 方針に任せ、市場注文だけを操作する上級モード">
        <input type="checkbox" checked={omakase} onChange={(e) => onOmakaseChange(e.target.checked)} />
        🤖 おまかせ農場 (市場だけ操作)
      </label>
      {omakase ? (
        <p className="action-hints">
          農作業と購入 (種など) は自動。<strong>販売 (SELL) はあなたの仕事</strong>です — 下の市場注文で売り時を決めて
          Submit してください。何も注文せず Submit すればターンだけ進みます。
        </p>
      ) : (
        <>
          {renderUnit('Farmer', 0, farmer, setFarmer)}
          {hands.map((h, i) => (
            <div key={i} className="hand-block">
              {renderUnit(`Hand ${i + 1}`, i + 1, h, (d) => setHand(i, d))}
              <select
                className="role-select"
                value={roles[i + 1] ?? 'MANUAL'}
                title="役割を割り当てると毎ターン自動で行動します"
                onChange={(e) => onRoleChange(i + 1, e.target.value as HandRole)}
              >
                {(Object.keys(ROLE_LABELS) as HandRole[]).map((r) => (
                  <option key={r} value={r}>
                    {ROLE_LABELS[r]}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </>
      )}
      <div className="action-section">
        <div className="action-section-header">
          <span>Market orders ({orders.length})</span>
          <button type="button" onClick={() => setOrders([...orders, { ...defaultMarket }])}>
            + Add
          </button>
        </div>
        {orders.map(renderOrder)}
      </div>
      <div className="action-row">
        <button
          type="button"
          className="submit-turn"
          onClick={submit}
          disabled={busy || state.done}
          style={{ flex: 1 }}
          title="ここで組んだ操作と市場注文でこのターンを実行 (盤面タップやスマートコマンドは自動実行されるため通常は不要)"
        >
          実行 (このターン)
        </button>
        <button
          type="button"
          onClick={draft.repeatLast}
          disabled={busy || state.done || !draft.hasLast}
          title="前ターンのユニット操作を再セット (市場注文は除く)"
        >
          ↻ Repeat
        </button>
      </div>
      {noopNotes.length > 0 && (
        <div className="noop-notes" role="status">
          <strong>Silently discarded last turn:</strong>
          <ul>
            {noopNotes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="action-hints">
        Moves available: {MOVE_OPS.join(', ')}. Tile ops apply to the unit's current cell.
      </div>
    </div>
  );
}
