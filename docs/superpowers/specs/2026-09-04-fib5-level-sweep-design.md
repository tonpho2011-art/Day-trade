# Fib 5-candle: entry-level sweep + 16-slot R:R cap (v1)

Locked 2026-09-04. Optimizes the existing Fibonacci 5-candle strategy only.
Does not change `autotrade.py` or the 3-agent committee.

## Goal

Find which Fib **entry level** is best on the cached 5m S&P sample under
rules that match the original limit-order strategy, with a real account
constraint (max 16 names). No extra spread/commission haircut — the
outlined strategy did not include one.

A level is “best” only if it has the highest total P/L **and** is
non-negative in both the first and second half of the sample. If no level
clears the half-sample gate, report the least-bad level and treat it as
**not tradeable**.

## Unchanged strategy

- 5+ same-color 5-minute candles, then an opposite close → pending limit.
- Longs and shorts. SL = START, TP = END.
- Fill on a later bar’s touch. Gap-through fills at that bar’s **open**
  (limit price improvement). No `fill_at_limit`.
- RTH 09:30–16:00 ET. No new pending in the last 15 minutes. Flatten at
  ~15:55. $3500 notional per fill.
- Data: Alpaca IEX 5m cache (`data/bars_5m_iex/`), 500 names, ~400 days.
- EMA filter off. Min-range filter off.

## What this pass adds

### 1. Bad-bar filter

Skip any bar where the open gaps more than 25% vs the previous bar’s
close, or the bar’s range is more than 25% of the previous close:

`abs(open / prev_close - 1) > 0.25` or `(high - low) / prev_close > 0.25`

That removes split/bad ticks (the BKNG $170 → $4,161 print) without
hardcoding symbols. On a bad bar: do not fill a pending, do not use
that bar’s high/low for SL/TP. If already in a position, flatten at
the **previous bar’s close**. Cancel any pending. Still count the bar
in the time index (do not drop it from the series, or 5-candle streaks
would silently change).

### 2. Entry-level sweep

Run five independent simulations:

| Level | Meaning |
|-------|---------|
| 0.25 | Shallow pullback (near END) |
| 0.382 | Classic Fib |
| 0.50 | Current default |
| 0.618 | Classic Fib (near START) |
| 0.70 | Deep pullback |

Each run uses the same bars, same SL/TP rules, same bad-bar filter.

### 3. No extra cost

Reported P/L is raw strategy P/L (fill, SL, TP, flatten). Do **not**
subtract bps, commissions, or spread. `apply_roundtrip_bps` may still
exist for optional `--robust` reports; it is **not** part of `--optimize`
scoring.

### 4. Portfolio overlay (max 16, best R:R)

Each symbol is still simulated alone (one pending + one position per
name). Then a **cross-symbol overlay** keeps at most 16 live positions:

1. Collect filled trades (with `entry_time`, `exit_time`, `entry_price`,
   `stop`, `target`, `side`, `symbol`).
2. Walk fills grouped by `entry_time` (same 5m timestamp).
3. `open` = fills already kept whose `exit_time` is still in the future.
4. Slots free = `16 - len(open)`.
5. Rank **this bar’s** new fills by:
   - **Actual R:R** descending: long `(target - entry) / (entry - stop)`;
     short `(entry - target) / (stop - entry)`. If denominator ≤ 0, R:R = 0.
   - Tie: larger `|start - end|` (use `|stop - target|`).
   - Tie: earlier is N/A at the same timestamp; use symbol alphabetically.
6. Keep the top `slots_free` fills. Do **not** flatten a weaker open
   position to make room. First-come across bars, R:R only among same-bar
   competitors.

`cap_concurrent(..., max_positions=16)` first-come-only is **not** the
selector. A new function (`select_portfolio`) implements the overlay.

## Winner rule

Use a **fixed** sample split so every level is judged on the same dates:
cutoff = median timestamp of the cached bars (not per-level median).

For each level, after the 16-slot overlay (no cost):

- `total_pnl`
- `first_half_pnl` (entry_time < cutoff)
- `second_half_pnl` (entry_time ≥ cutoff)

**Winner** = highest `total_pnl` among levels with
`first_half_pnl >= 0` and `second_half_pnl >= 0`.

If the eligible set is empty, print all five rows and say none is
tradeable. Do not change the default `FibConfig.level` unless a winner
exists; if one exists, set the CLI default to that level.

## CLI

```
python fib5.py --optimize --universe-size 500 --out data/fib5_optimize.json
```

Print a table: level, trades kept, win rate, P/L, first-half P/L,
second-half P/L, longs/shorts. Write the same plus the winner to JSON.

`--backtest` stays. Optional `--max-positions` default 16 for overlay
when `--optimize` (and usable later on a single `--backtest`).

## Tests

- Bad-bar: a 25%+ gap bar does not fill / does not produce the BKNG-style
  giant P/L on a toy series.
- R:R rank: two fills at the same time, cap 1, keeps the higher R:R.
- Tie-break: equal R:R, cap 1, keeps the larger range.
- Overlay: a third fill while 16 are open is skipped; after one exits,
  a later fill can enter.
- Sweep wiring: `select_portfolio` is what `--optimize` reports; no
  bps haircut on those stats.

## Out of scope

Min-range filter, EMA filter, changing SL/TP, live paper loop, committee
integration, fill-at-limit mode, and any bps/cost haircut as the
optimize objective.
