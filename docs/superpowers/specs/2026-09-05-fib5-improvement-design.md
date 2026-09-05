# Fib5 improvement: chairs + closed 30m (v1)

Locked 2026-09-05. Live paper still uses limit+bracket. No market chase.

## 1. Chairs = fills only

`16` means filled positions, not Alpaca working orders.

- Stop/take-profit legs do **not** take a chair.
- Resting entry limits do **not** take a chair, but we only rest
  `cap - held` entry limits (so we cannot fill past 16).
- Each cycle: rank current setups by the same R:R overlay as the
  backtest. Keep the top `room` tickets. Cancel weaker unfilled
  entry parents. Place missing ones as limits.

## 2. Closed 30m filter

A setup is valid only if the last **finished** 30-minute RTH candle is
the same color as the 5-candle impulse (green impulse → green 30m for
longs; red for shorts). The forming 30m bar is ignored.

`FibConfig.htf_minutes` default `0` (off). Pass `--htf-minutes 30` to enable.
