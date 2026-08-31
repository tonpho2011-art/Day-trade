# Day-trade

Rule-based stock scanning, signal generation, and (paper-money-only) automated
trading on top of `yfinance` and Alpaca's paper trading API.

## Setup

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and fill in `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from
https://app.alpaca.markets/paper/dashboard/overview. These are only required
for `--live-paper` mode on `bot.py`; `autotrade.py` always connects to Alpaca
(read-only unless `--live-paper`), even in dry run, since it needs the market
clock, current positions, and buying power to make decisions.

**Safety:** nothing in this repo can place a live-money order. `bot.py` and
`autotrade.py` default to a dry run (print-only, nothing submitted) unless you
pass `--live-paper`, and even then `src/daytrade/broker.py` only ever talks to
Alpaca's paper (fake money) endpoint.

## Commands

### screen.py -- stock screener / signal tool

```powershell
python screen.py AAPL                       # screen a single ticker
python screen.py AAPL MSFT NVDA             # screen multiple named tickers
python screen.py --losers                   # today's top losers (default count)
python screen.py --active                   # most active tickers by volume
python screen.py --strongbuy                # tickers with a strong-buy signal
python screen.py --strongsell                # tickers with a strong-sell signal
python screen.py --news-movers              # tickers moving on news
python screen.py --macro                    # macro/market overview
python screen.py --losers --count 25        # top losers, 25 results
python screen.py --active --count 30        # most active, 30 results
python screen.py --strongbuy --count 10     # strong-buy, 10 results
python screen.py --strongsell --count 10    # strong-sell, 10 results
python screen.py --news-movers --count 20   # news movers, 20 results
```

`--losers`, `--active`, `--strongbuy`, `--strongsell`, `--news-movers`, and
`--macro` are mutually exclusive -- pick one at a time. Bare tickers ignore
`--count`/mode flags.

### bot.py -- simple signal-driven paper bot

```powershell
python bot.py AAPL MSFT NVDA                              # dry-run once, no orders placed
python bot.py AAPL MSFT NVDA --live-paper                 # places real (paper/fake-money) orders on Alpaca
python bot.py AAPL MSFT NVDA --cash-per-trade 1000        # sets $ amount to risk per trade
python bot.py AAPL MSFT NVDA --max-positions 5            # caps number of open positions
python bot.py AAPL MSFT NVDA --loop                       # keeps running repeatedly instead of once
python bot.py AAPL MSFT NVDA --loop --interval-minutes 15 # loops, checking every 15 minutes
python bot.py AAPL MSFT NVDA --ignore-market-hours        # runs even when the market is closed
python bot.py AAPL MSFT NVDA --live-paper --cash-per-trade 1000 --max-positions 5 --loop --interval-minutes 15  # live-paper looped bot with sizing/position limits
python bot.py AAPL MSFT NVDA --live-paper --ignore-market-hours  # live-paper, runs outside market hours
```

### autotrade.py -- fully automated intraday loop

```powershell
python autotrade.py                                     # dry-run, loops continuously during market hours
python autotrade.py --once                              # dry-run, single pass then exit
python autotrade.py --live-paper                        # places real (paper/fake-money) orders, loops
python autotrade.py --live-paper --cash-per-trade 500    # live-paper with sizing, using default 7x leverage / 2% stop-loss / 4% take-profit
python autotrade.py --live-paper --once                 # places paper orders, single pass then exit
python autotrade.py --interval-minutes 5                # checks/rebalances every 5 minutes
python autotrade.py --universe-size 100                 # scans a universe of 100 tickers
python autotrade.py --cash-per-trade 500                # sets $ amount to risk per trade
python autotrade.py --leverage 7                        # multiplies cash-per-trade 7x using margin (paper only)
python autotrade.py --max-positions 8                   # caps number of open positions
python autotrade.py --stop-loss-pct 2                   # exits a position if it drops 2%
python autotrade.py --take-profit-pct 4                 # exits a position after a 4% gain
python autotrade.py --flatten-minutes-before-close 5    # closes all positions 5 min before market close
python autotrade.py --no-new-entries-minutes-before-close 15  # stops opening new trades 15 min before close
python autotrade.py --ignore-market-hours               # runs even when the market is closed
python autotrade.py --live-paper --once --ignore-market-hours  # single live-paper pass, ignoring market hours
python autotrade.py --live-paper --cash-per-trade 500 --leverage 7 --stop-loss-pct 2 --take-profit-pct 4  # live-paper with sizing, leverage, and risk exits
python autotrade.py --live-paper --cash-per-trade 500 --leverage 7 --stop-loss-pct 2 --take-profit-pct 4 --max-positions 8 --interval-minutes 5 --universe-size 100  # full live-paper config combining all the above
```

`--leverage` multiplies `--cash-per-trade` using Alpaca's margin buying
power -- real leverage even on a paper account, amplifying both gains and
losses by the same multiple.

### main.py -- hardcoded single backtest

```powershell
python main.py
```

No flags -- hardcoded to `run(symbol="AAPL", period="1y", interval="1d")` in
the source. Edit `main.py` directly to change the symbol/period/interval.

## Notes

- Every flag above takes a value where shown; combine as many as you want on
  one line -- they're split out individually so you can see what each one
  does in isolation.
- `--live-paper` is the only flag that actually submits orders (to Alpaca's
  paper/fake-money endpoint); without it, everything is dry-run/print-only.
- Trade/decision history for `bot.py` and `autotrade.py` is appended to
  `data/trade_log.csv`.
- This is a simple rule-based heuristic on top of free delayed/real-time
  data. It will generate losing trades -- that's normal, not a bug. Do not
  point this at a real-money account.

Commands
python autotrade.py --live-paper --cash-per-trade 500 --leverage 7 --stop-loss-pct 2 --take-profit-pct 4 --max-positions 8 --interval-minutes 5 --universe-size 100 (for everything)
cd C:\Users\namhe\Downloads\Daytrade (pull from this folder)
 .venv\Scripts\Activate.ps1 (fix modulenotfounderror)