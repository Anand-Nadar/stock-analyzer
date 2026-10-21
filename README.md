# StockPilot India

Local-first Indian stock research tool (Phases 1-6). Research aid only - not investment advice.
Quantitative rules decide candidates; the local AI only explains evidence that comes from your database.

## Setup (Linux)
```bash
cd /home/dell/workspace/stock-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                      # offline unit tests

python -m stockpilot init-db
python -m stockpilot load-master      # NSE EQUITY_L.csv (or --file path)
python -m stockpilot set-universe     # config/starter_universe.txt - edit it
python -m stockpilot download-prices
```

## Daily workflow
```bash
python -m stockpilot daily-update     # prices -> freshness check -> swing + long-term screens -> paper trades -> DB backup
python -m stockpilot screen-swing --show-rejected
python -m stockpilot screen-long --show-all
uvicorn stockpilot.web.app:app --reload     # dashboard on http://127.0.0.1:8000
```
Cron (weekdays 19:30, after NSE close):
`30 19 * * 1-5  cd /home/dell/workspace/stock-analyzer && .venv/bin/python -m stockpilot daily-update >> data/daily.log 2>&1`
Failures are logged to `data_refresh_log`, shown in red on the dashboard, and make the command exit non-zero. One failing step does not stop the others.

## Phase 3 - Fundamentals / long-term
Units: **money in Rs crore, shares in crore.**
```bash
python -m stockpilot import-financials --csv my_financials.csv   # template: config/financials_template.csv (stored as validated)
python -m stockpilot fetch-financials RELIANCE TCS               # yfinance, UNOFFICIAL, stored as NOT validated
python -m stockpilot validate-financials RELIANCE                # after you cross-check with filings
python -m stockpilot import-sectors --csv sectors.csv            # symbol,sector[,industry] - needed for peer P/E and sector exposure
python -m stockpilot screen-long
```
Score weights: growth 20, profitability 20, financial health 20, valuation 20, stability 10, momentum 10 (initial hypotheses).
* Missing data never scores zero: a pillar without data is dropped and **confidence** falls. Candidates need confidence >= 0.6.
* **Serious flags block candidacy** whatever the score: net loss, negative operating cash flow, debt/equity > 2, interest cover < 1.5, P/E > 60, stale financials (> 20 months).
* Banks/NBFCs/insurers skip leverage, ROCE, operating-margin and EV/EBITDA tests.
* Governance (pledging, auditor remarks, related-party dealings) is **not assessed** - always check manually.
* The fundamentals engine is **not backtested** (no point-in-time data), so its weights are unvalidated.

## Phase 4 - Backtesting and paper trading
```bash
python -m stockpilot backtest --oos-from 2025-06-01      # swing strategy vs simple baseline; in-sample / out-of-sample split
python -m stockpilot paper-open SYMBOL                   # from the latest swing run (BREAKOUT setups only)
python -m stockpilot paper-update                        # closes trades that hit stop/target/time (also run by daily-update)
python -m stockpilot paper-report
```
Signals use data up to day t and execute on t+1; costs = `round_trip_cost_pct` + slippage per side; a bar touching both stop and target counts as a stop;
gaps fill at the open. Reported: return, CAGR, max drawdown, Sharpe (rf=0), win rate, profit factor, R-multiples, exposure, turnover, benchmark.
**Treat results sceptically:** the universe is today's survivors (survivorship bias), dividends are excluded, and a handful of trades proves nothing.
If the strategy does not clearly beat the baseline *out of sample*, do not trust the scores.

## Phase 5 - Local AI (Ollama)
```bash
ollama pull llama3.1                                     # or set STOCKPILOT_OLLAMA_MODEL
python -m stockpilot research RELIANCE                   # deterministic report (no AI needed)
python -m stockpilot research RELIANCE --ai --save       # adds an LLM narrative; saved to data/reports/
python -m stockpilot compare TCS INFY
python -m stockpilot summarize-filing --file results.pdf
```
The model receives only numbered facts from your database and must cite them. Every number in its output is checked against those facts and
anything unverifiable is flagged (checks are warnings - still read the report). If Ollama is down, you get the deterministic report instead.
Filing summaries are checked against the source text the same way. Scanned PDFs need OCR first.

## Phase 6 - Portfolio
```bash
python -m stockpilot buy RELIANCE --qty 10 --price 2800 --date 2025-06-02 --fees 25
python -m stockpilot sell RELIANCE --qty 4 --price 3000
python -m stockpilot import-transactions --csv trades.csv       # symbol,date,side,quantity,price[,fees,notes]
python -m stockpilot portfolio                                  # FIFO P&L, weights, sector exposure, concentration warnings
python -m stockpilot backup-db                                  # keeps the newest 14 in data/backups/
```
Overselling and back-dated inconsistencies are rejected. Taxes are not calculated. (`portfolio_holdings` from v0.1 is unused; the ledger is `portfolio_transactions`.)

## Wider universe and a real backtest
```bash
python -m stockpilot load-index --index nifty100          # also: nifty50 / nifty200 / nifty500, or --file list.csv
python -m stockpilot download-prices --years 5            # --years also back-fills history you already have
python -m stockpilot check-data
python -m stockpilot backtest --oos-from 2025-06-01       # judge the OUT-OF-SAMPLE columns
python -m stockpilot backtest --oos-from 2025-06-01 --min-score 65 --max-hold 15   # try other settings
```
`load-index` records each company's industry as its sector (used for sector exposure and peer P/E), so you rarely need `import-sectors`.
Backtest flags: `--min-score --atr-mult --max-hold --max-positions --min-rr --risk-pct --cost-pct --capital`.
`backtest --regime-filter` only allows new trades while the Nifty is above its 200-day average. `backtest-analysis` breaks the latest backtest down by exit type, score bucket, setup and year, and shows costs and a 95% range for the average R-multiple (is the edge distinguishable from zero?).
`backtest --breakout-only` ignores near-breakout (buy-stop) setups; the same switch for live screening is `STOCKPILOT_BREAKOUT_ONLY=1`.
`momentum-backtest` tests a slow strategy: each month hold the strongest N stocks by 12-1 month return (`--top-n --lookback --skip --regime-filter`), compared with holding all the stocks equally and with the Nifty; it also lists today's top picks. Survivorship bias is especially strong for this test (today's index members are past winners).
Every setting you tune because it looked better in-sample is curve-fitting: pick settings on the in-sample period, then check the out-of-sample period once.

## Settings
Env vars: `STOCKPILOT_CAPITAL`, `_RISK_PCT`, `_MAX_POSITION_PCT`, `_COST_PCT`, `_DB_PATH`, `_OLLAMA_URL`, `_OLLAMA_MODEL`, `_HISTORY_YEARS`; the rest are in `stockpilot/config.py`.
Existing databases are migrated automatically (new columns/tables are added on `init-db` or first dashboard load).

## Known limitations
* yfinance and NSE endpoints are unofficial/changeable - validate against exchange data and filings before relying on anything.
* Not covered: intraday, options/futures, order execution, multi-user, tax computation.
