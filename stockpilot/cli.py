"""Command line interface:  python -m stockpilot <command> --help"""
from __future__ import annotations

import argparse
import json
import sys

from . import db
from .config import SETTINGS, PROJECT_ROOT
from .data import universe as uni
from .data.validation import freshness_report, find_gaps
from .risk import position_size

DISCLAIMER = "\nResearch aid only - not investment advice. Stops do not guarantee loss limits."


# ------------------------------ phase 1-2 ------------------------------
def cmd_init_db(a):
    with db.session() as conn:
        db.init_db(conn)
        uni.ensure_benchmark(conn)
    print(f"Database ready at {SETTINGS.db_path}")


def cmd_load_master(a):
    text = open(a.file).read() if a.file else uni.download_equity_list()
    df = uni.parse_equity_list(text)
    with db.session() as conn:
        db.init_db(conn)
        n = uni.upsert_stocks(conn, df)
        db.log_refresh(conn, "load_nse_master", "OK", f"{n} EQ-series securities")
    print(f"Loaded {n} NSE EQ-series securities.")


def cmd_set_universe(a):
    symbols = uni.read_symbol_file(a.file)
    with db.session() as conn:
        n, unknown = uni.set_universe(conn, symbols)
    print(f"Universe set to {n} stocks.")
    if unknown:
        print(f"Not found in security master (check spelling / run load-master): {', '.join(unknown)}")


def cmd_load_index(a):
    text = open(a.file).read() if a.file else uni.download_index_list(a.index)
    df = uni.parse_index_constituents(text)
    with db.session() as conn:
        db.init_db(conn)
        n, unknown, sectors = uni.apply_index_universe(conn, df)
        db.log_refresh(conn, "load_index", "OK", f"{a.index}: {n} stocks, {sectors} sectors")
    print(f"Universe set to {n} stocks from {a.index if not a.file else a.file}; sector recorded for {sectors}.")
    if unknown:
        print(f"{len(unknown)} symbols not in the NSE master (renamed/new?): {', '.join(unknown[:20])}{' ...' if len(unknown) > 20 else ''}")
    print("Next: python -m stockpilot download-prices --years 5")


def cmd_download(a):
    from .data.prices import update_prices
    with db.session() as conn:
        s = update_prices(conn, a.symbols or None, include_benchmark=not a.no_benchmark, years=a.years)
    print(f"Updated {s['ok']} symbols, {s['rows']} rows.")
    if s["empty"]:
        print("No data returned for:", ", ".join(s["empty"]))
    for f in s["failed"]:
        print("FAILED:", f)
    for sym, issues in s["issues"].items():
        print(f"Data issues in {sym}: {'; '.join(issues)}")


def cmd_check(a):
    with db.session() as conn:
        rep = freshness_report(conn)
        print(json.dumps(rep.as_dict(), indent=2))
        if a.gaps:
            for sym in [r["symbol"] for r in conn.execute("SELECT symbol FROM stocks WHERE in_universe=1")]:
                g = find_gaps(conn, sym)
                if g:
                    print(f"{sym}: gaps {g}")


def cmd_screen(a):
    from .swing import screen_swing
    if a.capital: SETTINGS.capital = a.capital
    if a.risk_pct: SETTINGS.risk_pct_per_trade = a.risk_pct
    with db.session() as conn:
        run = screen_swing(conn)
    print(f"Data as of {run['data_as_of']} | capital Rs {SETTINGS.capital:,.0f} | risk/trade {SETTINGS.risk_pct_per_trade}% | run #{run.get('run_id')}")
    cands = run["candidates"][: a.top]
    if not cands:
        print("No candidates passed all gates today. That is a valid outcome.")
    for r in cands:
        p = r["plan"]
        print(f"\n{r['symbol']:<12} score {r['score']:>5}  {r['setup']}  close {r['close']}")
        print(f"   entry {p['entry']} ({p['entry_type']}) | stop {p['stop_loss']} | target {p['target']} | net R:R {p['net_risk_reward']}")
        print(f"   qty {p['position']['quantity']} (Rs {p['position']['position_value']:,.0f}), max loss at stop Rs {p['position']['risk_at_stop']:,.0f}")
    if a.show_rejected:
        print("\nRejected:")
        for r in run["rejected"]:
            print(f"  {r['symbol']:<12} {r.get('score')}  -> {'; '.join(r['rejections'])}")
    print(DISCLAIMER)


def cmd_size(a):
    print(json.dumps(position_size(a.capital, a.risk_pct, a.entry, a.stop, a.max_position_pct).as_dict(), indent=2))


# ------------------------------ phase 3: fundamentals ------------------------------
def cmd_import_financials(a):
    from .fundamentals.importer import import_csv
    with db.session() as conn:
        db.init_db(conn)
        n, errs = import_csv(conn, a.csv)
        db.log_refresh(conn, "import_financials", "OK" if not errs else "PARTIAL", f"{n} rows, {len(errs)} errors")
    print(f"Imported {n} statement rows (amounts in Rs crore, shares in crore).")
    for e in errs:
        print("  skipped:", e)


def cmd_fetch_financials(a):
    from .fundamentals.importer import fetch_yfinance
    with db.session() as conn:
        db.init_db(conn)
        syms = a.symbols or [r["symbol"] for r in conn.execute("SELECT symbol FROM stocks WHERE in_universe=1")]
        out = fetch_yfinance(conn, syms)
        db.log_refresh(conn, "fetch_financials", "OK" if not out["failed"] else "PARTIAL", f"ok={len(out['ok'])} failed={len(out['failed'])}")
    print(f"Fetched {len(out['ok'])} symbols. No data: {out['empty'] or '-'}")
    for f in out["failed"]:
        print("FAILED:", f)
    print("These figures come from an unofficial source and are marked NOT validated. Cross-check with filings, then run `validate-financials SYMBOL`.")


def cmd_validate_fin(a):
    from .fundamentals.importer import mark_validated
    with db.session() as conn:
        print(f"Marked {mark_validated(conn, a.symbol)} statement rows validated for {a.symbol.upper()}.")


def cmd_import_sectors(a):
    from .fundamentals.importer import import_sectors_csv
    with db.session() as conn:
        print(f"Updated sector for {import_sectors_csv(conn, a.csv)} stocks.")


def cmd_screen_long(a):
    from .fundamentals.scoring import screen_long
    with db.session() as conn:
        run = screen_long(conn)
    print(f"Long-term screen, price data as of {run['data_as_of']} | run #{run.get('run_id')}")
    shown = run["results"] if a.show_all else run["candidates"][: a.top]
    if not shown:
        print("No CANDIDATE passed. Check that financials are imported (import-financials / fetch-financials).")
    for r in shown:
        print(f"\n{r['symbol']:<12} {r['status']:<22} score {r['score']}  confidence {r['confidence']}")
        if r["status"] != "NO_DATA":
            print("   " + " | ".join(f"{k} {v['points'] if v['points'] is not None else 'n/a'}/{v['weight']}" for k, v in r["pillars"].items()))
        for f in r["serious_flags"]:
            print("   SERIOUS:", f)
        for f in r["flags"]:
            print("   flag:", f)
    watch = [r for r in run["results"] if r["status"] == "WATCH_SERIOUS_WARNING"]
    if watch and not a.show_all:
        print(f"\n{len(watch)} high-scoring stocks were held back by serious flags: " + ", ".join(w["symbol"] for w in watch) + "  (use --show-all)")
    print("\nCandidates are worth RESEARCHING, not buying blindly. Governance is not assessed.")


# ------------------------------ phase 4: backtest & paper ------------------------------
def cmd_backtest(a):
    from dataclasses import replace
    from .backtest import run_and_store
    strategies = ("swing", "baseline") if a.strategy == "both" else (a.strategy,)
    overrides = {k: v for k, v in {
        "min_score": a.min_score, "atr_stop_multiple": a.atr_mult, "bt_max_hold_days": a.max_hold,
        "bt_max_positions": a.max_positions, "risk_pct_per_trade": a.risk_pct, "round_trip_cost_pct": a.cost_pct,
        "min_risk_reward": a.min_rr, "capital": a.capital,
        "breakout_only": True if a.breakout_only else None}.items() if v is not None}
    cfg = replace(SETTINGS, **overrides)
    if overrides:
        print(f"Settings overridden for this run: {overrides}")
        print("! Tuning warning: every setting you try and keep because it looked better on the in-sample period is a form of curve-fitting.\n"
              "  Judge only by the OUT-OF-SAMPLE columns, and don't keep re-tuning against them.\n")
    with db.session() as conn:
        rep = run_and_store(conn, cfg, a.start, a.end, a.oos_from, strategies, regime_filter=a.regime_filter)
    if a.regime_filter:
        print("Regime filter ON: new trades only while the Nifty is above its 200-day average.\n")
    keys = ["trades", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_rf0", "win_rate_pct", "profit_factor",
            "avg_r", "avg_bars_held", "avg_exposure_pct", "turnover_x", "net_pnl", "benchmark_return_pct", "benchmark_max_drawdown_pct"]
    cols = list(rep)
    print(f"{'metric':<28}" + "".join(f"{c:>24}" for c in cols))
    print(f"{'period':<28}" + "".join(f"{(rep[c]['start'] or '-') + '..' + (rep[c]['end'] or '-'):>24}" for c in cols))
    for k in keys:
        print(f"{k:<28}" + "".join(f"{str(rep[c]['metrics'].get(k, '-')):>24}" for c in cols))
    shown = set()
    for c in cols:
        for n in rep[c]["metrics"].get("notes", []):
            if (c, n) not in shown:
                print(f"! [{c}] {n}")
                shown.add((c, n))
    print("\nLimitations:")
    for l in rep[cols[0]]["limitations"]:
        print(" -", l)
    print("Results saved to backtest_runs.")


def cmd_bt_analysis(a):
    from .analysis import analyze_trades
    with db.session() as conn:
        q = ("SELECT id, strategy, config_json, metrics_json FROM backtest_runs WHERE id=?" if a.run_id else
             "SELECT id, strategy, config_json, metrics_json FROM backtest_runs "
             "WHERE run_at >= datetime((SELECT MAX(run_at) FROM backtest_runs), '-5 minutes') ORDER BY id")
        rows = conn.execute(q, (a.run_id,) if a.run_id else ()).fetchall()
    if not rows:
        sys.exit("No backtest runs found. Run `backtest` first.")
    for r in rows:
        trades = json.loads(r["metrics_json"])["trades"]
        an = analyze_trades(trades)
        cfg = json.loads(r["config_json"])
        print(f"\n===== run #{r['id']}  {r['strategy']}  {cfg.get('start')}..{cfg.get('end')} =====")
        o = an["overall"]
        if not o["n"]:
            print("no trades"); continue
        print(f"trades {o['n']} | win rate {o['win_pct']}% | avg R {o.get('avg_r')} (95% range {o.get('r_low')} to {o.get('r_high')}) | "
              f"net P&L Rs {o['net_pnl']:,.0f} | before costs Rs {an['gross_pnl']:,.0f} | costs Rs {an['costs']:,.0f}")
        for title, key in (("by exit type", "by_exit"), ("by score bucket", "by_score"), ("by setup", "by_setup"), ("by entry year", "by_year")):
            print(f"  {title}:")
            for k, v in an[key].items():
                print(f"    {str(k):<14} n={v['n']:<4} win {v.get('win_pct')}%  avg R {v.get('avg_r')}  net Rs {v.get('net_pnl'):,.0f}")
        for n in an["notes"]:
            print("  !", n)
    print("\nIf higher score buckets do not show a higher avg R, the score is not predictive.")


def cmd_momentum(a):
    from dataclasses import replace
    from .momentum import run_and_store_momentum, MOMENTUM_LIMITATIONS
    cfg = replace(SETTINGS, **({"capital": a.capital} if a.capital else {}), **({"round_trip_cost_pct": a.cost_pct} if a.cost_pct is not None else {}))
    with db.session() as conn:
        rep = run_and_store_momentum(conn, cfg, a.start, a.end, a.oos_from, a.top_n, a.lookback, a.skip, a.regime_filter)
    picks = rep.pop("_picks")
    cols = list(rep)
    keys = ["total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_rf0", "avg_exposure_pct", "turnover_x", "costs_paid",
            "positions_opened", "benchmark_return_pct", "benchmark_max_drawdown_pct"]
    print(f"{'metric':<28}" + "".join(f"{c:>30}" for c in cols))
    print(f"{'period':<28}" + "".join(f"{(rep[c]['start'] or '-') + '..' + (rep[c]['end'] or '-'):>30}" for c in cols))
    for k in keys:
        print(f"{k:<28}" + "".join(f"{str(rep[c]['metrics'].get(k, '-')):>30}" for c in cols))
    print("\nReturn by calendar year (%):")
    years = sorted({y for c in cols for y in rep[c]["metrics"].get("returns_by_year_pct", {})})
    print(f"{'year':<28}" + "".join(f"{c:>30}" for c in cols))
    for y in years:
        print(f"{y:<28}" + "".join(f"{str(rep[c]['metrics'].get('returns_by_year_pct', {}).get(y, '-')):>30}" for c in cols))
    print(f"\nCurrent top {a.top_n} by momentum as of {picks['as_of']} (12-1 month return %): {picks['picks']}")
    print("\nLimitations:")
    for l in MOMENTUM_LIMITATIONS:
        print(" -", l)
    print("Compare 'momentum' with 'equal_weight_all' (same stocks, no ranking) and with the benchmark. Saved to backtest_runs.")


def cmd_paper_open(a):
    from .paper import open_paper_trade
    with db.session() as conn:
        try:
            print("Opened paper trade:", json.dumps(open_paper_trade(conn, a.symbol, a.run_id, a.qty)))
        except ValueError as exc:
            sys.exit(f"Error: {exc}")


def cmd_paper_update(a):
    from .paper import update_paper_trades
    with db.session() as conn:
        closed = update_paper_trades(conn)
    print(f"{len(closed)} paper trade(s) closed." + "".join(f"\n  {c['symbol']}: {c['reason']} at {c['exit']} on {c['closed_at']}" for c in closed))


def cmd_paper_report(a):
    from .paper import paper_report
    with db.session() as conn:
        r = paper_report(conn)
    for t in r["trades"]:
        print(f"#{t['id']:<3} {t['symbol']:<12} {t['status']:<8} entry {t['entry']} stop {t['stop']} target {t['target']} x{t['qty']}  mark/exit {t['mark_or_exit']}  net P&L Rs {t['net_pnl']:,.0f}")
    print(f"\nClosed: {r['closed']} | win rate {r['win_rate_pct']}% | closed net P&L Rs {r['closed_net_pnl']:,.0f}")


# ------------------------------ phase 5: AI ------------------------------
def cmd_research(a):
    from .ai.facts import build_facts
    from .ai.report import plain_report, llm_report, save_report
    from .ai.ollama import OllamaError
    with db.session() as conn:
        try:
            facts = build_facts(conn, a.symbol)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
    body = plain_report(facts)
    if a.ai:
        try:
            r = llm_report(facts, a.model)
            body = f"{r['text']}\n\n--- verification ---\n" + ("\n".join("! " + n for n in r["notes"]) or "no problems detected (still read it critically)") \
                   + f"\n\n--- facts the model was given ---\n{r['facts_block']}\n\n--- deterministic report ---\n{body}"
        except OllamaError as exc:
            print(f"[AI unavailable: {exc}]\nShowing the deterministic report instead.\n")
    print(body)
    if a.save:
        asof = (facts.get("price") or {}).get("as_of", "nodata")
        print("\nSaved:", save_report(a.symbol, body, asof))


def cmd_compare(a):
    from .ai.facts import build_facts
    from .ai.report import compare, plain_report
    from .ai.ollama import OllamaError
    with db.session() as conn:
        fa, fb = build_facts(conn, a.a), build_facts(conn, a.b)
    try:
        r = compare(fa, fb, a.model)
        print(r["text"], "\n")
        for n in r["notes"]:
            print("!", n)
    except OllamaError as exc:
        print(f"[AI unavailable: {exc}]\n")
        print(plain_report(fa), "\n" + "=" * 60 + "\n", plain_report(fb))


def cmd_summarize(a):
    from .ai.filings import summarize
    from .ai.ollama import OllamaError
    try:
        r = summarize(a.file, a.model)
    except (OllamaError, ValueError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
    print(r["summary"])
    print(f"\n[{r['chunks']} chunks processed" + (", TRUNCATED: only the first part of the document was read" if r["truncated"] else "") + "]")
    if r["unverified_numbers"]:
        print("! Numbers in the summary that do not appear in the source text (treat as unreliable):", ", ".join(r["unverified_numbers"][:15]))


# ------------------------------ phase 6: portfolio & jobs ------------------------------
def _trade(side):
    def fn(a):
        from .portfolio import add_transaction
        with db.session() as conn:
            try:
                tid = add_transaction(conn, a.symbol, side, a.qty, a.price, a.date, a.fees, a.notes or "")
            except ValueError as exc:
                sys.exit(f"Error: {exc}")
        print(f"Recorded {side} #{tid}: {a.qty} {a.symbol.upper()} @ {a.price}")
    return fn


def cmd_import_txns(a):
    from .portfolio import import_transactions_csv
    with db.session() as conn:
        n, errs = import_transactions_csv(conn, a.csv)
    print(f"Imported {n} transactions.")
    for e in errs:
        print("  skipped:", e)


def cmd_portfolio(a):
    from .portfolio import portfolio_summary
    with db.session() as conn:
        p = portfolio_summary(conn)
    if not p["positions"]:
        print("No holdings. Record trades with `buy SYMBOL --qty N --price P`.")
    for x in p["positions"]:
        print(f"{x['symbol']:<12} qty {x['quantity']:<6} avg {x['avg_cost']:<9} last {x['price']}  value Rs {x['value'] or 0:>11,.0f}  "
              f"P&L {x['unrealized_pnl']} ({x['unrealized_pct']}%)  weight {x['weight_pct']}%  [{x['sector']}]")
    print(f"\nValue Rs {p['total_value']:,.0f} | cost Rs {p['total_cost']:,.0f} | unrealized Rs {p['unrealized_pnl']:,.0f} | realized Rs {p['realized_pnl']:,.0f}")
    print("Sector exposure:", p["sector_pct"])
    for w in p["warnings"]:
        print("!", w)
    print("Taxes are not calculated.")


def cmd_daily(a):
    from .jobs import daily_update
    with db.session() as conn:
        db.init_db(conn)
        res = daily_update(conn)
    for k, v in res.items():
        if k != "ok":
            print(f"{k:<18} {v}")
    sys.exit(0 if res["ok"] else 1)


def cmd_backup(a):
    from .jobs import backup_db
    with db.session() as conn:
        print("Backup written to", backup_db(conn))


def main(argv=None):
    p = argparse.ArgumentParser(prog="stockpilot", description="StockPilot India - research aid, not investment advice")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help=None):
        sp = sub.add_parser(name, help=help)
        sp.set_defaults(fn=fn)
        return sp

    add("init-db", cmd_init_db)
    s = add("load-master", cmd_load_master, "import NSE EQUITY_L.csv (download, or --file)"); s.add_argument("--file")
    s = add("set-universe", cmd_set_universe, "mark tracked symbols from a text file")
    s.add_argument("--file", default=str(PROJECT_ROOT / "config" / "starter_universe.txt"))
    s = add("load-index", cmd_load_index, "set universe + sectors from a NIFTY index list (downloads, or --file)")
    s.add_argument("--index", choices=sorted(uni.INDEX_FILES), default="nifty100"); s.add_argument("--file")
    s = add("download-prices", cmd_download)
    s.add_argument("symbols", nargs="*"); s.add_argument("--years", type=int); s.add_argument("--no-benchmark", action="store_true")
    s = add("check-data", cmd_check); s.add_argument("--gaps", action="store_true")
    s = add("screen-swing", cmd_screen)
    s.add_argument("--capital", type=float); s.add_argument("--risk-pct", type=float)
    s.add_argument("--top", type=int, default=10); s.add_argument("--show-rejected", action="store_true")
    s = add("size", cmd_size, "position-size calculator")
    s.add_argument("--capital", type=float, default=SETTINGS.capital); s.add_argument("--risk-pct", type=float, default=SETTINGS.risk_pct_per_trade)
    s.add_argument("--entry", type=float, required=True); s.add_argument("--stop", type=float, required=True)
    s.add_argument("--max-position-pct", type=float, default=SETTINGS.max_position_pct)

    s = add("import-financials", cmd_import_financials, "load statements from CSV (Rs crore)"); s.add_argument("--csv", required=True)
    s = add("fetch-financials", cmd_fetch_financials, "fetch annual statements via yfinance (unofficial)"); s.add_argument("symbols", nargs="*")
    s = add("validate-financials", cmd_validate_fin, "mark a symbol's statements as manually verified"); s.add_argument("symbol")
    s = add("import-sectors", cmd_import_sectors, "CSV: symbol,sector[,industry]"); s.add_argument("--csv", required=True)
    s = add("screen-long", cmd_screen_long)
    s.add_argument("--top", type=int, default=15); s.add_argument("--show-all", action="store_true")

    s = add("backtest", cmd_backtest, "backtest the swing strategy vs a simple baseline")
    s.add_argument("--start"); s.add_argument("--end"); s.add_argument("--oos-from", help="split: test out-of-sample from this date (YYYY-MM-DD)")
    s.add_argument("--strategy", choices=["both", "swing", "baseline"], default="both")
    s.add_argument("--min-score", type=float, help=f"entry score threshold (default {SETTINGS.min_score})")
    s.add_argument("--atr-mult", type=float, help=f"stop distance in ATRs (default {SETTINGS.atr_stop_multiple})")
    s.add_argument("--max-hold", type=int, help=f"max bars to hold (default {SETTINGS.bt_max_hold_days})")
    s.add_argument("--max-positions", type=int, help=f"max simultaneous positions (default {SETTINGS.bt_max_positions})")
    s.add_argument("--min-rr", type=float, help=f"min net risk/reward (default {SETTINGS.min_risk_reward})")
    s.add_argument("--risk-pct", type=float); s.add_argument("--cost-pct", type=float); s.add_argument("--capital", type=float)
    s.add_argument("--regime-filter", action="store_true", help="only enter while the Nifty is above its 200-day average")
    s.add_argument("--breakout-only", action="store_true", help="ignore near-breakout (buy-stop) setups")
    s = add("momentum-backtest", cmd_momentum, "monthly rotation into the strongest N stocks (low turnover) vs equal-weight and the Nifty")
    s.add_argument("--start"); s.add_argument("--end"); s.add_argument("--oos-from")
    s.add_argument("--top-n", type=int, default=10); s.add_argument("--lookback", type=int, default=252); s.add_argument("--skip", type=int, default=21)
    s.add_argument("--regime-filter", action="store_true"); s.add_argument("--capital", type=float); s.add_argument("--cost-pct", type=float)
    s = add("backtest-analysis", cmd_bt_analysis, "break the latest backtest down by score, setup, exit type, year")
    s.add_argument("--run-id", type=int)
    s = add("paper-open", cmd_paper_open, "open a paper trade from the latest swing run")
    s.add_argument("symbol"); s.add_argument("--run-id", type=int); s.add_argument("--qty", type=int)
    add("paper-update", cmd_paper_update, "close paper trades that hit stop/target/time")
    add("paper-report", cmd_paper_report)

    s = add("research", cmd_research, "full research report for one stock"); s.add_argument("symbol")
    s.add_argument("--ai", action="store_true", help="add a local-LLM narrative (needs Ollama)"); s.add_argument("--model"); s.add_argument("--save", action="store_true")
    s = add("compare", cmd_compare, "compare two stocks"); s.add_argument("a"); s.add_argument("b"); s.add_argument("--model")
    s = add("summarize-filing", cmd_summarize, "summarise a .txt/.md/.pdf filing with the local model")
    s.add_argument("--file", required=True); s.add_argument("--model")

    for side in ("buy", "sell"):
        s = add(side, _trade(side.upper()), f"record a {side} transaction")
        s.add_argument("symbol"); s.add_argument("--qty", type=int, required=True); s.add_argument("--price", type=float, required=True)
        s.add_argument("--date"); s.add_argument("--fees", type=float, default=0.0); s.add_argument("--notes")
    s = add("import-transactions", cmd_import_txns, "CSV: symbol,date,side,quantity,price[,fees,notes]"); s.add_argument("--csv", required=True)
    add("portfolio", cmd_portfolio)
    add("daily-update", cmd_daily, "prices -> checks -> screens -> paper update -> backup (for cron)")
    add("backup-db", cmd_backup)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
