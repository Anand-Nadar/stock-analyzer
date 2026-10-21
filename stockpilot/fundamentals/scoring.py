"""Long-term (fundamental) engine.

Score 0-100 with the blueprint's initial weights (a hypothesis to be validated, not a proven model):
  Growth 20 | Profitability 20 | Financial health 20 | Valuation 20 | Stability 10 | Momentum 10

Rules that matter more than the score:
  * Missing data never scores zero. A pillar with no usable inputs is dropped from the score and
    reduces `confidence` (share of weight that had real data behind it).
  * Serious red flags block a stock from being a CANDIDATE no matter how high the score is.
  * Governance (promoter pledging, auditor remarks, related-party dealings) is NOT assessed - always check manually.
All money in Rs crore, shares in crore (see importer).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from statistics import median

import pandas as pd

from ..config import SETTINGS, Settings
from ..data.prices import load_price_frame

WEIGHTS = {"growth": 20, "profitability": 20, "financial_health": 20, "valuation": 20, "stability": 10, "momentum": 10}


def is_financial(sector: str | None, industry: str | None = None) -> bool:
    t = f"{sector or ''} {industry or ''}".lower()
    return any(k in t for k in ("financ", "bank", "insurance", "nbfc"))


def _lin(x: float | None, bad: float, good: float) -> float | None:
    """0 at `bad`, 1 at `good`, linear in between (works for either direction)."""
    if x is None:
        return None
    return max(0.0, min(1.0, (x - bad) / (good - bad)))


def _n(x, nd: int = 1) -> str:
    """Compact number for detail strings."""
    return "n/a" if x is None else (str(x) if isinstance(x, bool) else f"{x:.{nd}f}")


def _safe_div(a, b):
    return None if a is None or b in (None, 0) else a / b


def _cagr(first, last, years):
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return None
    return ((last / first) ** (1 / years) - 1) * 100.0


def load_statements(conn: sqlite3.Connection, stock_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM financial_statements WHERE stock_id=? AND period_type='A' ORDER BY period_end",
                        (stock_id,)).fetchall()
    return [dict(r) for r in rows]


def compute_metrics(stmts: list[dict], price: float | None) -> dict:
    """Derive ratios from ascending annual statements. Anything not computable is None."""
    m: dict = {"n_years": len(stmts)}
    if not stmts:
        return m
    last = stmts[-1]
    m["latest_period"] = last["period_end"]
    m["all_validated"] = all(s.get("validated") for s in stmts)

    rev = [s.get("revenue") for s in stmts]
    npf = [s.get("net_profit") for s in stmts]
    yrs = len(stmts) - 1
    m["revenue_cagr"] = _cagr(rev[0], rev[-1], yrs)
    m["profit_cagr"] = _cagr(npf[0], npf[-1], yrs)
    pairs = [(a, b) for a, b in zip(rev[:-1], rev[1:]) if a is not None and b is not None]
    m["revenue_up_frac"] = (sum(b > a for a, b in pairs) / len(pairs)) if pairs else None
    m["revenue_yoy"] = ((pairs[-1][1] / pairs[-1][0] - 1) * 100.0) if pairs and pairs[-1][0] else None
    known = [p for p in npf if p is not None]
    m["profit_pos_frac"] = (sum(p > 0 for p in known) / len(known)) if known else None

    eq, ta, cl = last.get("equity"), last.get("total_assets"), last.get("current_liabilities")
    ebit, interest = last.get("operating_profit"), last.get("interest_expense")
    m["roe"] = _safe_div(last.get("net_profit"), eq) * 100.0 if _safe_div(last.get("net_profit"), eq) is not None else None
    cap_emp = (ta - cl) if ta is not None and cl is not None else None
    roce = _safe_div(ebit, cap_emp)
    m["roce"] = roce * 100.0 if roce is not None else None
    opm = _safe_div(ebit, last.get("revenue"))
    m["op_margin"] = opm * 100.0 if opm is not None else None
    m["debt_equity"] = _safe_div(last.get("total_debt"), eq)
    m["interest_cover"] = _safe_div(ebit, abs(interest)) if interest else None
    ocfm = _safe_div(last.get("operating_cash_flow"), last.get("revenue"))
    m["ocf_margin"] = ocfm * 100.0 if ocfm is not None else None
    ocf, np_ = last.get("operating_cash_flow"), last.get("net_profit")
    m["cash_conversion"] = (ocf / np_) if ocf is not None and np_ is not None and np_ > 0 else None
    m["net_loss"] = (np_ < 0) if np_ is not None else None
    m["ocf_negative"] = (ocf < 0) if ocf is not None else None

    sh = last.get("shares_outstanding")
    if price and sh:
        mcap = price * sh
        m["market_cap_cr"] = mcap
        eps = _safe_div(np_, sh)
        m["eps"] = eps
        m["pe"] = (price / eps) if eps and eps > 0 else None          # None when earnings are <= 0
        m["pb"] = _safe_div(mcap, eq) if eq and eq > 0 else None
        if last.get("ebitda") and last["ebitda"] > 0 and last.get("total_debt") is not None:
            m["ev_ebitda"] = (mcap + last["total_debt"] - (last.get("cash") or 0.0)) / last["ebitda"]
    return m


def momentum_metrics(df: pd.DataFrame, bench: pd.DataFrame | None) -> dict:
    out: dict = {}
    if len(df) >= 253:
        c = df["close"]
        out["ret_12m"] = float((c.iloc[-1] / c.iloc[-253] - 1) * 100.0)
        out["above_200dma"] = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])
        if bench is not None and len(bench) >= 253:
            b = bench["close"].reindex(df.index, method="ffill")
            if b.notna().iloc[-253] and b.notna().iloc[-1]:
                out["excess_12m"] = out["ret_12m"] - float((b.iloc[-1] / b.iloc[-253] - 1) * 100.0)
    return out


def _avg(parts: list[float | None]) -> float | None:
    vals = [p for p in parts if p is not None]
    return sum(vals) / len(vals) if vals else None


def score_long_term(symbol: str, m: dict, mom: dict, sector_pe_median: float | None, financial: bool,
                    as_of: date, s: Settings = SETTINGS) -> dict:
    res: dict = {"symbol": symbol, "is_financial": financial, "metrics": {**m, **mom}, "flags": [], "serious_flags": [],
                 "warnings": [], "pillars": {}}
    if not m.get("n_years"):
        res.update(status="NO_DATA", score=None, confidence=0.0)
        res["warnings"].append("no annual financial statements imported")
        return res
    if m["n_years"] < 3:
        res["warnings"].append(f"only {m['n_years']} annual period(s): growth/stability measures are weak")

    fin = financial  # banks/NBFCs: leverage, ROCE, operating margin and EV/EBITDA are not meaningful
    pe_rel = (m["pe"] / sector_pe_median) if m.get("pe") and sector_pe_median else None
    pillars: dict[str, tuple[float | None, str]] = {
        "growth": (_avg([_lin(m.get("revenue_cagr"), 0, 15), _lin(m.get("profit_cagr"), 0, 20), m.get("revenue_up_frac")]),
                   f"revenue CAGR {_n(m.get('revenue_cagr'))}, profit CAGR {_n(m.get('profit_cagr'))}, revenue up in {_n(m.get('revenue_up_frac'))} of years"),
        "profitability": (_avg([_lin(m.get("roe"), 5, 20), None if fin else _lin(m.get("roce"), 6, 22),
                                None if fin else _lin(m.get("op_margin"), 5, 25)]),
                          f"ROE {_n(m.get('roe'))}, ROCE {_n(m.get('roce'))}, operating margin {_n(m.get('op_margin'))}"),
        "financial_health": (None if fin else _avg([_lin(m.get("debt_equity"), 2.0, 0.3), _lin(m.get("interest_cover"), 1.5, 8),
                                                    _lin(m.get("ocf_margin"), 0, 12)]),
                             f"debt/equity {_n(m.get('debt_equity'))}, interest cover {_n(m.get('interest_cover'))}, OCF margin {_n(m.get('ocf_margin'))}"),
        "valuation": (_avg([_lin(m.get("pe"), 45, 12) if m.get("pe") is not None else (0.0 if m.get("eps") is not None else None),
                            _lin(m.get("pb"), 8, 1.5), None if fin else _lin(m.get("ev_ebitda"), 25, 8),
                            _lin(pe_rel, 1.5, 0.8)]),
                      f"P/E {_n(m.get('pe'))}, P/B {_n(m.get('pb'))}, EV/EBITDA {_n(m.get('ev_ebitda'))}, P/E vs sector median {_n(pe_rel)}"),
        "stability": (_avg([m.get("profit_pos_frac"), _lin(m.get("cash_conversion"), 0.5, 1.2)]),
                      f"profitable in {_n(m.get('profit_pos_frac'))} of years, cash conversion {_n(m.get('cash_conversion'))}"),
        "momentum": (_avg([_lin(mom.get("excess_12m"), -10, 25),
                           None if "above_200dma" not in mom else (1.0 if mom["above_200dma"] else 0.0)]),
                     f"12m return {_n(mom.get('ret_12m'))}, excess vs benchmark {_n(mom.get('excess_12m'))}, above 200DMA {_n(mom.get('above_200dma'))}"),
    }
    got = wsum = 0.0
    for k, (frac, detail) in pillars.items():
        w = WEIGHTS[k]
        res["pillars"][k] = {"weight": w, "fraction": None if frac is None else round(frac, 3), "points": None if frac is None else round(frac * w, 1),
                             "detail": detail if frac is not None else "no usable data"}
        if frac is not None:
            got += frac * w
            wsum += w
    res["confidence"] = round(wsum / 100.0, 2)
    res["score"] = round(got / wsum * 100.0, 1) if wsum else None

    # ---- red flags: serious ones block candidacy ----
    sf, f = res["serious_flags"], res["flags"]
    if m.get("net_loss"):
        sf.append("latest year net loss")
    if m.get("ocf_negative") and not fin:
        sf.append("negative operating cash flow")
    if not fin and m.get("debt_equity") is not None and m["debt_equity"] > 2.0:
        sf.append(f"very high leverage (debt/equity {m['debt_equity']:.2f})")
    if not fin and m.get("interest_cover") is not None and m["interest_cover"] < 1.5:
        sf.append(f"weak interest cover ({m['interest_cover']:.2f}x)")
    if m.get("pe") is not None and m["pe"] > 60:
        sf.append(f"excessive valuation (P/E {m['pe']:.0f})")
    try:
        age_m = (as_of.year - int(m["latest_period"][:4])) * 12 + (as_of.month - int(m["latest_period"][5:7]))
        if age_m > s.lt_stale_months:
            sf.append(f"financials stale (latest period {m['latest_period']})")
    except (KeyError, ValueError):
        pass
    if m.get("revenue_yoy") is not None and m["revenue_yoy"] < -10:
        f.append(f"revenue fell {m['revenue_yoy']:.0f}% last year")
    if m.get("cash_conversion") is not None and m["cash_conversion"] < 0.5:
        f.append("weak cash conversion (OCF < 50% of profit)")
    if not m.get("all_validated"):
        res["warnings"].append("some financials are not marked validated (cross-check with filings, then `validate-financials`)")
    res["warnings"].append("governance not assessed: check promoter pledging, auditor remarks and related-party dealings manually")

    if res["score"] is None:
        res["status"] = "NO_DATA"
    elif res["score"] >= s.lt_min_score and res["confidence"] >= s.lt_min_confidence:
        res["status"] = "WATCH_SERIOUS_WARNING" if sf else "CANDIDATE"
    else:
        res["status"] = "REJECT" if res["confidence"] >= s.lt_min_confidence else "LOW_CONFIDENCE"
    return res


def screen_long(conn: sqlite3.Connection, s: Settings = SETTINGS, persist: bool = True,
                extra_symbols: list[str] | None = None) -> dict:
    """Score every universe stock (plus `extra_symbols`) that has financial statements."""
    rows = conn.execute("SELECT id, symbol, sector, industry FROM stocks WHERE in_universe=1 ORDER BY symbol").fetchall()
    if extra_symbols:
        have = {r["symbol"] for r in rows}
        extra = [x.upper() for x in extra_symbols if x.upper() not in have]
        if extra:
            rows = list(rows) + conn.execute(
                f"SELECT id, symbol, sector, industry FROM stocks WHERE symbol IN ({','.join('?' * len(extra))})", extra).fetchall()
    bench_row = conn.execute("SELECT id FROM stocks WHERE symbol=?", (s.benchmark_symbol,)).fetchone()
    bench = load_price_frame(conn, bench_row["id"]) if bench_row else None
    if bench is not None and bench.empty:
        bench = None

    prepared, as_ofs = [], []
    for r in rows:
        df = load_price_frame(conn, r["id"])
        price = float(df["close"].iloc[-1]) if not df.empty else None
        if not df.empty:
            as_ofs.append(df.index[-1].date())
        m = compute_metrics(load_statements(conn, r["id"]), price)
        prepared.append((r, m, momentum_metrics(df, bench) if not df.empty else {}))

    by_sector: dict[str, list[float]] = {}
    for r, m, _ in prepared:
        if r["sector"] and m.get("pe"):
            by_sector.setdefault(r["sector"], []).append(m["pe"])
    medians = {k: median(v) for k, v in by_sector.items() if len(v) >= 3}   # need >= 3 peers

    as_of = max(as_ofs) if as_ofs else datetime.now().date()
    results = [score_long_term(r["symbol"], m, mom, medians.get(r["sector"]), is_financial(r["sector"], r["industry"]), as_of, s)
               for r, m, mom in prepared]
    order = {"CANDIDATE": 0, "WATCH_SERIOUS_WARNING": 1, "REJECT": 2, "LOW_CONFIDENCE": 3, "NO_DATA": 4}
    results.sort(key=lambda x: (order[x["status"]], -(x["score"] or -1)))
    run = {"engine": "LONG_TERM", "data_as_of": as_of.isoformat() if as_ofs else None, "results": results,
           "candidates": [x for x in results if x["status"] == "CANDIDATE"]}
    if persist:
        cfg = {"weights": WEIGHTS, "min_score": s.lt_min_score, "min_confidence": s.lt_min_confidence,
               "stale_months": s.lt_stale_months, "sector_pe_medians": medians}
        cur = conn.execute("INSERT INTO recommendation_runs (engine, data_as_of, config_json, results_json, notes) VALUES (?,?,?,?,?)",
                           ("LONG_TERM", run["data_as_of"], json.dumps(cfg), json.dumps(results, default=str),
                            "Candidates worth researching - not a buy recommendation. Governance not assessed."))
        run["run_id"] = cur.lastrowid
        for x in results:
            lp, sid = x["metrics"].get("latest_period"), conn.execute("SELECT id FROM stocks WHERE symbol=?", (x["symbol"],)).fetchone()
            if lp and sid:
                for k, v in x["metrics"].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        conn.execute("INSERT INTO fundamental_metrics (stock_id, period_end, metric, value) VALUES (?,?,?,?) "
                                     "ON CONFLICT(stock_id, period_end, metric) DO UPDATE SET value=excluded.value, calculated_at=datetime('now')",
                                     (sid["id"], lp, k, v))
        conn.commit()
    return run
