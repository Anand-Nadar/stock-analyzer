"""Local dashboard.  Run:  uvicorn stockpilot.web.app:app --reload   (binds to localhost by default)"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .. import db
from ..config import SETTINGS
from ..data.validation import freshness_report

app = FastAPI(title="StockPilot India")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _latest_run(conn, engine: str):
    row = conn.execute("SELECT * FROM recommendation_runs WHERE engine=? ORDER BY id DESC LIMIT 1", (engine,)).fetchone()
    if not row:
        return None
    data = json.loads(row["results_json"])
    base = {"id": row["id"], "run_at": row["run_at"], "data_as_of": row["data_as_of"], "config": json.loads(row["config_json"])}
    if engine == "SWING":
        return {**base, **data}
    return {**base, "results": data, "candidates": [r for r in data if r["status"] == "CANDIDATE"],
            "watch": [r for r in data if r["status"] == "WATCH_SERIOUS_WARNING"]}


def _render(request: Request, name: str, **ctx):
    return templates.TemplateResponse(request, name, {"settings": SETTINGS, **ctx})


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    with db.session() as conn:
        db.init_db(conn)
        fresh = freshness_report(conn).as_dict()
        run = _latest_run(conn, "SWING")
        log = conn.execute("SELECT run_at, task, status, details FROM data_refresh_log ORDER BY id DESC LIMIT 8").fetchall()
        failed = conn.execute("SELECT run_at, task, details FROM data_refresh_log WHERE status='FAILED' AND run_at >= datetime('now','-2 days') ORDER BY id DESC").fetchall()
    return _render(request, "dashboard.html", page="swing", fresh=fresh, run=run, log=log, failed=failed)


@app.get("/long-term", response_class=HTMLResponse)
def long_term(request: Request):
    with db.session() as conn:
        db.init_db(conn)
        run = _latest_run(conn, "LONG_TERM")
    return _render(request, "long_term.html", page="long", run=run)


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio(request: Request):
    from ..paper import paper_report
    from ..portfolio import portfolio_summary
    with db.session() as conn:
        db.init_db(conn)
        pf = portfolio_summary(conn)
        paper = paper_report(conn)
    return _render(request, "portfolio.html", page="portfolio", pf=pf, paper=paper)


@app.get("/backtests", response_class=HTMLResponse)
def backtests(request: Request):
    with db.session() as conn:
        db.init_db(conn)
        rows = conn.execute("SELECT id, strategy, run_at, config_json, costs_json, metrics_json FROM backtest_runs ORDER BY id DESC LIMIT 12").fetchall()
    runs = [{"id": r["id"], "strategy": r["strategy"], "run_at": r["run_at"], "config": json.loads(r["config_json"]),
             "costs": json.loads(r["costs_json"] or "{}"), "metrics": json.loads(r["metrics_json"] or "{}").get("metrics", {})} for r in rows]
    return _render(request, "backtests.html", page="backtests", runs=runs)


@app.get("/stock/{symbol}", response_class=HTMLResponse)
def stock(request: Request, symbol: str):
    from ..ai.facts import build_facts
    from ..ai.report import plain_report
    with db.session() as conn:
        db.init_db(conn)
        try:
            facts = build_facts(conn, symbol)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
    return _render(request, "stock.html", page="stock", facts=facts, report=plain_report(facts))


@app.get("/api/swing/latest")
def api_latest():
    with db.session() as conn:
        db.init_db(conn)
        run = _latest_run(conn, "SWING")
    return JSONResponse(run or {"detail": "no screening run yet; run `python -m stockpilot screen-swing`"})


@app.get("/api/data/freshness")
def api_freshness():
    with db.session() as conn:
        db.init_db(conn)
        return freshness_report(conn).as_dict()
