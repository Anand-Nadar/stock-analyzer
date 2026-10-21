"""Research reports: deterministic facts + optional local-LLM narrative + automatic verification of the narrative."""
from __future__ import annotations

import re
from pathlib import Path

from ..config import PROJECT_ROOT, SETTINGS, Settings
from . import ollama
from .facts import flatten_facts

FORBIDDEN = [r"price target", r"guarantee", r"\bwill (?:reach|rise|double|triple|soar|crash)\b", r"risk[- ]free",
             r"sure (?:thing|to)", r"can'?t lose"]

SYSTEM_PROMPT = """You are a careful equity research assistant for an individual investor in India.
Rules:
1. Use ONLY the numbered facts provided. Never use outside knowledge about the company, prices or news.
2. After every number or factual claim, cite the fact id in square brackets, e.g. [F12] or [F3, F9].
3. If something needed is not in the facts, write "not available". Do not guess or estimate.
4. Do NOT give price targets, forecasts, guarantees, or buy/sell instructions. Describe what the screens found and why.
5. Explicitly mention serious flags, data gaps, stale data and the fact that governance was not assessed.
6. Output plain text with exactly these headings: Summary, Swing view, Long-term view, Risks and data gaps. Maximum 350 words."""


def _facts_block(facts: dict) -> tuple[str, list[dict]]:
    flat = flatten_facts(facts)
    return "\n".join(f"{f['id']}: {f['path']} = {f['value']}" for f in flat), flat


# --------------------------- verification ---------------------------
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_CITE = re.compile(r"\[\s*F\d+(?:\s*,\s*F\d+)*\s*\]")


def _numbers(text: str) -> list[float]:
    text = _CITE.sub(" ", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    vals = []
    for tok in _NUM.findall(text):
        try:
            vals.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return vals


def verify_report(text: str, flat: list[dict]) -> dict:
    """Check the narrative against the facts. Warnings only: a human should still read the report."""
    ids = {f["id"] for f in flat}
    cited = {c.strip() for grp in re.findall(r"\[\s*(F\d+(?:\s*,\s*F\d+)*)\s*\]", text) for c in grp.split(",")}
    nums = [f["value"] for f in flat if isinstance(f["value"], (int, float)) and not isinstance(f["value"], bool)]
    unverified = []
    for v in _numbers(text):
        if float(v).is_integer() and (abs(v) < 10 or 1990 <= abs(v) <= 2100):
            continue   # small counts and years are not checked
        if not any(abs(v - f) <= max(0.06, 0.015 * abs(f)) for f in nums):
            unverified.append(v)
    return {"unknown_citations": sorted(cited - ids), "has_citations": bool(cited), "unverified_numbers": unverified,
            "forbidden_phrases": [p for p in FORBIDDEN if re.search(p, text, re.I)]}


def verification_notes(v: dict) -> list[str]:
    notes = []
    if not v["has_citations"]:
        notes.append("the narrative contains no fact citations")
    if v["unknown_citations"]:
        notes.append(f"cites fact ids that do not exist: {', '.join(v['unknown_citations'])}")
    if v["unverified_numbers"]:
        notes.append(f"numbers not found in the facts (treat as unreliable): {', '.join(str(x) for x in v['unverified_numbers'][:12])}")
    if v["forbidden_phrases"]:
        notes.append("contains language the assistant is not supposed to use (forecast/guarantee/price target) - ignore it")
    return notes


# --------------------------- deterministic report ---------------------------
def plain_report(facts: dict) -> str:
    g = facts.get
    lines = [f"{facts['symbol']} - {g('company') or 'n/a'}" + (f" ({g('sector')})" if g("sector") else "")]
    p = g("price")
    if p:
        lines += ["", f"Price snapshot (as of {p['as_of']}, {p['age_days']} day(s) old)",
                  f"  Close {p['close']} | 20/50/200-day averages {p['sma20']} / {p['sma50']} / {p['sma200']} | RSI14 {p['rsi14']}",
                  f"  Returns: 20d {p['return_20d_pct']}%, 60d {p['return_60d_pct']}% | volume vs 20d average {p['volume_ratio_20d']}x"]
    sw = g("swing")
    if sw:
        lines += ["", f"Swing screen: {'QUALIFIES' if sw['qualifies'] else 'does not qualify'} (score {sw['score']}, setup {sw['setup']})"]
        for k, c in sw.get("components", {}).items():
            lines.append(f"  {k}: {c['points']:.0f}/{c['max']} - {c['detail']}")
        if sw["rejections"]:
            lines.append("  Rejected because: " + "; ".join(sw["rejections"]))
        if sw.get("plan"):
            pl = sw["plan"]
            lines.append(f"  Conditional plan: entry {pl['entry']}, stop {pl['stop_loss']}, target {pl['target']}, net R:R {pl['net_risk_reward']}, "
                         f"qty {pl['quantity']} (loss at stop Rs {pl['risk_at_stop']:,.0f}). Invalid if price closes below the stop.")
    lt = g("long_term")
    if lt:
        lines += ["", f"Long-term screen: {lt['status']} (score {lt['score']}, data confidence {lt['confidence']})"]
        for k, c in lt["pillars"].items():
            lines.append(f"  {k}: {c['points'] if c['points'] is not None else 'n/a'}/{c['weight']} - {c['detail']}")
        if lt["serious_flags"]:
            lines.append("  SERIOUS FLAGS: " + "; ".join(lt["serious_flags"]))
        if lt["flags"]:
            lines.append("  Other flags: " + "; ".join(lt["flags"]))
        for w in lt["warnings"]:
            lines.append("  Note: " + w)
    if facts["missing"]:
        lines += ["", "Data gaps: " + "; ".join(facts["missing"])]
    lines += ["", "Research aid only - not investment advice."]
    return "\n".join(lines)


# --------------------------- LLM report ---------------------------
def llm_report(facts: dict, model: str | None = None, s: Settings = SETTINGS) -> dict:
    block, flat = _facts_block(facts)
    text = ollama.chat([{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Facts about {facts['symbol']}:\n{block}\n\nWrite the research summary."}], model, s=s)
    v = verify_report(text, flat)
    return {"text": text, "facts_block": block, "verification": v, "notes": verification_notes(v)}


COMPARE_PROMPT = """You compare two Indian stocks for a research note using ONLY the numbered facts for each (prefixed A and B).
Cite ids like [A12] or [B7]. If a metric is missing for one of them, say "not available". No price targets, forecasts or buy/sell instructions.
Cover: where each is stronger, where each is weaker, serious flags, data gaps. Maximum 300 words."""


def compare(facts_a: dict, facts_b: dict, model: str | None = None, s: Settings = SETTINGS) -> dict:
    ba, fa = _facts_block(facts_a)
    bb, fb = _facts_block(facts_b)
    pre = lambda txt, p: re.sub(r"^F(\d+):", rf"{p}\1:", txt, flags=re.M)
    text = ollama.chat([{"role": "system", "content": COMPARE_PROMPT},
                        {"role": "user", "content": f"A = {facts_a['symbol']}\n{pre(ba, 'A')}\n\nB = {facts_b['symbol']}\n{pre(bb, 'B')}"}], model, s=s)
    # verify numbers against the union of both fact sets; citations use A/B prefixes
    flat = [{**f, "id": "A" + f["id"][1:]} for f in fa] + [{**f, "id": "B" + f["id"][1:]} for f in fb]
    ids = {f["id"] for f in flat}
    cited = set(re.findall(r"\b[AB]\d+\b", text))
    v = verify_report(text, flat)          # numbers are checked against both fact sets
    v["unknown_citations"] = sorted(cited - ids)
    v["has_citations"] = bool(cited)
    return {"text": text, "verification": v, "notes": verification_notes(v)}


def save_report(symbol: str, body: str, as_of: str) -> Path:
    out = PROJECT_ROOT / "data" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol.upper()}_{as_of}.md"
    path.write_text(body)
    return path
