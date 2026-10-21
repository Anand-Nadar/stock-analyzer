from stockpilot.risk import position_size


def test_blueprint_example():
    # Rs 3L capital, 0.5% risk, entry 500, stop 485 -> 100 shares, Rs 50,000
    ps = position_size(300_000, 0.5, 500, 485, max_position_pct=20)
    assert ps.quantity == 100
    assert ps.position_value == 50_000
    assert ps.risk_at_stop == 1_500
    assert ps.limited_by == "RISK"


def test_concentration_cap():
    ps = position_size(300_000, 0.5, 500, 499, max_position_pct=10)   # risk allows 1500 shares
    assert ps.quantity == 60 and ps.limited_by == "CONCENTRATION"


def test_cash_cap():
    ps = position_size(300_000, 1.0, 500, 480, max_position_pct=100, available_cash=10_000)
    assert ps.quantity == 20 and ps.limited_by == "CASH"


def test_invalid_stop():
    assert position_size(300_000, 0.5, 500, 510).quantity == 0
    assert position_size(300_000, 0.5, 500, 510).limited_by == "INVALID"


def test_budget_too_small():
    ps = position_size(100_000, 0.1, 5000, 4000)   # budget Rs100, risk/share Rs1000
    assert ps.quantity == 0 and ps.note
