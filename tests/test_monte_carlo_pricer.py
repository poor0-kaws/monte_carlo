from math import sqrt
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import monte_carlo_pricer as pricer


class FakeTicker:
    def __init__(self, history_frame=None, options=None, option_chain_result=None):
        self._history_frame = history_frame
        self.options = options or []
        self._option_chain_result = option_chain_result

    def history(self, period="2y", auto_adjust=False):
        return self._history_frame

    def option_chain(self, expiration_date):
        return self._option_chain_result


def make_price_history(prices):
    index = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=index)


def make_option_table(strikes, prices):
    return pd.DataFrame({"strike": strikes, "lastPrice": prices})


def test_get_stock_history_returns_history():
    history = make_price_history([100.0, 101.0, 102.0])
    ticker = FakeTicker(history_frame=history)

    result = pricer.get_stock_history(ticker)

    pd.testing.assert_frame_equal(result, history)


def test_get_stock_history_raises_when_history_is_empty():
    empty_history = pd.DataFrame()
    ticker = FakeTicker(history_frame=empty_history)

    with pytest.raises(ValueError, match="Stock history is empty"):
        pricer.get_stock_history(ticker)


def test_get_current_stock_price_returns_latest_close():
    history = make_price_history([100.0, 101.5, 103.25])

    result = pricer.get_current_stock_price(history)

    assert result == 103.25


def test_get_current_stock_price_raises_when_no_close_values_exist():
    history = pd.DataFrame({"Close": [np.nan, np.nan]})

    with pytest.raises(ValueError, match="closing stock price"):
        pricer.get_current_stock_price(history)


def test_get_nearest_expiration_date_returns_first_date():
    ticker = FakeTicker(options=["2026-04-20", "2026-04-27"])

    result = pricer.get_nearest_expiration_date(ticker)

    assert result == "2026-04-20"


def test_get_nearest_expiration_date_raises_when_no_dates_exist():
    ticker = FakeTicker(options=[])

    with pytest.raises(ValueError, match="expiration dates"):
        pricer.get_nearest_expiration_date(ticker)


def test_get_at_the_money_option_row_returns_nearest_strike():
    option_table = make_option_table([260.0, 270.0, 280.0], [10.0, 5.0, 2.0])

    result = pricer.get_at_the_money_option_row(option_table, stock_price=268.0)

    assert float(result["strike"]) == 270.0


def test_get_at_the_money_strike_returns_shared_strike_value():
    option_table = make_option_table([260.0, 270.0, 280.0], [10.0, 5.0, 2.0])

    result = pricer.get_at_the_money_strike(option_table, stock_price=273.0)

    assert result == 270.0


def test_get_option_row_for_strike_returns_exact_match_when_possible():
    option_table = make_option_table([260.0, 270.0, 280.0], [10.0, 5.0, 2.0])

    result = pricer.get_option_row_for_strike(option_table, strike_price=270.0)

    assert float(result["lastPrice"]) == 5.0


def test_get_option_row_for_strike_uses_closest_match_when_exact_is_missing():
    option_table = make_option_table([265.0, 275.0], [8.0, 4.0])

    result = pricer.get_option_row_for_strike(option_table, strike_price=270.0)

    assert float(result["strike"]) == 265.0


def test_calculate_annual_volatility_matches_manual_formula():
    prices = [100.0, 102.0, 101.0, 104.0, 103.0]
    history = make_price_history(prices)

    result = pricer.calculate_annual_volatility(history)

    log_returns = np.log(history["Close"] / history["Close"].shift(1)).dropna()
    expected = float(log_returns.std()) * sqrt(pricer.TRADING_DAYS_PER_YEAR)

    assert result == pytest.approx(expected)


def test_get_risk_free_rate_converts_percent_to_decimal(monkeypatch):
    treasury_history = make_price_history([4.85, 4.90, 5.10])

    def fake_ticker(symbol):
        assert symbol == pricer.TREASURY_BILL_TICKER
        return FakeTicker(history_frame=treasury_history)

    monkeypatch.setattr(pricer.yf, "Ticker", fake_ticker)

    result = pricer.get_risk_free_rate()

    assert result == pytest.approx(0.051)


def test_get_risk_free_rate_uses_fallback_when_download_fails(monkeypatch):
    def fake_ticker(symbol):
        raise RuntimeError("network problem")

    monkeypatch.setattr(pricer.yf, "Ticker", fake_ticker)

    result = pricer.get_risk_free_rate()

    assert result == pricer.DEFAULT_RISK_FREE_RATE


def test_get_days_to_expiration_counts_future_days():
    today = pd.Timestamp("2026-04-17")

    result = pricer.get_days_to_expiration("2026-04-20", today=today)

    assert result == 3


def test_get_days_to_expiration_raises_for_same_day_expiration():
    today = pd.Timestamp("2026-04-20")

    with pytest.raises(ValueError, match="expires today or earlier"):
        pricer.get_days_to_expiration("2026-04-20", today=today)


def test_get_time_step_in_years_returns_one_trading_day():
    result = pricer.get_time_step_in_years()

    assert result == pytest.approx(1 / pricer.TRADING_DAYS_PER_YEAR)


def test_get_years_to_expiration_turns_days_into_year_fraction():
    result = pricer.get_years_to_expiration(63)

    assert result == pytest.approx(63 / pricer.TRADING_DAYS_PER_YEAR)


def test_generate_random_shocks_returns_expected_shape():
    result = pricer.generate_random_shocks(
        simulation_count=4,
        days_to_expiration=3,
        random_seed=123,
    )

    assert result.shape == (4, 3)


def test_generate_random_shocks_is_repeatable_with_same_seed():
    first = pricer.generate_random_shocks(
        simulation_count=3,
        days_to_expiration=2,
        random_seed=7,
    )
    second = pricer.generate_random_shocks(
        simulation_count=3,
        days_to_expiration=2,
        random_seed=7,
    )

    np.testing.assert_allclose(first, second)


def test_generate_random_shocks_raises_for_bad_inputs():
    with pytest.raises(ValueError, match="Simulation count"):
        pricer.generate_random_shocks(simulation_count=0, days_to_expiration=3)

    with pytest.raises(ValueError, match="Days to expiration"):
        pricer.generate_random_shocks(simulation_count=3, days_to_expiration=0)


def test_simulate_price_paths_returns_one_starting_column_plus_future_days():
    market_inputs = pricer.MarketInputs(
        ticker="AAPL",
        stock_price=100.0,
        expiration_date="2026-04-20",
        strike_price=100.0,
        call_market_price=2.0,
        put_market_price=1.5,
        annual_volatility=0.20,
        risk_free_rate=0.05,
        days_to_expiration=3,
    )

    result = pricer.simulate_price_paths(
        market_inputs=market_inputs,
        simulation_count=5,
        random_seed=123,
    )

    assert result.shape == (5, 4)
    assert np.all(result[:, 0] == 100.0)


def test_simulate_price_paths_matches_manual_formula_when_volatility_is_zero():
    market_inputs = pricer.MarketInputs(
        ticker="AAPL",
        stock_price=100.0,
        expiration_date="2026-04-20",
        strike_price=100.0,
        call_market_price=2.0,
        put_market_price=1.5,
        annual_volatility=0.0,
        risk_free_rate=0.05,
        days_to_expiration=3,
    )

    result = pricer.simulate_price_paths(
        market_inputs=market_inputs,
        simulation_count=2,
        random_seed=123,
    )

    dt = 1 / pricer.TRADING_DAYS_PER_YEAR
    one_day_growth = np.exp(0.05 * dt)

    assert result[0, 1] == pytest.approx(100.0 * one_day_growth)
    assert result[0, 2] == pytest.approx(100.0 * one_day_growth**2)
    assert result[0, 3] == pytest.approx(100.0 * one_day_growth**3)


def test_plot_simulated_price_paths_saves_graph_file(tmp_path):
    market_inputs = pricer.MarketInputs(
        ticker="AAPL",
        stock_price=100.0,
        expiration_date="2026-04-20",
        strike_price=100.0,
        call_market_price=2.0,
        put_market_price=1.5,
        annual_volatility=0.20,
        risk_free_rate=0.05,
        days_to_expiration=3,
    )
    price_paths = np.array(
        [
            [100.0, 101.0, 102.0, 103.0],
            [100.0, 99.0, 98.5, 99.5],
        ]
    )
    output_file = tmp_path / "paths.png"

    result = pricer.plot_simulated_price_paths(
        price_paths=price_paths,
        market_inputs=market_inputs,
        output_file=str(output_file),
    )

    assert result == output_file
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_get_ending_prices_returns_last_column():
    price_paths = np.array(
        [
            [100.0, 101.0, 102.0],
            [100.0, 99.0, 98.0],
        ]
    )

    result = pricer.get_ending_prices(price_paths)

    np.testing.assert_allclose(result, np.array([102.0, 98.0]))


def test_calculate_call_payoffs_uses_max_of_stock_minus_strike_and_zero():
    ending_prices = np.array([95.0, 100.0, 110.0])

    result = pricer.calculate_call_payoffs(ending_prices, strike_price=100.0)

    np.testing.assert_allclose(result, np.array([0.0, 0.0, 10.0]))


def test_calculate_put_payoffs_uses_max_of_strike_minus_stock_and_zero():
    ending_prices = np.array([95.0, 100.0, 110.0])

    result = pricer.calculate_put_payoffs(ending_prices, strike_price=100.0)

    np.testing.assert_allclose(result, np.array([5.0, 0.0, 0.0]))


def test_discount_future_value_brings_future_money_back_to_today():
    result = pricer.discount_future_value(
        average_future_value=10.0,
        risk_free_rate=0.05,
        years_to_expiration=1.0,
    )

    assert result == pytest.approx(10.0 * np.exp(-0.05))


def test_calculate_option_price_from_payoffs_uses_discounted_average_payoff():
    payoffs = np.array([0.0, 10.0, 20.0])

    result = pricer.calculate_option_price_from_payoffs(
        payoffs=payoffs,
        risk_free_rate=0.05,
        years_to_expiration=1.0,
    )

    expected = float(payoffs.mean()) * np.exp(-0.05)
    assert result == pytest.approx(expected)


def test_price_european_options_returns_call_and_put_model_prices():
    market_inputs = pricer.MarketInputs(
        ticker="AAPL",
        stock_price=100.0,
        expiration_date="2026-04-20",
        strike_price=100.0,
        call_market_price=7.5,
        put_market_price=4.5,
        annual_volatility=0.20,
        risk_free_rate=0.05,
        days_to_expiration=252,
    )
    price_paths = np.array(
        [
            [100.0, 110.0],
            [100.0, 90.0],
            [100.0, 105.0],
        ]
    )

    result = pricer.price_european_options(price_paths, market_inputs)

    expected_call_payoffs = np.array([10.0, 0.0, 5.0])
    expected_put_payoffs = np.array([0.0, 10.0, 0.0])
    expected_call_price = float(expected_call_payoffs.mean()) * np.exp(-0.05)
    expected_put_price = float(expected_put_payoffs.mean()) * np.exp(-0.05)

    assert result.call_model_price == pytest.approx(expected_call_price)
    assert result.put_model_price == pytest.approx(expected_put_price)
    assert result.call_market_price == 7.5
    assert result.put_market_price == 4.5


def test_load_market_inputs_builds_one_clean_market_inputs_object(monkeypatch):
    stock_history = make_price_history([200.0, 205.0, 210.0])
    calls = make_option_table([205.0, 210.0, 215.0], [7.0, 5.5, 3.0])
    puts = make_option_table([205.0, 210.0, 215.0], [2.5, 4.0, 6.5])
    option_chain = SimpleNamespace(calls=calls, puts=puts)
    fake_ticker = FakeTicker(
        history_frame=stock_history,
        options=["2026-04-20"],
        option_chain_result=option_chain,
    )

    monkeypatch.setattr(pricer.yf, "Ticker", lambda symbol: fake_ticker)
    monkeypatch.setattr(pricer, "calculate_annual_volatility", lambda history: 0.22)
    monkeypatch.setattr(pricer, "get_risk_free_rate", lambda: 0.035)
    monkeypatch.setattr(pricer, "get_days_to_expiration", lambda expiration_date: 3)

    result = pricer.load_market_inputs("AAPL")

    assert result.ticker == "AAPL"
    assert result.stock_price == 210.0
    assert result.expiration_date == "2026-04-20"
    assert result.strike_price == 210.0
    assert result.call_market_price == 5.5
    assert result.put_market_price == 4.0
    assert result.annual_volatility == 0.22
    assert result.risk_free_rate == 0.035
    assert result.days_to_expiration == 3
