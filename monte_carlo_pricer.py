"""
Task 1: load market data for the Monte Carlo option pricer.

This file does not price the option yet.
It only gathers the inputs the pricing step will need later.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252
DEFAULT_TICKER = "AAPL"
VOLATILITY_LOOKBACK_DAYS = 252
TREASURY_BILL_TICKER = "^IRX"
TREASURY_BILL_PERCENT_DIVISOR = 100
DEFAULT_RISK_FREE_RATE = 0.04
SIMULATION_COUNT = 1000
PLOT_OUTPUT_FILE = "simulation_paths.png"
AVERAGE_PATH_COLOR = "#0B6E4F"
START_PRICE_COLOR = "#C1121F"


@dataclass
class MarketInputs:
    """
    Stores all the raw inputs we need before we do the simulation.
    """

    ticker: str
    stock_price: float
    expiration_date: str
    strike_price: float
    call_market_price: float
    put_market_price: float
    annual_volatility: float
    risk_free_rate: float
    days_to_expiration: int


@dataclass
class OptionPricingResults:
    """
    Stores the model prices and the market prices side by side.
    """

    call_model_price: float
    put_model_price: float
    call_market_price: float
    put_market_price: float


def get_stock_history(ticker: yf.Ticker, period: str = "2y") -> pd.DataFrame:
    """
    Download recent stock history.

    We use 2 years so we have enough data to:
    1. get the latest stock price
    2. calculate 1 year of volatility
    """

    history = ticker.history(period=period, auto_adjust=False)

    if history.empty:
        raise ValueError("Stock history is empty.")

    return history


def get_current_stock_price(history: pd.DataFrame) -> float:
    """
    Use the latest closing price as today's stock price.
    """

    latest_close = history["Close"].dropna()

    if latest_close.empty:
        raise ValueError("Could not find a closing stock price.")

    return float(latest_close.iloc[-1])


def get_nearest_expiration_date(ticker: yf.Ticker) -> str:
    """
    Pick the earliest expiration date available.
    """

    expiration_dates = ticker.options

    if not expiration_dates:
        raise ValueError("No option expiration dates were found.")

    return expiration_dates[0]


def get_at_the_money_option_row(
    option_table: pd.DataFrame,
    stock_price: float,
) -> pd.Series:
    """
    Pick the option whose strike is closest to the stock price.

    That is what "at the money" means here.
    """

    if option_table.empty:
        raise ValueError("Option table is empty.")

    clean_table = option_table.dropna(subset=["strike", "lastPrice"]).copy()

    if clean_table.empty:
        raise ValueError("Option table does not contain usable rows.")

    clean_table["distance_from_stock_price"] = (
        clean_table["strike"] - stock_price
    ).abs()

    best_row_index = clean_table["distance_from_stock_price"].idxmin()
    best_row = clean_table.loc[best_row_index]

    return best_row


def get_at_the_money_strike(option_table: pd.DataFrame, stock_price: float) -> float:
    """
    Find the strike that sits closest to the current stock price.

    We use this one shared strike for both the call and the put.
    That keeps the comparison fair and easy to understand.
    """

    best_row = get_at_the_money_option_row(option_table, stock_price)
    return float(best_row["strike"])


def get_option_row_for_strike(
    option_table: pd.DataFrame,
    strike_price: float,
) -> pd.Series:
    """
    Find the row for one strike price.

    If the exact strike is missing, use the closest one instead.
    """

    if option_table.empty:
        raise ValueError("Option table is empty.")

    clean_table = option_table.dropna(subset=["strike", "lastPrice"]).copy()

    if clean_table.empty:
        raise ValueError("Option table does not contain usable rows.")

    exact_match = clean_table[clean_table["strike"] == strike_price]

    if not exact_match.empty:
        return exact_match.iloc[0]

    clean_table["distance_from_strike"] = (clean_table["strike"] - strike_price).abs()
    best_row_index = clean_table["distance_from_strike"].idxmin()
    return clean_table.loc[best_row_index]


def calculate_annual_volatility(history: pd.DataFrame) -> float:
    """
    Estimate volatility with a common method:
    daily log returns, then annualize them.
    """

    close_prices = history["Close"].dropna()
    recent_close_prices = close_prices.tail(VOLATILITY_LOOKBACK_DAYS + 1)

    if len(recent_close_prices) < 2:
        raise ValueError("Not enough price history to calculate volatility.")

    log_returns = np.log(recent_close_prices / recent_close_prices.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Log returns are empty, so volatility cannot be calculated.")

    daily_volatility = float(log_returns.std())
    annual_volatility = daily_volatility * sqrt(TRADING_DAYS_PER_YEAR)

    return annual_volatility


def get_risk_free_rate() -> float:
    """
    Try to pull a short-term Treasury bill yield from Yahoo Finance.

    Yahoo's `^IRX` value is shown as a percent, so 5.10 means 5.10%.
    We turn that into 0.0510 so the math is easier later.

    If the download fails, we use a simple fallback.
    """

    try:
        treasury_history = yf.Ticker(TREASURY_BILL_TICKER).history(period="5d")

        if treasury_history.empty:
            return DEFAULT_RISK_FREE_RATE

        latest_close = treasury_history["Close"].dropna()

        if latest_close.empty:
            return DEFAULT_RISK_FREE_RATE

        latest_rate_percent = float(latest_close.iloc[-1])
        return latest_rate_percent / TREASURY_BILL_PERCENT_DIVISOR
    except Exception:
        return DEFAULT_RISK_FREE_RATE


def get_days_to_expiration(
    expiration_date: str,
    today: pd.Timestamp | None = None,
) -> int:
    """
    Count how many whole days remain until expiration.
    """

    if today is None:
        today = pd.Timestamp.today().normalize()

    days_to_expiration = int((pd.Timestamp(expiration_date) - today).days)

    if days_to_expiration <= 0:
        raise ValueError("The chosen option already expires today or earlier.")

    return days_to_expiration


def get_time_step_in_years() -> float:
    """
    One trading day expressed as part of one year.

    Example:
    If we treat one year as 252 trading days,
    then one day is 1 / 252 of a year.
    """

    return 1 / TRADING_DAYS_PER_YEAR


def get_years_to_expiration(days_to_expiration: int) -> float:
    """
    Turn days into years for discounting math.
    """

    if days_to_expiration <= 0:
        raise ValueError("Days to expiration must be greater than 0.")

    return days_to_expiration / TRADING_DAYS_PER_YEAR


def generate_random_shocks(
    simulation_count: int,
    days_to_expiration: int,
    random_seed: int | None = 42,
) -> np.ndarray:
    """
    Create the random Z values for the simulation.

    Each row is one possible future path.
    Each column is one future day.
    """

    if simulation_count <= 0:
        raise ValueError("Simulation count must be greater than 0.")

    if days_to_expiration <= 0:
        raise ValueError("Days to expiration must be greater than 0.")

    rng = np.random.default_rng(random_seed)
    shocks = rng.standard_normal((simulation_count, days_to_expiration))
    return shocks


def simulate_price_paths(
    market_inputs: MarketInputs,
    simulation_count: int = SIMULATION_COUNT,
    random_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate many possible future stock-price paths.

    Geometric Brownian motion uses this idea:
    next price = current price * growth part * randomness part

    The growth part comes from the risk-free rate.
    The randomness part comes from volatility and random shocks.
    """

    dt = get_time_step_in_years()
    shocks = generate_random_shocks(
        simulation_count=simulation_count,
        days_to_expiration=market_inputs.days_to_expiration,
        random_seed=random_seed,
    )

    paths = np.zeros((simulation_count, market_inputs.days_to_expiration + 1))
    paths[:, 0] = market_inputs.stock_price

    drift = (
        market_inputs.risk_free_rate - 0.5 * (market_inputs.annual_volatility**2)
    ) * dt
    diffusion_scale = market_inputs.annual_volatility * sqrt(dt)

    for day_index in range(1, market_inputs.days_to_expiration + 1):
        previous_prices = paths[:, day_index - 1]
        day_shocks = shocks[:, day_index - 1]

        growth_factor = np.exp(drift + diffusion_scale * day_shocks)
        new_prices = previous_prices * growth_factor
        paths[:, day_index] = new_prices

    return paths


def print_simulation_summary(price_paths: np.ndarray) -> None:
    """
    Print a small summary of the simulated paths.
    """

    ending_prices = price_paths[:, -1]

    print(f"Simulated paths: {price_paths.shape[0]}")
    print(f"Days per path: {price_paths.shape[1] - 1}")
    print(f"Average ending stock price: ${ending_prices.mean():.2f}")
    print(f"Lowest ending stock price: ${ending_prices.min():.2f}")
    print(f"Highest ending stock price: ${ending_prices.max():.2f}")


def get_ending_prices(price_paths: np.ndarray) -> np.ndarray:
    """
    Pull out the last column.

    That last column is the stock price at expiration
    for each simulated path.
    """

    if price_paths.size == 0:
        raise ValueError("Price paths are empty, so there are no ending prices.")

    return price_paths[:, -1]


def calculate_call_payoffs(ending_prices: np.ndarray, strike_price: float) -> np.ndarray:
    """
    Call payoff at expiration:
    max(stock price - strike price, 0)
    """

    return np.maximum(ending_prices - strike_price, 0.0)


def calculate_put_payoffs(ending_prices: np.ndarray, strike_price: float) -> np.ndarray:
    """
    Put payoff at expiration:
    max(strike price - stock price, 0)
    """

    return np.maximum(strike_price - ending_prices, 0.0)


def discount_future_value(
    average_future_value: float,
    risk_free_rate: float,
    years_to_expiration: float,
) -> float:
    """
    Bring a future average payoff back to today's dollars.
    """

    return average_future_value * np.exp(-risk_free_rate * years_to_expiration)


def calculate_option_price_from_payoffs(
    payoffs: np.ndarray,
    risk_free_rate: float,
    years_to_expiration: float,
) -> float:
    """
    Option price = discounted average payoff.
    """

    if payoffs.size == 0:
        raise ValueError("Payoffs are empty, so the option price cannot be calculated.")

    average_payoff = float(payoffs.mean())
    return discount_future_value(average_payoff, risk_free_rate, years_to_expiration)


def price_european_options(
    price_paths: np.ndarray,
    market_inputs: MarketInputs,
) -> OptionPricingResults:
    """
    Price both the call and the put from the simulated ending prices.
    """

    ending_prices = get_ending_prices(price_paths)
    years_to_expiration = get_years_to_expiration(market_inputs.days_to_expiration)

    call_payoffs = calculate_call_payoffs(ending_prices, market_inputs.strike_price)
    put_payoffs = calculate_put_payoffs(ending_prices, market_inputs.strike_price)

    call_model_price = calculate_option_price_from_payoffs(
        payoffs=call_payoffs,
        risk_free_rate=market_inputs.risk_free_rate,
        years_to_expiration=years_to_expiration,
    )
    put_model_price = calculate_option_price_from_payoffs(
        payoffs=put_payoffs,
        risk_free_rate=market_inputs.risk_free_rate,
        years_to_expiration=years_to_expiration,
    )

    return OptionPricingResults(
        call_model_price=call_model_price,
        put_model_price=put_model_price,
        call_market_price=market_inputs.call_market_price,
        put_market_price=market_inputs.put_market_price,
    )


def print_option_price_comparison(pricing_results: OptionPricingResults) -> None:
    """
    Print model prices next to market prices.
    """

    call_difference = pricing_results.call_model_price - pricing_results.call_market_price
    put_difference = pricing_results.put_model_price - pricing_results.put_market_price

    print()
    print("Option price comparison:")
    print(f"Model call price: ${pricing_results.call_model_price:.2f}")
    print(f"Market call price: ${pricing_results.call_market_price:.2f}")
    print(f"Call difference: ${call_difference:.2f}")
    print()
    print(f"Model put price: ${pricing_results.put_model_price:.2f}")
    print(f"Market put price: ${pricing_results.put_market_price:.2f}")
    print(f"Put difference: ${put_difference:.2f}")


def plot_simulated_price_paths(
    price_paths: np.ndarray,
    market_inputs: MarketInputs,
    output_file: str = PLOT_OUTPUT_FILE,
) -> Path:
    """
    Create a graph of the simulated stock-price paths.

    Each thin blue line is one possible future.
    The thick green line is the average path.
    The red dashed line is today's starting stock price.
    """

    if price_paths.size == 0:
        raise ValueError("Price paths are empty, so there is nothing to plot.")

    day_numbers = np.arange(price_paths.shape[1])
    average_path = price_paths.mean(axis=0)

    figure, axis = plt.subplots(figsize=(12, 7))

    for one_path in price_paths:
        axis.plot(day_numbers, one_path, color="#1D4ED8", alpha=0.05, linewidth=1)

    axis.plot(
        day_numbers,
        average_path,
        color=AVERAGE_PATH_COLOR,
        linewidth=3,
        label="Average simulated path",
    )
    axis.axhline(
        y=market_inputs.stock_price,
        color=START_PRICE_COLOR,
        linestyle="--",
        linewidth=2,
        label="Today's stock price",
    )

    axis.set_title(
        f"{market_inputs.ticker} Monte Carlo Simulation Paths",
        fontsize=16,
        pad=16,
    )
    axis.set_xlabel("Days into the future")
    axis.set_ylabel("Simulated stock price ($)")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    output_path = Path(output_file)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return output_path


def load_market_inputs(stock_symbol: str = DEFAULT_TICKER) -> MarketInputs:
    """
    Gather every input we need before simulation starts.
    """

    ticker = yf.Ticker(stock_symbol)
    stock_history = get_stock_history(ticker)
    stock_price = get_current_stock_price(stock_history)

    expiration_date = get_nearest_expiration_date(ticker)
    option_chain = ticker.option_chain(expiration_date)

    strike_price = get_at_the_money_strike(option_chain.calls, stock_price)
    call_row = get_option_row_for_strike(option_chain.calls, strike_price)
    put_row = get_option_row_for_strike(option_chain.puts, strike_price)

    annual_volatility = calculate_annual_volatility(stock_history)
    risk_free_rate = get_risk_free_rate()
    days_to_expiration = get_days_to_expiration(expiration_date)

    return MarketInputs(
        ticker=stock_symbol,
        stock_price=stock_price,
        expiration_date=expiration_date,
        strike_price=strike_price,
        call_market_price=float(call_row["lastPrice"]),
        put_market_price=float(put_row["lastPrice"]),
        annual_volatility=annual_volatility,
        risk_free_rate=risk_free_rate,
        days_to_expiration=days_to_expiration,
    )


def print_market_inputs(market_inputs: MarketInputs) -> None:
    """
    Print the loaded data in a readable way.
    """

    print(f"Ticker: {market_inputs.ticker}")
    print(f"Current stock price: ${market_inputs.stock_price:.2f}")
    print(f"Nearest expiration date: {market_inputs.expiration_date}")
    print(f"At-the-money strike price: ${market_inputs.strike_price:.2f}")
    print(f"Market call price: ${market_inputs.call_market_price:.2f}")
    print(f"Market put price: ${market_inputs.put_market_price:.2f}")
    print(f"Annual volatility: {market_inputs.annual_volatility:.4f}")
    print(f"Risk-free rate: {market_inputs.risk_free_rate:.4f}")
    print(f"Days to expiration: {market_inputs.days_to_expiration}")


def main() -> None:
    market_inputs = load_market_inputs()
    print_market_inputs(market_inputs)
    print()

    price_paths = simulate_price_paths(market_inputs)
    print_simulation_summary(price_paths)
    pricing_results = price_european_options(price_paths, market_inputs)
    print_option_price_comparison(pricing_results)

    output_path = plot_simulated_price_paths(price_paths, market_inputs)
    print(f"Saved simulation graph to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
