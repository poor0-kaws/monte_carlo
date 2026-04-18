# Monte Carlo Option Pricer

This project prices an Apple call option and put option with a Monte Carlo simulation.

In simple words, the program:
- gets Apple stock and options data
- simulates 1000 possible future stock paths
- calculates the call and put payoff for each path
- averages those payoffs
- discounts them back to today's value
- compares the model prices to the live market prices

## AI Disclosure

This project was made with AI assistance.

## What The Project Does

The script currently:
- uses `AAPL` as the ticker
- downloads live market data with `yfinance`
- picks the nearest option expiration date
- picks the strike price closest to the current stock price
- estimates volatility from recent price history
- gets a short-term risk-free rate
- simulates 1000 stock-price paths with geometric Brownian motion
- prices a European call option
- prices a European put option
- compares model prices to market prices
- saves a graph of the simulated stock paths

## How It Works

The flow is:

1. Get today's stock price.
2. Get the nearest expiration date.
3. Find the at-the-money strike.
4. Estimate volatility.
5. Simulate many future stock-price paths.
6. Compute call and put payoffs at expiration.
7. Average the payoffs.
8. Discount the average payoffs back to today.

## Monte Carlo In Plain English

Monte Carlo means:

1. Try something many times.
2. Add randomness each time.
3. Average the results.

Instead of guessing one future stock price, this project simulates 1000 possible futures.

## Core Formula

The stock simulation uses geometric Brownian motion.

The basic idea is:

`next price = current price × growth part × random shock part`

In code form, that becomes:

`new_price = previous_price * exp(drift + diffusion * shock)`

What the parts mean:
- `previous_price`: the stock price from the day before
- `drift`: the steady growth part
- `diffusion`: how strongly randomness affects the stock
- `shock`: the random `Z` value from a normal distribution

## Option Payoff Formulas

At expiration:

- Call payoff: `max(stock price - strike price, 0)`
- Put payoff: `max(strike price - stock price, 0)`

Then the program:
- calculates the payoff for every simulated path
- averages the payoffs
- discounts that average back to today

That discounted average is the model price.

## Why Discounting Matters

Money later is worth less than money now.

So after the program finds the average future payoff, it brings that value back to today's dollars using the risk-free rate.

## European Options

This project prices European options.

That means the option is only exercised at expiration, not early.

## Project Files

- `monte_carlo_pricer.py`
  The main script.
- `tests/test_monte_carlo_pricer.py`
  The tests.
- `requirements.txt`
  The dependency list.
- `simulation_paths.png`
  The graph created by the script.

## Requirements

You need:
- Python 3
- internet access for live Yahoo Finance data

Main libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `pytest`

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Project

```bash
.venv/bin/python monte_carlo_pricer.py
```

The script will:
- fetch live Apple data
- simulate 1000 paths
- print the model and market option prices
- save `simulation_paths.png`

## Example Output

```text
Ticker: AAPL
Current stock price: $270.23
Nearest expiration date: 2026-04-20
At-the-money strike price: $270.00
Market call price: $1.79
Market put price: $1.60
Annual volatility: 0.2378
Risk-free rate: 0.0360
Days to expiration: 3

Simulated paths: 1000
Days per path: 3
Average ending stock price: $270.04
Lowest ending stock price: $249.11
Highest ending stock price: $292.70

Option price comparison:
Model call price: $2.86
Market call price: $1.79
Call difference: $1.07

Model put price: $2.82
Market put price: $1.60
Put difference: $1.22
```

Your numbers will change because market data changes.

## Run The Tests

```bash
.venv/bin/pytest -q
```

The tests cover:
- data loading
- strike selection
- volatility math
- risk-free rate logic
- random shock generation
- path simulation
- payoff math
- discounting math
- graph creation

## The Graph

The saved graph shows:
- many thin blue lines for the simulated stock paths
- one thick green line for the average path
- one red dashed line for today's stock price

This makes it easier to see the range of possible futures.

## Assumptions

This version keeps things simple:
- the ticker is hardcoded to `AAPL`
- the simulation count is fixed at `1000`
- the option chosen is the nearest expiration and closest strike
- volatility is estimated from about 1 year of daily log returns
- the option style is European

## Limitations

This is a good learning project, but not a full production trading model.

Some limits:
- no dividends
- no early exercise
- no changing volatility over time
- one default ticker
- simple contract selection rules

## Why Model Price And Market Price Can Differ

It is normal for the model price and the market price to be different.

That can happen because:
- volatility is only an estimate
- the real market includes supply and demand
- the model is simplified
- 1000 simulations is still a limited sample

## Good Next Improvements

Useful next steps:
- let the user choose the ticker
- let the user choose the simulation count
- add a histogram of ending prices
- add a histogram of payoffs
- compare Monte Carlo pricing to Black-Scholes pricing
- save outputs to CSV

## Learning Summary

The core idea is:
- a stock can move in many possible ways
- we simulate many of those ways
- each path gives one possible payoff
- we average the payoffs
- then we discount that value back to today

That is the heart of Monte Carlo option pricing.
