# Portfolio Optimization for Wealth Management

This sub-project implements Modern Portfolio Theory (MPT) components, specifically **Mean-Variance Optimization (MVO)** and **Efficient Frontier** generation.

## Key Features

- **Data Loading**: Integration with `yfinance` to fetch live historical data.
- **Risk & Return Metrics**: Calculations for annual returns, standard deviation (volatility), and Sharpe ratio.
- **Optimization**: Finding the Maximum Sharpe Ratio and Minimum Volatility portfolios.
- **Visualization**: Plotting the Efficient Frontier and asset weight distributions.

## How it Works

MVO uses the expected returns and the covariance of assets to find the set of portfolios that offer the highest expected return for a given level of risk (or the lowest risk for a given level of return). This set of portfolios is known as the **Efficient Frontier**.

### Tickers Used (Example)

Typically, a diversified portfolio across sectors:

- AAPL (Tech)
- MSFT (Tech)
- AMZN (Consumer)
- GOOGL (Comm)
- JPM (Financial)
- JNJ (Healthcare)
- V (Fin Services)
- PG (Consumer Staple)
- XOM (Energy)
- SPY (Market Proxy)

### Key Financial Metrics

#### 📈 Volatility (Standard Deviation)

Volatility measures the degree of variation in a stock's price over time. In this tool, we calculate the **annualized standard deviation** of daily returns.

- **High Volatility**: The stock price fluctuates significantly (e.g., small-cap tech).
- **Low Volatility**: The stock price is relatively stable (e.g., utility companies or treasury bonds).

#### 📊 Sharpe Ratio

The Sharpe Ratio is the gold standard for measuring **risk-adjusted return**. It answers the question: *"Was the extra return worth the extra risk?"*

- **Formula**: `(Portfolio Return - Risk-Free Rate) / Volatility`
- **Interpretation**: A higher Sharpe ratio (e.g., > 1.0) means the portfolio is generating excellent returns for each unit of risk taken.

##### How we calculate each element

1. **Portfolio Return**: We take the **mean daily returns** of each asset over the lookback period, multiply by their respective **weights**, and then **annualize** the result by multiplying by 252 (trading days in a year).
2. **Risk-Free Rate**: In this implementation, we use a constant assumption (e.g., 2%). In a professional setting, this is typically the yield on the 10-year US Treasury bond.
3. **Volatility**: We use the **Covariance Matrix** of the assets and the portfolio weights to calculate the portfolio's variance. The square root of this variance gives us the **standard deviation (volatility)**.

### ⚙️ The Optimization Process: How we get the Weights

We don't just "guess" the weights. We use a mathematical optimizer (**SLSQP**) to find the exact combination of assets that satisfies our goals.

1. **Initial Guess**: The algorithm starts with an "equal weight" portfolio (e.g., 10% for each of 10 stocks).
2. **Objective Function**: We tell the optimizer what to optimize (e.g., "Make the Sharpe Ratio as high as possible").
3. **Constraints**:
    - **Sum to 1**: The total allocation must always be 100%.
    - **Long-Only**: We set bounds of `(0, 1)`, meaning we don't allow "short selling" (negative weights).
4. **Iteration**: The optimizer tries thousands of combinations in milliseconds to find the one that mathematically wins.

## Getting Started

1. Install dependencies: `pip install yfinance pandas numpy scipy matplotlib plotly`
2. Open `portfolio_optimization.ipynb` to see the step-by-step analysis.
