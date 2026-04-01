import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, prices, risk_free_rate=0.02):
        """
        Initialize with historical prices.
        
        Args:
            prices (pd.DataFrame): Adjusted close prices.
            risk_free_rate (float): Annual risk-free rate (default 2%).
        """
        self.prices = prices
        self.tickers = prices.columns
        self.risk_free_rate = risk_free_rate
        
        # Calculate daily returns
        self.returns = prices.pct_change().dropna()
        
        # Annualize returns and covariance
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        
    def portfolio_performance(self, weights):
        """
        Calculate expected return, standard deviation, and Sharpe ratio.
        """
        returns = np.sum(self.mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / std
        return returns, std, sharpe

    def _neg_sharpe_ratio(self, weights):
        return -self.portfolio_performance(weights)[2]

    def _volatility(self, weights):
        return self.portfolio_performance(weights)[1]

    def optimize_maximum_sharpe(self):
        """
        Find weights that maximize the Sharpe ratio.
        """
        num_assets = len(self.tickers)
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(self._neg_sharpe_ratio, initial_guess, 
                          args=args, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        return result.x

    def optimize_minimum_volatility(self):
        """
        Find weights that minimize portfolio volatility.
        """
        num_assets = len(self.tickers)
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(self._volatility, initial_guess, 
                          args=args, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        return result.x

    def get_efficient_frontier(self, target_returns):
        """
        Calculate minimum volatility for various target returns.
        """
        num_assets = len(self.tickers)
        efficient_frontier = []
        
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target}
            )
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets,]
            
            result = minimize(self._volatility, initial_guess, 
                              method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                efficient_frontier.append(result.fun)
            else:
                efficient_frontier.append(None)
                
        return efficient_frontier
