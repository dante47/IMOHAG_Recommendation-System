import numpy as np
import pandas as pd

def revenue(price, demand):
    return price * demand

def price_elasticity(prices, demands):
    prices = np.asarray(prices)
    demands = np.asarray(demands)
    if len(prices) < 2 or len(demands) < 2:
        return np.array([])
    pct_price = np.diff(prices) / prices[:-1]
    pct_demand = np.diff(demands) / demands[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        elasticities = pct_demand / pct_price
    return elasticities

def estimate_optimal_price(prices, demands):
    # naive: return price that maximizes revenue in the observed series
    revenues = prices * demands
    idx = np.nanargmax(revenues)
    return prices[idx], revenues[idx]

if __name__ == '__main__':
    print('Pricing tools ready')
