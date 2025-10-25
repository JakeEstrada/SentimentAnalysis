import pandas as pd
import numpy as np

#impact_score = sign(r_t) * f(sigma t)  (f()= impact score function)
# return rt is the market-adjusted return Et
# volatility sigmat is the total stock volatility sigma ra,t (including both
# market-driven and idiosyncratic componenets) instead of daily volatility sigmat

def est_impact_scores():
    df = pd.read_csv('historical_prices_return_volatility.csv')
    
    #return rt/market-adj return
    returns = df['market_adj_return']

    #volatility sigmat
    total_volatility = returns.rolling(window=3, min_periods=3).std()
    total_volatility = total_volatility.fillna(method='bfill')

    #normalized return
    zr = (returns - returns.mean()) / returns.std()

    #normalized volatility
    z_sig = (total_volatility - total_volatility.mean()) / total_volatility.std()

    #impact score initialized with zeros
    impact_score = np.zeros(len(df))

    #impact score function
    IS_function_mask = abs(zr) > 0.5
    impact_score[IS_function_mask] = np.sign(zr[IS_function_mask]) * (1 + (abs(zr[IS_function_mask]) > 1).astype(int) + (z_sig[IS_function_mask] > 1).astype(int))


    df['impact_score'] = impact_score.astype(int)

    df.to_csv('historical_prices_impact.csv', index=False)

    print('Created historical_prices_impact.csv successfully')
    print(f"Impact score distribution:\n{df['impact_score'].value_counts().sort_index()}")
if __name__ == '__main__':
    est_impact_scores()