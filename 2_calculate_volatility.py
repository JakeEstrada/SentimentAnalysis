import pandas as pd
import numpy as np
from datetime import datetime

def calc_market_adj_return_and_volatility():

    historical_prices = pd.read_csv('historical_prices.csv')

    historical_prices['date'] = pd.to_datetime(historical_prices['date'])
    start_date = '2009-01-01'
    end_date = '2014-12-31'

    historical_prices_filtered = historical_prices[(historical_prices['date'] >= start_date) & (historical_prices['date'] <= end_date)].copy()

    print(f'Total length of historical prices from 2009-2014: {len(historical_prices_filtered)}')
    length = len(historical_prices_filtered)
    historical_prices_filtered.sort_values(['symbol', 'date'], inplace=True)
    historical_prices_filtered = historical_prices_filtered[historical_prices_filtered['close'] > 0]
    removed = length - len(historical_prices_filtered)

    if removed > 0:
        print(f'removed {removed} rows with a close price of 0')
    df = []

    symbols = historical_prices_filtered['symbol'].unique()
    print(f'Calculating return & volatility for {len(symbols)} symbols')

    for s in symbols:
        symbol_data = historical_prices_filtered[historical_prices_filtered['symbol'] == s].copy()

        if len(symbol_data) < 4:
            continue

        #calc daily log return    
        symbol_data['daily_return'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))

        #calc 3 day volatility
        symbol_data['daily_volatility'] = symbol_data['daily_return'].rolling(window=3, min_periods=3).std()

        if s == '^GSPC':
            symbol_data['market_return'] = symbol_data['daily_return']
            symbol_data['alpha'] = 1.0
            symbol_data['beta'] = 0.0
            symbol_data['market_adj_return'] = 0.0
            symbol_data['market_adj_volatility'] = 0.0
            continue

        sp_data = historical_prices_filtered[historical_prices_filtered['symbol'] == '^GSPC'].copy()

        if len(sp_data) == 0:
            print('s&p data not found ')
            continue

        sp_data = sp_data.copy()
        sp_data['market_return'] = np.log(sp_data['close'] / sp_data['close'].shift(1))

        #put stock and S&P data together
        all_symbols = pd.merge(symbol_data, sp_data[['date', 'market_return']], on='date', how='left')

        all_symbols = all_symbols.dropna(subset=['daily_return', 'market_return'])

        if len(all_symbols) < 10:
            print('length of all_symbols is less than 10')
            continue

        #calc alpha and beta
        cov = all_symbols['daily_return'].cov(all_symbols['market_return'])
        var = all_symbols['market_return'].var()

        if var == 0:
            beta = 0.0
        else:
            beta = cov / var

        alpha = all_symbols['daily_return'].mean() - beta * all_symbols['market_return'].mean()

        #calc market adjusted returns
        all_symbols['market_adj_return'] = all_symbols['daily_return'] - (alpha + beta * all_symbols['market_return'])

        #calc market adjusted volatility
        all_symbols['market_adj_volatility'] = all_symbols['market_adj_return'].rolling(window=3, min_periods=3).std()

        all_symbols['beta'] = beta
        all_symbols['alpha'] = alpha

        df.append(all_symbols)

    if df:
        final_df = pd.concat(df, ignore_index=True)

        columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                   'daily_return', 'daily_volatility', 'market_return', 'beta', 'alpha',
                   'market_adj_return', 'market_adj_volatility']

        for c in columns:
            if c not in final_df.columns:
                final_df[c] = np.nan
        
        final_df['daily_volatility'] = final_df['daily_volatility'].fillna(method='bfill')
        final_df['market_adj_volatility'] = final_df['market_adj_volatility'].fillna(method='bfill')
        
        final_df = final_df[columns]

        final_df.to_csv('historical_prices_return_volatility.csv', index=False)
        print('final csv completed')
        print(f'shape of historical_prices_return_volatility.csv: {len(final_df)}')
        print(f'Columns: {list(final_df.columns)}')

    else:
        print('No df created Error')

if __name__ == "__main__":
    calc_market_adj_return_and_volatility()

