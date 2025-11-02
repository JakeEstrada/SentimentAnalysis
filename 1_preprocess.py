import numpy as np
import pandas as pd
from datetime import datetime


def split(df, train_years, test_year):
        train_df = df[df['date'].dt.year.isin(train_years)].copy()
        test_df = df[df['date'].dt.year == test_year].copy()
        return train_df, test_df

def preproccess():
    dtm_df = pd.read_csv('../phase_1/vectorized_news_dtm.csv')
    tfidf_df = pd.read_csv('../phase_1/vectorized_news_tfidf.csv')
    curated_df = pd.read_csv('../phase_1/vectorized_news_curated.csv')

    dtm_df['date'] = pd.to_datetime(dtm_df['date'])
    tfidf_df['date'] = pd.to_datetime(tfidf_df['date'])
    curated_df['date'] = pd.to_datetime(curated_df['date'])

    dtm_df['news_vector'] = dtm_df['news_vector'].apply(eval)
    tfidf_df['news_vector'] = tfidf_df['news_vector'].apply(eval)
    curated_df['news_vector'] = curated_df['news_vector'].apply(eval)
    
    for name, df in [('DTM', dtm_df), ('TF-IDF', tfidf_df), ('Custom', curated_df)]:
        print(f"{name} date range: {df['date'].min()} to {df['date'].max()}")



    train_years = [2015, 2016, 2017, 2018, 2019]
    test_year = 2020

    dtm_train_df, dtm_test_df = split(dtm_df, train_years, test_year)
    tfidf_train_df, tfidf_test_df = split(tfidf_df, train_years, test_year)
    curated_train_df, curated_test_df = split(curated_df, train_years, test_year)

    
    print(f"dtm Train: {len(dtm_train_df)}, Test: {len(dtm_test_df)}")
    print(f"tfidf  Train: {len(tfidf_train_df)}, Test: {len(tfidf_test_df)}")
    print(f"curated  Train: {len(curated_train_df)}, Test: {len(curated_test_df)}")


    dfs = {
         'dtm': (dtm_train_df, dtm_test_df),
         'tfidf': (tfidf_train_df, tfidf_test_df),
         'curated': (curated_train_df, curated_test_df)
    }

    for name, (train, test) in dfs.items():
        train.to_csv(f'{name}_train.csv', index=False)
        test.to_csv(f'{name}_test.csv', index=False)
        print(f"Saved {name}_train.csv ({len(train)} rows) and {name}_test.csv ({len(test)} rows)")
    
    return dfs

preproccess()
