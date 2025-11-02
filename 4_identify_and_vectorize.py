import numpy as np
import pandas as pd
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime, timedelta

def id_and_vectorize():
    sentiment_words = ['loss', 'fell', 'short', 'lower', 'sell',
                       'high', 'growth', 'buy', 'bullish', 'strong']
    
    print('reading data')
    all_news = pd.read_csv('all_news.csv')
    print(f'len of all news: {len(all_news)}')

    print('aggregating news over 3 days')
    aggregated_news = aggregate_news(all_news)

    print('preprocessing aggregated news')
    aggregated_news = preproccess(aggregated_news)

    print('vectorize news')
    vectorize(aggregated_news, sentiment_words)

    print('complete')

def aggregate_news(all_news):
    print('aggregating over 3 days')

    all_news['date'] = pd.to_datetime(all_news['date'])
    all_news = all_news.sort_values(['symbol', 'date'])

    aggregate_news = []
    
    for s in all_news['symbol'].unique():
        s_news = all_news[all_news['symbol'] == s].copy()

        unique_dates = s_news['date'].unique()

        for i, curr_date in enumerate(unique_dates):
            window_dates = []
            window_dates.append(curr_date)

            for prev_days in [1,2]:
                target = curr_date - pd.Timedelta(days=prev_days)
                if target in unique_dates:
                    window_dates.append(target)

            window = s_news[s_news['date'].isin(window_dates)]

            combined_window = ' '.join(
                window['article'].fillna('')
            )
        
            aggregate_news.append({
                'date': curr_date,
                'symbol': s,
                'news': combined_window
            })
        
    df_aggregated = pd.DataFrame(aggregate_news)
    df_aggregated.to_csv('aggregated_news.csv', index=False)
    print(f'Aggregated_news.csv created {len(df_aggregated)} rows')

    return df_aggregated

def preproccess(aggregated_news):

    def clean(text):
        if pd.isna(text) or text == "":
            return ""
        
        
        text = re.sub(r'Loading... Market News and Data brought*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Market News and Data brought*', '', text, flags=re.IGNORECASE)
        if re.search(r'before we continue', text, flags=re.IGNORECASE):
            text = "" 
        if re.search(r'Login Register Premium Services Finanical News Latest Earnings', text, flags=re.IGNORECASE):
            text = ""
        if re.search(r'Bull or bear, Seeking Alpha', text, flags=re.IGNORECASE):
            text = ""

        

        text = re.sub(r'<.*?>', '', text)

        text = re.sub(r'https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()

        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        return ' '.join(tokens)
    
    aggregated_news['ProcessedNews'] = aggregated_news['news'].apply(clean)

    print('text preprocssed')

    return aggregated_news

def vectorize(aggredated_news, sentiment_words):
    impact_scores = pd.read_csv('historical_prices_impact.csv')
    impact_scores['date'] = pd.to_datetime(impact_scores['date'])

    merged_df = pd.merge(aggredated_news,
                         impact_scores[['date', 'symbol', 'impact_score']],
                         on=['date', 'symbol'],
                         how='inner')
    
    print(f'merged_df: {merged_df.shape}')
    print(f"Columns in merged_df: {merged_df.columns.tolist()}")

    ne_merged_df = merged_df[merged_df['ProcessedNews'].str.strip() != ""].copy()
    print(f'with empty articles: {len(merged_df)}')
    print(f'without empty articles: {len(ne_merged_df)}')

    print('Creating DTM vectors')
    dtm_vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_features=30)
    dtm_vectors = dtm_vectorizer.fit_transform(ne_merged_df['ProcessedNews'])

    print('creating TD-IDF vectors')
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=30)
    tfidf_vectors = tfidf_vectorizer.fit_transform(ne_merged_df['ProcessedNews'])

    print('creatings curated vectors')
    def curated(text):
        vector = [text.count(word) for word in sentiment_words]
        return vector
    
    curated_vectors = ne_merged_df['ProcessedNews'].apply(curated)

    dtm_df = pd.DataFrame({'date': ne_merged_df['date'],
                           'symbol': ne_merged_df['symbol'],
                           'news_vector': [vec.tolist() for vec in dtm_vectors.toarray()],
                           'impact_score': ne_merged_df['impact_score']
                           })

    tfidf_df = pd.DataFrame({'date': ne_merged_df['date'],
                           'symbol': ne_merged_df['symbol'],
                           'news_vector': [vec.tolist() for vec in tfidf_vectors.toarray()],
                           'impact_score': ne_merged_df['impact_score']
                           })
    
    curated_df = pd.DataFrame({'date': ne_merged_df['date'],
                           'symbol': ne_merged_df['symbol'],
                           'news_vector': list(curated_vectors),
                           'impact_score': ne_merged_df['impact_score']
                           })


    dtm_df.to_csv('vectorized_news_dtm.csv', index=False)
    tfidf_df.to_csv('vectorized_news_tfidf.csv', index=False)
    curated_df.to_csv('vectorized_news_curated.csv', index=False)

    print('all vectorization methods completed')
    print(f'dtm vectors shape: {dtm_vectors.shape}')
    print(f'tf-idf vectors: {tfidf_vectors.shape}')
    print(f'curated vectors: {len(sentiment_words)} features')

if __name__ == '__main__':
    id_and_vectorize()