import yfinance as yf
import requests
import pandas as pd
import time
import os
import re
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from playwright.sync_api import sync_playwright

def filter_tickers(tickers):
    filtered = []
    for t in tickers:
        t = str(t).strip().upper()

        if len(t) > 5:
            continue

        if any(c.isdigit() for c in t.replace('^', '')):
            continue

        if not t.replace('^', '').isalpha():
            continue

        filtered.append(t)

    return list(set(filtered))

def batches_download(tickers, batch_size=30):
    all_ticker_data = []

    total_batches = (len(tickers) -1) // batch_size + 1

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f'batch {batch_num}/{total_batches}')

        try:
            batch_data = yf.download(
                batch,
                start='2009-01-01',
                end='2020-12-31',
                group_by='ticker',
                progress=False,
                timeout=30,
            )

            #process tickers
            batch_success = 0
            for t in batch:
                if t in batch_data:
                    t_df = batch_data[t]
            
                    if not t_df.empty:
                        t_df = t_df.reset_index()
                        t_df['symbol'] = t
                        
                        if t_df['Open'].isna().all() or t_df['Close'].isna().all():
                            print(f'No price data for {t} - skipping')
                            continue
                        
                        if 'Close' in t_df.columns:
                            t_df = t_df[['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
                            t_df.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']

                        #format date
                        t_df['date'] = pd.to_datetime(t_df['date']).dt.strftime('%Y-%m-%d')
                        
                        all_ticker_data.append(t_df)
                        batch_success += 1  
                    else:
                        print(f'failed ticker {t}')
                else:
                    print(f'failed ticker {t}')

            
            print(f'Success {batch_success}/{len(batch)}')
      

            if batch_num < total_batches:
                time.sleep(2)
                
        
        except Exception as e:
            print(f'Failed batch: {e}')
            continue    

    return all_ticker_data

def get_stock_prices():
    "Download prices for all tickers in datasets"

    try:
        analyst_ratings = pd.read_csv('analyst_ratings.csv')
        headlines = pd.read_csv('headlines.csv')
    except FileNotFoundError:
        print("Datasets not found")
        return None
    
    all_tickers = set(analyst_ratings['stock'].dropna().unique()) | set (headlines['stock'].dropna().unique())
    all_tickers = [t for t in all_tickers if pd.notna(t)]

    
    print(f"All tickers length {len(all_tickers)}")

    all_tickers = filter_tickers(all_tickers)
    print(f'All tickers after filter {len(all_tickers)}')
    all_tickers.append('^GSPC')

    all_ticker_data = batches_download(all_tickers, batch_size=30)


    if all_ticker_data:
        historical_prices = pd.concat(all_ticker_data, ignore_index=True)
        historical_prices = historical_prices.dropna(subset=['open', 'close', 'volume'])
       #historical_prices['date'] = pd.to_datetime(historical_prices['date']).dt.strftime('%Y-%m-%d')

        historical_prices.to_csv('historical_prices.csv', index=False, float_format='%.2f')
        print(f"Saved historical csv")
        print(f"Total row {len(historical_prices)}")
    else:
        print('historical_prices.csv failed')
    
    return historical_prices

def is_blocked_content(content):
    blocked_indicators = [
        "Pardon Our Interruption",
        "something about your browser made us think you were a bot",
        "access denied", 
        "cloudflare",
        "captcha",
        "bot detection"
    ]
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in blocked_indicators)

# def scrape_article_content_enhanced(url):
def scrape_article(url):
    #try simple request for static
    
    content = try_simple(url)
    if content and is_blocked_content(content):
        print(f'Blocked by {urlparse(url).netloc}')
        return "Blocked by site SR"
    
    if content and "success" in content.lower() and len(content) > 200:
        return content
    
    content = try_playwright(url)
    if content and is_blocked_content(content):
        return "Blocked by site PW"
    
    #if simple request failed or site uses JS use playwrite
    return content or "Content not found"
# def try_simple_request(url):
def try_simple(url):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
    }

    for i in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=3)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                article_div = soup.find("div", class_=lambda x: x and "article-body" in x)
                if not article_div:
                    article_div = soup.find("article")

                if article_div:
                    article_text = article_div.get_text(separator="\n", strip=True)

                    if len(article_text) > 200: 
                        return article_text
                                    
                # simple request didn't work
                break
            else:
                print(f"Get request #{i+1} failed with {response.status_code}")
                time.sleep(2)
        except Exception as e:
            print(f"Simple request error: {e}")
            break
    return None

# def try_playwright_scraping(url):
def try_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
                })
            
            page.goto(url, wait_until='domcontentloaded',timeout=7000)

            content = extract_playwright(page)

            browser.close()

            
            if content and len(content) > 100:
                return content 
            else:
                return "Content not found"
    
    except Exception as e:
        print(f"Playwrite error on {url}: {e}")
        return f"Scrape error: {str(e)}"
    
# def extract_content_playwright(page):
def extract_playwright(page):
    selectors = [
        'div.new-story-content',
        'div.story-content',
        'article .content',
        'div[data-testid="article-content"]',
        'article .article-body',
        'main .content',
        'main .article-body',
        'div.article-content',
        'div.post-content',
        'article div[class*="content"]',
        'main div[class*="content"]',
        'div[class*="article-body"]',
        'div[class*="article-content"]',
        'div[class*="post-content"]',
        'div[class*="story-content"]',
        'div[class*="content"]',
        'div[class*="article"]',
        'div[class*="post"]',
        'article',
        'main',
    ]

    for s in selectors:
        try:
            element = page.query_selector(s)
            if element:
                content = element.inner_text()
                if content and len(content.strip()) > 200:
                    clean = clean_text(content)
                    if len(clean) > 200:
                        return clean
        except:
            continue

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    for s in soup(["script", "style", "nav", "header", "footer", "aside"]):
        s.decompose()

    selectors = [
        {"name": "div", "class_": "article-content"},
        {"name": "div", "class_": "story-content"},
        {"name": "div", "class_": "post-content"},
        {"name": "article", "class_": None},
        {"name": "main", "class_": None}
    ]

    for s in selectors:
        elem = soup.find(s["name"], class_=s["class_"])
        
        if elem:
            text = elem.get_text(separator='\n', strip=True)
            clean = clean_text(text)
            if len(clean) > 200:
                return clean
            
    body = soup.find('body')
    if body:
        noise = [
            'nav', 'header', 'footer', 'aside', '.navbar', '.sidebar', 
            '.advertisement', '.comments', '.related-posts'
        ]

        for n in noise:
            for e in body.select(n):
                e.decompose()

        text = body.get_text(separator='\n', strip=True)
        clean = clean_text(text)
        return clean
    
    return None

# def clean_content(text):
def clean_text(text):
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

# def filter_datasets_by_date():
def filter_dates():
    print('Filtering dates to 2009-2014')

    try:
        analyst_ratings = pd.read_csv('analyst_ratings.csv')
        headlines = pd.read_csv('headlines.csv')
        print(f"Read {len(analyst_ratings)} analyst_ratings and {len(headlines)} headlines")

        analyst_ratings = analyst_ratings.rename(columns={analyst_ratings.columns[0]: 'id', 'stock': 'symbol', 'url': 'URL' })
        headlines = headlines.rename(columns={headlines.columns[0]: 'id', 'stock': 'symbol', 'url': 'URL'})

    except FileNotFoundError:
        print("CSV reading error")
        return
    
    analyst_ratings['date'] = analyst_ratings['date'].str[:10]
    headlines['date'] = headlines['date'].str[:10]

    start = '2009-01-01'
    end = '2014-12-31'

    analyst_ratings = analyst_ratings[
        (analyst_ratings['date'] >= start) & (analyst_ratings['date'] <= end)
    ]
    
    headlines = headlines[
        (headlines['date'] >= start) & (headlines['date'] <= end)
    ]

    
    analyst_ratings.to_csv('analyst_ratings_filtered.csv', index=False)
    headlines.to_csv('headlines_filtered.csv', index=False)

    return analyst_ratings, headlines

def get_even_articles(articles_amount=8000):
    analyst = pd.read_csv('analyst_ratings_filtered.csv')
    headlines = pd.read_csv('headlines_filtered.csv')

    analyst['csv'] = 'analyst'
    headlines['csv'] = 'headlines'

    combined = pd.concat([analyst, headlines], ignore_index=True)
    combined['date'] = combined['date'].str[:10]
    combined['year'] = pd.to_datetime(combined['date']).dt.year

    years = [2009, 2010, 2011, 2012, 2013, 2014]
    csvs = ['analyst', 'headlines']

    articles_for_year = articles_amount // (len(years) * len(csvs))

    articles = []
    for y in years:
        for c in csvs:
            group_articles = combined[(combined['year'] == y) & (combined['csv'] == c)]

            if len(group_articles) >= articles_for_year:
                year_articles = group_articles.sample(n=articles_for_year, random_state=42)
            else:
                year_articles = group_articles

            articles.append(year_articles)
    
    even_articles = pd.concat(articles, ignore_index=True)

    return even_articles


# def download_articles_hybrid():
def download_articles():
    print("Starting Article scraping with simple request and playwrite")


    even_articles = get_even_articles(articles_amount=8000)

    analyst_ratings = even_articles[even_articles['csv'] == 'analyst'].drop(columns=['csv', 'year'])
    headlines = even_articles[even_articles['csv'] == 'headlines'].drop(columns=['csv', 'year'])


    for df in [analyst_ratings, headlines]:
        df['article'] = None
        df['method'] = None

    # def scrape_dataset_hybrid(df, dataset_name):
    def scrape_dataset(df, dataset):
        scraped_count = 0
        simple_count = 0
        playwright_count = 0
        total_count = len(df)

        print(f'Scaping {dataset}, {total_count}-articles')

        for i, row in df.iterrows():
            url = row['URL']
            date = row['date']
            domain = urlparse(url).netloc if 'URL' in row else 'unknown'
            print(f'{i+1}/{total_count}.  {domain}: {url[:50]}.')

            article = scrape_article(url)

            if "simple request successful" in str(article):
                df.at[i, 'method'] = 'simple'
                simple_count += 1
            elif article and len(article) > 100 and "error" not in article.lower():
                df.at[i, 'method'] = 'playwright'
                playwright_count += 1

            df.at[i, 'article'] = article
            scraped_count += 1

            time.sleep(0.2)

            if(i + 1) % 10 == 0:
                print(f'Progess: {i+1}/{total_count}')
                print(f'Scaped Count: {scraped_count}')
                print(f'Simple Scraped: {simple_count}')
                print(f'Playwright Scraped: {playwright_count}')

        return df, scraped_count, simple_count, playwright_count
    
    print("Starting scraping")

    analyst_ratings, analyst_count, analyst_simple_count, analyst_playwright_count = scrape_dataset(analyst_ratings, "analyst_ratings")

    headlines, headlines_count, headlines_simple_count, headlines_playwright_count = scrape_dataset(headlines, "headlines")


    analyst_ratings.to_csv('analyst_ratings_articles.csv', index=False)
    headlines.to_csv('headlines_articles.csv', index=False)

    print(f'Total articles: {analyst_count + headlines_count}')
    print(f'Successful Simple requests: {analyst_simple_count + headlines_simple_count}')
    print(f'Successful playwright requests: {analyst_playwright_count + headlines_playwright_count}')


    return analyst_ratings, headlines

# def create_all_news_final():
def merge_csvs():
    print("creating all_news.csv")

    try:
        analyst_ratings = pd.read_csv('analyst_ratings_articles.csv')
        headlines = pd.read_csv('headlines_articles.csv')

        print('Read datasets')
        print(f'Analyst ratings {len(analyst_ratings)} articles')
        print(f'Headlines {len(headlines)} articles')

        
        analyst_ratings['date'] = analyst_ratings['date'].str[:10]
        headlines['date'] = headlines['date'].str[:10]

        all_news = pd.concat([analyst_ratings, headlines], ignore_index=True)

        print(f'After merge: {len(all_news)} total articles')

        #check for dates
        first_count = len(all_news)
        all_news = all_news[
            (all_news['date'] >= '2009-01-01') & (all_news['date'] <= '2014-12-31')
        ]
        
        final_count = len(all_news)
        print(f'After date filter- {final_count} articles')
        
        needed_columns = ['id', 'headline', 'URL', 'article', 'publisher', 'date', 'symbol']
        available_columns = [c for c in needed_columns if c in all_news.columns]
        all_news = all_news[available_columns]

        all_news = all_news.drop_duplicates(subset=['URL'])

        print(f'After removing duplicates: {len(all_news)} articles')

        all_news = all_news[
            ~all_news['article'].str.contains('error|blocked|not found|too short|never miss a trade again', case=False, na=False)
        ]
        
        print(f'Successful articles scarped {len(all_news)} articles')

        all_news.to_csv('all_news.csv', index=False)

        print(f'Created all_news.csv. {len(all_news)} articles in csv, from 2009-2014')
        
        return all_news
    
    except Exception as e:
        print(f'Error creating all_news.csv: {e}')
        return None

def test():
    print('testing scraping')
    analyst_ratings, headlines = filter_dates()

    test_analyst = analyst_ratings.head(10).copy()
    test_headlines = headlines.head(10).copy()

    for df in [test_analyst, test_headlines]:
        df['article'] = None
    
    print("testing analyst rating")
    for i, row in test_analyst.iterrows():
        url = row['URL']
        article = scrape_article(url)
        test_analyst.at[i, 'article'] = article
        time.sleep(0.9)

    print('testing headlines')
    for i, row in test_headlines.iterrows():
        url = row['URL']
        article = scrape_article(url)
        test_headlines.at[i, 'article'] = article
        time.sleep(0.9)


    needed_columns = ['id', 'headline', 'URL', 'article', 'publisher', 'date', 'symbol']
    for df in [test_analyst, test_headlines]:
        for c in needed_columns:
            if c not in df.columns:
                if c == 'id':
                    df['id'] = range(len(df)) 
    test_analyst.to_csv('test_analyst.csv', index=False)
    test_headlines.to_csv('test_headlines.csv', index=False)

    all_news = merge_csvs()
    if all_news is not None:
        print(f'merge successful created {len(all_news)} articles')
    else:
        print('merge failed')

    print('Test completed')
    return all_news

def run():
    print('Starting full scraping')
    start_time = time.time()

    try:
        print('Scraping articles')
        analyst_ratings, headlines = download_articles()

        print('Creating all_news.csv')
        all_news = merge_csvs()
        
        end_time = time.time()
        total_time_minutes = (end_time-start_time) / 60
        total_hours = total_time_minutes / 60

        print('All_news.csv created')
        print(f'Total time: {total_time_minutes} minutes, {total_hours} hours ')

        if all_news is not None:
            print(f'All_news lengths {len(all_news)} articles')
    except Exception as e:
        print(f'Run() failed : {e}')

if __name__ == "__main__":
    run()

