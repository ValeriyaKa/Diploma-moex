"""
backend/collector/news_historical_finbert.py
=============================================
Collects historical news and analyzes sentiment with FinBERT.

Strategy (multi-source):
  1. GDELT DOC API  — free, no auth, covers 2015+. Returns article titles
     matching a keyword query. We re-score them with FinBERT instead of
     using raw GDELT tone.
  2. RBC RSS feeds  — current news in XML format (no JS).
  3. Individual RBC article pages — server-rendered, for fetching full text
     of articles whose URLs we already know.

Outputs:
  data/news_finbert_historical.csv  — daily sentiment per ticker
  data/news_sentiment.json          — latest ticker/market sentiment
  data/news_articles_analyzed.csv   — raw articles with sentiment scores

Usage:
    python -m backend.collector.news_historical_finbert
    python -m backend.collector.news_historical_finbert --from 2022-01-01 --to 2026-05-22
    python -m backend.collector.news_historical_finbert --ticker SBER
    python -m backend.collector.news_historical_finbert --skip-fetch
    python -m backend.collector.news_historical_finbert --merge
"""
import os, re, json, time, logging, argparse
from datetime import datetime, timedelta, date
from collections import defaultdict
from xml.etree import ElementTree

import requests
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

NO_PROXY = {"http": None, "https": None}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# ================================================================
# TICKER CONFIG
# ================================================================

# Search queries for GDELT DOC API and ticker matching
TICKER_QUERIES = {
    "GAZP": ["газпром", "gazprom"],
    "LKOH": ["лукойл", "lukoil"],
    "NVTK": ["новатэк", "novatek"],
    "ROSN": ["роснефть", "rosneft"],
    "TATN": ["татнефть", "tatneft"],
    "SNGS": ["сургутнефтегаз"],
    "SIBN": ["газпром нефть"],
    "BANEP": ["башнефть"],
    "SBER": ["сбербанк", "sberbank"],
    "VTBR": ["втб банк", "vtb"],
    "T": ["тинькофф", "tinkoff", "т-банк"],
    "MOEX": ["московская биржа", "мосбиржа"],
    "GMKN": ["норникель", "norilsk nickel"],
    "PLZL": ["полюс золото", "polyus"],
    "ALRS": ["алроса", "alrosa"],
    "CHMF": ["северсталь", "severstal"],
    "NLMK": ["нлмк", "nlmk"],
    "MAGN": ["ммк", "mmk"],
    "YDEX": ["яндекс", "yandex"],
    "OZON": ["озон", "ozon"],
    "VKCO": ["вконтакте", "vk company"],
    "POSI": ["позитив", "positive tech"],
    "MGNT": ["магнит", "magnit"],
    "X5":   ["x5 retail", "пятёрочка"],
    "LENT": ["лента ритейл"],
    "FIXP": ["фикс прайс", "fix price"],
    "MTSS": ["мтс", "mts"],
    "RTKM": ["ростелеком", "rostelecom"],
    "PIKK": ["пик группа", "pik"],
    "SMLT": ["самолёт", "самолет"],
    "AFLT": ["аэрофлот", "aeroflot"],
    "FESH": ["fesco", "феско"],
    "UWGN": ["объединённая вагонная"],
    "SFIN": ["сфи группа"],
    "CBOM": ["мкб банк"],
}

# Corporate-event keywords → 2× weight
IMPACT_KEYWORDS_RU = [
    "дивиденд", "отчёт", "отчет", "прибыль", "выручка", "убыток",
    "IPO", "SPO", "buyback", "обратный выкуп", "допэмиссия",
    "рекомендация", "прогноз", "целевая цена", "рейтинг",
    "годовой отчёт", "годовой отчет", "квартальный", "МСФО", "РСБУ",
    "капитализация", "акции", "котировки",
    "контракт", "сделка", "партнёрство", "слияние", "поглощение",
    "санкции", "ограничения", "запрет",
    "реструктуризация", "банкротство", "дефолт",
    "экспорт", "импорт", "нефть", "газ",
]


# ================================================================
# SOURCE 1: GDELT DOC API
# ================================================================

def _gdelt_query(query, start_date, end_date, max_records=250):
    """
    Query GDELT DOC 2.0 API for article metadata.
    Returns list of {url, title, date, tone, domain, language}.
    Free, no auth, covers 2015-present.
    Date format: YYYYMMDDHHMMSS
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    sd = start_date.strftime("%Y%m%d%H%M%S")
    ed = end_date.strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "startdatetime": sd,
        "enddatetime": ed,
        "format": "json",
        "sort": "datedesc",
    }

    try:
        r = requests.get(base, params=params, headers=HEADERS,
                         timeout=30, proxies=NO_PROXY)
        if r.status_code != 200:
            log.debug(f"GDELT {r.status_code} for query={query}")
            return []

        data = r.json()
        articles = data.get("articles", [])
        results = []
        for a in articles:
            results.append({
                "url": a.get("url", ""),
                "title": a.get("title", ""),
                "date": a.get("seendate", "")[:10].replace("T", ""),
                "tone": a.get("tone", 0),
                "domain": a.get("domain", ""),
                "language": a.get("language", ""),
                "source": "gdelt",
            })
        return results
    except Exception as e:
        log.debug(f"GDELT error: {e}")
        return []


def _collect_gdelt_articles(date_from, date_to, tickers):
    """
    Collect articles from GDELT for all tickers, chunked by quarter.
    GDELT has a limit per request, so we split time range into quarters.
    """
    dt_from = datetime.strptime(date_from, "%Y-%m-%d")
    dt_to = datetime.strptime(date_to, "%Y-%m-%d")

    ticker_articles = defaultdict(list)
    seen_urls = set()

    # Split into ~90-day chunks
    chunk_start = dt_from
    chunk_num = 0
    while chunk_start < dt_to:
        chunk_end = min(chunk_start + timedelta(days=89), dt_to)
        chunk_num += 1

        for i, (ticker, queries) in enumerate(TICKER_QUERIES.items()):
            if ticker not in tickers:
                continue

            for query in queries[:1]:  # first query per ticker
                articles = _gdelt_query(
                    query, chunk_start, chunk_end, max_records=250
                )
                new = 0
                for art in articles:
                    if art["url"] not in seen_urls:
                        seen_urls.add(art["url"])
                        ticker_articles[ticker].append(art)
                        new += 1

                time.sleep(0.5)  # rate limit

            if (i + 1) % 10 == 0 and chunk_num == 1:
                print(f"    Chunk 1: {i+1}/{len(tickers)} tickers...")

        period = f"{chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        total = sum(len(v) for v in ticker_articles.values())
        print(f"  Chunk {chunk_num}: {period} — total articles: {total}")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(1)  # rate limit between chunks

    return ticker_articles


# ================================================================
# SOURCE 2: RBC RSS FEEDS (current/recent news)
# ================================================================

RBC_RSS_URLS = [
    "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "https://rssexport.rbc.ru/rbcnews/news/20/full.rss",
]


def _collect_rbc_rss():
    """Parse RBC RSS feeds for recent articles."""
    articles = []
    for rss_url in RBC_RSS_URLS:
        try:
            r = requests.get(rss_url, headers=HEADERS, timeout=15, proxies=NO_PROXY)
            if r.status_code != 200:
                continue
            root = ElementTree.fromstring(r.content)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                description = item.findtext("description", "")

                # Parse date
                article_date = None
                if pub_date:
                    for fmt in [
                        "%a, %d %b %Y %H:%M:%S %z",
                        "%a, %d %b %Y %H:%M:%S %Z",
                        "%Y-%m-%dT%H:%M:%S",
                    ]:
                        try:
                            article_date = datetime.strptime(
                                pub_date.strip(), fmt
                            ).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue

                if title and article_date:
                    articles.append({
                        "title": title,
                        "text": description[:500] if description else "",
                        "url": link,
                        "date": article_date,
                        "source": "rbc_rss",
                    })
            time.sleep(0.3)
        except Exception as e:
            log.debug(f"RBC RSS error: {e}")

    return articles


# ================================================================
# SOURCE 3: RBC ARTICLE PAGE PARSER (for enriching with full text)
# ================================================================

def _fetch_rbc_article_text(url, session=None):
    """
    Fetch full text from a single RBC article page.
    Article pages are server-rendered (unlike /search/ and /tags/).
    """
    from bs4 import BeautifulSoup

    sess = session or requests.Session()
    try:
        r = sess.get(url, headers=HEADERS, timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")

        # Article text — multiple selector patterns
        for cls in [
            re.compile(r"article__text"),
            re.compile(r"article__body"),
            re.compile(r"ArticleText"),
        ]:
            div = soup.find("div", class_=cls)
            if div:
                paragraphs = div.find_all("p")
                return " ".join(p.get_text(strip=True) for p in paragraphs)[:2000]

        return ""
    except Exception:
        return ""


# ================================================================
# SOURCE 4: INTERFAX RSS
# ================================================================

INTERFAX_RSS_URLS = [
    "https://www.interfax.ru/rss.asp",          # main feed
    "https://www.interfax.ru/business/rss.asp",  # business
]


def _collect_interfax_rss():
    """Parse Interfax RSS feeds for recent financial articles."""
    articles = []
    for rss_url in INTERFAX_RSS_URLS:
        try:
            r = requests.get(rss_url, headers=HEADERS, timeout=15, proxies=NO_PROXY)
            if r.status_code != 200:
                continue
            root = ElementTree.fromstring(r.content)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                description = item.findtext("description", "")

                article_date = None
                if pub_date:
                    for fmt in [
                        "%a, %d %b %Y %H:%M:%S %z",
                        "%a, %d %b %Y %H:%M:%S %Z",
                        "%a, %d %b %Y %H:%M:%S",
                    ]:
                        try:
                            article_date = datetime.strptime(
                                pub_date.strip(), fmt
                            ).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue

                if title and article_date:
                    articles.append({
                        "title": title,
                        "text": description[:500] if description else "",
                        "url": link,
                        "date": article_date,
                        "source": "interfax_rss",
                    })
            time.sleep(0.3)
        except Exception as e:
            log.debug(f"Interfax RSS error: {e}")

    return articles


# ================================================================
# SOURCE 5: LENTA.RU ARCHIVE PAGES
# ================================================================

def _collect_lenta_archive(date_from, date_to, tickers):
    """
    Crawl Lenta.ru daily archive pages for financial news.
    Lenta archive: https://lenta.ru/YYYY/MM/DD/
    Pages are server-rendered HTML.
    """
    from bs4 import BeautifulSoup

    dt_from = datetime.strptime(date_from, "%Y-%m-%d").date()
    dt_to = datetime.strptime(date_to, "%Y-%m-%d").date()

    session = requests.Session()
    all_articles = []
    current = dt_from
    days_total = (dt_to - dt_from).days + 1
    day_count = 0

    # Build lowercase query set for matching
    query_map = {}
    for ticker, queries in TICKER_QUERIES.items():
        if ticker not in tickers:
            continue
        for q in queries:
            query_map[q.lower()] = ticker

    while current <= dt_to:
        day_count += 1
        # Only every 7th day (weekly sampling) to be reasonable
        if current.weekday() == 0 or day_count == 1:  # Mondays + first day
            url = f"https://lenta.ru/{current.strftime('%Y/%m/%d')}/"
            try:
                r = session.get(url, headers=HEADERS, timeout=15, proxies=NO_PROXY)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    # Find article links and titles
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        title = a.get_text(strip=True)
                        if not title or len(title) < 15:
                            continue
                        # Only finance-related articles
                        if not href.startswith("/news/") and not href.startswith("/articles/"):
                            continue
                        full_url = f"https://lenta.ru{href}" if href.startswith("/") else href

                        # Check if matches any ticker
                        title_lower = title.lower()
                        for q, ticker in query_map.items():
                            if q in title_lower:
                                all_articles.append({
                                    "title": title,
                                    "text": "",
                                    "url": full_url,
                                    "date": current.isoformat(),
                                    "source": "lenta",
                                    "ticker": ticker,
                                })
                                break
                time.sleep(0.2)
            except Exception:
                pass

        current += timedelta(days=1)
        if day_count % 90 == 0:
            print(f"    Lenta: day {day_count}/{days_total}, articles: {len(all_articles)}")

    return all_articles


# ================================================================
# FINBERT SENTIMENT
# ================================================================

_finbert = None


def _get_finbert():
    global _finbert
    if _finbert is not None:
        return _finbert
    try:
        from transformers import pipeline
        _finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        print("FinBERT loaded")
        return _finbert
    except Exception as e:
        print(f"FinBERT unavailable: {e}")
        return None


# Translation map for Russian financial terms → English (for FinBERT)
RU_EN_FINANCIAL = {
    "рост": "growth", "падение": "decline", "прибыль": "profit",
    "убыток": "loss", "дивиденд": "dividend", "выручка": "revenue",
    "выросл": "increased", "снизил": "decreased", "рекорд": "record",
    "обвал": "crash", "ралли": "rally", "санкци": "sanctions",
    "инфляци": "inflation", "ставк": "rate", "нефт": "oil",
    "газ ": "gas ", "банк": "bank", "акци": "shares",
    "биржа": "exchange", "индекс": "index", "курс": "exchange rate",
    "рубл": "ruble", "доллар": "dollar", "экспорт": "export",
    "импорт": "import", "сделк": "deal", "отчёт": "report",
    "отчет": "report", "кризис": "crisis", "дефолт": "default",
    "банкрот": "bankruptcy", "повыш": "increase", "пониж": "decrease",
    "рекомендац": "recommendation", "прогноз": "forecast",
    "капитализац": "capitalization", "доходност": "yield",
    "квартальн": "quarterly", "годов": "annual",
    "чистая прибыль": "net profit",
    "EBITDA": "EBITDA", "МСФО": "IFRS", "РСБУ": "RAS",
    "обратный выкуп": "buyback", "допэмиссия": "share offering",
    "целевая цена": "target price", "рейтинг": "rating",
    "слияние": "merger", "поглощение": "acquisition",
}


def _translate_for_finbert(text):
    """Translate key Russian financial terms to English for FinBERT."""
    result = text
    for ru, en in RU_EN_FINANCIAL.items():
        result = result.replace(ru, en)
    return result


def analyze_with_finbert(text):
    """Analyze text sentiment with FinBERT. Returns -1.0 to +1.0."""
    finbert = _get_finbert()
    if finbert is None:
        return 0.0

    try:
        translated = _translate_for_finbert(text)
        result = finbert(translated[:512])[0]
        label = result["label"]
        score = result["score"]
        if label == "positive":
            return round(score, 3)
        elif label == "negative":
            return round(-score, 3)
        return 0.0
    except Exception:
        return 0.0


# ================================================================
# PARSE DATE HELPERS
# ================================================================

def _parse_article_date(date_str):
    """Parse date from various formats to YYYY-MM-DD."""
    if not date_str:
        return None

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # Try common formats
    for fmt in ["%d.%m.%Y", "%Y-%m-%d", "%d %B %Y", "%d.%m.%Y %H:%M",
                "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(date_str.strip()[:19], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Russian month names
    months_ru = {
        "янв": "01", "фев": "02", "мар": "03", "апр": "04",
        "мая": "05", "май": "05", "июн": "06", "июл": "07",
        "авг": "08", "сен": "09", "окт": "10", "ноя": "11", "дек": "12",
    }
    for ru, num in months_ru.items():
        if ru in date_str.lower():
            parts = re.findall(r"\d+", date_str)
            if len(parts) >= 2:
                day = parts[0].zfill(2)
                year = parts[-1] if len(parts[-1]) == 4 else f"20{parts[-1]}"
                return f"{year}-{num}-{day}"

    # GDELT seendate format: YYYYMMDD
    if re.match(r"^\d{8}$", date_str):
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    return None


# ================================================================
# MAIN COLLECTION
# ================================================================

def collect_historical_news(date_from="2022-01-01", date_to=None,
                            tickers=None,
                            cache_file="data/news_articles_cache.json"):
    """
    Collect historical news articles, analyze with FinBERT,
    aggregate daily per ticker.
    """
    if date_to is None:
        date_to = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    if tickers is None:
        tickers = list(TICKER_QUERIES.keys())

    # Load cache
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded cache: {sum(len(v) for v in cache.values())} articles")

    has_enough = all(
        ticker in cache and len(cache[ticker]) >= 10
        for ticker in tickers
    )

    if not has_enough:
        # ---- Source 1: GDELT DOC API (primary, covers 2015+) ----
        print("\n[Source 1] Collecting from GDELT DOC API...")
        gdelt_articles = _collect_gdelt_articles(date_from, date_to, tickers)
        for ticker, articles in gdelt_articles.items():
            existing = cache.get(ticker, [])
            existing_urls = {a.get("url", "") for a in existing}
            new = [a for a in articles if a["url"] not in existing_urls]
            cache[ticker] = existing + new

        # ---- Source 2: RBC RSS (supplement for recent news) ----
        print("\n[Source 2] Collecting from RBC RSS feeds...")
        rss_articles = _collect_rbc_rss()
        print(f"  RBC RSS articles: {len(rss_articles)}")

        # ---- Source 3: Interfax RSS ----
        print("\n[Source 3] Collecting from Interfax RSS...")
        interfax_articles = _collect_interfax_rss()
        print(f"  Interfax articles: {len(interfax_articles)}")

        # Match RSS articles (RBC + Interfax) to tickers
        all_rss = rss_articles + interfax_articles
        rss_matched = 0
        for art in all_rss:
            combined = (art.get("title", "") + " " + art.get("text", "")).lower()
            for ticker, queries in TICKER_QUERIES.items():
                if ticker not in tickers:
                    continue
                for q in queries:
                    if q.lower() in combined:
                        existing = cache.get(ticker, [])
                        existing_urls = {a.get("url", "") for a in existing}
                        if art.get("url") not in existing_urls:
                            cache.setdefault(ticker, []).append(art)
                            rss_matched += 1
                        break
        print(f"  RSS matched to tickers: {rss_matched}")

        # ---- Source 4: Lenta.ru archive (weekly sampling) ----
        print("\n[Source 4] Collecting from Lenta.ru archive...")
        lenta_articles = _collect_lenta_archive(date_from, date_to, tickers)
        print(f"  Lenta articles matched: {len(lenta_articles)}")
        for art in lenta_articles:
            ticker = art.pop("ticker", None)
            if ticker:
                existing = cache.get(ticker, [])
                existing_urls = {a.get("url", "") for a in existing}
                if art.get("url") not in existing_urls:
                    cache.setdefault(ticker, []).append(art)

        # Save cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    total_cached = sum(len(v) for v in cache.values())
    print(f"\nTotal articles cached: {total_cached}")
    for ticker in sorted(cache.keys()):
        if cache[ticker]:
            print(f"  {ticker}: {len(cache[ticker])} articles")

    # ---- Analyze sentiment with FinBERT ----
    print("\nAnalyzing sentiment with FinBERT...")
    all_records = []

    for ticker, articles in cache.items():
        if ticker not in tickers:
            continue

        for article in articles:
            # Parse date from various fields
            raw_date = (article.get("date")
                        or article.get("date_str")
                        or "")
            article_date = _parse_article_date(raw_date)
            if not article_date:
                continue

            # Check date range
            if article_date < date_from or article_date > date_to:
                continue

            # Analyze with FinBERT
            text = f"{article.get('title', '')}. {article.get('text', '')}"
            sentiment = analyze_with_finbert(text)

            # Corporate events get 2× weight
            is_corporate = any(kw in text.lower() for kw in IMPACT_KEYWORDS_RU)

            all_records.append({
                "date": article_date,
                "ticker": ticker,
                "sentiment": sentiment,
                "is_corporate": is_corporate,
                "title": article.get("title", "")[:200],
            })

    if not all_records:
        print("No articles to analyze!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"Total analyzed: {len(df)} articles, {df['ticker'].nunique()} tickers")
    print(f"Corporate events: {df['is_corporate'].sum()}")

    # ---- Aggregate daily per ticker ----
    df["weighted_sentiment"] = df.apply(
        lambda r: r["sentiment"] * 2.0 if r["is_corporate"] else r["sentiment"],
        axis=1,
    )

    df_agg = df.groupby(["date", "ticker"]).agg(
        tone=("weighted_sentiment", "mean"),
        count=("sentiment", "count"),
        corporate_count=("is_corporate", "sum"),
    ).reset_index()

    # Pivot to wide format
    df_tone = df_agg.pivot_table(
        index="date", columns="ticker", values="tone", aggfunc="mean"
    ).reset_index()
    df_tone.columns = ["date"] + [f"sent_{c}" for c in df_tone.columns[1:]]

    df_count = df_agg.pivot_table(
        index="date", columns="ticker", values="count", aggfunc="sum"
    ).reset_index()
    df_count.columns = ["date"] + [f"news_count_{c}" for c in df_count.columns[1:]]

    df_pivot = df_tone.merge(df_count, on="date", how="outer")

    # Market sentiment
    mkt = df.groupby("date")["weighted_sentiment"].mean().reset_index()
    mkt.columns = ["date", "market_sentiment"]
    mkt_count = df.groupby("date")["sentiment"].count().reset_index()
    mkt_count.columns = ["date", "market_news_count"]
    df_pivot = df_pivot.merge(mkt, on="date", how="outer")
    df_pivot = df_pivot.merge(mkt_count, on="date", how="outer")

    # Rolling features
    df_pivot.sort_values("date", inplace=True)
    sent_cols = [c for c in df_pivot.columns if c.startswith("sent_")]
    for col in sent_cols:
        df_pivot[f"{col}_3d"] = (
            df_pivot[col].rolling(3, min_periods=1).mean().round(3)
        )
        df_pivot[f"{col}_7d"] = (
            df_pivot[col].rolling(7, min_periods=1).mean().round(3)
        )

    if "market_sentiment" in df_pivot.columns:
        df_pivot["market_sentiment_3d"] = (
            df_pivot["market_sentiment"].rolling(3, min_periods=1).mean().round(3)
        )
        df_pivot["market_sentiment_7d"] = (
            df_pivot["market_sentiment"].rolling(7, min_periods=1).mean().round(3)
        )

    # ---- Save ----
    out_path = "data/news_finbert_historical.csv"
    df_pivot.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df_pivot)}")
    print(f"  Columns: {len(df_pivot.columns)}")
    if len(df_pivot) > 0:
        print(f"  Date range: {df_pivot['date'].min()} -> {df_pivot['date'].max()}")
        print(f"  Tickers: {len(sent_cols)}")

    # Save latest sentiment JSON
    ticker_sentiment = {}
    for ticker in TICKER_QUERIES:
        col = f"sent_{ticker}"
        if col in df_pivot.columns:
            last = df_pivot[col].dropna().tail(5)
            if not last.empty:
                ticker_sentiment[ticker] = round(float(last.mean()), 3)

    mkt_vals = df_pivot.get(
        "market_sentiment", pd.Series(dtype=float)
    ).dropna().tail(5)
    market_sent = round(float(mkt_vals.mean()), 3) if not mkt_vals.empty else 0.0

    with open("data/news_sentiment.json", "w") as f:
        json.dump({
            "ticker_sentiment": ticker_sentiment,
            "market_sentiment": market_sent,
            "updated": date.today().isoformat(),
            "source": "GDELT+RBC_FinBERT",
        }, f, indent=2)

    print(f"Saved: data/news_sentiment.json ({len(ticker_sentiment)} tickers)")

    # Save raw articles for inspection
    articles_path = "data/news_articles_analyzed.csv"
    df[["date", "ticker", "sentiment", "is_corporate", "title"]].to_csv(
        articles_path, index=False
    )
    print(f"Saved: {articles_path} ({len(df)} articles)")

    return df_pivot


# ================================================================
# MERGE WITH GDELT HISTORICAL DATA
# ================================================================

def merge_with_gdelt(finbert_path="data/news_finbert_historical.csv",
                     gdelt_path="data/news_sentiment_historical.csv",
                     output_path="data/news_sentiment_historical_merged.csv"):
    """
    Merge FinBERT data with existing GDELT data.
    FinBERT takes priority where both exist, GDELT fills gaps.
    """
    dfs = []

    if os.path.exists(finbert_path):
        df_fb = pd.read_csv(finbert_path)
        df_fb["date"] = pd.to_datetime(df_fb["date"]).dt.strftime("%Y-%m-%d")
        dfs.append(("finbert", df_fb))
        print(f"FinBERT: {len(df_fb)} rows")

    if os.path.exists(gdelt_path):
        df_gd = pd.read_csv(gdelt_path)
        df_gd["date"] = pd.to_datetime(df_gd["date"]).dt.strftime("%Y-%m-%d")
        dfs.append(("gdelt", df_gd))
        print(f"GDELT: {len(df_gd)} rows")

    if not dfs:
        print("No data to merge!")
        return

    if len(dfs) == 1:
        df_merged = dfs[0][1]
    else:
        # FinBERT first, then fill with GDELT
        df_fb = dfs[0][1].set_index("date")
        df_gd = dfs[1][1].set_index("date")

        all_cols = set(df_fb.columns) | set(df_gd.columns)

        df_merged = pd.DataFrame(
            index=sorted(set(df_fb.index) | set(df_gd.index))
        )
        for col in sorted(all_cols):
            if col in df_fb.columns and col in df_gd.columns:
                df_merged[col] = df_fb[col].combine_first(df_gd[col])
            elif col in df_fb.columns:
                df_merged[col] = df_fb[col]
            else:
                df_merged[col] = df_gd[col]

        df_merged = df_merged.reset_index().rename(columns={"index": "date"})

    df_merged.to_csv(output_path, index=False)
    print(f"\nMerged: {output_path}")
    print(f"  Rows: {len(df_merged)}")
    print(f"  Columns: {len(df_merged.columns)}")

    return df_merged


# ================================================================
# ENRICH GDELT URLS WITH FINBERT
# ================================================================

def enrich_gdelt_with_finbert(urls_csv="data/gdelt_articles_urls.csv",
                              max_articles=2000):
    """
    Take URLs collected by news_gdelt_raw, fetch article text,
    and re-score with FinBERT. Updates the cache used by
    collect_historical_news.
    """
    from bs4 import BeautifulSoup

    if not os.path.exists(urls_csv):
        print(f"No {urls_csv} found. Run news_gdelt_raw first.")
        return

    df = pd.read_csv(urls_csv)
    print(f"Loaded {len(df)} GDELT article records")

    # Filter to only ticker-specific records (not _MARKET)
    df = df[df["ticker"] != "_MARKET"].copy()

    # Deduplicate by URL
    df = df.drop_duplicates(subset=["url"])
    print(f"Unique article URLs: {len(df)}")

    # Limit to manageable number
    if len(df) > max_articles:
        # Sample proportionally by ticker
        df = df.groupby("ticker").apply(
            lambda x: x.sample(min(len(x), max_articles // df["ticker"].nunique()),
                               random_state=42)
        ).reset_index(drop=True)
        print(f"Sampled to {len(df)} articles")

    # Fetch article texts and analyze with FinBERT
    session = requests.Session()
    cache_file = "data/news_articles_cache.json"
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    success = 0
    errors = 0
    for idx, row in df.iterrows():
        url = row["url"]
        ticker = row["ticker"]
        article_date = row.get("date", "")

        # Try to fetch article text
        try:
            r = session.get(url, headers=HEADERS, timeout=10, proxies=NO_PROXY)
            if r.status_code != 200:
                errors += 1
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # Extract title
            title = ""
            title_el = soup.find("h1") or soup.find("title")
            if title_el:
                title = title_el.get_text(strip=True)[:300]

            # Extract article text
            text = ""
            for sel in [
                ("div", {"class": re.compile(r"article__text|article__body|text-content|article-text")}),
                ("article", {}),
                ("div", {"class": re.compile(r"content|body")}),
            ]:
                div = soup.find(sel[0], sel[1])
                if div:
                    paragraphs = div.find_all("p")
                    if paragraphs:
                        text = " ".join(p.get_text(strip=True) for p in paragraphs)[:2000]
                        break

            if not title and not text:
                errors += 1
                continue

            article = {
                "title": title,
                "text": text,
                "url": url,
                "date": article_date,
                "source": "gdelt_enriched",
            }

            existing = cache.get(ticker, [])
            existing_urls = {a.get("url", "") for a in existing}
            if url not in existing_urls:
                cache.setdefault(ticker, []).append(article)
                success += 1

            if (success + errors) % 50 == 0:
                print(f"  Progress: {success} fetched, {errors} errors, "
                      f"{idx+1}/{len(df)} total")
                # Save cache periodically
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            time.sleep(0.2)  # rate limit

        except Exception as e:
            errors += 1
            continue

    # Final save
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\nEnrichment complete: {success} articles fetched, {errors} errors")
    print(f"Cache now has {sum(len(v) for v in cache.values())} total articles")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default="2022-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--ticker", default=None,
                        help="Single ticker to process (default: all)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching, only re-analyze cached articles")
    parser.add_argument("--merge", action="store_true",
                        help="Merge FinBERT results with GDELT data")
    parser.add_argument("--from-gdelt", action="store_true",
                        help="Enrich GDELT URLs with article text + FinBERT")
    parser.add_argument("--max-articles", type=int, default=2000,
                        help="Max articles to fetch when using --from-gdelt")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else None

    if args.from_gdelt:
        print("=" * 60)
        print("Enriching GDELT articles with FinBERT")
        print("=" * 60)
        enrich_gdelt_with_finbert(max_articles=args.max_articles)
        print("\nNow run without --from-gdelt to analyze with FinBERT:")
        print("  python -m backend.collector.news_historical_finbert --merge")
    else:
        print("=" * 60)
        print("Historical News Collector + FinBERT Sentiment")
        print(f"Period: {args.date_from} to {args.date_to}")
        print(f"Sources: GDELT DOC API + RBC RSS + Interfax + Lenta.ru + FinBERT")
        print("=" * 60)

        df = collect_historical_news(
            args.date_from, args.date_to,
            tickers=tickers,
        )

        if args.merge:
            print("\n" + "=" * 60)
            print("Merging with GDELT data...")
            merge_with_gdelt()
