"""
backend/collector/news_ru.py
=============================
Collects historical news sentiment from Russian financial media.

Working sources (tested May 2026):
  1. RBC AJAX API   — JSON endpoint, returns 20 articles per page with pagination
  2. Kommersant     — HTML search, articles with dates

Sentiment is computed via a Russian financial lexicon (no GPU needed).

Output format is 100% compatible with train_models.py:
  data/news_sentiment_historical.csv  (daily sentiment per ticker)
  data/news_sentiment.json            (latest sentiment for inference)

Usage:
    # Collect historical data (default: last 2 years)
    python -m backend.collector.news_ru --mode historical

    # Collect historical for specific period
    python -m backend.collector.news_ru --mode historical --from 2024-05-01 --to 2026-05-15

    # Daily update (RSS, for cron/scheduler)
    python -m backend.collector.news_ru --mode daily

    # Append daily data to existing historical CSV
    python -m backend.collector.news_ru --mode daily --append

Requirements:
    pip install requests beautifulsoup4 lxml feedparser
"""

import os
import re
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, date
from urllib.parse import quote_plus
from collections import defaultdict

import requests
from bs4 import BeautifulSoup

try:
    import feedparser
except ImportError:
    feedparser = None

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ================================================================
# TICKER -> SEARCH KEYWORDS (Russian + English)
# ================================================================

TICKER_KEYWORDS = {
    "GAZP":  ["Газпром", "Gazprom"],
    "LKOH":  ["Лукойл", "Lukoil"],
    "NVTK":  ["Новатэк", "Novatek"],
    "ROSN":  ["Роснефть", "Rosneft"],
    "TATN":  ["Татнефть", "Tatneft"],
    "SNGS":  ["Сургутнефтегаз"],
    "SIBN":  ["Газпром нефть"],
    "BANEP": ["Башнефть"],
    "SBER":  ["Сбербанк", "Сбер"],
    "VTBR":  ["ВТБ банк", "VTB"],
    "T":     ["Тинькофф", "Т-Банк", "Т-Технологии", "Tinkoff"],
    "MOEX":  ["Мосбиржа", "Московская биржа"],
    "SFIN":  ["SFI", "СФИ"],
    "CBOM":  ["МКБ банк", "Московский кредитный банк"],
    "GMKN":  ["Норникель", "Nornickel"],
    "PLZL":  ["Полюс золото", "Polyus"],
    "ALRS":  ["Алроса", "АЛРОСА"],
    "CHMF":  ["Северсталь", "Severstal"],
    "NLMK":  ["НЛМК"],
    "MAGN":  ["ММК", "Магнитогорский"],
    "YDEX":  ["Яндекс", "Yandex", "YDEX"],
    "OZON":  ["Озон", "Ozon"],
    "VKCO":  ["VK Company", "ВКонтакте"],
    "POSI":  ["Positive Technologies", "Позитив"],
    "MGNT":  ["Магнит"],
    "X5":    ["X5 Group", "Пятёрочка", "X5 Retail"],
    "LENT":  ["Лента ритейл"],
    "FIXP":  ["Fix Price"],
    "MTSS":  ["МТС телеком"],
    "RTKM":  ["Ростелеком"],
    "PIKK":  ["ПИК группа", "PIK"],
    "SMLT":  ["Самолёт девелопер", "Самолет"],
    "AFLT":  ["Аэрофлот"],
    "FESH":  ["ДВМП", "FESCO"],
    "UWGN":  ["ОВК вагон"],
}

# Global/market keywords — captures the overall news backdrop for MOEX
GLOBAL_KEYWORDS = {
    "russia_economy":  ["экономика России", "российская экономика"],
    "sanctions":       ["санкции Россия", "санкции против"],
    "oil_opec":        ["нефть ОПЕК", "цена нефти Brent", "нефть Urals"],
    "cbr_rate":        ["ключевая ставка ЦБ", "Центробанк ставка", "ставка Банка России"],
    "ruble":           ["курс рубля", "доллар рубль", "рубль укрепился", "рубль ослабел"],
    "geopolitics":     ["геополитика Россия", "международные отношения"],
    "inflation":       ["инфляция Россия", "инфляция ускорилась", "рост цен"],
    "imoex":           ["индекс Мосбиржи", "IMOEX", "рынок акций"],
    "moex_exchange":   ["Московская биржа торги", "биржевой оборот", "IPO на Мосбирже"],
    "bonds_ofz":       ["ОФЗ доходность", "гособлигации", "рынок облигаций"],
    "dividends_market":["дивидендный сезон", "дивидендные отсечки"],
    "gdp_budget":      ["ВВП России", "бюджет России", "дефицит бюджета"],
    "export_trade":    ["экспорт нефти", "торговый баланс", "экспорт газа"],
}


# ================================================================
# RUSSIAN FINANCIAL SENTIMENT LEXICON
# ================================================================

POSITIVE_WORDS = {
    "рост", "растёт", "растет", "выросли", "вырос", "выросла", "повышение",
    "увеличение", "прибыль", "доход", "дивиденды", "рекорд", "рекордный",
    "максимум", "ралли", "бычий", "восстановление", "восстановился",
    "позитив", "позитивный", "оптимизм", "улучшение", "подъём", "подъем",
    "укрепление", "укрепился", "спрос", "превысил", "превышает",
    "сделка", "контракт", "партнёрство", "партнерство", "инвестиции",
    "расширение", "модернизация", "buyback", "обратный",
}

NEGATIVE_WORDS = {
    "падение", "упал", "упали", "упала", "снижение", "снизился", "снизилась",
    "обвал", "обвалился", "кризис", "убыток", "убытки", "потери",
    "минимум", "медвежий", "коррекция", "распродажа", "обесценивание",
    "негатив", "негативный", "пессимизм", "ухудшение", "давление",
    "ослабление", "ослабел", "дефицит",
    "санкции", "штраф", "банкротство", "дефолт", "реструктуризация",
    "делистинг", "заморозка", "заблокировал", "ограничение",
    "долг", "задолженность", "риск", "риски",
}

AMPLIFIERS = {"резко", "значительно", "существенно", "сильно", "рекордно"}
NEGATORS = {"не", "ни", "без", "нет"}


def compute_headline_sentiment(text: str) -> float:
    """Compute sentiment score for a Russian headline. Returns [-1, 1]."""
    text_lower = text.lower()
    words = re.findall(r"[а-яёa-z]+", text_lower)

    pos_count = 0
    neg_count = 0

    for i, word in enumerate(words):
        is_negated = (i > 0 and words[i - 1] in NEGATORS)
        amplifier = 1.5 if (i > 0 and words[i - 1] in AMPLIFIERS) else 1.0

        if word in POSITIVE_WORDS:
            if is_negated:
                neg_count += amplifier
            else:
                pos_count += amplifier
        elif word in NEGATIVE_WORDS:
            if is_negated:
                pos_count += amplifier
            else:
                neg_count += amplifier

    # Multi-word phrases
    for phrase in ["выше ожиданий", "сильные результаты", "выше прогноза",
                   "обратный выкуп", "повысил прогноз"]:
        if phrase in text_lower:
            pos_count += 2
    for phrase in ["ниже ожиданий", "слабые результаты", "ниже прогноза",
                   "понизил прогноз", "отмена дивидендов"]:
        if phrase in text_lower:
            neg_count += 2

    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos_count - neg_count) / total))


# ================================================================
# SOURCE 1: RBC AJAX API (JSON — works perfectly!)
# ================================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _safe_get(url: str, retries: int = 3, delay: float = 2.0):
    """HTTP GET with retry."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = delay * (attempt + 1) * 3
                log.warning(f"  Rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                log.debug(f"  HTTP {r.status_code}: {url[:80]}")
                return None
        except Exception as e:
            log.debug(f"  Request error: {e}")
            time.sleep(delay)
    return None


def fetch_rbc(keyword: str, date_from: str, date_to: str, max_pages: int = 5) -> list[dict]:
    """
    Fetch news from RBC AJAX API (JSON).
    date_from/date_to: DD.MM.YYYY format
    Returns list of {"date": "YYYY-MM-DD", "title": "...", "source": "rbc"}
    """
    results = []
    cursor = None

    for page in range(max_pages):
        url = (
            f"https://www.rbc.ru/search/ajax/"
            f"?query={quote_plus(keyword)}"
            f"&dateFrom={date_from}&dateTo={date_to}"
            f"&project=rbcnews"
        )
        if cursor:
            url += f"&cursor={cursor}"

        resp = _safe_get(url)
        if not resp:
            break

        try:
            data = resp.json()
        except Exception:
            break

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            title = item.get("title", "")
            pub_date = item.get("publish_date_t") or item.get("publish_date", "")

            dt_str = None
            if pub_date:
                # Try ISO format first
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(pub_date))
                if m:
                    dt_str = m.group(0)
                else:
                    # Try timestamp
                    try:
                        ts = int(pub_date)
                        dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass

            if not dt_str:
                # Extract from publish_date string like "2025-03-12T00:01:12+03:00"
                raw = item.get("publish_date", "")
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(raw))
                if m:
                    dt_str = m.group(0)

            if title:
                results.append({
                    "date": dt_str,
                    "title": title,
                    "source": "rbc",
                })

        # Pagination
        more = data.get("moreExists", False)
        cursor = data.get("endCursor")
        if not more or not cursor:
            break

        time.sleep(1.0)

    return results


# ================================================================
# SOURCE 2: KOMMERSANT (HTML — works!)
# ================================================================

def fetch_kommersant(keyword: str, date_from: str, date_to: str, max_pages: int = 3) -> list[dict]:
    """
    Fetch news from Kommersant search.
    date_from/date_to: YYYY-MM-DD format
    """
    results = []

    for page in range(1, max_pages + 1):
        url = (
            f"https://www.kommersant.ru/search/results"
            f"?search_query={quote_plus(keyword)}"
            f"&dateFrom={date_from}&dateTo={date_to}"
            f"&sort_type=1&page={page}"
        )

        resp = _safe_get(url)
        if not resp:
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Kommersant uses <article> tags with <h2> links
        articles = soup.select("article")
        if not articles:
            break

        found = 0
        for article in articles:
            link = article.select_one("h2 a")
            if not link:
                continue

            title = link.get_text(strip=True)
            if not title:
                continue

            # Date: look for patterns in article text
            dt_str = None
            article_text = article.get_text()
            # Pattern: "DD.MM.YYYY"
            m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", article_text)
            if m:
                dt_str = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

            results.append({
                "date": dt_str,
                "title": title,
                "source": "kommersant",
            })
            found += 1

        if found == 0:
            break

        time.sleep(1.5)

    return results


# ================================================================
# RSS FEEDS FOR DAILY UPDATES
# ================================================================

RSS_FEEDS = {
    "rbc":        "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "rbc_fin":    "https://rssexport.rbc.ru/rbcnews/news/eco/full.rss",
    "kommersant": "https://www.kommersant.ru/RSS/news.xml",
    "ria_econ":   "https://ria.ru/export/rss2/economy/index.xml",
}


def collect_rss_today() -> list[dict]:
    """Collect today's news from RSS feeds."""
    if feedparser is None:
        log.warning("feedparser not installed, skipping RSS")
        return []

    results = []
    today_str = date.today().isoformat()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:50]:
                title = entry.get("title", "")
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    dt_str = date(pub.tm_year, pub.tm_mon, pub.tm_mday).isoformat()
                else:
                    dt_str = today_str

                results.append({
                    "date": dt_str,
                    "title": title,
                    "source": source,
                })
        except Exception as e:
            log.warning(f"  RSS error ({source}): {e}")

    log.info(f"  RSS collected {len(results)} headlines from {len(RSS_FEEDS)} feeds")
    return results


# ================================================================
# MATCH HEADLINES TO TICKERS & COMPUTE SENTIMENT
# ================================================================

def match_headlines_to_tickers(
    headlines: list[dict],
) -> dict:
    """
    Match headlines to tickers and compute daily sentiment.
    Returns: {
        "ticker_daily": {ticker: {date: {"sentiment": float, "count": int}}},
        "global_daily": {date: {"sentiment": float, "count": int}},
    }
    """
    ticker_daily = defaultdict(lambda: defaultdict(lambda: {"scores": [], "count": 0}))
    global_daily = defaultdict(lambda: {"scores": [], "count": 0})

    for item in headlines:
        dt = item.get("date")
        title = item.get("title", "")
        if not dt or not title:
            continue

        sentiment = compute_headline_sentiment(title)
        title_lower = title.lower()

        # Match tickers
        for ticker, keywords in TICKER_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in title_lower:
                    ticker_daily[ticker][dt]["scores"].append(sentiment)
                    ticker_daily[ticker][dt]["count"] += 1
                    break

        # Match global keywords
        for category, keywords in GLOBAL_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in title_lower:
                    global_daily[dt]["scores"].append(sentiment)
                    global_daily[dt]["count"] += 1
                    break

    # Aggregate to means
    result_ticker = {}
    for ticker, dates in ticker_daily.items():
        result_ticker[ticker] = {}
        for dt, data in dates.items():
            if data["scores"]:
                result_ticker[ticker][dt] = {
                    "sentiment": np.mean(data["scores"]),
                    "count": data["count"],
                }

    result_global = {}
    for dt, data in global_daily.items():
        if data["scores"]:
            result_global[dt] = {
                "sentiment": np.mean(data["scores"]),
                "count": data["count"],
            }

    return {"ticker_daily": result_ticker, "global_daily": result_global}


# ================================================================
# HISTORICAL COLLECTION
# ================================================================

def collect_historical(
    start_date: str = "2024-05-01",
    end_date: str | None = None,
    sleep_between: float = 2.0,
) -> pd.DataFrame:
    """
    Collect historical news from RBC AJAX API + Kommersant.
    Breaks the range into 3-month chunks for better coverage.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    os.makedirs("data", exist_ok=True)

    # Break into 3-month chunks
    chunks = []
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=89), e)
        chunks.append((s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        s = chunk_end + timedelta(days=1)

    log.info(f"Collecting {len(chunks)} chunks from {start_date} to {end_date}")
    log.info(f"Sources: RBC AJAX API + Kommersant")

    all_headlines = []
    total_tickers = len(TICKER_KEYWORDS) + len(GLOBAL_KEYWORDS)
    current = 0

    # --- Ticker news ---
    for ticker, keywords in TICKER_KEYWORDS.items():
        current += 1
        keyword = keywords[0]
        log.info(f"[{current}/{total_tickers}] [{ticker}] Searching: '{keyword}'")

        ticker_count = 0
        for cs, ce in chunks:
            # RBC (DD.MM.YYYY format)
            df_rbc = datetime.strptime(cs, "%Y-%m-%d").strftime("%d.%m.%Y")
            dt_rbc = datetime.strptime(ce, "%Y-%m-%d").strftime("%d.%m.%Y")

            items = fetch_rbc(keyword, df_rbc, dt_rbc, max_pages=5)
            all_headlines.extend(items)
            ticker_count += len(items)
            time.sleep(sleep_between)

            # Kommersant (YYYY-MM-DD format)
            items2 = fetch_kommersant(keyword, cs, ce, max_pages=2)
            all_headlines.extend(items2)
            ticker_count += len(items2)
            time.sleep(sleep_between)

        log.info(f"  Found {ticker_count} headlines")

    # --- Global news ---
    for category, keywords in GLOBAL_KEYWORDS.items():
        current += 1
        keyword = keywords[0]
        log.info(f"[{current}/{total_tickers}] [GLOBAL:{category}] '{keyword}'")

        global_count = 0
        for cs, ce in chunks:
            df_rbc = datetime.strptime(cs, "%Y-%m-%d").strftime("%d.%m.%Y")
            dt_rbc = datetime.strptime(ce, "%Y-%m-%d").strftime("%d.%m.%Y")

            items = fetch_rbc(keyword, df_rbc, dt_rbc, max_pages=3)
            all_headlines.extend(items)
            global_count += len(items)
            time.sleep(sleep_between)

        log.info(f"  Found {global_count} headlines")

    log.info(f"\nTotal headlines collected: {len(all_headlines)}")

    # Remove headlines without dates
    with_dates = [h for h in all_headlines if h.get("date")]
    log.info(f"Headlines with dates: {with_dates.__len__()}")

    # Match to tickers and compute sentiment
    matched = match_headlines_to_tickers(with_dates)

    tickers_found = len(matched["ticker_daily"])
    global_days = len(matched["global_daily"])
    log.info(f"Matched: {tickers_found} tickers, {global_days} global days")

    return _build_output(matched, start_date, end_date)


def collect_daily(append: bool = False) -> pd.DataFrame:
    """Collect today's news from RSS feeds."""
    os.makedirs("data", exist_ok=True)

    headlines = collect_rss_today()
    matched = match_headlines_to_tickers(headlines)

    today = date.today().isoformat()
    df_new = _build_output(matched, today, today)

    if append and os.path.exists("data/news_sentiment_historical.csv"):
        df_old = pd.read_csv("data/news_sentiment_historical.csv")
        df_old["date"] = pd.to_datetime(df_old["date"]).dt.strftime("%Y-%m-%d")
        df_old = df_old[df_old["date"] != today]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.sort_values("date", inplace=True)
        df_all = _add_training_features(df_all)
        df_all.to_csv("data/news_sentiment_historical.csv", index=False)
        _save_inference_json(df_all)
        log.info(f"Appended {len(df_new)} rows, total: {len(df_all)}")
        return df_all
    else:
        df_new.to_csv("data/news_sentiment_daily.csv", index=False)
        log.info(f"Saved {len(df_new)} rows to data/news_sentiment_daily.csv")
        return df_new


# ================================================================
# OUTPUT BUILDER (compatible with train_models.py)
# ================================================================

def _build_output(matched: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Build output CSV in the same format as the GDELT version."""
    dates = pd.date_range(start_date, end_date, freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    data = {"date": date_strs}
    for ticker in TICKER_KEYWORDS:
        ticker_data = matched["ticker_daily"].get(ticker, {})
        data[f"sent_{ticker}"] = [
            ticker_data.get(d, {}).get("sentiment", np.nan) for d in date_strs
        ]
        data[f"news_count_{ticker}"] = [
            ticker_data.get(d, {}).get("count", 0) for d in date_strs
        ]

    global_data = matched["global_daily"]
    data["market_sentiment"] = [
        global_data.get(d, {}).get("sentiment", np.nan) for d in date_strs
    ]
    data["market_news_count"] = [
        global_data.get(d, {}).get("count", 0) for d in date_strs
    ]

    df = pd.DataFrame(data)

    # Forward-fill NaN sentiment (weekends, days without news)
    sent_cols = [c for c in df.columns if c.startswith("sent_")]
    df[sent_cols] = df[sent_cols].ffill().fillna(0)
    if "market_sentiment" in df.columns:
        df["market_sentiment"] = df["market_sentiment"].ffill().fillna(0)

    df = _add_training_features(df)

    df.to_csv("data/news_sentiment_historical.csv", index=False)
    _save_inference_json(df)

    log.info(f"Saved: data/news_sentiment_historical.csv ({len(df)} rows)")
    return df


def _add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling news features used by train_models.py."""
    if df.empty:
        return df
    df = df.copy()

    for col in [c for c in df.columns if c.startswith("sent_")]:
        df[f"{col}_3d"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_7d"] = df[col].rolling(7, min_periods=1).mean()

    for col in [c for c in df.columns if c.startswith("news_count_")]:
        df[f"{col}_3d"] = df[col].rolling(3, min_periods=1).sum()
        df[f"{col}_7d"] = df[col].rolling(7, min_periods=1).sum()

    if "market_sentiment" in df.columns:
        df["market_sentiment_3d"] = df["market_sentiment"].rolling(3, min_periods=1).mean()
        df["market_sentiment_7d"] = df["market_sentiment"].rolling(7, min_periods=1).mean()
    if "market_news_count" in df.columns:
        df["market_news_count_3d"] = df["market_news_count"].rolling(3, min_periods=1).sum()
        df["market_news_count_7d"] = df["market_news_count"].rolling(7, min_periods=1).sum()

    return df


def _save_inference_json(df: pd.DataFrame):
    """Save latest sentiment as JSON for inference."""
    ticker_sentiment = {}
    for ticker in TICKER_KEYWORDS:
        col = f"sent_{ticker}"
        if col in df.columns:
            last_vals = df[col].dropna().tail(5)
            if not last_vals.empty:
                ticker_sentiment[ticker] = round(float(last_vals.mean()), 3)

    market_sent = 0.0
    if "market_sentiment" in df.columns:
        last_market = df["market_sentiment"].dropna().tail(5)
        if not last_market.empty:
            market_sent = round(float(last_market.mean()), 3)

    sentiment_json = {
        "ticker_sentiment": ticker_sentiment,
        "market_sentiment": market_sent,
        "updated": datetime.now().isoformat(),
        "source": "RBC AJAX API + Kommersant",
    }

    with open("data/news_sentiment.json", "w", encoding="utf-8") as f:
        json.dump(sentiment_json, f, indent=2, ensure_ascii=False)

    log.info(f"Saved: data/news_sentiment.json ({len(ticker_sentiment)} tickers)")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="Russian news sentiment collector v2")
    parser.add_argument(
        "--mode", choices=["historical", "daily"], default="historical",
        help="historical = RBC API + Kommersant search; daily = RSS feeds"
    )
    parser.add_argument("--from", dest="date_from", default=None)
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between requests")
    parser.add_argument("--append", action="store_true",
                        help="Append daily data to existing historical CSV")
    args = parser.parse_args()

    if args.date_from is None:
        args.date_from = (date.today() - timedelta(days=365 * args.years)).isoformat()

    print("=" * 60)
    print("Russian News Sentiment Collector v2")
    print(f"Mode: {args.mode}")
    print(f"Period: {args.date_from} to {args.date_to}")
    print(f"Sources: RBC AJAX API + Kommersant")
    print("=" * 60)

    if args.mode == "historical":
        df = collect_historical(args.date_from, args.date_to, sleep_between=args.sleep)
        print(f"\nDone! {len(df)} rows")
        # Show stats
        sent_cols = [c for c in df.columns if c.startswith("sent_") and "_3d" not in c and "_7d" not in c]
        non_zero = sum((df[c] != 0).sum() for c in sent_cols)
        print(f"Non-zero sentiment values: {non_zero}")
        print("Files:")
        print("  data/news_sentiment_historical.csv  <- for training")
        print("  data/news_sentiment.json            <- for predictions")
    else:
        df = collect_daily(append=args.append)
        print(f"\nDone! {len(df)} rows")
        if args.append:
            print("  Appended to data/news_sentiment_historical.csv")
        else:
            print("  Saved to data/news_sentiment_daily.csv")
