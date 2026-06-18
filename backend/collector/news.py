"""
backend/collector/news.py
=========================
Collects news from Russian and international RSS feeds,
analyzes sentiment using FinBERT (ProsusAI/finbert) for financial context.
Falls back to dictionary-based approach if FinBERT unavailable.

Sources:
  Russian:  RIA Novosti, RBC, TASS, Finam
  International: Reuters, Investing.com

Stores results in 'news_sentiment' table.

Usage:
    python -m backend.collector.news
"""
import os
import re
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)

# ================================================================
# RSS FEED SOURCES
# ================================================================

FEEDS = {
    # Russian sources
    "ria": {
        "url": "https://ria.ru/export/rss2/index.xml",
        "lang": "ru",
        "name": "RIA Novosti",
    },
    "rbc": {
        "url": "https://rssexport.rbc.ru/rbcnews/news/20/full.rss",
        "lang": "ru",
        "name": "RBC",
    },
    "tass": {
        "url": "https://tass.ru/rss/v2.xml",
        "lang": "ru",
        "name": "TASS",
    },
    "finam": {
        "url": "https://www.finam.ru/analysis/conews/rsspoint/",
        "lang": "ru",
        "name": "Finam",
    },
    # International sources
    "investing": {
        "url": "https://www.investing.com/rss/news.rss",
        "lang": "en",
        "name": "Investing.com",
    },
}

# ================================================================
# TICKER DETECTION - map company names to tickers
# ================================================================

TICKER_KEYWORDS = {
    "SBER": ["sberbank", "sber", "sber ", "sberbankbank",
             "a]a[", "ca[h", "caeh", "cayhfayr"],
    "GAZP": ["gazprom", "u'cghjv", "ufpghjv", "ufpghjvf"],
    "LKOH": ["lukoil", "lukojl", "kerj]k", "keyj]k"],
    "YDEX": ["yandex", "ydex", "zydtrc", "zyltrc"],
    "NVTK": ["novatek", "yjdfn'r", "yjdfntyr"],
    "ROSN": ["rosneft", "hjcytanm", "hjcytaom"],
    "GMKN": ["norilsk", "nornickel", "yjhybrtkmybrtkm",
             "yjhybrtl", "yjhbkmcr"],
    "MGNT": ["magnit", "vfuybn"],
    "TATN": ["tatneft", "nfnytanm"],
    "MOEX": ["moex", "vjca",
             "vjcrjdcrfz ,bh;f", "vjca"],
    "VTBR": ["vtb", "dn,"],
    "PLZL": ["polyus", "gjk.c", "pol.c"],
    "MTSS": ["mts", "vnc"],
    "T": ["tinkoff", "tcs", "t-bank", "t-technologies",
          "nrch'lbn", "nbymrjaa", "т-технологии"],
    "PIKK": ["pik", "gbr "],
    "SNGS": ["surgut", "cehuen", "cehuenytatufp"],
    "ALRS": ["alrosa", "fkhjcf"],
    "AFLT": ["aeroflot", "f'hjakjn"],
    "CHMF": ["severstal", "ctdthcnfkm"],
    "NLMK": ["nlmk", "ykv"],
    "OZON": ["ozon", "jpjy"],
    "VKCO": ["vkontakte", "vk ", "drjynfrnt"],
    "POSI": ["positive", "gjpbnbd"],
    "X5": ["x5 ", "x5retail", "gznthj", "x5 group"],
}

# Political / geopolitical keywords
POLITICAL_KEYWORDS_RU = [
    "sanctions", "conflict", "war",
    "cfyrwbb", "rjyakbrn", "djqyf",
    "ytanm", "ufp", "jg'r",
    "yato", "yfnj",
    "trump", "nhfvg",
    "china", "rbnfq",
    "fed ", "apc", "ecb",
    "central bank", "wtynhfkmysq ,fyr",
    "interest rate", "cnfdrf",
    "inflation", "byakzwbz",
    "gdp", "ddg",
    "recession", "htwtccbz",
    "geopoliti", "utjgjkbnb",
]

# ================================================================
# SIMPLE SENTIMENT ANALYSIS (dictionary-based)
# ================================================================

# Russian positive/negative financial words
POSITIVE_RU = [
    "hjcn", "edbkbxtybt", "ghb,skm", "ljhj;ftn",
    "gjdsitybt", "dshfc", "ghb,sdftv",
    "jgnbvbpv", "ecgtiysq", "htpekmnfn",
    "htrjhlysq", "vfrcbvev", "ecbktybt",
    "lbdbltyl", "dsreg", "cjukfcjdfybt",
]

NEGATIVE_RU = [
    "gfltybt", "cyb;tybt", "e,snjr", "rjhhtrwbz",
    "rjkkfgc", "rhbpbc", "hbcr", "yfhenitybt",
    "cnfuyfwbz", "byakzwbz", "cfrywbb",
    "ifnlfey", "ltkbcnbyu", "htwtccbz",
    ",fyrhjncndj", "ljku", "j,dfk",
]

POSITIVE_EN = [
    "growth", "increase", "profit", "rise", "gain",
    "rally", "surge", "boost", "optimism", "recovery",
    "record high", "upgrade", "beat", "strong",
    "dividend", "buyback", "deal",
]

NEGATIVE_EN = [
    "fall", "decline", "loss", "drop", "crash",
    "crisis", "risk", "cut", "weak", "default",
    "sanction", "war", "conflict", "recession",
    "bankrupt", "layoff", "downgrade", "miss",
]


_finbert_pipeline = None
_finbert_available = None


def _get_finbert():
    """Lazy-load FinBERT pipeline. Returns None if unavailable."""
    global _finbert_pipeline, _finbert_available
    if _finbert_available is False:
        return None
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        log.info("FinBERT loaded successfully")
        return _finbert_pipeline
    except Exception as e:
        log.warning(f"FinBERT unavailable, falling back to dictionary: {e}")
        _finbert_available = False
        return None


def _translate_title_for_finbert(title: str, lang: str) -> str:
    """
    For Russian titles, translate key financial terms to English
    so FinBERT (English model) can understand the context.
    Simple keyword replacement — not full translation.
    """
    if lang != "ru":
        return title

    replacements = {
        "рост": "growth", "падение": "decline", "прибыль": "profit",
        "убыток": "loss", "дивиденд": "dividend", "выручка": "revenue",
        "выросл": "increased", "снизил": "decreased", "рекорд": "record",
        "обвал": "crash", "ралли": "rally", "санкци": "sanctions",
        "инфляци": "inflation", "ставк": "rate", "нефт": "oil",
        "газ ": "gas ", "банк": "bank", "акци": "shares",
        "биржа": "exchange", "индекс": "index", "курс": "exchange rate",
        "рубл": "ruble", "доллар": "dollar", "экспорт": "export",
        "импорт": "import", "сделк": "deal", "IPO": "IPO",
        "buyback": "buyback", "отчёт": "report", "отчет": "report",
        "кризис": "crisis", "дефолт": "default", "банкрот": "bankruptcy",
        "оптимизм": "optimism", "пессимизм": "pessimism",
        "повыш": "increase", "пониж": "decrease",
        "рекомендац": "recommendation", "прогноз": "forecast",
        "капитализац": "capitalization", "доходност": "yield",
    }
    text = title
    for ru, en in replacements.items():
        text = text.replace(ru, en)
    return text


def analyze_sentiment(title: str, lang: str = "ru") -> float:
    """
    FinBERT-based sentiment analysis: -1.0 to +1.0.
    Falls back to dictionary-based if FinBERT unavailable.
    """
    finbert = _get_finbert()

    if finbert is not None:
        try:
            text = _translate_title_for_finbert(title, lang)
            result = finbert(text[:512])[0]
            label = result["label"]
            score = result["score"]
            if label == "positive":
                return round(score, 3)
            elif label == "negative":
                return round(-score, 3)
            else:  # neutral
                return 0.0
        except Exception as e:
            log.debug(f"FinBERT error: {e}")

    # Fallback: dictionary-based
    text = title.lower()
    pos_words = POSITIVE_RU if lang == "ru" else POSITIVE_EN
    neg_words = NEGATIVE_RU if lang == "ru" else NEGATIVE_EN

    pos_count = sum(1 for w in pos_words if w in text)
    neg_count = sum(1 for w in neg_words if w in text)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return round((pos_count - neg_count) / total, 3)


def detect_tickers(title: str) -> list[str]:
    """Find which tickers are mentioned in the title."""
    text = title.lower()
    found = []
    for ticker, keywords in TICKER_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                found.append(ticker)
                break
    return found


def is_political(title: str) -> bool:
    """Check if news is political/geopolitical."""
    text = title.lower()
    return any(kw in text for kw in POLITICAL_KEYWORDS_RU)


# ================================================================
# RSS PARSER
# ================================================================

def _fetch_article_text(url: str) -> str:
    """
    Fetch full article text from URL.
    Returns cleaned text or empty string on failure.
    """
    if not url or len(url) < 10:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MOEXPredictor/1.0)"}
        r = requests.get(url, timeout=10, headers=headers,
                         proxies={"http": None, "https": None})
        r.encoding = "utf-8"
        if r.status_code != 200:
            return ""

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Try common article selectors
        article = (soup.find("article") or
                   soup.find("div", class_=re.compile(r"article|content|text|body", re.I)) or
                   soup.find("div", {"itemprop": "articleBody"}))

        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            # Fallback: all <p> tags
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)

        # Clean up
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]  # limit to ~3000 chars for FinBERT
    except Exception as e:
        log.debug(f"Article fetch error {url[:60]}: {e}")
        return ""


def parse_rss(url: str, source: str, lang: str, fetch_full_text: bool = True) -> list[dict]:
    """Parse RSS feed and extract news items with optional full article text."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MOEXPredictor/1.0)"
        }
        r = requests.get(url, timeout=15, headers=headers,
                         proxies={"http": None, "https": None})
        r.encoding = "utf-8"
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"[{source}] RSS fetch error: {e}")
        return []

    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")

        # Try to get description/summary from RSS
        desc_el = item.find("description")
        desc_text = ""
        if desc_el is not None and desc_el.text:
            # Strip HTML from description
            desc_text = re.sub(r"<[^>]+>", "", desc_el.text).strip()

        if title_el is None or title_el.text is None:
            continue

        title = title_el.text.strip()
        article_url = link_el.text.strip() if link_el is not None and link_el.text else ""

        # Parse date
        pub_date = None
        if pub_el is not None and pub_el.text:
            try:
                from email.utils import parsedate_to_datetime
                pub_date = parsedate_to_datetime(pub_el.text).isoformat()
            except Exception:
                pub_date = datetime.now().isoformat()
        else:
            pub_date = datetime.now().isoformat()

        # Get full text: RSS description + article body
        full_text = desc_text
        if fetch_full_text and article_url:
            body = _fetch_article_text(article_url)
            if body:
                full_text = body
            time.sleep(0.3)  # be nice to servers

        # Analyze sentiment on full text (title + body)
        analysis_text = f"{title}. {full_text}" if full_text else title
        sentiment = analyze_sentiment(analysis_text, lang)

        # Detect tickers in title + description
        detect_text = f"{title} {desc_text} {full_text[:500]}"
        tickers = detect_tickers(detect_text)
        political = is_political(detect_text)

        items.append({
            "published": pub_date,
            "source": source,
            "title": title,
            "url": article_url,
            "sentiment": sentiment,
            "tickers": tickers,
            "is_political": political,
        })

    return items


# ================================================================
# COLLECT FROM ALL SOURCES
# ================================================================

def collect_all_news() -> list[dict]:
    """Collect news from all RSS feeds."""
    all_news = []
    for source_id, feed in FEEDS.items():
        log.info(f"Fetching {feed['name']}...")
        items = parse_rss(feed["url"], source_id, feed["lang"])
        all_news.extend(items)
        log.info(f"  Got {len(items)} items")
        time.sleep(0.5)

    log.info(f"\nTotal news items: {len(all_news)}")

    # Stats
    with_tickers = [n for n in all_news if n["tickers"]]
    political = [n for n in all_news if n["is_political"]]
    positive = [n for n in all_news if n["sentiment"] > 0.1]
    negative = [n for n in all_news if n["sentiment"] < -0.1]
    log.info(f"  With tickers: {len(with_tickers)}")
    log.info(f"  Political: {len(political)}")
    log.info(f"  Positive: {len(positive)}")
    log.info(f"  Negative: {len(negative)}")

    return all_news


# ================================================================
# GET SENTIMENT PER TICKER (for ML features)
# ================================================================

def get_ticker_sentiment(news: list[dict]) -> dict[str, float]:
    """
    Aggregate sentiment per ticker for use as ML feature.
    Returns dict: {'SBER': 0.35, 'GAZP': -0.12, ...}
    """
    from collections import defaultdict
    scores = defaultdict(list)

    for item in news:
        for ticker in item["tickers"]:
            scores[ticker].append(item["sentiment"])

    result = {}
    for ticker, values in scores.items():
        if values:
            result[ticker] = round(sum(values) / len(values), 3)

    return result


def get_market_sentiment(news: list[dict]) -> float:
    """
    Overall market sentiment from political/macro news.
    Used as a global feature for all tickers.
    """
    political = [n for n in news if n["is_political"]]
    if not political:
        return 0.0
    avg = sum(n["sentiment"] for n in political) / len(political)
    return round(avg, 3)


# ================================================================
# SAVE TO DATABASE
# ================================================================

def save_news_to_db(news: list[dict]):
    """Save news to Supabase via REST API."""
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()

    sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

    saved = 0
    for item in news:
        try:
            sb.table("news_sentiment").upsert({
                "published":    item["published"],
                "source":       item["source"],
                "title":        item["title"][:500],
                "url":          item["url"][:500],
                "sentiment":    item["sentiment"],
                "tickers":      item["tickers"],
                "is_political": item["is_political"],
            }, on_conflict="url").execute()
            saved += 1
        except Exception as e:
            log.debug(f"Save error: {e}")

    log.info(f"Saved {saved}/{len(news)} news items to DB")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    from dotenv import load_dotenv
    load_dotenv()

    news = collect_all_news()

    # Ticker sentiments
    ticker_sent = get_ticker_sentiment(news)
    if ticker_sent:
        print("\nTicker sentiments:")
        for t, s in sorted(ticker_sent.items()):
            emoji = "+" if s > 0 else "-" if s < 0 else "="
            print(f"  {t}: {emoji}{abs(s):.3f}")

    # Market sentiment
    market = get_market_sentiment(news)
    print(f"\nOverall market sentiment: {market:+.3f}")

    # Save to Supabase
    save_news_to_db(news)

    # Save sentiments to local file for use in predictions
    import json
    os.makedirs("data", exist_ok=True)
    sentiments = {
        "ticker_sentiment": get_ticker_sentiment(news),
        "market_sentiment": get_market_sentiment(news),
    }
    with open("data/news_sentiment.json", "w") as f:
        json.dump(sentiments, f)
    print("Saved data/news_sentiment.json")