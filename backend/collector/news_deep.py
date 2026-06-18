"""
backend/collector/news_deep.py
===============================
Deep news collector — fetches FULL article text from Russian sources,
analyzes with ENSEMBLE of FinBERT + ruBERT for robust sentiment.

Key improvements over news_historical_finbert.py:
  1. Full article text parsing (not just headlines)
  2. Dual-model NLP: FinBERT (financial) + ruBERT (Russian)
  3. More sources: RBC API, Finam, Kommersant, Interfax, SmartLab
  4. Better ticker matching (title + body)
  5. Chunked text analysis (up to 3000 chars per article)

Sources:
  1. RBC AJAX API     — search by keyword, returns JSON with pagination
  2. Finam            — financial news RSS + article pages
  3. Kommersant       — search HTML, server-rendered
  4. Interfax         — RSS feeds
  5. SmartLab         — company news pages

Output (100% compatible with train_models.py):
  data/news_sentiment_historical_merged.csv  — daily sentiment per ticker
  data/news_sentiment.json                   — latest ticker/market sentiment

Usage:
    python -m backend.collector.news_deep
    python -m backend.collector.news_deep --from 2024-01-01 --to 2026-05-26
    python -m backend.collector.news_deep --ticker SBER
    python -m backend.collector.news_deep --skip-fetch   # only re-analyze cached
    python -m backend.collector.news_deep --merge         # merge with existing data
"""
import os
import re
import json
import time
import logging
import argparse
import hashlib
from datetime import datetime, timedelta, date
from collections import defaultdict
from urllib.parse import quote_plus
from pathlib import Path

import requests
from bs4 import BeautifulSoup
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

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ================================================================
# TICKER CONFIG
# ================================================================

TICKER_QUERIES = {
    "GAZP":  ["Газпром"],
    "LKOH":  ["Лукойл", "ЛУКОЙЛ"],
    "NVTK":  ["Новатэк", "НОВАТЭК"],
    "ROSN":  ["Роснефть"],
    "TATN":  ["Татнефть"],
    "SNGS":  ["Сургутнефтегаз"],
    "SIBN":  ["Газпром нефть"],
    "SBER":  ["Сбербанк", "Сбер"],
    "VTBR":  ["ВТБ"],
    "T":     ["Тинькофф", "Т-Банк", "Т-Технологии"],
    "MOEX":  ["Мосбиржа", "Московская биржа"],
    "GMKN":  ["Норникель"],
    "PLZL":  ["Полюс"],
    "ALRS":  ["Алроса", "АЛРОСА"],
    "CHMF":  ["Северсталь"],
    "NLMK":  ["НЛМК"],
    "MAGN":  ["ММК", "Магнитогорский"],
    "YDEX":  ["Яндекс", "Yandex"],
    "OZON":  ["Озон", "Ozon"],
    "VKCO":  ["VK Company", "ВКонтакте", "VK "],
    "POSI":  ["Positive Technologies", "Позитив"],
    "MGNT":  ["Магнит"],
    "X5":    ["X5 Group", "Пятёрочка", "X5 Retail"],
    "LENT":  ["Лента ритейл", "Лента сеть"],
    "FIXP":  ["Fix Price", "Фикс Прайс"],
    "MTSS":  ["МТС"],
    "RTKM":  ["Ростелеком"],
    "PIKK":  ["ПИК группа", "ПИК компания"],
    "SMLT":  ["Самолёт", "Самолет"],
    "AFLT":  ["Аэрофлот"],
    "FESH":  ["FESCO", "ДВМП"],
    "UWGN":  ["ОВК", "Объединённая вагонная"],
}

# Global/market keywords
GLOBAL_QUERIES = [
    "индекс Мосбиржи IMOEX",
    "ключевая ставка ЦБ",
    "курс рубля доллар",
    "нефть Brent цена",
    "санкции Россия экономика",
    "инфляция Россия",
    "ВВП России экономика",
    "IPO Мосбиржа",
    "дивидендный сезон акции",
]

# Impact keywords: articles mentioning these get 2× weight
IMPACT_KEYWORDS = [
    "дивиденд", "отчёт", "отчет", "прибыль", "выручка", "убыток",
    "IPO", "SPO", "buyback", "обратный выкуп", "допэмиссия",
    "МСФО", "РСБУ", "целевая цена", "рейтинг",
    "контракт", "сделка", "слияние", "поглощение",
    "санкции", "банкротство", "дефолт",
]


# ================================================================
# HTTP HELPERS
# ================================================================

def _safe_get(url, retries=3, delay=2.0, timeout=15):
    """HTTP GET with retry and backoff."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, proxies=NO_PROXY)
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


def _extract_article_text(html, url=""):
    """Extract main text content from HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript", "figure",
                     "figcaption", "button"]):
        tag.decompose()

    # Site-specific selectors
    selectors = [
        # RBC
        ("div", {"class": re.compile(r"article__text|article__body")}),
        # Kommersant
        ("div", {"class": re.compile(r"b-article__text|article_text_wrapper")}),
        # Interfax
        ("div", {"class": re.compile(r"textMTL|article__text")}),
        ("article", {"itemprop": "articleBody"}),
        # Finam
        ("div", {"class": re.compile(r"finfin-local-plugin-content|post-body")}),
        # SmartLab
        ("div", {"class": re.compile(r"content|topic-text")}),
        # Generic
        ("article", {}),
        ("div", {"class": re.compile(r"text-content|article-text|entry-content")}),
        ("div", {"itemprop": "articleBody"}),
    ]

    text = ""
    for tag_name, attrs in selectors:
        el = soup.find(tag_name, attrs)
        if el:
            paragraphs = el.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
            else:
                text = el.get_text(separator=" ", strip=True)
            if len(text) > 100:
                break

    if not text or len(text) < 100:
        # Fallback: all <p> tags
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)

    # Clean
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common boilerplate
    for noise in ["Подпишитесь на", "Читайте также", "Материалы по теме",
                  "Следите за нами", "Все права защищены", "©"]:
        idx = text.find(noise)
        if idx > 200:
            text = text[:idx]

    return text[:4000]  # limit for NLP models


# ================================================================
# SOURCE 1: RBC AJAX API
# ================================================================

def _fetch_rbc(keyword, date_from, date_to, max_pages=8):
    """
    Fetch from RBC AJAX API. Returns list of articles with full text.
    date_from/date_to: DD.MM.YYYY format
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
            article_url = item.get("fronturl", "") or item.get("url", "")
            pub_date = item.get("publish_date_t") or item.get("publish_date", "")

            dt_str = None
            if pub_date:
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(pub_date))
                if m:
                    dt_str = m.group(0)
                else:
                    try:
                        ts = int(pub_date)
                        dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass

            if title and dt_str:
                results.append({
                    "date": dt_str,
                    "title": title,
                    "url": article_url,
                    "text": "",  # will fetch full text later
                    "source": "rbc",
                })

        more = data.get("moreExists", False)
        cursor = data.get("endCursor")
        if not more or not cursor:
            break
        time.sleep(1.0)

    return results


def _fetch_rbc_article(url):
    """Fetch full text from RBC article page."""
    resp = _safe_get(url, timeout=10)
    if not resp:
        return ""
    return _extract_article_text(resp.text, url)


# ================================================================
# SOURCE 2: FINAM
# ================================================================

FINAM_RSS = "https://www.finam.ru/analysis/conews/rsspoint/"


def _fetch_finam_rss():
    """Fetch articles from Finam RSS."""
    articles = []
    resp = _safe_get(FINAM_RSS)
    if not resp:
        return articles

    from xml.etree import ElementTree
    try:
        root = ElementTree.fromstring(resp.content)
    except Exception:
        return articles

    for item in root.iter("item"):
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        pub_date = item.findtext("pubDate", "")
        description = item.findtext("description", "")

        dt_str = None
        if pub_date:
            for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"]:
                try:
                    dt_str = datetime.strptime(pub_date.strip(), fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue

        desc_text = re.sub(r"<[^>]+>", "", description).strip() if description else ""

        if title:
            articles.append({
                "date": dt_str or date.today().isoformat(),
                "title": title,
                "url": link,
                "text": desc_text,
                "source": "finam",
            })

    return articles


def _fetch_finam_article(url):
    """Fetch full text from Finam article page."""
    resp = _safe_get(url, timeout=10)
    if not resp:
        return ""
    return _extract_article_text(resp.text, url)


# ================================================================
# SOURCE 3: KOMMERSANT
# ================================================================

def _fetch_kommersant(keyword, date_from, date_to, max_pages=5):
    """Fetch from Kommersant search. date_from/date_to: YYYY-MM-DD."""
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

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("article")
        if not articles:
            break

        for article in articles:
            link_el = article.select_one("h2 a") or article.select_one("a")
            if not link_el:
                continue

            title = link_el.get_text(strip=True)
            href = link_el.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://www.kommersant.ru{href}"

            dt_str = None
            m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", article.get_text())
            if m:
                dt_str = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

            # Try to get description/summary
            desc_el = article.select_one(".search_results_text, .article_subheader, p")
            desc = desc_el.get_text(strip=True) if desc_el else ""

            if title:
                results.append({
                    "date": dt_str,
                    "title": title,
                    "url": href,
                    "text": desc,
                    "source": "kommersant",
                })

        time.sleep(1.5)

    return results


def _fetch_kommersant_article(url):
    """Fetch full text from Kommersant article page."""
    resp = _safe_get(url, timeout=10)
    if not resp:
        return ""
    return _extract_article_text(resp.text, url)


# ================================================================
# SOURCE 4: INTERFAX RSS
# ================================================================

INTERFAX_RSS = [
    "https://www.interfax.ru/rss.asp",
    "https://www.interfax.ru/business/rss.asp",
]


def _fetch_interfax_rss():
    """Fetch from Interfax RSS feeds."""
    from xml.etree import ElementTree
    articles = []

    for rss_url in INTERFAX_RSS:
        resp = _safe_get(rss_url)
        if not resp:
            continue
        try:
            root = ElementTree.fromstring(resp.content)
        except Exception:
            continue

        for item in root.iter("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            desc = item.findtext("description", "")

            dt_str = None
            if pub_date:
                for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                            "%a, %d %b %Y %H:%M:%S"]:
                    try:
                        dt_str = datetime.strptime(pub_date.strip(), fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue

            desc_text = re.sub(r"<[^>]+>", "", desc).strip() if desc else ""

            if title:
                articles.append({
                    "date": dt_str or date.today().isoformat(),
                    "title": title,
                    "url": link,
                    "text": desc_text,
                    "source": "interfax",
                })
        time.sleep(0.3)

    return articles


# ================================================================
# SOURCE 5: SMARTLAB COMPANY PAGES
# ================================================================

def _fetch_smartlab_news(ticker, max_pages=3):
    """Fetch news from SmartLab company page."""
    # SmartLab ticker mapping
    sl_map = {
        "SBER": "sber", "GAZP": "gazp", "LKOH": "lkoh",
        "YDEX": "ydex", "GMKN": "gmkn", "ROSN": "rosn",
        "NVTK": "nvtk", "TATN": "tatn", "MGNT": "mgnt",
        "VTBR": "vtbr", "MOEX": "moex", "PLZL": "plzl",
        "CHMF": "chmf", "NLMK": "nlmk", "OZON": "ozon",
        "MTSS": "mtss", "T": "tcsg", "AFLT": "aflt",
        "ALRS": "alrs", "SNGS": "sngs", "PIKK": "pikk",
        "POSI": "posi", "VKCO": "vkco", "X5": "x5",
        "MAGN": "magn", "RTKM": "rtkm", "SIBN": "sibn",
    }
    sl_ticker = sl_map.get(ticker)
    if not sl_ticker:
        return []

    articles = []
    for page in range(1, max_pages + 1):
        url = f"https://smart-lab.ru/q/{sl_ticker}/news/page{page}/"
        resp = _safe_get(url, timeout=10)
        if not resp:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        news_items = soup.select(".news-list-item, .topic, tr")

        for item in news_items:
            link_el = item.select_one("a")
            if not link_el:
                continue

            title = link_el.get_text(strip=True)
            href = link_el.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://smart-lab.ru{href}"

            if not title or len(title) < 10:
                continue

            # Try to find date
            dt_str = None
            date_el = item.select_one(".date, time, .news-date")
            if date_el:
                m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", date_el.get_text())
                if m:
                    dt_str = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

            articles.append({
                "date": dt_str,
                "title": title,
                "url": href,
                "text": "",
                "source": "smartlab",
            })

        time.sleep(1.0)

    return articles


# ================================================================
# NLP: DUAL MODEL SENTIMENT (FinBERT + ruBERT)
# ================================================================

_finbert = None
_rubert = None


def _get_finbert():
    """Load ProsusAI/finbert (English financial sentiment)."""
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


def _get_rubert():
    """Load Russian sentiment model."""
    global _rubert
    if _rubert is not None:
        return _rubert
    try:
        from transformers import pipeline
        # Try blanchefort/rubert-base-cased-sentiment-rusentiment first
        for model_name in [
            "blanchefort/rubert-base-cased-sentiment-rusentiment",
            "blanchefort/rubert-base-cased-sentiment",
            "seara/rubert-tiny2-russian-sentiment",
        ]:
            try:
                _rubert = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    truncation=True,
                    max_length=512,
                )
                print(f"ruBERT loaded: {model_name}")
                return _rubert
            except Exception:
                continue
        print("No ruBERT model available")
        return None
    except Exception as e:
        print(f"ruBERT unavailable: {e}")
        return None


# Translation map for FinBERT (Russian → English financial terms)
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
    "чистая прибыль": "net profit", "EBITDA": "EBITDA",
    "МСФО": "IFRS", "РСБУ": "RAS",
    "обратный выкуп": "buyback", "допэмиссия": "share offering",
    "целевая цена": "target price", "рейтинг": "rating",
    "слияние": "merger", "поглощение": "acquisition",
}


def _translate_for_finbert(text):
    """Translate key Russian terms to English for FinBERT."""
    result = text
    for ru, en in RU_EN_FINANCIAL.items():
        result = result.replace(ru, en)
    return result


def _finbert_score(text):
    """Get FinBERT sentiment score: -1.0 to +1.0."""
    finbert = _get_finbert()
    if finbert is None:
        return None
    try:
        translated = _translate_for_finbert(text)
        result = finbert(translated[:512])[0]
        label = result["label"]
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        return 0.0
    except Exception:
        return None


def _rubert_score(text):
    """Get ruBERT sentiment score: -1.0 to +1.0."""
    rubert = _get_rubert()
    if rubert is None:
        return None
    try:
        result = rubert(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        # Model labels vary: POSITIVE/NEGATIVE/NEUTRAL or positive/negative/neutral
        if any(p in label for p in ["positive", "pos"]):
            return score
        elif any(n in label for n in ["negative", "neg"]):
            return -score
        return 0.0
    except Exception:
        return None


def analyze_sentiment_ensemble(text):
    """
    Ensemble sentiment: average of FinBERT + ruBERT.
    For long texts, analyze chunks and average.
    Returns: float in [-1, 1]
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    # For long texts, split into chunks and average
    chunks = []
    if len(text) > 600:
        # First chunk: title + opening (most important)
        chunks.append(text[:512])
        # Middle chunk
        mid = len(text) // 2
        chunks.append(text[max(0, mid - 256):mid + 256])
        # Last chunk: conclusion
        if len(text) > 1200:
            chunks.append(text[-512:])
    else:
        chunks = [text]

    scores = []
    for chunk in chunks:
        fb = _finbert_score(chunk)
        rb = _rubert_score(chunk)

        chunk_scores = [s for s in [fb, rb] if s is not None]
        if chunk_scores:
            scores.append(np.mean(chunk_scores))

    if not scores:
        return 0.0

    # Weight first chunk (title+opening) more heavily
    if len(scores) > 1:
        weights = [2.0] + [1.0] * (len(scores) - 1)
        return round(float(np.average(scores, weights=weights)), 4)
    return round(float(scores[0]), 4)


# ================================================================
# MAIN COLLECTION
# ================================================================

CACHE_FILE = "data/news_deep_cache.json"


def _url_hash(url):
    """Short hash for deduplication."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _load_cache():
    """Load article cache."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    """Save article cache."""
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def collect_articles(date_from="2024-01-01", date_to=None,
                     tickers=None, fetch_text=True,
                     sleep_between=1.5):
    """
    Collect articles from all sources.
    Returns cache dict: {ticker: [articles], "_GLOBAL": [articles]}
    """
    if date_to is None:
        date_to = date.today().isoformat()

    if tickers is None:
        tickers = list(TICKER_QUERIES.keys())

    cache = _load_cache()
    existing_urls = set()
    for articles in cache.values():
        for a in articles:
            if a.get("url"):
                existing_urls.add(a["url"])

    initial_count = sum(len(v) for v in cache.values())
    print(f"Cache loaded: {initial_count} articles, {len(existing_urls)} unique URLs")

    def _add_article(key, article):
        url = article.get("url", "")
        if url and url in existing_urls:
            return False
        if url:
            existing_urls.add(url)
        cache.setdefault(key, []).append(article)
        return True

    # ---- Source 1: RBC AJAX API (primary, best coverage) ----
    print("\n[1/5] RBC AJAX API...")

    # Break into 3-month chunks
    dt_from = datetime.strptime(date_from, "%Y-%m-%d")
    dt_to = datetime.strptime(date_to, "%Y-%m-%d")
    chunks = []
    cs = dt_from
    while cs < dt_to:
        ce = min(cs + timedelta(days=89), dt_to)
        chunks.append((cs, ce))
        cs = ce + timedelta(days=1)

    for i, (ticker, queries) in enumerate(TICKER_QUERIES.items()):
        if ticker not in tickers:
            continue

        ticker_new = 0
        for keyword in queries[:2]:  # up to 2 keywords
            for chunk_start, chunk_end in chunks:
                df_rbc = chunk_start.strftime("%d.%m.%Y")
                dt_rbc = chunk_end.strftime("%d.%m.%Y")

                articles = _fetch_rbc(keyword, df_rbc, dt_rbc, max_pages=8)
                for art in articles:
                    if _add_article(ticker, art):
                        ticker_new += 1
                time.sleep(sleep_between)

        if ticker_new > 0:
            print(f"  {ticker}: +{ticker_new} new")

        if (i + 1) % 10 == 0:
            _save_cache(cache)
            print(f"  Progress: {i+1}/{len(tickers)} tickers")

    # Global queries
    for query in GLOBAL_QUERIES:
        global_new = 0
        for chunk_start, chunk_end in chunks:
            df_rbc = chunk_start.strftime("%d.%m.%Y")
            dt_rbc = chunk_end.strftime("%d.%m.%Y")
            articles = _fetch_rbc(query, df_rbc, dt_rbc, max_pages=3)
            for art in articles:
                if _add_article("_GLOBAL", art):
                    global_new += 1
            time.sleep(sleep_between)
        if global_new > 0:
            print(f"  GLOBAL ({query[:30]}): +{global_new}")

    _save_cache(cache)

    # ---- Source 2: Kommersant ----
    print("\n[2/5] Kommersant...")
    for i, (ticker, queries) in enumerate(TICKER_QUERIES.items()):
        if ticker not in tickers:
            continue

        ticker_new = 0
        for keyword in queries[:1]:
            for chunk_start, chunk_end in chunks:
                cs_str = chunk_start.strftime("%Y-%m-%d")
                ce_str = chunk_end.strftime("%Y-%m-%d")
                articles = _fetch_kommersant(keyword, cs_str, ce_str, max_pages=3)
                for art in articles:
                    if _add_article(ticker, art):
                        ticker_new += 1
                time.sleep(sleep_between)

        if ticker_new > 0:
            print(f"  {ticker}: +{ticker_new}")

    _save_cache(cache)

    # ---- Source 3: Finam RSS ----
    print("\n[3/5] Finam RSS...")
    finam_articles = _fetch_finam_rss()
    finam_matched = 0
    for art in finam_articles:
        combined = (art.get("title", "") + " " + art.get("text", "")).lower()
        matched = False
        for ticker, queries in TICKER_QUERIES.items():
            if ticker not in tickers:
                continue
            for q in queries:
                if q.lower() in combined:
                    if _add_article(ticker, art):
                        finam_matched += 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            # Global news
            _add_article("_GLOBAL", art)
    print(f"  Finam: {len(finam_articles)} total, {finam_matched} matched to tickers")

    # ---- Source 4: Interfax RSS ----
    print("\n[4/5] Interfax RSS...")
    interfax_articles = _fetch_interfax_rss()
    interfax_matched = 0
    for art in interfax_articles:
        combined = (art.get("title", "") + " " + art.get("text", "")).lower()
        matched = False
        for ticker, queries in TICKER_QUERIES.items():
            if ticker not in tickers:
                continue
            for q in queries:
                if q.lower() in combined:
                    if _add_article(ticker, art):
                        interfax_matched += 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            _add_article("_GLOBAL", art)
    print(f"  Interfax: {len(interfax_articles)} total, {interfax_matched} matched")

    # ---- Source 5: SmartLab ----
    print("\n[5/5] SmartLab...")
    for ticker in tickers[:15]:  # top tickers only
        sl_articles = _fetch_smartlab_news(ticker, max_pages=2)
        sl_new = 0
        for art in sl_articles:
            if _add_article(ticker, art):
                sl_new += 1
        if sl_new:
            print(f"  {ticker}: +{sl_new}")
        time.sleep(1.0)

    _save_cache(cache)

    total = sum(len(v) for v in cache.values())
    new = total - initial_count
    print(f"\nCollection complete: {total} articles ({new} new)")

    # ---- Fetch full text for articles that don't have it ----
    if fetch_text:
        print("\nFetching full article text...")
        fetched = 0
        errors = 0
        total_to_fetch = sum(
            1 for articles in cache.values()
            for a in articles
            if len(a.get("text", "")) < 100 and a.get("url")
        )
        print(f"  Articles needing text: {total_to_fetch}")

        for key, articles in cache.items():
            for art in articles:
                if len(art.get("text", "")) >= 100:
                    continue
                url = art.get("url", "")
                if not url:
                    continue

                try:
                    text = ""
                    if "rbc.ru" in url:
                        text = _fetch_rbc_article(url)
                    elif "kommersant.ru" in url:
                        text = _fetch_kommersant_article(url)
                    elif "finam.ru" in url:
                        text = _fetch_finam_article(url)
                    else:
                        resp = _safe_get(url, timeout=10, retries=1)
                        if resp:
                            text = _extract_article_text(resp.text, url)

                    if text and len(text) > 100:
                        art["text"] = text
                        fetched += 1
                    else:
                        errors += 1

                    if (fetched + errors) % 50 == 0:
                        print(f"  Fetched: {fetched}/{total_to_fetch}, errors: {errors}")
                        _save_cache(cache)

                    time.sleep(0.5)
                except Exception:
                    errors += 1

        _save_cache(cache)
        print(f"  Text fetched: {fetched}, errors: {errors}")

    return cache


# ================================================================
# ANALYZE & BUILD OUTPUT
# ================================================================

def analyze_and_build(cache, date_from="2024-01-01", date_to=None, tickers=None):
    """
    Analyze all cached articles with FinBERT+ruBERT ensemble,
    build daily sentiment CSV.
    """
    if date_to is None:
        date_to = date.today().isoformat()
    if tickers is None:
        tickers = list(TICKER_QUERIES.keys())

    print(f"\nAnalyzing sentiment (FinBERT + ruBERT ensemble)...")

    all_records = []
    total_articles = sum(len(v) for k, v in cache.items() if k != "_GLOBAL")
    analyzed = 0

    for key, articles in cache.items():
        if key == "_GLOBAL":
            continue
        if key not in tickers:
            continue

        for art in articles:
            # Parse date
            dt_str = art.get("date")
            if not dt_str:
                continue
            # Normalize date
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(dt_str))
            if not m:
                continue
            article_date = m.group(0)

            if article_date < date_from or article_date > date_to:
                continue

            # Build analysis text: title + body
            title = art.get("title", "")
            text = art.get("text", "")
            analysis_text = f"{title}. {text}" if text else title

            # Ensemble sentiment
            if "sentiment_ensemble" in art:
                sentiment = art["sentiment_ensemble"]
            else:
                sentiment = analyze_sentiment_ensemble(analysis_text)
                art["sentiment_ensemble"] = sentiment

            is_corporate = any(kw in analysis_text.lower() for kw in IMPACT_KEYWORDS)
            has_text = len(text) > 100

            all_records.append({
                "date": article_date,
                "ticker": key,
                "sentiment": sentiment,
                "is_corporate": is_corporate,
                "has_full_text": has_text,
                "title": title[:200],
            })

            analyzed += 1
            if analyzed % 100 == 0:
                print(f"  Analyzed: {analyzed}/{total_articles}")

    # Global/market articles
    for art in cache.get("_GLOBAL", []):
        dt_str = art.get("date")
        if not dt_str:
            continue
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(dt_str))
        if not m:
            continue
        article_date = m.group(0)
        if article_date < date_from or article_date > date_to:
            continue

        title = art.get("title", "")
        text = art.get("text", "")
        analysis_text = f"{title}. {text}" if text else title

        if "sentiment_ensemble" in art:
            sentiment = art["sentiment_ensemble"]
        else:
            sentiment = analyze_sentiment_ensemble(analysis_text)
            art["sentiment_ensemble"] = sentiment

        all_records.append({
            "date": article_date,
            "ticker": "_GLOBAL",
            "sentiment": sentiment,
            "is_corporate": False,
            "has_full_text": len(text) > 100,
            "title": title[:200],
        })

    _save_cache(cache)  # save with cached sentiment scores

    if not all_records:
        print("No articles to analyze!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"\nTotal analyzed: {len(df)} articles")
    print(f"  With full text: {df['has_full_text'].sum()} ({df['has_full_text'].mean()*100:.0f}%)")
    print(f"  Corporate events: {df['is_corporate'].sum()}")
    print(f"  Tickers: {df['ticker'].nunique()}")

    # Save raw analyzed articles
    articles_path = "data/news_deep_analyzed.csv"
    df.to_csv(articles_path, index=False)
    print(f"  Saved: {articles_path}")

    # ---- Build daily pivot ----
    # Corporate events get 2× weight
    df["weighted_sentiment"] = df.apply(
        lambda r: r["sentiment"] * 2.0 if r["is_corporate"] else r["sentiment"],
        axis=1,
    )

    # Separate ticker vs global
    df_tickers = df[df["ticker"] != "_GLOBAL"].copy()
    df_global = df.copy()  # all articles for market sentiment

    # Pivot: daily sentiment per ticker
    if not df_tickers.empty:
        df_agg = df_tickers.groupby(["date", "ticker"]).agg(
            tone=("weighted_sentiment", "mean"),
            count=("sentiment", "count"),
        ).reset_index()

        df_tone = df_agg.pivot_table(
            index="date", columns="ticker", values="tone", aggfunc="mean"
        ).reset_index()
        df_tone.columns = ["date"] + [f"sent_{c}" for c in df_tone.columns[1:]]

        df_count = df_agg.pivot_table(
            index="date", columns="ticker", values="count", aggfunc="sum"
        ).reset_index()
        df_count.columns = ["date"] + [f"news_count_{c}" for c in df_count.columns[1:]]

        df_pivot = df_tone.merge(df_count, on="date", how="outer")
    else:
        df_pivot = pd.DataFrame({"date": []})

    # Market sentiment
    mkt = df_global.groupby("date")["weighted_sentiment"].mean().reset_index()
    mkt.columns = ["date", "market_sentiment"]
    mkt_count = df_global.groupby("date")["sentiment"].count().reset_index()
    mkt_count.columns = ["date", "market_news_count"]

    df_pivot = df_pivot.merge(mkt, on="date", how="outer")
    df_pivot = df_pivot.merge(mkt_count, on="date", how="outer")

    # Sort and fill date gaps
    df_pivot.sort_values("date", inplace=True)

    # Rolling features
    sent_cols = [c for c in df_pivot.columns if c.startswith("sent_")]
    for col in sent_cols:
        df_pivot[f"{col}_3d"] = df_pivot[col].rolling(3, min_periods=1).mean().round(4)
        df_pivot[f"{col}_7d"] = df_pivot[col].rolling(7, min_periods=1).mean().round(4)

    count_cols = [c for c in df_pivot.columns if c.startswith("news_count_")]
    for col in count_cols:
        df_pivot[f"{col}_3d"] = df_pivot[col].rolling(3, min_periods=1).sum()
        df_pivot[f"{col}_7d"] = df_pivot[col].rolling(7, min_periods=1).sum()

    if "market_sentiment" in df_pivot.columns:
        df_pivot["market_sentiment_3d"] = (
            df_pivot["market_sentiment"].rolling(3, min_periods=1).mean().round(4)
        )
        df_pivot["market_sentiment_7d"] = (
            df_pivot["market_sentiment"].rolling(7, min_periods=1).mean().round(4)
        )
    if "market_news_count" in df_pivot.columns:
        df_pivot["market_news_count_3d"] = (
            df_pivot["market_news_count"].rolling(3, min_periods=1).sum()
        )
        df_pivot["market_news_count_7d"] = (
            df_pivot["market_news_count"].rolling(7, min_periods=1).sum()
        )

    # Save
    out_path = "data/news_deep_historical.csv"
    df_pivot.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df_pivot)}, Columns: {len(df_pivot.columns)}")
    if len(df_pivot) > 0:
        print(f"  Date range: {df_pivot['date'].min()} → {df_pivot['date'].max()}")

    # Save latest sentiment JSON
    _save_sentiment_json(df_pivot)

    return df_pivot


def _save_sentiment_json(df_pivot):
    """Save latest sentiment as JSON for inference."""
    ticker_sentiment = {}
    for ticker in TICKER_QUERIES:
        col = f"sent_{ticker}"
        if col in df_pivot.columns:
            last = df_pivot[col].dropna().tail(5)
            if not last.empty:
                ticker_sentiment[ticker] = round(float(last.mean()), 4)

    mkt_vals = df_pivot.get("market_sentiment", pd.Series(dtype=float)).dropna().tail(5)
    market_sent = round(float(mkt_vals.mean()), 4) if not mkt_vals.empty else 0.0

    with open("data/news_sentiment.json", "w") as f:
        json.dump({
            "ticker_sentiment": ticker_sentiment,
            "market_sentiment": market_sent,
            "updated": date.today().isoformat(),
            "source": "RBC+Kommersant+Finam+Interfax+SmartLab (FinBERT+ruBERT)",
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved: data/news_sentiment.json ({len(ticker_sentiment)} tickers)")


# ================================================================
# MERGE WITH EXISTING HISTORICAL DATA
# ================================================================

def merge_with_existing(deep_path="data/news_deep_historical.csv",
                        existing_path="data/news_sentiment_historical_merged.csv",
                        output_path="data/news_sentiment_historical_merged.csv"):
    """
    Merge deep-collected data with existing historical data.
    Deep data takes priority where both exist.
    """
    dfs = []

    if os.path.exists(deep_path):
        df_deep = pd.read_csv(deep_path)
        df_deep["date"] = pd.to_datetime(df_deep["date"]).dt.strftime("%Y-%m-%d")
        dfs.append(("deep", df_deep))
        print(f"Deep data: {len(df_deep)} rows")

    if os.path.exists(existing_path) and existing_path != deep_path:
        df_old = pd.read_csv(existing_path)
        df_old["date"] = pd.to_datetime(df_old["date"]).dt.strftime("%Y-%m-%d")
        dfs.append(("existing", df_old))
        print(f"Existing data: {len(df_old)} rows")

    if not dfs:
        print("No data to merge!")
        return

    if len(dfs) == 1:
        df_merged = dfs[0][1]
    else:
        # Deep data first (priority), then fill with existing
        df_new = dfs[0][1].set_index("date")
        df_old = dfs[1][1].set_index("date")

        all_cols = set(df_new.columns) | set(df_old.columns)
        df_merged = pd.DataFrame(
            index=sorted(set(df_new.index) | set(df_old.index))
        )
        for col in sorted(all_cols):
            if col in df_new.columns and col in df_old.columns:
                df_merged[col] = df_new[col].combine_first(df_old[col])
            elif col in df_new.columns:
                df_merged[col] = df_new[col]
            else:
                df_merged[col] = df_old[col]

        df_merged = df_merged.reset_index().rename(columns={"index": "date"})

    df_merged.to_csv(output_path, index=False)
    print(f"\nMerged: {output_path}")
    print(f"  Rows: {len(df_merged)}, Columns: {len(df_merged.columns)}")

    # Update JSON
    _save_sentiment_json(df_merged)

    return df_merged


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Deep news collector with full-text analysis (FinBERT + ruBERT)"
    )
    parser.add_argument("--from", dest="date_from", default="2024-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--ticker", default=None, help="Single ticker")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching, only re-analyze cached articles")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip fetching full article text (headers only)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing historical data after analysis")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Seconds between requests")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else None

    print("=" * 65)
    print("Deep News Collector + FinBERT/ruBERT Ensemble Sentiment")
    print(f"Period: {args.date_from} → {args.date_to}")
    print(f"Sources: RBC API + Kommersant + Finam + Interfax + SmartLab")
    print(f"NLP: FinBERT (financial) + ruBERT (Russian) ensemble")
    print("=" * 65)

    if args.skip_fetch:
        cache = _load_cache()
        print(f"Using cached articles: {sum(len(v) for v in cache.values())}")
    else:
        cache = collect_articles(
            args.date_from, args.date_to,
            tickers=tickers,
            fetch_text=not args.no_text,
            sleep_between=args.sleep,
        )

    df = analyze_and_build(cache, args.date_from, args.date_to, tickers)

    if args.merge:
        print("\n" + "=" * 65)
        print("Merging with existing historical data...")
        merge_with_existing()

    # Stats
    if not df.empty:
        sent_cols = [c for c in df.columns
                     if c.startswith("sent_") and "_3d" not in c and "_7d" not in c]
        non_zero = sum((df[c].fillna(0) != 0).sum() for c in sent_cols)
        total_cells = sum(len(df) for _ in sent_cols)
        print(f"\nCoverage: {non_zero}/{total_cells} non-zero sentiment values "
              f"({non_zero/total_cells*100:.1f}%)")

    print("\nFiles:")
    print("  data/news_deep_cache.json              <- article cache")
    print("  data/news_deep_analyzed.csv             <- raw analyzed articles")
    print("  data/news_deep_historical.csv           <- daily sentiment (training)")
    print("  data/news_sentiment.json                <- latest (inference)")
    if args.merge:
        print("  data/news_sentiment_historical_merged.csv <- merged (training)")
