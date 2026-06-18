"""
frontend/app.py
═══════════════════════════════════════════════════════════════
Streamlit интерфейс для прогнозов MOEX.

КАК РАБОТАЕТ STREAMLIT:
- При любом действии пользователя весь скрипт выполняется заново
- @st.cache_data(ttl=120) — кэширует результат функции на 120 сек
- session_state — словарь сохраняется между запусками
- st.columns(4) — делит страницу на 4 колонки
- st.container(border=True) — карточка с рамкой

ЗАПУСК:
    streamlit run frontend/app.py

ВАЖНО: FastAPI должен быть запущен на :8000 в другом терминале.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests, os
from datetime import date
from dotenv import load_dotenv
load_dotenv()

# Адрес FastAPI сервера
API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Настройка страницы (должна быть ПЕРВОЙ командой Streamlit)
st.set_page_config(
    page_title             = "MOEX Predictor",
    page_icon              = "📈",
    layout                 = "wide",
    initial_sidebar_state  = "expanded",
)


# ═══════════════════════════════════════════════════════════════
# ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ С КЭШЕМ
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=120)
def fetch_predictions():
    """Последние прогнозы по всем акциям."""
    try:
        r = requests.get(f"{API}/api/predictions", timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_candles(
    ticker: str,
    interval: str = "1d",
    limit: int = 252,
    date_from: str | None = None,
    date_to: str | None = None,
):
    """Свечи для графика."""
    try:
        params = {"interval": interval, "limit": limit}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        r = requests.get(
            f"{API}/api/candles/{ticker}",
            params=params,
            timeout=10,
        )
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_indicators(ticker: str, limit: int = 60):
    """Индикаторы RSI/MACD."""
    try:
        r = requests.get(
            f"{API}/api/indicators/{ticker}",
            params={"limit": limit},
            timeout=10,
        )
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_securities():
    """Список акций."""
    try:
        r = requests.get(f"{API}/api/securities", timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_ticker_preds(ticker: str, days: int = 60):
    """История прогнозов для тикера."""
    try:
        r = requests.get(
            f"{API}/api/predictions/{ticker}",
            params={"days": days},
            timeout=10,
        )
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=120)
def fetch_hourly_forecast(ticker: str):
    """Почасовой прогноз на следующий торговый день."""
    try:
        r = requests.get(f"{API}/api/hourly-forecast/{ticker}", timeout=15)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════
# ЦВЕТА И ОФОРМЛЕНИЕ СИГНАЛОВ
# ═══════════════════════════════════════════════════════════════

SIG_COLOR = {"BUY": "#1D9E75", "SELL": "#E24B4A", "HOLD": "#888780"}
SIG_EMOJI = {"BUY": "↑ BUY", "SELL": "↓ SELL", "HOLD": "→ HOLD"}
SIG_RU = {
    "BUY":  "Рекомендуется покупка",
    "SELL": "Рекомендуется продажа",
    "HOLD": "Нейтрально — держать",
}


def delta_color(val):
    """Зелёный для роста, красный для падения."""
    return "#1D9E75" if val is not None and float(val) >= 0 else "#E24B4A"


TICKER_NAMES = {
    "SBER": "Сбербанк", "GAZP": "Газпром", "LKOH": "Лукойл",
    "GMKN": "Норникель", "YDEX": "Яндекс", "MGNT": "Магнит",
    "ROSN": "Роснефть", "NVTK": "Новатэк", "MTSS": "МТС",
    "PLZL": "Полюс Золото", "CHMF": "Северсталь", "ALRS": "АЛРОСА",
    "MOEX": "Мосбиржа", "VTBR": "ВТБ", "TATN": "Татнефть",
    "SNGS": "Сургутнефтегаз", "NLMK": "НЛМК", "POLY": "Полиметалл",
    "PHOR": "ФосАгро", "IRAO": "Интер РАО", "RUAL": "Русал",
    "MAGN": "ММК", "AFKS": "АФК Система", "PIKK": "ПИК",
    "OZON": "Озон", "TCSG": "Т-Банк", "FIVE": "X5 Group",
    "RTKM": "Ростелеком", "HYDR": "РусГидро", "FEES": "ФСК ЕЭС",
    "TRNFP": "Транснефть", "CBOM": "МКБ", "SMLT": "Самолёт",
    "ENPG": "Эн+ Груп", "FLOT": "Совкомфлот",
}


def get_ticker_label(ticker: str, securities: list[dict] | None = None) -> str:
    """Возвращает 'TICKER — Название' для отображения в UI."""
    # Сначала ищем в справочнике API
    if securities:
        for s in securities:
            if s.get("ticker") == ticker and s.get("short_name"):
                return f"{ticker} — {s['short_name']}"
    # Затем в локальном словаре
    name = TICKER_NAMES.get(ticker)
    return f"{ticker} — {name}" if name else ticker


def build_ticker_options(securities: list[dict], predictions: list[dict]) -> list[str]:
    """Собирает список тикеров из справочника и прогнозов, чтобы переход с карточек не сбрасывался."""
    seen = set()
    tickers = []
    for items in (securities, predictions):
        if isinstance(items, dict):
            items = items.get("data") or items.get("items") or []
        if not isinstance(items, list):
            continue
        for source in items:
            if not isinstance(source, dict):
                continue
            ticker = (source.get("ticker") or "").strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return sorted(tickers) or ["SBER"]


# ═══════════════════════════════════════════════════════════════
# САЙДБАР (левая панель)
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📈 MOEX Predictor")
    st.caption("Прогнозирование акций РФ")
    st.divider()

    page = st.radio(
        "Навигация",
        [
            "🏠 Дашборд",
            "📊 Анализ акции",
            "📋 История прогнозов",
            "📈 Точность модели",
            "ℹ️ О системе",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Источник: MOEX ISS API")
    st.caption("Модель: LightGBM + LSTM")
    st.caption("Обновление: 20:00 МСК")

    if st.button("🔄 Обновить данные", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**🔬 Режим исследования**")
    research_mode = st.toggle("Выбрать дату анализа", value=False)
    if research_mode:
        research_date = st.date_input(
            "Дата анализа (as of)",
            value=date.today(),
            min_value=date(2022, 1, 1),
            max_value=date.today(),
            format="DD.MM.YYYY",
            key="research_date_input",
        )
        st.session_state["research_date"] = research_date.isoformat()
        st.caption(f"Прогноз: на следующий торговый день после {research_date:%d.%m.%Y}")
    else:
        st.session_state.pop("research_date", None)

    # Перенаправление с карточек
    if st.session_state.get("goto_page"):
        page = st.session_state.pop("goto_page")


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА 1: ДАШБОРД
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Дашборд":
    st.title("Прогнозы на следующий торговый день")
    st.caption(f"Сформировано: {date.today():%d.%m.%Y}")

    preds = fetch_predictions()

    if not preds:
        st.warning(
            "Прогнозы пока не сформированы.\n\n"
            "Это нормально — модели ещё не обучены. "
            "После обучения в DataSphere прогнозы появятся автоматически."
        )

        # Показываем хотя бы список акций из справочника
        st.divider()
        st.subheader("Отслеживаемые акции")
        secs = fetch_securities()
        if secs:
            df_secs = pd.DataFrame(secs)
            st.dataframe(
                df_secs,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker":     "Тикер",
                    "short_name": "Название",
                    "sector":     "Сектор",
                    "is_active":  "Активна",
                },
            )
        st.stop()

    # ── Фильтры ──
    c1, c2, c3 = st.columns(3)
    with c1:
        sig_filter = st.multiselect(
            "Сигнал", ["BUY", "SELL", "HOLD"],
            default=["BUY", "SELL", "HOLD"],
        )
    with c2:
        sectors = sorted({p.get("sector", "—") for p in preds})
        sec_filter = st.multiselect(
            "Сектор", sectors, default=sectors,
        )
    with c3:
        min_conf = st.slider("Мин. уверенность", 0.0, 1.0, 0.0, 0.05)

    filtered = [
        p for p in preds
        if p.get("signal") in sig_filter
        and p.get("sector") in sec_filter
        and (p.get("confidence") or 0) >= min_conf
    ]

    # ── Метрики ──
    st.divider()
    buys  = sum(1 for p in filtered if p.get("signal") == "BUY")
    sells = sum(1 for p in filtered if p.get("signal") == "SELL")
    holds = sum(1 for p in filtered if p.get("signal") == "HOLD")
    avg_conf = (
        sum(p.get("confidence", 0) for p in filtered) / max(len(filtered), 1) * 100
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("↑ BUY",  buys)
    m2.metric("↓ SELL", sells)
    m3.metric("→ HOLD", holds)
    m4.metric("Средняя уверенность", f"{avg_conf:.0f}%")
    st.divider()

    # ── Карточки акций ──
    sorted_p = sorted(
        filtered,
        key=lambda p: (p.get("signal") != "BUY", -(p.get("confidence") or 0)),
    )

    for row_start in range(0, len(sorted_p), 4):
        row_preds = sorted_p[row_start:row_start + 4]
        cols = st.columns(4)
        for col, pred in zip(cols, row_preds):
            t      = pred.get("ticker", "—")
            sig    = pred.get("signal", "HOLD")
            delta  = pred.get("predicted_delta", 0) or 0
            conf   = (pred.get("confidence", 0) or 0) * 100
            price  = pred.get("current_close")
            sc     = SIG_COLOR[sig]
            ds     = "+" if delta >= 0 else ""

            with col:
                with st.container(border=True):
                    _card_name = pred.get("short_name") or TICKER_NAMES.get(t, t)
                    ca, cb = st.columns([2, 1])
                    ca.markdown(f"**{t}** · {_card_name}")
                    cb.markdown(
                        f'<span style="color:{sc};font-weight:600;'
                        f'font-size:13px">{SIG_EMOJI[sig]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(_card_name)
                    if price:
                        st.markdown(f"**{float(price):,.1f} ₽**")
                    st.markdown(
                        f'<span style="color:{delta_color(delta)};font-size:15px">'
                        f'{ds}{delta:.1f}%</span>'
                        f'<span style="color:gray;font-size:12px"> · '
                        f'{conf:.0f}%</span>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Подробнее →", key=f"btn_{t}",
                                 use_container_width=True):
                        st.session_state["sel_ticker"] = t
                        st.session_state["goto_page"] = "📊 Анализ акции"
                        st.rerun()


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА 2: АНАЛИЗ АКЦИИ
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Анализ акции":
    secs = fetch_securities()
    preds = fetch_predictions()
    ticker_list = build_ticker_options(secs, preds)

    sel_default = st.session_state.get("sel_ticker", ticker_list[0])
    def_idx = ticker_list.index(sel_default) if sel_default in ticker_list else 0

    # Показываем "TICKER — Название" в выпадающем списке
    ticker_labels = [get_ticker_label(t, secs) for t in ticker_list]
    selected_label = st.selectbox("Выберите акцию", ticker_labels, index=def_idx)
    ticker = selected_label.split(" — ")[0].strip()
    st.session_state["sel_ticker"] = ticker

    # Режим исследования — показываем данные за выбранную дату
    research_date = st.session_state.get("research_date")
    if research_date:
        try:
            import exchange_calendars as xcals
            _moex = xcals.get_calendar("XMOS")
            next_td = _moex.next_session(research_date).date()
        except Exception:
            next_td = (pd.to_datetime(research_date) + pd.Timedelta(days=1)).date()
            while next_td.weekday() >= 5:
                next_td += pd.Timedelta(days=1)
        st.info(
            f"🔬 Режим исследования: анализ за **{pd.to_datetime(research_date):%d.%m.%Y}** "
            f"→ прогноз на **{next_td:%d.%m.%Y}**"
        )
        # Ищем исторический прогноз за эту дату
        hist = fetch_ticker_preds(ticker, 120)
        pred = next(
            (p for p in hist if str(p.get("target_date", "")).startswith(
                next_td.isoformat())),
            None
        )
        if pred is None:
            st.warning(f"Сохранённого прогноза на {next_td:%d.%m.%Y} нет в базе.")
    else:
        research_date = None
        # Если есть прогноз — показываем
        pred = next((p for p in preds if p["ticker"] == ticker), None)

    # ── Шапка: цена + сигнал ──────────────────────────────────────
    # Быстро загружаем последнюю свечу для актуальной цены
    _quick = fetch_candles(ticker, "1d", 2, date_to=research_date if research_date else None)
    _curr_close = float(pd.DataFrame(_quick)["close"].iloc[-1]) if _quick else None
    _prev_close = float(pd.DataFrame(_quick)["close"].iloc[-2]) if _quick and len(_quick) >= 2 else None
    _day_chg    = (_curr_close / _prev_close - 1) * 100 if (_curr_close and _prev_close) else None
    _chg_color  = "#1D9E75" if (_day_chg or 0) >= 0 else "#E24B4A"
    _chg_sign   = "+" if (_day_chg or 0) >= 0 else ""

    # Название компании из справочника
    _sec_info   = next((s for s in fetch_securities() if s.get("ticker") == ticker), {})
    _company    = _sec_info.get("short_name", ticker)

    st.markdown(
        f'<div style="margin-bottom:4px">'
        f'<span style="font-size:28px;font-weight:700">{ticker}</span>'
        f'<span style="font-size:16px;color:#888;margin-left:10px">{_company}</span>'
        f'</div>'
        f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:16px">'
        + (f'<span style="font-size:36px;font-weight:700">{_curr_close:,.1f} ₽</span>'
           f'<span style="font-size:18px;color:{_chg_color};font-weight:600">'
           f'{_chg_sign}{_day_chg:.2f}%</span>'
           f'<span style="font-size:13px;color:#aaa">за день</span>'
           if _curr_close else '') +
        f'</div>',
        unsafe_allow_html=True,
    )

    if pred:
        sig   = pred.get("signal", "HOLD")
        delta = pred.get("predicted_delta", 0) or 0
        conf  = (pred.get("confidence", 0) or 0) * 100
        sc    = SIG_COLOR[sig]
        ds    = "+" if delta >= 0 else ""

        # Баннер прогноза
        bg = (
            "rgba(29,158,117,.1)" if sig == "BUY"
            else "rgba(226,75,74,.1)" if sig == "SELL"
            else "rgba(136,135,128,.1)"
        )
        st.markdown(
            f'<div style="background:{bg};border-left:4px solid {sc};'
            f'padding:14px 20px;border-radius:8px;margin-bottom:16px;'
            f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
            f'<div>'
            f'<div style="font-size:11px;color:{sc};font-weight:600;text-transform:uppercase;margin-bottom:4px">'
            f'Прогноз модели · на {pred.get("target_date", "следующий день")}</div>'
            f'<div style="font-size:20px;font-weight:700;color:{sc}">{SIG_RU[sig]}</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:22px;font-weight:700;color:{delta_color(delta)}">'
            f'{ds}{delta:.1f}%</div>'
            f'<div style="font-size:12px;color:#888">уверенность модели: <b>{conf:.0f}%</b></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Факторы влияния (бывший SHAP) ──
        tf = pred.get("top_features") or {}
        if tf:
            feat_names = {
                # Технические индикаторы
                "rsi_14":           ("RSI (14 дней)",          "Индекс относительной силы — показывает перекупленность (>70) или перепроданность (<30) акции"),
                "macd_hist":        ("MACD гистограмма",       "Разница между быстрой и медленной скользящими средними — сигнал смены тренда"),
                "macd":             ("MACD линия",             "Разность EMA(12) и EMA(26) — основной трендовый осциллятор"),
                "macd_signal":      ("MACD сигнальная",        "Сглаженная линия MACD — пересечения дают сигналы покупки/продажи"),
                "bb_position":      ("Полосы Боллинджера",     "Положение цены в полосах Боллинджера — 0 = нижняя граница, 1 = верхняя"),
                "bb_width":         ("Ширина полос Боллинджера", "Ширина канала — узкие полосы предшествуют сильному движению"),
                "atr_14":           ("Волатильность ATR",      "Средний истинный диапазон за 14 дней — мера дневных колебаний цены"),
                "vol_ratio":        ("Объём / средний",        "Текущий объём торгов относительно 20-дневного среднего — высокий подтверждает движение"),
                "volume":           ("Объём торгов",           "Количество проторгованных акций за день"),
                "sma_20":           ("SMA (20 дней)",          "Простая скользящая средняя за 20 дней — краткосрочный тренд"),
                "ema_12":           ("EMA (12 дней)",          "Экспоненциальная скользящая средняя — быстрый тренд"),
                "ema_26":           ("EMA (26 дней)",          "Экспоненциальная скользящая средняя — медленный тренд"),
                "stoch_k":          ("Стохастик %K",           "Быстрый стохастический осциллятор — текущая цена в диапазоне за 14 дней"),
                "stoch_d":          ("Стохастик %D",           "Сглаженный стохастик — сигнальная линия"),
                "williams_r":       ("Williams %R",            "Осциллятор перекупленности/перепроданности (-100 до 0)"),
                "obv":              ("OBV (баланс. объём)",    "Кумулятивный объём с учётом направления цены — подтверждение тренда"),
                "adx_14":           ("ADX (сила тренда)",      "Средний индекс направленного движения — >25 = сильный тренд"),
                "cci_14":           ("CCI (14 дней)",          "Индекс товарного канала — отклонение цены от среднего"),

                # Изменения цены (returns)
                "close_pct_1":      ("Изменение за 1 день",    "Процентное изменение цены за предыдущий торговый день"),
                "close_pct_3":      ("Изменение за 3 дня",     "Процентное изменение цены за последние 3 торговых дня"),
                "close_pct_5":      ("Изменение за 5 дней",    "Процентное изменение цены за последнюю неделю (5 торговых дней)"),
                "ret_1d":           ("Доходность 1 день",      "Однодневная доходность акции"),
                "ret_3d":           ("Доходность 3 дня",       "Доходность за 3 торговых дня"),
                "ret_5d":           ("Доходность 5 дней",      "Доходность за неделю — показывает краткосрочный импульс"),
                "ret_10d":          ("Доходность 10 дней",     "Доходность за 2 недели — среднесрочный тренд"),
                "ret_20d":          ("Доходность 20 дней",     "Доходность за месяц — оценка текущего тренда"),
                "log_ret_1":        ("Лог-доходность 1 день",  "Логарифмическая доходность за день — нормализованная мера изменения"),

                # Волатильность
                "volatility_10":    ("Волатильность 10 дней",  "Стандартное отклонение дневной доходности за 10 дней"),
                "volatility_20":    ("Волатильность 20 дней",  "Стандартное отклонение дневной доходности за 20 дней"),
                "volatility_5":     ("Волатильность 5 дней",   "Краткосрочная волатильность за неделю"),

                # Ценовые уровни
                "high_low_range":   ("Дневной диапазон",       "Разница между максимумом и минимумом дня в процентах"),
                "close_to_high":    ("Цена к максимуму",       "Насколько близко закрытие к дневному максимуму (0–1)"),
                "close_to_low":     ("Цена к минимуму",        "Насколько близко закрытие к дневному минимуму (0–1)"),
                "gap":              ("Гэп открытия",           "Разница между ценой открытия и вчерашним закрытием"),
                "upper_shadow":     ("Верхняя тень свечи",     "Длина верхнего фитиля — давление продавцов"),
                "lower_shadow":     ("Нижняя тень свечи",      "Длина нижнего фитиля — давление покупателей"),
                "body_ratio":       ("Тело свечи",             "Размер тела свечи к полному диапазону — решительность движения"),

                # Новости и сентимент
                "news_sentiment":   ("Настроение новостей",    "Тональность новостей по акции — положительная (рост) или отрицательная (падение)"),
                "market_sentiment": ("Настроение рынка",       "Общий тон рыночных новостей по всем отслеживаемым акциям"),
                "sentiment_ma_3":   ("Сентимент (ср. 3 дня)",  "Сглаженное настроение новостей за 3 дня"),
                "news_count":       ("Кол-во новостей",        "Число новостей за день — повышенное внимание к акции"),

                # Макроэкономика
                "imoex":            ("Индекс IMOEX",           "Значение индекса Мосбиржи — общий уровень рынка"),
                "imoex_ret":        ("Изменение IMOEX",        "Дневная доходность индекса Мосбиржи — движение всего рынка"),
                "usd_rub":          ("Курс USD/RUB",           "Курс доллара к рублю — влияет на экспортёров и импортёров"),
                "usd_rub_ret":      ("Изменение USD/RUB",      "Дневное изменение курса доллара — валютный риск"),
                "brent":            ("Нефть Brent",            "Цена нефти Brent — ключевой фактор для нефтегазовых компаний"),
                "brent_ret":        ("Изменение Brent",        "Дневное изменение цены нефти"),
                "gold":             ("Золото",                 "Цена золота — индикатор риск-аппетита и хедж"),
                "key_rate":         ("Ключевая ставка ЦБ",     "Ставка Банка России — влияет на стоимость кредитов и привлекательность акций"),

                # Фундаментальные
                "pe_ratio":         ("P/E мультипликатор",     "Цена/Прибыль — дорого (>15) или дёшево (<10) оценена акция"),
                "pb_ratio":         ("P/B мультипликатор",     "Цена/Балансовая стоимость — оценка относительно активов"),
                "div_yield":        ("Дивидендная доходность",  "Годовая дивидендная доходность акции в процентах"),
                "market_cap":       ("Капитализация",          "Рыночная стоимость компании — размер бизнеса"),
                "P/E":              ("P/E мультипликатор",     "Цена/Прибыль — оценка стоимости акции"),

                # Календарные
                "month":            ("Месяц",                  "Номер месяца — сезонные эффекты (январский эффект, дивидендный сезон)"),
                "day_of_week":      ("День недели",            "День недели — статистические паттерны (эффект понедельника и т.п.)"),
                "day_of_month":     ("День месяца",            "Число месяца — начало/конец месяца влияет на потоки"),
                "quarter":          ("Квартал",                "Квартал года — отчётные периоды и ребалансировки"),
                "is_month_end":     ("Конец месяца",           "Последние дни месяца — ребалансировка фондов"),
                "is_month_start":   ("Начало месяца",          "Первые дни месяца — приток ликвидности"),
                "week_of_year":     ("Неделя года",            "Номер недели в году"),
            }
            with st.expander("📊 Факторы влияния на прогноз", expanded=True):
                st.caption(
                    "Модель объясняет своё решение через вклад каждого индикатора. "
                    "Чем длиннее полоса — тем сильнее этот фактор повлиял на прогноз."
                )
                for feat, imp in list(tf.items())[:10]:
                    # Fallback для неизвестных признаков: показываем имя переменной
                    name, desc = feat_names.get(
                        feat,
                        (feat.replace("_", " ").title(), "Технический признак модели"),
                    )
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(
                            min(float(imp), 1.0),
                            text=f"**{name}** — {desc}",
                        )
                    with col_b:
                        st.markdown(
                            f'<div style="text-align:right;padding-top:6px;'
                            f'font-size:13px;color:#888">вклад: {imp:.3f}</div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.info(
            f"Прогноза для {ticker} пока нет. "
            "Появится после обучения моделей в DataSphere."
        )

    st.divider()

    # ── Графики и индикаторы ──
    tab1, tab2, tab3, tab4 = st.tabs(["📈 График цены", "📉 Индикаторы", "🎯 Точность", "📅 Сезонность"])

    with tab1:
        # ── Период в стиле TradingView ──
        PERIODS = ["1 день", "5 дней", "1 мес", "3 мес", "6 мес", "1 год", "2 года"]
        if "chart_period" not in st.session_state:
            st.session_state["chart_period"] = "6 мес"
        if "chart_type" not in st.session_state:
            st.session_state["chart_type"] = "Линия"

        pcols = st.columns([1]*len(PERIODS) + [2])
        for i, p in enumerate(PERIODS):
            with pcols[i]:
                active = st.session_state["chart_period"] == p
                if st.button(
                    p,
                    key=f"period_{p}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state["chart_period"] = p
                    st.rerun()
        with pcols[-1]:
            chart_type = st.radio(
                "", ["Линия", "Свечи"],
                index=0 if st.session_state["chart_type"] == "Линия" else 1,
                horizontal=True, label_visibility="collapsed",
                key="chart_type_radio",
            )
            st.session_state["chart_type"] = chart_type

        period  = st.session_state["chart_period"]

        period_map = {
            "1 день":  ("1h", 10),
            "5 дней":  ("1h", 40),
            "1 мес":   ("1d", 22),
            "3 мес":   ("1d", 66),
            "6 мес":   ("1d", 130),
            "1 год":   ("1d", 252),
            "2 года":  ("1d", 504),
        }
        interval, limit = period_map[period]
        candles_data = fetch_candles(
            ticker, interval, limit,
            date_to=research_date if research_date else None
        )

        if candles_data:
            df_c = pd.DataFrame(candles_data)
            df_c["time"] = pd.to_datetime(df_c["time"])
            for col in ["open", "high", "low", "close", "volume"]:
                df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

            inds_for_chart = fetch_indicators(ticker, min(limit, 500)) if interval == "1d" else []
            df_overlay = pd.DataFrame(inds_for_chart) if inds_for_chart else pd.DataFrame()
            if not df_overlay.empty:
                df_overlay["time"] = pd.to_datetime(df_overlay["time"])
                for col in ["sma_20", "bb_upper", "bb_lower"]:
                    if col in df_overlay:
                        df_overlay[col] = pd.to_numeric(df_overlay[col], errors="coerce")

            # Цвет графика — красный если падает, зелёный если растёт
            _first = float(df_c["close"].iloc[0])
            _last  = float(df_c["close"].iloc[-1])
            _line_color = "#1D9E75" if _last >= _first else "#E24B4A"
            _fill_color = "rgba(29,158,117,.12)" if _last >= _first else "rgba(226,75,74,.10)"

            # Горизонтальная линия закрытия предыдущего дня
            _prev_day_close = float(df_c["close"].iloc[-2]) if len(df_c) >= 2 else None

            fig = go.Figure()

            if chart_type == "Свечи":
                candle_hover = [
                    f"<b>{time:%d.%m.%Y %H:%M}</b><br>"
                    f"Открытие: {open_:.2f}<br>"
                    f"Максимум: {high:.2f}<br>"
                    f"Минимум: {low:.2f}<br>"
                    f"Закрытие: {close:.2f}"
                    for time, open_, high, low, close in zip(
                        df_c["time"], df_c["open"], df_c["high"], df_c["low"], df_c["close"]
                    )
                ]
                fig.add_trace(go.Candlestick(
                    x = df_c["time"],
                    open  = df_c["open"],   high  = df_c["high"],
                    low   = df_c["low"],    close = df_c["close"],
                    name  = ticker,
                    increasing_line_color = "#1D9E75",
                    decreasing_line_color = "#E24B4A",
                    increasing_fillcolor  = "#1D9E75",
                    decreasing_fillcolor  = "#E24B4A",
                    text=candle_hover,
                    hoverinfo="text",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x = df_c["time"], y = df_c["close"],
                    mode = "lines", name = ticker,
                    line = dict(color=_line_color, width=2.2),
                    fill = "tozeroy",
                    fillcolor = _fill_color,
                    hovertemplate="<b>%{x|%d.%m.%Y %H:%M}</b><br>%{y:,.2f} ₽<extra></extra>",
                ))
                # Линия закрытия предыдущего дня (как в TradingView)
                if _prev_day_close and interval in ("1h", "1d"):
                    fig.add_hline(
                        y=_prev_day_close,
                        line_dash="dot",
                        line_color="rgba(150,150,150,0.5)",
                        line_width=1,
                        annotation_text=f"Закрытие предыдущего дня  {_prev_day_close:,.1f}",
                        annotation_position="bottom right",
                        annotation_font_size=11,
                        annotation_font_color="rgba(150,150,150,0.8)",
                    )

            if not df_overlay.empty:
                fig.add_trace(go.Scatter(
                    x=df_overlay["time"], y=df_overlay["sma_20"],
                    mode="lines", name="SMA 20",
                    line=dict(color="#D79A2B", width=1.6),
                    hovertemplate="SMA 20: %{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df_overlay["time"], y=df_overlay["bb_upper"],
                    mode="lines", name="Bollinger верх",
                    line=dict(color="rgba(92,103,125,.45)", width=1, dash="dot"),
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=df_overlay["time"], y=df_overlay["bb_lower"],
                    mode="lines", name="Bollinger низ",
                    line=dict(color="rgba(92,103,125,.45)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(92,103,125,.06)",
                    hoverinfo="skip",
                ))

            if interval == "1d":
                fig.add_trace(go.Scatter(
                    x=df_c["time"],
                    y=df_c["close"],
                    mode="markers",
                    name="Выбор дня",
                    customdata=df_c["time"].dt.date.astype(str),
                    marker=dict(size=24, color="rgba(0,0,0,0.01)"),
                    hoverinfo="skip",
                    showlegend=False,
                ))

            fig.update_layout(
                height=560,
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=12, t=18, b=0),
                legend=dict(
                    orientation="h",
                    y=1.02,
                    x=0,
                    bgcolor="rgba(255,255,255,.72)",
                ),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hoverlabel=dict(bgcolor="white", bordercolor="rgba(92,103,125,.25)"),
            )
            fig.update_yaxes(
                title_text="Цена, ₽",
                fixedrange=False,
                gridcolor="rgba(92,103,125,.12)",
                zeroline=False,
                showline=True,
                linecolor="rgba(92,103,125,.18)",
            )
            fig.update_xaxes(
                gridcolor="rgba(92,103,125,.08)",
                showline=True,
                linecolor="rgba(92,103,125,.18)",
            )
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            if interval == "1h":
                fig.update_xaxes(rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[19, 10], pattern="hour"),
                ])
            chart_state = None
            try:
                chart_state = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"price_chart_{ticker}_{interval}_{limit}",
                    on_select="rerun",
                    selection_mode="points",
                )
            except TypeError:
                st.plotly_chart(fig, use_container_width=True)

            if pred and candles_data:
                last_c = float(df_c["close"].iloc[-1])
                delta = pred.get("predicted_delta", 0) or 0
                pred_c = float(pred.get("predicted_close") or last_c * (1 + delta / 100))
                signal = pred.get("signal", "HOLD")
                sc2 = SIG_COLOR.get(signal, SIG_COLOR["HOLD"])
                target_date = pred.get("target_date")
                if not target_date:
                    target_date = (df_c["time"].iloc[-1] + pd.offsets.BDay(1)).date().isoformat()

                hourly_forecast = fetch_hourly_forecast(ticker)
                hourly_points = hourly_forecast.get("points") or []
                st.markdown("**Прогноз на следующий торговый день**")
                fc1, fc2 = st.columns([1, 2])

                if hourly_points:
                    df_fh = pd.DataFrame(hourly_points)
                    df_fh["time"] = pd.to_datetime(df_fh["time"])
                    df_fh["predicted_close"] = pd.to_numeric(df_fh["predicted_close"], errors="coerce")
                    df_fh["cumulative_delta"] = pd.to_numeric(df_fh["cumulative_delta"], errors="coerce")
                    final_close = float(df_fh["predicted_close"].iloc[-1])
                    final_delta = float(df_fh["cumulative_delta"].iloc[-1])

                    with fc1:
                        st.metric("Старт", f"{last_c:,.2f} ₽")
                        st.metric("К закрытию", f"{final_close:,.2f} ₽", f"{final_delta:+.2f}%")
                    with fc2:
                        forecast_fig = go.Figure()
                        forecast_fig.add_trace(go.Scatter(
                            x=df_fh["time"],
                            y=df_fh["predicted_close"],
                            mode="lines+markers",
                            name="Почасовой прогноз",
                            line=dict(color=sc2, width=2.8),
                            marker=dict(size=7, color=sc2),
                            hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.2f} ₽<extra></extra>",
                        ))
                        forecast_fig.add_hline(
                            y=last_c,
                            line_dash="dot",
                            line_color="rgba(92,103,125,.45)",
                            annotation_text="текущая цена",
                            annotation_position="bottom right",
                        )
                        forecast_fig.update_layout(
                            height=300,
                            xaxis_rangeslider_visible=False,
                            margin=dict(l=0, r=12, t=10, b=0),
                            showlegend=False,
                            hovermode="x unified",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            hoverlabel=dict(bgcolor="white", bordercolor="rgba(92,103,125,.25)"),
                        )
                        forecast_fig.update_yaxes(
                            title_text="Цена, ₽",
                            gridcolor="rgba(92,103,125,.12)",
                            zeroline=False,
                            showline=True,
                            linecolor="rgba(92,103,125,.18)",
                        )
                        forecast_fig.update_xaxes(
                            title_text=f"{pd.to_datetime(hourly_forecast.get('target_date')):%d.%m.%Y}",
                            gridcolor="rgba(92,103,125,.08)",
                            tickformat="%H:%M",
                            showline=True,
                            linecolor="rgba(92,103,125,.18)",
                        )
                        st.plotly_chart(forecast_fig, use_container_width=True)
                else:
                    with fc1:
                        st.metric("Текущая цена", f"{last_c:,.2f} ₽")
                        st.metric("Ожидаемая цена", f"{pred_c:,.2f} ₽", f"{delta:+.2f}%")
                    with fc2:
                        st.caption("Почасовой прогноз недоступен: нужны загруженные 1h-свечи и lstm_hourly модель.")
                        forecast_fig = go.Figure()
                        forecast_fig.add_trace(go.Scatter(
                            x=["Сейчас", f"Цель {pd.to_datetime(target_date):%d.%m.%Y}"],
                            y=[last_c, pred_c],
                            mode="lines+markers+text",
                            name="Прогноз",
                            line=dict(color=sc2, width=3),
                            marker=dict(size=12, color=[SIG_COLOR["HOLD"], sc2]),
                            text=[f"{last_c:,.2f} ₽", f"{pred_c:,.2f} ₽"],
                            textposition=["bottom center", "top center"],
                            hovertemplate="%{x}<br>%{y:.2f} ₽<extra></extra>",
                        ))
                        forecast_fig.update_layout(
                            height=230,
                            xaxis_rangeslider_visible=False,
                            margin=dict(l=0, r=12, t=10, b=0),
                            showlegend=False,
                            hovermode="x unified",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            hoverlabel=dict(bgcolor="white", bordercolor="rgba(92,103,125,.25)"),
                        )
                        forecast_fig.update_yaxes(
                            title_text="Цена, ₽",
                            gridcolor="rgba(92,103,125,.12)",
                            zeroline=False,
                            showline=True,
                            linecolor="rgba(92,103,125,.18)",
                        )
                        forecast_fig.update_xaxes(showgrid=False)
                        st.plotly_chart(forecast_fig, use_container_width=True)

            if interval == "1d":
                detail_key = f"intraday_day_{ticker}"
                if isinstance(chart_state, dict):
                    selection = chart_state.get("selection")
                else:
                    selection = getattr(chart_state, "selection", None)
                if isinstance(selection, dict):
                    points = selection.get("points", [])
                else:
                    points = getattr(selection, "points", []) if selection is not None else []
                if points:
                    selected_raw = points[0].get("customdata") or points[0].get("x")
                    if selected_raw:
                        selected_from_chart = pd.to_datetime(selected_raw).date().isoformat()
                        st.session_state[detail_key] = selected_from_chart

                day_options = list(dict.fromkeys(df_c["time"].dt.date.astype(str).tolist()))
                selected_day = st.session_state.get(detail_key, day_options[-1])
                if selected_day not in day_options:
                    selected_day = day_options[-1]

                min_day = pd.to_datetime(day_options[0]).date()
                max_day = pd.to_datetime(day_options[-1]).date()
                typed_day = st.date_input(
                    "Дата для часовой детализации",
                    value=pd.to_datetime(selected_day).date(),
                    min_value=min_day,
                    max_value=max_day,
                    format="DD.MM.YYYY",
                    key=f"day_input_{ticker}_{selected_day}",
                )
                selected_day = typed_day.isoformat()
                st.session_state[detail_key] = selected_day

                next_day = (pd.to_datetime(selected_day) + pd.Timedelta(days=1)).date().isoformat()
                hourly_data = fetch_candles(
                    ticker,
                    "1h",
                    200,
                    date_from=selected_day,
                    date_to=next_day,
                )

                if hourly_data:
                    df_h = pd.DataFrame(hourly_data)
                    df_h["time"] = pd.to_datetime(df_h["time"])
                    for col in ["open", "high", "low", "close", "volume"]:
                        df_h[col] = pd.to_numeric(df_h[col], errors="coerce")

                    hfig = go.Figure()
                    hourly_hover = [
                        f"<b>{time:%d.%m.%Y %H:%M}</b><br>"
                        f"Открытие: {open_:.2f}<br>"
                        f"Максимум: {high:.2f}<br>"
                        f"Минимум: {low:.2f}<br>"
                        f"Закрытие: {close:.2f}"
                        for time, open_, high, low, close in zip(
                            df_h["time"], df_h["open"], df_h["high"], df_h["low"], df_h["close"]
                        )
                    ]
                    hfig.add_trace(go.Candlestick(
                        x=df_h["time"],
                        open=df_h["open"],
                        high=df_h["high"],
                        low=df_h["low"],
                        close=df_h["close"],
                        name=f"{ticker} 1ч",
                        increasing_line_color="#1D9E75",
                        decreasing_line_color="#E24B4A",
                        increasing_fillcolor="#1D9E75",
                        decreasing_fillcolor="#E24B4A",
                        text=hourly_hover,
                        hoverinfo="text",
                    ))
                    hfig.update_layout(
                        height=390,
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=0, r=12, t=22, b=0),
                        legend=dict(
                            orientation="h",
                            y=1.03,
                            x=0,
                            bgcolor="rgba(255,255,255,.72)",
                        ),
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        hoverlabel=dict(bgcolor="white", bordercolor="rgba(92,103,125,.25)"),
                    )
                    hfig.update_yaxes(
                        title_text="Цена, ₽",
                        gridcolor="rgba(92,103,125,.12)",
                        zeroline=False,
                        showline=True,
                        linecolor="rgba(92,103,125,.18)",
                    )
                    hfig.update_xaxes(
                        gridcolor="rgba(92,103,125,.08)",
                        showline=True,
                        linecolor="rgba(92,103,125,.18)",
                    )
                    hfig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
                    st.plotly_chart(hfig, use_container_width=True)
                else:
                    st.info(f"Для {ticker} за {selected_day} нет часовых свечей.")
        else:
            st.warning("Нет данных свечей")

    with tab2:
        inds = fetch_indicators(ticker, 60)
        if inds:
            df_i = pd.DataFrame(inds)
            df_i["time"] = pd.to_datetime(df_i["time"])

            # RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_i["time"], y=df_i["rsi_14"],
                name="RSI(14)", line=dict(color="#7F77DD", width=1.5),
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#E24B4A",
                              opacity=0.5, annotation_text="70 (перекуплен)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#1D9E75",
                              opacity=0.5, annotation_text="30 (перепродан)")
            fig_rsi.update_layout(
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.subheader("RSI (Relative Strength Index)")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD
            mh = df_i["macd_hist"].astype(float)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(
                x=df_i["time"], y=mh, name="MACD гистограмма",
                marker_color=["#1D9E75" if v > 0 else "#E24B4A" for v in mh],
            ))
            fig_macd.update_layout(
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.subheader("MACD гистограмма")
            st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.warning("Нет данных индикаторов")

    with tab3:
        hp = fetch_ticker_preds(ticker, 60)
        if hp:
            df_hp = pd.DataFrame(hp)
            df_hp = df_hp.dropna(subset=["is_correct"])
            if not df_hp.empty:
                acc = df_hp["is_correct"].mean() * 100
                a1, a2, a3 = st.columns(3)
                a1.metric("Точность направления", f"{acc:.0f}%")
                a2.metric("Всего прогнозов", len(df_hp))
                buy_df = df_hp[df_hp["signal"] == "BUY"]
                if len(buy_df) > 0:
                    a3.metric("Точность BUY",
                              f"{buy_df['is_correct'].mean()*100:.0f}%")
            else:
                st.info("Результаты появятся после первого торгового дня.")
        else:
            st.info("Истории прогнозов пока нет.")

    with tab4:
        st.subheader("Историческая сезонная динамика")
        st.caption(
            "Показывает среднюю доходность акции по месяцам и дням недели "
            "на основе исторических данных. Модель использует эти паттерны при обучении "
            "(признаки month, day_of_week). Сезонность основана на прошлых данных "
            "и не является гарантированным прогнозом."
        )

        # Загружаем 3 года истории для надёжной статистики
        seas_candles = fetch_candles(ticker, "1d", 756)
        if seas_candles and len(seas_candles) >= 60:
            df_s = pd.DataFrame(seas_candles)
            df_s["time"]  = pd.to_datetime(df_s["time"])
            df_s["close"] = pd.to_numeric(df_s["close"], errors="coerce")
            df_s["ret"]   = df_s["close"].pct_change() * 100
            df_s["month"] = df_s["time"].dt.month
            df_s["dow"]   = df_s["time"].dt.dayofweek  # 0=пн

            MONTHS_RU = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",
                         7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}
            DAYS_RU   = {0:"Пн",1:"Вт",2:"Ср",3:"Чт",4:"Пт"}

            # ── Средняя доходность по месяцам ──
            monthly = (
                df_s.dropna(subset=["ret"])
                .groupby("month")["ret"]
                .agg(["mean","count","std"])
                .reset_index()
            )
            monthly["label"] = monthly["month"].map(MONTHS_RU)
            monthly["color"] = monthly["mean"].apply(
                lambda x: "#1D9E75" if x >= 0 else "#E24B4A"
            )

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=monthly["label"],
                y=monthly["mean"].round(2),
                marker_color=monthly["color"],
                text=monthly["mean"].apply(lambda x: f"{x:+.2f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Средняя доходность: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
                name="Средняя доходность",
            ))
            fig_m.add_hline(y=0, line_color="rgba(150,150,150,0.4)", line_width=1)
            fig_m.update_layout(
                title=dict(text="Средняя доходность по месяцам", font_size=15),
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis=dict(
                    ticksuffix="%",
                    gridcolor="rgba(92,103,125,.1)",
                    zeroline=False,
                ),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_m, use_container_width=True)

            # ── Средняя доходность по дням недели ──
            daily_dow = (
                df_s[df_s["dow"].isin([0,1,2,3,4])]
                .dropna(subset=["ret"])
                .groupby("dow")["ret"]
                .agg(["mean","count"])
                .reset_index()
            )
            daily_dow["label"] = daily_dow["dow"].map(DAYS_RU)
            daily_dow["color"] = daily_dow["mean"].apply(
                lambda x: "#1D9E75" if x >= 0 else "#E24B4A"
            )

            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=daily_dow["label"],
                y=daily_dow["mean"].round(3),
                marker_color=daily_dow["color"],
                text=daily_dow["mean"].apply(lambda x: f"{x:+.3f}%"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Средняя доходность: %{y:.3f}%<extra></extra>",
                name="Ср. доходность",
            ))
            fig_d.add_hline(y=0, line_color="rgba(150,150,150,0.4)", line_width=1)
            fig_d.update_layout(
                title=dict(text="Средняя доходность по дням недели", font_size=15),
                height=280,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis=dict(
                    ticksuffix="%",
                    gridcolor="rgba(92,103,125,.1)",
                    zeroline=False,
                ),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_d, use_container_width=True)

            # ── Сводная таблица ──
            best_m  = monthly.loc[monthly["mean"].idxmax(), "label"]
            worst_m = monthly.loc[monthly["mean"].idxmin(), "label"]
            best_d  = daily_dow.loc[daily_dow["mean"].idxmax(), "label"]
            worst_d = daily_dow.loc[daily_dow["mean"].idxmin(), "label"]

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Лучший месяц",  best_m,
                      f"{monthly.loc[monthly['mean'].idxmax(),'mean']:+.2f}%")
            s2.metric("Худший месяц",  worst_m,
                      f"{monthly.loc[monthly['mean'].idxmin(),'mean']:+.2f}%")
            s3.metric("Лучший день",   best_d,
                      f"{daily_dow.loc[daily_dow['mean'].idxmax(),'mean']:+.3f}%")
            s4.metric("Худший день",   worst_d,
                      f"{daily_dow.loc[daily_dow['mean'].idxmin(),'mean']:+.3f}%")

            st.caption(
                f"На основе {len(df_s)} торговых дней · "
                f"период: {df_s['time'].min():%d.%m.%Y} — {df_s['time'].max():%d.%m.%Y}"
            )

            # ── Поведение модели на историческом периоде ──────────────
            st.divider()
            st.subheader("Поведение модели на историческом периоде")
            st.caption(
                "Скользящая точность прогнозов модели — показывает насколько стабильно "
                "модель работала в разные периоды времени. Падения могут указывать на "
                "сложные рыночные условия или необходимость дообучения."
            )

            hp_all = fetch_ticker_preds(ticker, 365)
            if hp_all and len(hp_all) >= 5:
                df_hp2 = pd.DataFrame(hp_all).dropna(subset=["is_correct"])
                df_hp2["target_date"] = pd.to_datetime(df_hp2["target_date"])
                df_hp2 = df_hp2.sort_values("target_date")
                df_hp2["is_correct_num"] = df_hp2["is_correct"].astype(float)

                # Скользящая точность (окно 10 прогнозов)
                df_hp2["rolling_acc"] = (
                    df_hp2["is_correct_num"]
                    .rolling(min(10, len(df_hp2)), min_periods=3)
                    .mean() * 100
                )
                # Накопленная точность
                df_hp2["cum_acc"] = df_hp2["is_correct_num"].expanding().mean() * 100

                fig_perf = go.Figure()

                # Фон: зелёный выше 50%, красный ниже
                fig_perf.add_hrect(
                    y0=50, y1=100,
                    fillcolor="rgba(29,158,117,.04)",
                    line_width=0,
                )
                fig_perf.add_hrect(
                    y0=0, y1=50,
                    fillcolor="rgba(226,75,74,.04)",
                    line_width=0,
                )

                # Скользящая точность
                fig_perf.add_trace(go.Scatter(
                    x=df_hp2["target_date"],
                    y=df_hp2["rolling_acc"].round(1),
                    mode="lines",
                    name="Скользящая точность (10 прогнозов)",
                    line=dict(color="#4f6ef7", width=2),
                    hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Точность: %{y:.1f}%<extra></extra>",
                ))

                # Накопленная точность
                fig_perf.add_trace(go.Scatter(
                    x=df_hp2["target_date"],
                    y=df_hp2["cum_acc"].round(1),
                    mode="lines",
                    name="Накопленная точность",
                    line=dict(color="#D79A2B", width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Накопл.: %{y:.1f}%<extra></extra>",
                ))

                # Линия 50% (случайное угадывание)
                fig_perf.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="rgba(150,150,150,0.5)",
                    annotation_text="50% — случайное угадывание",
                    annotation_position="bottom right",
                    annotation_font_size=11,
                    annotation_font_color="rgba(150,150,150,0.8)",
                )

                # Точки сигналов
                colors_map = {"BUY": "#1D9E75", "SELL": "#E24B4A", "HOLD": "#888"}
                for sig_val in ["BUY", "SELL"]:
                    mask = df_hp2["signal"] == sig_val
                    if mask.any():
                        sub = df_hp2[mask]
                        fig_perf.add_trace(go.Scatter(
                            x=sub["target_date"],
                            y=sub["rolling_acc"],
                            mode="markers",
                            name=sig_val,
                            marker=dict(
                                size=7,
                                color=[
                                    "#1D9E75" if c else "#E24B4A"
                                    for c in sub["is_correct"]
                                ],
                                symbol="circle",
                                line=dict(
                                    width=1,
                                    color=colors_map[sig_val],
                                ),
                            ),
                            hovertemplate=(
                                f"<b>%{{x|%d.%m.%Y}}</b><br>"
                                f"Сигнал: {sig_val}<br>"
                                f"Верно: %{{customdata}}<extra></extra>"
                            ),
                            customdata=sub["is_correct"].map({True: "✓", False: "✗"}),
                        ))

                fig_perf.update_layout(
                    height=340,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", y=1.08, x=0,
                        bgcolor="rgba(255,255,255,.7)",
                    ),
                    yaxis=dict(
                        title="Точность, %",
                        range=[0, 105],
                        ticksuffix="%",
                        gridcolor="rgba(92,103,125,.1)",
                        zeroline=False,
                    ),
                    xaxis=dict(
                        gridcolor="rgba(92,103,125,.08)",
                        showline=True,
                        linecolor="rgba(92,103,125,.18)",
                    ),
                )
                st.plotly_chart(fig_perf, use_container_width=True)

                # Итоговые метрики
                m1, m2, m3, m4 = st.columns(4)
                total    = len(df_hp2)
                correct  = df_hp2["is_correct_num"].sum()
                buys     = df_hp2[df_hp2["signal"] == "BUY"]
                sells    = df_hp2[df_hp2["signal"] == "SELL"]
                m1.metric("Всего прогнозов",  total)
                m2.metric("Общая точность",   f"{correct/total*100:.1f}%")
                m3.metric("Точность BUY",
                    f"{buys['is_correct_num'].mean()*100:.1f}%" if len(buys) else "—")
                m4.metric("Точность SELL",
                    f"{sells['is_correct_num'].mean()*100:.1f}%" if len(sells) else "—")
            else:
                st.info("История прогнозов появится после нескольких торговых дней.")
        else:
            st.info("Недостаточно исторических данных для сезонного анализа.")


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА 3: ИСТОРИЯ ПРОГНОЗОВ
# ═══════════════════════════════════════════════════════════════

elif page == "📋 История прогнозов":
    st.title("История прогнозов")
    preds = fetch_predictions()
    if preds:
        df_all = pd.DataFrame(preds)
        df_all["Сигнал"] = df_all["signal"].map({
            "BUY":  "↑ Купить",
            "SELL": "↓ Продать",
            "HOLD": "→ Держать",
        })
        df_all["Уверенность"] = (
            df_all["confidence"].astype(float) * 100
        ).round(0).astype(int).astype(str) + "%"
        df_all["Прогноз Δ"] = df_all["predicted_delta"].apply(
            lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
        )
        st.dataframe(
            df_all[["ticker", "short_name", "Сигнал",
                    "Прогноз Δ", "Уверенность", "sector"]].rename(columns={
                "ticker":     "Тикер",
                "short_name": "Компания",
                "sector":     "Сектор",
            }),
            use_container_width=True,
            hide_index=True,
            height=600,
        )
    else:
        st.info("Прогнозы пока не сформированы.")


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА 4: ТОЧНОСТЬ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════

elif page == "📈 Точность модели":
    st.title("Точность прогнозов")
    try:
        r = requests.get(f"{API}/api/accuracy", timeout=10)
        if r.ok:
            acc = r.json()
            if acc:
                cols = st.columns(len(acc))
                for col, (sig, data) in zip(cols, acc.items()):
                    col.metric(
                        f"Точность {sig}",
                        f"{data['pct']}%",
                        f"{data['correct']}/{data['total']} верных",
                    )
            else:
                st.info(
                    "Данные точности появятся через несколько торговых дней "
                    "после первых прогнозов."
                )
    except Exception:
        st.error("Не удалось загрузить статистику")


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА 5: О СИСТЕМЕ
# ═══════════════════════════════════════════════════════════════

elif page == "ℹ️ О системе":
    st.title("О системе")
    st.markdown("""
### Архитектура

**Источник данных:** MOEX ISS API — официальный бесплатный API Московской биржи.

**ML модель:** Ансамбль LightGBM + LSTM с SHAP-объяснениями.
- LightGBM классифицирует направление по 18 техническим признакам
- LSTM улавливает последовательные паттерны в ценовом ряду за 60 дней
- SHAP показывает какие факторы повлияли на каждый прогноз

**Обновление данных:** ежедневно в 19:30 МСК (после торгов).
**Генерация прогнозов:** ежедневно в 20:00 МСК.

**Стек:** FastAPI · asyncpg · PostgreSQL (Supabase) · Streamlit · Plotly · Yandex DataSphere

> ⚠️ Прогнозы создаются в исследовательских целях.
> Не являются финансовыми рекомендациями.
    """)

    st.divider()
    st.subheader("Статус сервисов")
    try:
        r = requests.get(f"{API}/health", timeout=5)
        if r.ok:
            h = r.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("API", "✅ Работает" if h["status"] == "ok" else "⚠️")
            c2.metric("База данных",
                      "✅" if h["database"] == "ok" else "❌")
            c3.metric("Тикеров", h.get("tickers", 0))
    except Exception:
        st.error("API недоступен. Запусти uvicorn в другом терминале.")
