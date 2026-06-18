"""
backend/collector/run.py
═══════════════════════════════════════════════════════════════
CLI для ручного запуска задач сборщика.

ПРИМЕРЫ:
    # Загрузить историю одной акции (для теста ~1 мин)
    python -m backend.collector.run --load-history --ticker SBER --from 2023-01-01

    # Загрузить все 20 акций с 2021 года (~20 мин)
    python -m backend.collector.run --load-history

    # Обновить свежие данные (для ежедневного запуска)
    python -m backend.collector.run --daily-update
"""
import argparse
import asyncio
import logging

# Загружаем .env в переменные окружения
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
)


async def cmd_load_history(ticker: str | None, date_from: str):
    """Команда загрузки истории."""
    from backend.db.database import get_pool
    from backend.collector.moex import load_history, load_history_all
    
    await get_pool()  # Подключаемся к БД
    
    if ticker:
        # Одна конкретная акция
        await load_history(ticker.upper(), date_from)
    else:
        # Все 20 акций
        await load_history_all(date_from)


async def cmd_daily_update():
    """Ежедневное обновление."""
    from backend.db.database import get_pool
    from backend.collector.moex import daily_update
    
    await get_pool()
    await daily_update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOEX Data Collector")
    parser.add_argument("--load-history", action="store_true",
                        help="Первоначальная загрузка истории")
    parser.add_argument("--daily-update", action="store_true",
                        help="Ежедневное обновление")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Тикер акции (только для --load-history)")
    parser.add_argument("--from", type=str, default="2021-01-01",
                        dest="date_from", help="Начальная дата")
    
    args = parser.parse_args()

    if args.load_history:
        asyncio.run(cmd_load_history(args.ticker, args.date_from))
    elif args.daily_update:
        asyncio.run(cmd_daily_update())
    else:
        parser.print_help()

#         (dip) PS C:\Diploma\project>     python -m backend.collector.run --daily-update
# 2026-04-22 09:48:41,225  INFO      === Обновление 2026-04-19 → 2026-04-22 ===
# 2026-04-22 09:48:49,018  ERROR     [SBER] ✗ 'NoneType' object is not subscriptable
# 2026-04-22 09:49:48,168  ERROR     [GAZP] ✗ 'NoneType' object is not subscriptable
# 2026-04-22 09:49:49,216  ERROR     [LKOH] ✗ 'NoneType' object is not subscriptable
# 2026-04-22 09:50:48,584  ERROR     [NVTK] ✗ 'NoneType' object is not subscriptable
# 2026-04-22 09:50:49,661  ERROR     [ROSN] ✗ 'NoneType' object is not subscriptable
# 2026-04-22 09:51:48,813  ERROR     [GMKN] ✗ 'NoneType' object is not subscriptable
# 2026-04-22 09:51:49,840  ERROR     [MGNT] ✗ 'NoneType' object is not subscriptable
