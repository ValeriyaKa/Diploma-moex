"""
fix_tickers.py — One-time migration script (v2).
Updates old tickers (YNDX→YDEX, FIVE→X5, TCSG→T) in Supabase tables.

Strategy (avoids FK and RLS errors):
  1. RENAME the old security row first (securities: YNDX → YDEX)
     This satisfies the foreign key for subsequent data table updates
     and avoids RLS INSERT restrictions (we UPDATE, not INSERT).
  2. Then rename ticker in candles, indicators, predictions.
  3. Finally update short_name/sector on the renamed security.

Usage:
    python fix_tickers.py                  # dry run (show what would change)
    python fix_tickers.py --apply          # actually apply changes
    python fix_tickers.py --apply --reload # apply + reload candles for new tickers
"""

import os, argparse, time
from datetime import date
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ================================================================
# TICKER MIGRATIONS
# ================================================================

MIGRATIONS = [
    {
        "old": "YNDX",
        "new": "YDEX",
        "short_name": "Яндекс (YDEX)",
        "sector": "IT",
        "reason": "Делистинг YNDX, переход на YDEX после реструктуризации Yandex N.V.",
    },
    {
        "old": "FIVE",
        "new": "X5",
        "short_name": "X5 Group",
        "sector": "Ритейл",
        "reason": "Редомициляция X5 Retail Group → X5 Group на MOEX.",
    },
    {
        "old": "TCSG",
        "new": "T",
        "short_name": "Т-Технологии",
        "sector": "Финансы",
        "reason": "Ребрендинг TCS Group → Т-Технологии, тикер TCSG → T.",
    },
]

DATA_TABLES = ["candles", "indicators", "predictions"]


def count_rows(table: str, ticker: str) -> int:
    """Count rows in table for a ticker."""
    try:
        result = supabase.table(table)\
            .select("ticker", count="exact")\
            .eq("ticker", ticker)\
            .limit(1)\
            .execute()
        return result.count or 0
    except Exception:
        return -1


def security_exists(ticker: str) -> bool:
    """Check if a security entry exists."""
    rows = supabase.table("securities")\
        .select("ticker")\
        .eq("ticker", ticker)\
        .execute().data
    return len(rows) > 0


def rename_security(old: str, new: str, short_name: str, sector: str, dry_run: bool):
    """
    Rename the security row: update ticker, short_name, sector.
    Uses UPDATE (not INSERT) to avoid RLS policy violations.
    """
    if not security_exists(old):
        # Old security doesn't exist — maybe already migrated or never existed
        if security_exists(new):
            print(f"  securities: '{new}' already exists, skipping")
            return True
        else:
            print(f"  securities: neither '{old}' nor '{new}' exists!")
            print(f"  WARNING: You need to manually insert '{new}' in Supabase Dashboard:")
            print(f"    INSERT INTO securities (ticker, short_name, sector, is_active)")
            print(f"    VALUES ('{new}', '{short_name}', '{sector}', true);")
            return False

    if security_exists(new):
        # New ticker already exists — just deactivate old
        print(f"  securities: '{new}' already exists, deactivating '{old}'")
        if not dry_run:
            supabase.table("securities")\
                .update({"is_active": False})\
                .eq("ticker", old)\
                .execute()
        return True

    if dry_run:
        print(f"  [DRY RUN] securities: would rename '{old}' → '{new}'")
        return True

    try:
        supabase.table("securities")\
            .update({
                "ticker": new,
                "short_name": short_name,
                "sector": sector,
                "is_active": True,
            })\
            .eq("ticker", old)\
            .execute()
        print(f"  securities: renamed '{old}' → '{new}' ({short_name})")
        return True
    except Exception as e:
        print(f"  securities: ERROR renaming '{old}' → '{new}': {e}")
        print(f"\n  FALLBACK: Run this SQL in Supabase Dashboard → SQL Editor:")
        print(f"    UPDATE securities SET ticker='{new}', short_name='{short_name}', ")
        print(f"      sector='{sector}', is_active=true WHERE ticker='{old}';")
        return False


def migrate_data_table(old: str, new: str, table: str, dry_run: bool) -> int:
    """Rename ticker in a data table (candles, indicators, predictions)."""
    count = count_rows(table, old)
    if count <= 0:
        print(f"  {table}: no rows with '{old}', skipping")
        return 0

    if dry_run:
        print(f"  [DRY RUN] {table}: would rename {count} rows '{old}' → '{new}'")
        return count

    try:
        supabase.table(table)\
            .update({"ticker": new})\
            .eq("ticker", old)\
            .execute()
        print(f"  {table}: renamed {count} rows '{old}' → '{new}'")
        return count
    except Exception as e:
        print(f"  {table}: ERROR: {e}")
        print(f"  FALLBACK SQL: UPDATE {table} SET ticker='{new}' WHERE ticker='{old}';")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fix outdated MOEX tickers in Supabase")
    parser.add_argument("--apply", action="store_true",
                        help="Actually apply changes (default: dry run)")
    parser.add_argument("--reload", action="store_true",
                        help="After migration, reload candles for new tickers")
    args = parser.parse_args()

    dry_run = not args.apply

    print("=" * 60)
    print("MOEX Ticker Migration v2")
    print("=" * 60)
    if dry_run:
        print("MODE: DRY RUN (pass --apply to execute)\n")
    else:
        print("MODE: APPLYING CHANGES\n")

    all_ok = True

    for mig in MIGRATIONS:
        old, new = mig["old"], mig["new"]
        print(f"\n{'─' * 50}")
        print(f"  {old} → {new}")
        print(f"  {mig['reason']}")
        print(f"{'─' * 50}")

        # Show current state
        for table in ["securities"] + DATA_TABLES:
            old_c = count_rows(table, old)
            new_c = count_rows(table, new)
            if old_c > 0 or new_c > 0:
                print(f"  {table:15s}: '{old}'={old_c:>6}   '{new}'={new_c:>6}")

        # STEP 1: Rename security FIRST (satisfies FK constraint)
        print(f"\n  Step 1: Rename security...")
        ok = rename_security(old, new, mig["short_name"], mig["sector"], dry_run)
        if not ok:
            print(f"  ⚠ Cannot proceed with data tables until security is fixed.")
            print(f"    Fix it manually, then re-run this script.")
            all_ok = False
            continue

        time.sleep(0.3)

        # STEP 2: Rename in data tables
        print(f"  Step 2: Rename data...")
        for table in DATA_TABLES:
            migrate_data_table(old, new, table, dry_run)
            time.sleep(0.3)

        time.sleep(0.5)

    # Verification
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")
    for mig in MIGRATIONS:
        old, new = mig["old"], mig["new"]
        old_total = sum(count_rows(t, old) for t in DATA_TABLES)
        new_total = sum(count_rows(t, new) for t in DATA_TABLES)
        old_sec = "exists" if security_exists(old) else "gone"
        new_sec = "exists" if security_exists(new) else "MISSING"
        status = "OK" if old_total == 0 and new_sec == "exists" else "NEEDS WORK"
        print(f"  {old:5s} → {new:5s}  |  old_rows={old_total:>6}  new_rows={new_total:>6}  "
              f"old_sec={old_sec:7s}  new_sec={new_sec:7s}  [{status}]")

    # Post-migration steps
    print(f"\n{'=' * 60}")
    print("NEXT STEPS")
    print(f"{'=' * 60}")
    print("""
1. Download fresh candles for new tickers:

    python load_candles.py --ticker YDEX --from 2024-06-01
    python load_candles.py --ticker X5 --from 2024-04-01
    python load_candles.py --ticker T --from 2024-11-01

2. Retrain models:
    python -m datasphere.train_models
""")

    if args.apply and args.reload:
        print("Auto-reloading candles for new tickers...")
        import subprocess, sys
        python = sys.executable  # use the same Python/virtualenv
        for ticker, from_date in [("YDEX", "2024-06-01"), ("X5", "2024-04-01"), ("T", "2024-11-01")]:
            print(f"\n  Loading {ticker} from {from_date}...")
            subprocess.run([
                python, "load_candles.py",
                "--ticker", ticker, "--from", from_date,
            ])
            time.sleep(1)
        print("\nDone!")


if __name__ == "__main__":
    main()
