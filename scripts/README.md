# Scripts

Utility scripts are grouped by purpose:

```text
scripts/data/        data loading, cleaning, and feature repair
scripts/ml/          model prediction, threshold tuning, and walk-forward helpers
scripts/evaluation/  backtests, baseline comparison, and error analysis
scripts/automation/  scheduled local maintenance scripts
```

Run them from the project root so imports and relative paths resolve correctly:

```powershell
python scripts/evaluation/rolling_backtest.py
python scripts/ml/generate_predictions.py
python scripts/data/collect_moex_data.py
```
