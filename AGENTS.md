# AGENTS.md — Coding Conventions and Working Notes

This file documents conventions and practical tips for agents working in this
repository. Follow these rules for any edits you make.

Scope: entire repository unless otherwise noted.

## General Principles

- Keep changes minimal and focused on the task; avoid unrelated refactors.
- Preserve existing style, naming, and module boundaries.
- Prefer adding helpers over invasive rewrites.
- Do not introduce noisy diffs (e.g., line‑ending flips, mass reformatting).

## Line Endings & Formatting

- Default EOL on Windows is CRLF. Do not change EOLs unintentionally.
- Keep the editor’s EOL to CRLF for .py files. If you must add lines, match the
  surrounding file’s EOL.
- Avoid trailing whitespace. Ensure a newline at file end.

## Imports

- Place imports at the top of the file (no in‑function imports) unless there is
  a strong reason (e.g., heavy optional dependency).
- Reuse already imported modules/utilities (e.g., `pandas as pd`,
  `load_csv_with_date_index`).

## Patching Discipline

- Make the smallest possible diff to achieve the intended behavior.
- Do not change logging, formatting, or comments outside the edited scope.
- Keep function order and public API stable unless explicitly requested.

## Module Boundaries (Design Pattern)

- Screening pipeline (multi‑ticker):
  - `JdStockFilteringManager` contains filter functions that operate on
    `{ticker: DataFrame}` and return a ticker list for a given day index.
  - Use `screening_stocks_by_func` for orchestration.
- Single‑ticker analysis (time‑series within one symbol):
  - Add small helpers to `JdStockDataManager` (e.g., computations on a single
    DataFrame, date searches) instead of overloading the screening path.

## Data Loading Policy

- Local CSVs live under `StockData/{ticker}.csv`. Load via
  `jd_io_utils.load_csv_with_date_index` when possible.
- When local CSV is missing (e.g., ETFs like QQQ, SPY, indices), use
  `FinanceDataReader.DataReader(ticker, start_date)` as a fallback.
- Normalize the DataFrame to include minimal columns
  `['Open', 'High', 'Low', 'Close', 'Volume']`; if `Volume` is missing, fill 0;
  if `Close` is missing, return an empty DataFrame.
- Do not persist fetched fallback data to CSV unless explicitly asked.

## Calculations

- Daily percent change (linear) is
  `change_pct = (Close / Close.shift(1) - 1) * 100`.
- Guard against insufficient rows (length < 2) before using `shift(1)` when you
  want an early return; otherwise allow the operation to yield NaN and filter.

## Logging

- Use existing logging configuration (`logging_conf.py`).
- Log external fetch failures (`FinanceDataReader`) with concise context; do not
  spam logs.

## CLI & Menu

- New runnable behaviors should be added as small `run_*` functions in
  `main.py`, then wired to the menu prompt and choice branch.
- Keep prompts short and provide safe defaults.

## Performance & Caching

- Reuse caches provided by the data layer (`jdDataGetter`) when screening.
- For single‑ticker helpers, prefer in‑memory computations unless persistence is
  requested.

## Git & Commit Style

- Use Conventional Commit‑style messages when possible:
  `feat(scope): summary` / `fix(scope): summary`.
- Summaries should describe what changed and why; include notable behavioral
  notes (e.g., ETF fallback, no persistence).

## External Services

- `FinanceDataReader` is the preferred remote source for equities/ETFs/indices
  when local CSV is missing.
- Handle network failures gracefully and return empty DataFrames on errors.

