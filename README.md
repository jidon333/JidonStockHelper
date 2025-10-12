# JidonStockHelper
US Stock helper. It provides useful technical indicators, chart graph and stock screener. 

![chart sample](https://github.com/jidon333/JidonStockHelper/assets/16080882/57839f25-2acc-4887-81f0-e0ddb0cf1c43)

![FA50_chart_2023-12-08](https://github.com/jidon333/JidonStockHelper/assets/16080882/3e542b18-b1ab-4059-a1b0-57dc400cb28d)
![MI_Index_chart_2023-12-08](https://github.com/jidon333/JidonStockHelper/assets/16080882/d02d3605-29b7-4dac-a34b-75b1ee04e8c0)
![MTT_chart_2023-12-08](https://github.com/jidon333/JidonStockHelper/assets/16080882/153a52f3-b76b-4ae5-87d8-f1110b5e67d4)

**Overview**
- Stock data toolkit for US markets: indicators, screeners, and charts (PyQt5 + Matplotlib).
- Works with local CSVs under `StockData/`, with metadata in `StockData/MetaData/`.
- If a requested symbol has no local CSV (e.g., ETFs QQQ/SPY), the app can fetch data via FinanceDataReader as a fallback for single‑ticker analysis.

**Key Features**
- Technical indicators: TR/ATR/ADR/SMA/EMA, relative metrics (TRS/ATRS/ATRS150).
- Screeners: MTT, High ADR Swing, Bull Snort, RS 8/10, Young, Good RS, Hope from bottom, Gap.
- Industry ranking (short/long term) and Top10 per industry.
- Count/Index charts: MTT, FA50, Momentum Index (NYSE/NASDAQ/S&P500), ATR Expansion.
- PyQt5 interactive stock chart with MA/EMA overlays and annotations.

**Menu (main.py)**
- 1: Stock Data Chart
- 2: Momentum Index Chart
- 3: Sync local CSV and generate metadata
- 4: Generate up/down data (local)
- 5: Process local stock data
- 6: Download stock data (overwrite)
- 7: Generate ATRS Ranking
- 8: Generate Industry Ranking
- 9: Generate screening result (XLSX)
- 10: MTT Index Chart
- 11: FA50 Index Chart
- 12: Generate all indicators and screening outputs
- 13: Power gap history screen
- 14: ATR Expansion Chart
- 15: Investigate ticker drop days (single ticker, with ETF fallback)

**Requirements**
- Python 3.9+
- pandas, numpy, matplotlib, PyQt5, FinanceDataReader, yahooquery, pandas_market_calendars, rich, openpyxl

**Run**
- `python main.py` then choose an option from the menu.
- For single‑ticker drop‑day search (option 15), enter a symbol (e.g., `QQQ`) and a threshold (e.g., `-3`).

**Data Layout**
- `StockData/{TICKER}.csv`: OHLCV time series per symbol.
- `StockData/MetaData/`: Up/Down aggregates, rankings, screenshots, etc.
- UI assets in `UI/` (Qt form, font, image).

**Notes**
- If an ETF/Index is not found locally, option 15 fetches data via FinanceDataReader without persisting to disk.
- Logging: rotating file + rich console; see `logging_conf.py`.

---

## 한국어 안내

### 개요
- 미국 주식 데이터를 대상으로 지표 계산, 스크리닝, 차트 시각화를 제공합니다(PyQt5 + Matplotlib).
- 기본적으로 `StockData/`의 로컬 CSV를 사용하고, 메타데이터는 `StockData/MetaData/`에 생성됩니다.
- 단일 티커 분석(예: QQQ, SPY) 중 로컬 CSV가 없으면 FinanceDataReader를 통해 데이터를 가져오는 폴백을 지원합니다(옵션 15).

### 주요 기능
- 기술 지표: TR/ATR/ADR/SMA/EMA, TRS/ATRS/ATRS150 등
- 스크리닝: MTT, High ADR Swing, Bull Snort, RS 8/10, Young, Good RS, Hope from bottom, Gap
- 산업 랭킹: 단/장기 산업 점수 및 산업 내 Top10 계산
- 차트: 종목 차트(MA/EMA 오버레이, 각종 보조정보), 모멘텀 인덱스(NYSE/NASDAQ/S&P500), 카운트(MTT/FA50), ATR Expansion
- PyQt5 기반 인터랙티브 탐색(키보드 이동/검색/MA 토글)

### 메뉴(메인 실행)
- 1: Stock Data Chart
- 2: Momentum Index Chart
- 3: 로컬 CSV 동기화 및 메타데이터 생성
- 4: 로컬 업/다운 데이터 생성
- 5: 로컬 주가 데이터 가공
- 6: 웹에서 다운로드(로컬 덮어쓰기)
- 7: ATRS 랭킹 생성
- 8: 산업 랭킹 생성
- 9: 스크리닝 결과 XLSX 생성
- 10: MTT Index Chart
- 11: FA50 Index Chart
- 12: 모든 지표/스크리닝 결과 일괄 생성
- 13: Power gap 히스토리 스크린
- 14: ATR Expansion Chart
- 15: 단일 티커 하락일 탐색(ETF 폴백 지원)

### 실행 방법
- `python main.py` 실행 후 메뉴에서 원하는 번호를 선택하세요.
- 단일 티커 하락일 탐색(옵션 15): 티커(예: QQQ)와 하락 임계값(예: -3)을 입력하면 최근 5년 내 해당 조건을 만족한 날짜들을 출력합니다.
  - 로컬 CSV가 없을 경우 FinanceDataReader를 통해 데이터를 가져오며, 디스크에는 저장하지 않습니다.

### 필요 라이브러리
- Python 3.9+
- pandas, numpy, matplotlib, PyQt5, FinanceDataReader, yahooquery, pandas_market_calendars, rich, openpyxl

### 데이터 구조
- `StockData/{TICKER}.csv`: 각 티커의 OHLCV 데이터
- `StockData/MetaData/`: 업/다운 집계, 랭킹, 스크린샷 등 결과물
- `UI/`: Qt 폼(.ui), 폰트, 이미지 등

### 참고 사항
- ETF/지수(QQQ, SPY, US500 등) 분석 시 로컬 CSV가 없으면 FinanceDataReader 폴백으로 데이터를 가져와 단일 티커 분석을 수행합니다(옵션 15).
- 로깅: 파일 로테이팅 + 콘솔(RichHandler). 자세한 설정은 `logging_conf.py`를 참고하세요.
