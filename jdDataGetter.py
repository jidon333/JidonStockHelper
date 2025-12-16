
"""

데이터 수집 및 로컬 파일 로딩, 캐싱 관련 로직

"""

import concurrent.futures
import datetime as dt
import json
import logging
import os
import time

import FinanceDataReader as fdr
import pandas as pd
import pandas_market_calendars as mcal


# io_utils.py에서 폴더 상수와 CSV/JSON/Pickle I/O 함수를 import
from jd_io_utils import (
    load_csv_with_date_index, save_df_to_csv,
    save_to_json, load_from_json,
    save_pickle, load_pickle,
)
from jdGlobal import (
    DATA_FOLDER,
    METADATA_FOLDER,
    sync_fail_ticker_list,
    exception_ticker_list
)

#미국 주식시장 달력
nyse = mcal.get_calendar("NYSE")

logger = logging.getLogger(__name__)




class JdDataGetter:
    """
    데이터 수집(get), 로컬 CSV 로딩, 캐싱 등을 전담하는 클래스.

    외부 API 호출, 타임아웃/재시도 로직, 로컬 파일 로딩 관련 함수들로 구성
    """


    def __init__(self, max_workers: int = 1, timeout_sec: int = 10):
        self.max_workers = max_workers
        self.timeout_sec = timeout_sec

        # 필요 시 캐시 변수
        self.cache_stock_list = pd.DataFrame()
        self.cache_stock_datas = {}
        # 추가로 필요한 전역 예외/상태 변수는 여기서 정의 가능


        
    # ------------------------------------------------------------
    # (1) get_fdr_stock_list
    # ------------------------------------------------------------
    def get_fdr_stock_list(self, market: str, days_num: int = 365*5, ignore_no_local_tickers: bool = True) -> pd.DataFrame:
        """
        FinanceDataReader를 통해 시장(예: 'NYSE', 'NASDAQ', 'S&P500')의
        종목 리스트를 가져오고, (옵션) 로컬 CSV 폴더에 없는 티커는 제외.
        로컬 pickle 캐시(cache_fdr_{market}_list.pkl) 사용 가능.
        """
        if market not in ["NYSE", "NASDAQ", "S&P500"]:
            logger.warning("[get_fdr_stock_list] Invalid market: %s", market)
            return pd.DataFrame()

        # 캐시가 있으면 캐시 반환
        cache_file = f"cache_fdr_{market}_list"  # pickle 파일명
        loaded_df = load_pickle(cache_file, folder=METADATA_FOLDER)
        if isinstance(loaded_df, pd.DataFrame) and not loaded_df.empty:
            return loaded_df

        # 웹에서 새로 가져오기
        try:
            stock_list = fdr.StockListing(market)
        except Exception:
            logger.exception("[get_fdr_stock_list] Error fetching list for %s", market)
            return pd.DataFrame()

        # 로컬 CSV 폴더에 없는 티커 제외
        if ignore_no_local_tickers:
            local_files = [
                os.path.splitext(f)[0] for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")
            ]
            stock_list = stock_list[stock_list["Symbol"].isin(local_files)]

        # 캐시 저장
        save_pickle(stock_list, cache_file, folder=METADATA_FOLDER)
        return stock_list
    

    # ------------------------------------------------------------
    # (2) get_local_stock_list
    # ------------------------------------------------------------
    def get_local_stock_list(self) -> pd.DataFrame:
        """
        NYSE, NASDAQ 종목 리스트를 합쳐서, 로컬 CSV 파일이 존재하는 종목만 필터링.
        한 번 불러오면 self.cache_stock_list에 저장하여 재사용.
        """
        if not self.cache_stock_list.empty:
            return self.cache_stock_list

        nyse_list = self.get_fdr_stock_list("NYSE")
        nasdaq_list = self.get_fdr_stock_list("NASDAQ")
        all_list = pd.concat([nyse_list, nasdaq_list])

        local_files = [
            os.path.splitext(f)[0] for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")
        ]
        filtered = all_list[all_list["Symbol"].isin(local_files)]

        self.cache_stock_list = filtered
        return filtered


    # ------------------------------------------------------------
    # (3) get_stock_datas_from_csv
    # ------------------------------------------------------------
    def get_stock_datas_from_csv(
        self,
        stock_list: pd.DataFrame,
        days_num: int = 365*5,
        use_cache_data: bool = False
    ) -> dict:
        """
        로컬 CSV에서 지정한 기간(days_num)만큼의 주가 데이터를 읽어
        {ticker: DataFrame} 형태로 반환.
        use_cache_data=True면, pickle 캐시('cache_getStockDatas_dic')를 우선 사용.
        """
        if use_cache_data:
            cached_dic = load_pickle("cache_getStockDatas_dic", folder=METADATA_FOLDER)
            if cached_dic is not None:
                #print("[get_stock_datas_from_csv] Loaded from cache.")
                self.cache_stock_datas = cached_dic
                return cached_dic

        logger.info("[get_stock_datas_from_csv] Loading from CSV files...")

        stock_datas_dic = {}
        today_date = dt.date.today()
        start_date = today_date - dt.timedelta(days=days_num)

        for idx, ticker in enumerate(stock_list["Symbol"]):
            df = load_csv_with_date_index(ticker, data_dir=DATA_FOLDER, start_date=start_date, end_date=today_date)
            if df.empty:
                continue
            stock_datas_dic[ticker] = df
            logger.debug(
                "[%s/%s] %s - Data loaded successfully.",
                idx + 1,
                len(stock_list["Symbol"]),
                ticker,
            )


        save_pickle(stock_datas_dic, "cache_getStockDatas_dic", folder=METADATA_FOLDER)
        self.cache_stock_datas = stock_datas_dic
        return stock_datas_dic
    
    # ------------------------------------------------------------
    # (4) fetch_data_with_timeout_process
    # ------------------------------------------------------------
    def fetch_data_with_timeout_process(self, executor, ticker, start_date, timeout_sec=10):
        """
        외부에서 주입받은 executor를 이용해 fdr.DataReader를 별도 프로세스에서 호출.
        - 만약 timeout_sec 내에 결과가 오지 않으면 (None, True) 반환 (타임아웃)
        - 예외 발생 시 (None, False) 반환
        - 정상적인 경우 (DataFrame, False) 반환

        :param executor: 외부에서 재사용하려고 만든 ProcessPoolExecutor 인스턴스
        :param ticker: 데이터를 가져올 종목(티커)
        :param start_date: 시작 날짜
        :param timeout_sec: 결과를 기다릴 최대 시간(초)
        :return: (data, is_timeout)
        """
        future = executor.submit(fdr.DataReader, ticker, start_date)
        try:
            data = future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning(
                "[Timeout] %s data request did not complete within %s seconds.",
                ticker,
                timeout_sec,
            )
            return None, True
        except Exception:
            future.cancel()
            logger.exception(
                "[Exception] An error occurred while requesting data for %s (start_date=%s)",
                ticker,
                start_date,
            )
            return None, False
        return data, False


    # ------------------------------------------------------------
    # (5) get_stock_data_with_retries
    # ------------------------------------------------------------
    def get_stock_data_with_retries(self, executor, ticker, trading_day, 
                                    timeout_sec=10, max_retries=3, retry_delay=5, long_wait=300):
        """
        특정 티커 데이터를 가져오되, 타임아웃/에러 발생 시 지정된 횟수(max_retries)로 재시도.
        - 첫 호출 후 실패하면 재시도(타임아웃 시 executor를 교체해서 Hang 방지)
        - (stock_data, is_timeout, executor) 튜플을 반환하여, executor가 바뀔 수도 있음.

        :param executor: 외부에서 만든 executor (ProcessPoolExecutor 등)
        :param ticker: 종목 티커
        :param trading_day: 시작 날짜(또는 날짜 인덱스 중 하나)
        :param timeout_sec: 1회 호출 타임아웃
        :param max_retries: 최대 재시도 횟수
        :param retry_delay: 재시도 시 기본 대기 시간(초)
        :param long_wait: 타임아웃 발생 시의 대기 시간(기본 300초=5분)
        :return: (stock_data, is_timeout, executor)
        """
        # 첫 시도
        stock_data, is_timeout = self.fetch_data_with_timeout_process(executor, ticker, trading_day, timeout_sec)
        if stock_data is not None and not stock_data.empty:
            return stock_data, is_timeout, executor

        logger.warning("%s initial call failed. Starting retries.", ticker)
        for retryCnt in range(max_retries):

            # 이전 호출이 타임아웃이라면, executor 교체
            if is_timeout:
                logger.warning("[%s] Timeout detected. Replacing the executor.", ticker)
                executor.shutdown(wait=False)
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)  # 새 executor 생성
                wait_time = long_wait
            else:
                wait_time = retry_delay * (retryCnt + 1)
            
            logger.info(
                "[%s] Retry %s/%s - waiting %s seconds before retrying...",
                ticker,
                retryCnt + 1,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)

            stock_data, is_timeout = self.fetch_data_with_timeout_process(executor, ticker, trading_day, timeout_sec)

            if is_timeout:
                logger.warning("[%s] Timeout detected again. Replacing the executor.", ticker)
                executor.shutdown(wait=False)
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
                return pd.DataFrame(), is_timeout, executor

            elif stock_data is not None and not stock_data.empty:
                # 정상적으로 값을 얻었다면 종료
                logger.info("%s retry successful!", ticker)
                return stock_data, is_timeout, executor
            else:
                # 타임아웃은 아니지만 데이터를 못얻었다면 그냥 빈 DataFrame 반환
                logger.warning("%s data call failed. Assigning an empty DataFrame.", ticker)
                return pd.DataFrame(), is_timeout, executor

        # 모든 재시도 실패
        logger.error("%s data call ultimately failed. Returning empty DataFrame.", ticker)
        return pd.DataFrame(), is_timeout, executor


    # ------------------------------------------------------------
    # (6) get_datas_from_web
    # ------------------------------------------------------------
    def get_datas_from_web(self, stock_list, trading_days):
        """
        여러 티커에 대해 웹 데이터를 가져와 out_data_dic[ticker] = stock_data 형태로 저장.
        외부에서 executor를 만들고 인자로 넣어준 뒤, get_stock_data_with_retries로 호출.

        :param stock_list: 종목 리스트 (DataFrame)
        :param trading_days: 날짜 목록(예: nyse.schedule(...).index)
        """
        i = 0
        stockNums = stock_list.shape[0]
        out_data_dic = {}

        # 설정 값들
        max_retries = 3
        retry_delay = 5   # Base retry delay(초)
        long_wait = 300   # 타임아웃 발생 시 대기(초)

        # 기본으로 사용할 executor를 먼저 생성
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        try:
            for ticker in stock_list['Symbol']:
                logger.debug("%s stock data from %s", ticker, trading_days[0])
                
                # get_stock_data_with_retries를 호출할 때, 현재 executor를 전달
                # 그리고 반환값으로 executor를 다시 받는다. (API hang으로 executor가 shutdown 되면 함수 내부에서 재생성해서 반환)
                stock_data, is_timeout, executor = self.get_stock_data_with_retries(
                    executor, ticker, trading_days[0],
                    timeout_sec=10,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    long_wait=long_wait
                )
                
                if not stock_data.empty:
                    stock_data.reset_index(inplace=True)
                    stock_data.rename(columns={'index': 'Date'}, inplace=True)
                    stock_data.set_index('Date', inplace=True)
                    stock_data['Symbol'] = ticker
                    stock_data['Name'] = stock_list.loc[stock_list['Symbol'] == ticker, 'Name'].values[0]
                    stock_data['Industry'] = stock_list.loc[stock_list['Symbol'] == ticker, 'Industry'].values[0]


                out_data_dic[ticker] = stock_data
                i += 1
                logger.debug("%0.2f%% Done", i / stockNums * 100)

        finally:
            executor.shutdown(wait=False)


        return out_data_dic

    # ------------------------------------------------------------
    # (7) sync_csv_from_web
    # ------------------------------------------------------------
    def sync_csv_from_web(self, days_num=14):
        logger.info("[sync_csv_from_web] Start sync...")

        stock_list = self.get_local_stock_list()  # 로컬에 있는 종목들
        schedule = nyse.schedule(
            start_date=dt.date.today() - dt.timedelta(days=days_num),
            end_date=dt.date.today()
        )
        trading_days = schedule.index

        # 1) 웹 데이터
        web_data_dic = self.get_datas_from_web(stock_list, trading_days)

        # 2) 로컬 CSV
        local_dic = self.get_stock_datas_from_csv(stock_list)

        merged_data_dic = {}
        i = 0
        total = len(stock_list)

        for ticker in stock_list['Symbol']:
            csv_df = local_dic.get(ticker, pd.DataFrame())
            web_df = web_data_dic.get(ticker, pd.DataFrame())

            if csv_df.empty or web_df.empty:
                sync_fail_ticker_list.append(ticker)
                continue

            # forward fill을 사용하여 NaN값을 이전 행의 값으로 대체
            # IsOriginData_NaN 레이블을 추가하고, 기본값으로 False를 할당
            # 'Open'이 Nan이면 IsOriginData_NaN을 True로 설정
            # remove duplicate index from csvData.
            web_df.ffill(inplace=True)
            web_df["IsOriginData_NaN"] = False
            web_df.loc[web_df['Open'].isnull(), 'IsOriginData_NaN'] = True
            csv_df = csv_df[~csv_df.index.isin(web_df.index)]

            merged = pd.concat([csv_df, web_df])
            merged_data_dic[ticker] = merged


        # 실패 목록
        with open('sync_fail_list.txt', 'w') as f:
            for tk in sync_fail_ticker_list:
                f.write(tk + '\n')

        return merged_data_dic


    # ------------------------------------------------------------
    # (8) download_stock_datas_from_web
    # ------------------------------------------------------------
    def download_stock_datas_from_web(self, days_num=365*5, exclude_not_in_local_csv=True):
        logger.info("[download_stock_datas_from_web] Start download...")

        if not exclude_not_in_local_csv:
            nyse_list = self.get_fdr_stock_list("NYSE", days_num, ignore_no_local_tickers=False)
            nasdaq_list = self.get_fdr_stock_list("NASDAQ", days_num, ignore_no_local_tickers=False)
            all_list = pd.concat([nyse_list, nasdaq_list])
        else:
            all_list = self.get_local_stock_list()

        schedule = nyse.schedule(
            start_date=dt.date.today() - dt.timedelta(days=days_num),
            end_date=dt.date.today()
        )
        trading_days = schedule.index

        web_dic = self.get_datas_from_web(all_list, trading_days)
        merged_data_dic = {}
        i = 0
        total = len(web_dic.keys())

        for ticker, webData in web_dic.items():
            if webData.empty:
                sync_fail_ticker_list.append(ticker)
                continue

            merged_data_dic[ticker] = webData

            i += 1
            logger.debug("[download_stock_datas_from_web] %s done %s/%s", ticker, i, total)

        # 실패 목록
        with open('download_fail_list.txt', 'w') as f:
            for tk in sync_fail_ticker_list:
                f.write(tk + '\n')

        return merged_data_dic
