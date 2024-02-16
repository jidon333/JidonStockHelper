

import FinanceDataReader as fdr

import yahooquery as yq 
from yahooquery import Ticker

import pickle
import math

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


import os
import json
import datetime as dt
import time

from jdGlobal import get_yes_no_input
from jdGlobal import data_folder
from jdGlobal import metadata_folder
from jdGlobal import screenshot_folder
from jdGlobal import filteredStocks_folder
from jdGlobal import profiles_folder

import openpyxl
from openpyxl.styles import PatternFill, Font, Color


nyse = mcal.get_calendar('NYSE')


exception_ticker_list = {}
sync_fail_ticker_list = []

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

if not os.path.exists(metadata_folder):
    os.makedirs(metadata_folder)

if not os.path.exists(filteredStocks_folder):
    os.makedirs(filteredStocks_folder)

if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

if not os.path.exists(profiles_folder):
    os.makedirs(profiles_folder)



# for test
stockIterateLimit = 99999

# 딕셔너리를 JSON 파일로 저장
def save_to_json(data, filename):
    full_path = os.path.join(metadata_folder, f'{filename}.json')
    with open(full_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"dictionary data is saved as {full_path} ")

# JSON 파일을 딕셔너리로 로드
def load_from_json(filename):
    full_path = os.path.join(metadata_folder, f'{filename}.json')

    with open(full_path, 'r', encoding='utf-8') as file:
        loaded_data = json.load(file)
    return loaded_data


class JdStockDataManager:
    def __init__(self):
        self.index_data = fdr.DataReader('US500')
        self.stock_GICS_df = pd.DataFrame()

        self.long_term_industry_rank_df = pd.DataFrame()
        self.short_term_industry_rank_df = pd.DataFrame()

        self.atrs_ranking_df = pd.DataFrame()


        # ---------- cache datas -------------#
        self.reset_caches()

    def reset_caches(self):
        self.cache_StockListFromLocalCsv = pd.DataFrame()
        self.cache_getStockDatasFromCsv_out_tickers = None
        self.cache_getStockDatasFromCsv_out_stock_datas_dic = None

# ------------------- private -----------------------------------------------

    def _get_csv_names(self):
        csv_names =[os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith('.csv')]
        return csv_names



    def _CookIndexData(self, index_data, n = 14):
        index_new_data = index_data

        # TR(True Range) 계산
        high = index_new_data['High']
        low = index_new_data['Low']
        prev_close = index_new_data['Close'].shift(1)

        d1 = high - low
        d2 = np.abs(high - prev_close)
        d3 = np.abs(low - prev_close)
        
        tr = np.maximum(d1, d2)
        tr = np.maximum(tr, d3)

        # ATR(Average True Range)
        atr = tr.rolling(n).mean()
        index_new_data['ATR'] = atr

        # TC(True Change) 계산
        tc = (index_new_data['Close'] - index_new_data['Close'].shift(1)) / atr
        index_new_data['TC'] = tc

        # ATC(Average True Change) 계산
        atc = tc.rolling(n).mean()
        index_new_data['ATC'] = atc

        return index_new_data

    def _CookStockData(self, stock_data : pd.DataFrame):

        new_data = stock_data

        try:
            # MRS 계산
            n = 20
            rs = (stock_data['Close'] / self.index_data['Close']) * 100
            rs_ma = rs.rolling(n).mean()
            mrs = ((rs / rs_ma) - 1) * 100

            # MRS를 주식 데이터에 추가
            new_data['RS'] = mrs

            # 50MA
            ma50 = stock_data['Close'].rolling(window=50).mean()
            new_data['50MA'] = ma50

            # 150MA
            ma150 = stock_data['Close'].rolling(window=150).mean()
            new_data['150MA'] = ma150

            # 200MA
            ma200 = stock_data['Close'].rolling(window=200).mean()
            new_data['200MA'] = ma200

            # 150MA Slope
            ma_diff = stock_data['150MA'].diff()
            new_data['MA150_Slope'] = ma_diff / 2

            # 200MA Slope
            ma_diff = stock_data['200MA'].diff()
            new_data['MA200_Slope'] = ma_diff / 2


            # TR 계산
            high = stock_data['High']
            low = stock_data['Low']
            prev_close = stock_data['Close'].shift(1)

            d1 = high - low
            d2 = np.abs(high - prev_close)
            d3 = np.abs(low - prev_close)
            
            tr = np.maximum(d1, d2)
            tr = np.maximum(tr, d3)

            new_data['TR'] = tr

            # DR% (Daily Range)
            daily_range_percentages = high / low

            # ADR% (20-days)
            n = 20
            adr = daily_range_percentages.rolling(n).mean()
            adr = 100 * (adr - 1)
            new_data['ADR'] = adr


            # ATR 계산
            n = 14
            atr = tr.rolling(n).mean()
            new_data['ATR'] = atr

            # TC(True Change) 계산
            tc = (stock_data['Close'] - stock_data['Close'].shift(1)) / atr
            new_data['TC'] = tc

            # ATC(Average True Change) 계산
            atc = tc.rolling(n).mean()
            new_data['ATC'] = atc

            # True Range의 합계, 최대 고가, 최소 저가 계산
            TrueRangeSum = tr.rolling(window=n).sum()
            TrueHighMax = new_data['High'].rolling(window=n).max()
            TrueLowMin = new_data['Low'].rolling(window=n).min()

            # Choppiness와 표준편차는 눈으로 보는게 낫다. 갭 상승, 돌파 같은 주요 모멘텀을 고려하지 않기 때문.
            # # Choppiness Index 계산
            # new_data['ChoppinessIndex'] = 100 * np.log10(TrueRangeSum / (TrueHighMax - TrueLowMin)) / np.log10(n)

            # # 표준편차
            # new_data['STD'] = new_data['Close'].rolling(window=14).std()

            new_index_data = self._CookIndexData(self.index_data, 14)
        
            # TRS(True Relative Strength)
            sp500_tc = new_index_data['TC']
            stock_tc = tc
            trs = stock_tc - sp500_tc
            new_data['TRS'] = trs

            # ATRS (14 days Average True Relative Strength)
            atrs = trs.rolling(n).mean()
            new_data['ATRS'] = atrs

            atrs_exp = trs.ewm(span=14, adjust=False).mean()
            new_data['ATRS_Exp'] = atrs_exp

            n = 150  # 이동평균 윈도우 크기
            # ATRS150 (150 days Average True Relative Strength)

            if len(new_data) < n:
                atrs150 = trs.rolling(len(new_data)).mean()
                new_data['ATRS150'] = atrs150
            else:
                atrs150 = trs.rolling(n).mean()
                new_data['ATRS150'] = atrs150

            # EMA 
            if len(new_data) < n:
                new_data['ATRS150_Exp'] = trs.ewm(span=len(new_data), adjust=False).mean()
            else:
                new_data['ATRS150_Exp'] = trs.ewm(span=n, adjust=False).mean()


            new_data = new_data.reindex(columns=['Symbol', 'Name', 'Industry',
                                                 'Open', 'High', 'Low', 'Close', 'Adj Close',
                                                 'Volume', 'RS','50MA', '150MA', '200MA',
                                                 'MA150_Slope', 'MA200_Slope', 
                                                 'ADR', 'TR', 'ATR', 'TC', 'ATC', 'TRS', 'ATRS', 'ATRS_Exp', 'ATRS150', 'ATRS150_Exp',
                                                 'IsOriginData_NaN'])
            

            new_data = new_data.round(5)


        except Exception as e:
            print(e)
            raise

        return new_data

    def _getDatasFromWeb(self, stock_list, trading_days, out_data_dic):
        # 모든 주식에 대해 해당 기간의 가격 데이터 가져오기
        i = 0
        stockNums = stock_list.shape[0]

        max_retries = 3
        retry_delay = 5  # seconds

        for ticker in stock_list['Symbol']:
            stock_data = fdr.DataReader(ticker, trading_days[0]) 

            if stock_data.empty:
                for retryCnt in range(max_retries):
                    print('fdr.DataReader({}) request failed. Retry request {} seconds later. '.format(ticker, retry_delay * (retryCnt+1)))
                    time.sleep(retry_delay * (retryCnt+1))
                    stock_data = fdr.DataReader(ticker, trading_days[0])
                    if stock_data.empty != True:
                        print('fdr.DataReader({}) request success!'.format(ticker))
                        break         
            
            stock_data['Symbol'] = ticker
            stock_data['Name'] = stock_list.loc[stock_list['Symbol'] == ticker, 'Name'].values[0]
            stock_data['Industry'] = stock_list.loc[stock_list['Symbol'] == ticker, 'Industry'].values[0]

            try:
                stock_data = self._CookStockData(stock_data)

                # 딕셔너리에 데이터 추가
                out_data_dic[ticker] = stock_data
                i = i+1
                print(f"{i/stockNums*100:.2f}% Done")
                
                if i > stockIterateLimit:
                    break


            except Exception as e:
                print(f"An error occurred: {e}")
                name = stock_list.loc[stock_list['Symbol'] == ticker, 'Name'].values[0]
                exception_ticker_list[ticker] = name

        with open("DataReader_exception.json", "w") as outfile:
            json.dump(exception_ticker_list, outfile)

    def _getAllDatasFromWeb(self, daysNum = 5*365, all_list = pd.DataFrame()):
        print("--- getAllDatasFromWeb ---")

        if all_list.empty:
            # 모든 상장 종목 가져오기
            nyse_list = self.get_fdr_stock_list('NYSE', daysNum)
            nasdaq_list = self.get_fdr_stock_list('NASDAQ', daysNum)
            all_list = pd.concat([nyse_list, nasdaq_list])

        # 미국 주식시장의 거래일 가져오기
        schedule = nyse.schedule(start_date=dt.date.today() - dt.timedelta(days=daysNum), end_date=dt.date.today())
        trading_days = nyse.schedule(start_date=dt.date.today() - dt.timedelta(days=daysNum), end_date=dt.date.today()).index


        # ticker: data dictionary.
        out_data_dic = {}
        self._getDatasFromWeb(all_list, trading_days, out_data_dic)

        return out_data_dic

    def _ExportDatasToCsv(self, data_dic):
        i = 0
        for ticker, data in data_dic.items():
            try:
                save_path = os.path.join('StockData', f"{ticker}.csv")
                data.to_csv(save_path, encoding='utf-8-sig')
                i = i+1
                print(f"{ticker}.csv", "is saved! {}/{}".format(i, len(data_dic.items())))
            except Exception as e:
                print(f"An error occurred: {e}")

    def getStockListFromLocalCsv(self):
        
        # [Optimize] 메모리 캐시가 있으면 먼저 메모리 캐시 사용
        if not self.cache_StockListFromLocalCsv.empty:
            return self.cache_StockListFromLocalCsv

        # [Optimize] 아니면 데이터 캐시 사용후 메모리에 적재
        try:
            with open('cache_StockListFromLocalCsv', "rb") as f:
                all_list = pickle.load(f)
                self.cache_StockListFromLocalCsv = all_list
                return all_list

        except FileNotFoundError:
            nyse_list = self.get_fdr_stock_list('NYSE')
            nasdaq_list = self.get_fdr_stock_list('NASDAQ')
            all_list = pd.concat([nyse_list, nasdaq_list])

            # all_list에서 Symbol이 csv_names에 있는 경우만 추려냄
            all_list = all_list[all_list['Symbol'].isin(self._get_csv_names())]

            # 결과 로컬에 캐싱.(가끔 전체 주식 리스트 업데이트할때 캐시 지울 필요가 있따.)
            with open('cache_StockListFromLocalCsv', "wb") as f:
                pickle.dump(all_list, f)

        return all_list

    def _SyncStockDatas(self, daysToSync = 14):
        print("-------------------SyncStockDatas-----------------\n ") 

        all_list = self.getStockListFromLocalCsv()
        
        sync_data_dic = {}

        stock_datas_fromWeb = self._getAllDatasFromWeb(daysToSync, all_list)

        tickers = []
        stock_datas_fromCsv = {}
        self.getStockDatasFromCsv(all_list, tickers, stock_datas_fromCsv)


        i = 0
        tickerNum = len(tickers)

        # tickers를 csv 파일 리스트로부터 가져오기 때문에 최근 상장한 주식은 포함하지 못하는 단점이 있다.
        # 이건 주기적으로 전체 데이터를 받거나 해야될듯?
        for ticker in tickers:
            csvData = stock_datas_fromCsv.get(ticker, pd.DataFrame())
            webData = stock_datas_fromWeb.get(ticker, pd.DataFrame())

            if csvData.empty or webData.empty:
                sync_fail_ticker_list.append(ticker)
                continue

            # 새로운 데이터프레임을 생성하여 webData_copy에 할당합니다.
            webData_copy = webData.copy()

            # IsOriginData_NaN 레이블을 추가하고, 기본값으로 False를 할당합니다.
            csvData['IsOriginData_NaN'] = False
            webData_copy['IsOriginData_NaN'] = False

            # forward fill을 사용하여 NaN값을 이전 값으로 대체하면서, IsOriginData_NaN 레이블을 변경합니다.
            #webData_copy.fillna(method='ffill', inplace=True)
            webData_copy.ffill(inplace=True)
            webData_copy.loc[webData['Open'].isnull(), 'IsOriginData_NaN'] = True

            webData = webData_copy

            # remove duplicate index from csvData.
            csvData = csvData[~csvData.index.isin(webData.index)]


            try:
                # concatenate the two dataframes
                df = pd.concat([csvData, webData])
                df = self._CookStockData(df)
                sync_data_dic[ticker] = df

                i = i+1
                print(ticker, ' sync Done. {}/{}'.format(i, tickerNum))

            except Exception as e:
                print(f"An error occurred during sync: {e}")
                name = webData.loc[webData['Symbol'] == ticker, 'Name'].values[0]
                exception_ticker_list[ticker] = name

        self._ExportDatasToCsv(sync_data_dic)

        with open('sync_fail_list.txt', 'w') as f:
            outputTexts = str()
            for ticker in sync_fail_ticker_list:
                outputTexts += str(ticker) + '\n'
            f.write(outputTexts)

    def _getCloseChanges_df(self, stock_list, ticker, start_date, end_date):
        try:
            save_path = os.path.join('StockData', f"{ticker}.csv")
            data = pd.read_csv(save_path)
            data.set_index('Date', inplace=True)
            returns = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            name = stock_list.loc[stock_list['Symbol'] == ticker, 'Name'].values[0]
            exception_ticker_list[ticker] = name
            returns = pd.Series()


        return returns

    def _getUpDownChanges_df(self, stock_list, start_date, end_date):
        # 모든 종목에 대한 전일 대비 수익률 계산
        # all_returns = pd.DataFrame()
        # for ticker in stock_list['Symbol']:
        #     returns = self._getCloseChanges_df(stock_list, ticker, start_date, end_date)
        #     all_returns[ticker] = returns
        l = [self._getCloseChanges_df(stock_list, ticker, start_date, end_date) for ticker in stock_list['Symbol']]
        all_returns = pd.concat(l, axis=1, sort=True)
        all_returns.columns = all_returns.columns.to_list()

        # all_returns의 각 행은 날짜, 각 열은 종목을 의미.
        # sum의 axis = 0은 모든 행을 더하고, axis = 1은 모든 열을 더한다.
        # (all_returns > 0)으로 all_returns의 모든 값을 True or False로 변경하고
        # sum 함수를 이용해 모든 열을 더해 상승 종목 수와 하락 종목 수를 구한다.
        daily_changes = pd.DataFrame(index=all_returns.index, columns=['up', 'down'])
        daily_changes['up'] = (all_returns > 0).sum(axis=1)
        daily_changes['down'] = (all_returns < 0).sum(axis=1)
        daily_changes['sum'] = daily_changes['up'] - daily_changes['down']
        daily_changes['ma150_changes'] = daily_changes['sum'].rolling(150).mean()

        return daily_changes


# ------------------- public -----------------------------------------------

    def get_fdr_stock_list(self, market : str, daysNum = 365*5, bIgnore_no_local_tickers = True):
        """
        bIgnore_no_local_tickers: Set False if you want to get all stock list from web when you have no local stock data.
        """
        fdr_stock_list = pd.DataFrame()
        bHaveCache = False 
        cacheFileName = f"cache_fdr_{market}_list"

        if market != 'NASDAQ' and market != 'NYSE' and market != 'S&P500':
            print(f'get_fdr_stock_list(), invalid market type {0}!', market)
            return fdr_stock_list

        try:
            with open(cacheFileName, "rb") as f:
                fdr_stock_list = pickle.load(f)
                bHaveCache = True
        except Exception as e:
            print(e)
            bHaveCache = False


        if not bHaveCache:
            fdr_stock_list = fdr.StockListing(market)
            if bIgnore_no_local_tickers:
                fdr_stock_list = fdr_stock_list[fdr_stock_list['Symbol'].isin(self._get_csv_names())]

            print('there\'s no cache. save the result newly.')
            with open(cacheFileName, "wb") as f:
                pickle.dump(fdr_stock_list, f)

        return fdr_stock_list
    

    def cookUpDownDatas(self, daysNum = 365*5):
        # S&P 500 지수의 모든 종목에 대해 매일 상승/하락한 종목 수 계산
        nyse_list = self.get_fdr_stock_list('NYSE', daysNum)
        nyse_list = nyse_list[nyse_list['Symbol'].isin(self._get_csv_names())]

        nasdaq_list = self.get_fdr_stock_list('NASDAQ', daysNum)
        nasdaq_list = nasdaq_list[nasdaq_list['Symbol'].isin(self._get_csv_names())]

        sp500_list = self.get_fdr_stock_list('S&P500', daysNum)
        sp500_list = sp500_list[sp500_list['Symbol'].isin(self._get_csv_names())]


        # 미국 주식시장의 거래일 가져오기
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.schedule(start_date=dt.date.today() - dt.timedelta(days=daysNum), end_date=dt.date.today()).index
        valid_start_date = trading_days[0]
        valid_end_date = trading_days[-1]

        daily_changes_nyse_df = self._getUpDownChanges_df(nyse_list, valid_start_date, valid_end_date)
        daily_changes_nyse_df.to_csv(os.path.join(metadata_folder, 'up_down_nyse.csv'))

        daily_changes_nasdaq_df = self._getUpDownChanges_df(nasdaq_list, valid_start_date, valid_end_date)
        daily_changes_nasdaq_df.to_csv(os.path.join(metadata_folder, 'up_down_nasdaq.csv'))

        daily_changes_sp500_df = self._getUpDownChanges_df(sp500_list, valid_start_date, valid_end_date)
        daily_changes_sp500_df.to_csv(os.path.join(metadata_folder, 'up_down_sp500.csv'))

        with open("up_down_exception.json", "w") as outfile:
                json.dump(exception_ticker_list, outfile, indent = 4)

        return daily_changes_nyse_df, daily_changes_nasdaq_df, daily_changes_sp500_df


    def cook_filter_count_data(self, filter_func, fileName : str, daysNum = 365, bAccumulateToExistingData = True):
        out_tickers = []
        out_stock_datas_dic = {}

        
        stock_data_len = 365*5 # 기본 데이터는 든든하게 미리 챙겨두기
        stock_list = self.getStockListFromLocalCsv()
        bUseCachedCSV = bAccumulateToExistingData # 갱신이 아니라 새로 데이터를 뽑는 경우 역시나 든든하게..
        self.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, stock_data_len, bUseCachedCSV)

        # 뭔가 내부 함수 에러나면 라이브러리 업그레이드부터 할 것 =ㅅ=;
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.schedule(start_date=dt.date.today() - dt.timedelta(days=daysNum), end_date=dt.date.today()).index
        valid_start_date = trading_days[0]


        today_str = dt.date.today()
        today_schedule = nyse.schedule(start_date=today_str, end_date=today_str)

        
        # 오늘이 거래일이라면 -2가 마지막 거래일
        end_date_index = 2
        filter_index_offset = 1

        # 오늘이 휴일인경우 -1이 마지막 거래일 맞음
        if today_schedule.empty:
            end_date_index = 1
            filter_index_offset = 0



        valid_end_date = trading_days[-end_date_index] # 어제가 마지막 거래일. trading_days[-1]은 오늘이고 trading_days[-2]가 어제다. (오늘이 거래일이라면)


        days = []
        cnts = []
        trading_days_num = len(trading_days)


        for i in range(end_date_index, trading_days_num):
            day = trading_days[-i]
            search_start_time = time.time()
            selected_tickers = []
            selected_tickers = filter_func(out_stock_datas_dic, -i + filter_index_offset) # filter_func[-1]은 어제이고 filter_func[-2]은 어저깨다. 1 더해줘야한다. (오늘이 거래일이라면)
            cnt = len(selected_tickers)

            #search_end_time = time.time()
            #execution_time = search_end_time - search_start_time
            #print(f"Search time elapsed: {execution_time}sec")
            days.append(day)
            cnts.append(cnt)
            print(f'{fileName} cnt of {day}: {cnt}')

        # days와 cnts 리스트로 데이터프레임 생성
        days.reverse()
        cnts.reverse()
        data = {'Date': days, 'Count': cnts}
        new_df = pd.DataFrame(data)
        new_df['Date'] = pd.to_datetime(data['Date'])
        new_df.set_index('Date', inplace=True)
        save_path = os.path.join(metadata_folder, f'{fileName}.csv')
        if bAccumulateToExistingData:
            local_df = pd.read_csv(save_path)
            local_df['Date'] = pd.to_datetime(local_df['Date'])
            local_df.set_index('Date', inplace=True)

            # 중복 인덱스 제거
            local_df = local_df[~local_df.index.isin(new_df.index)]

            concat_df = pd.concat([local_df, new_df])
            concat_df.to_csv(save_path, encoding='utf-8-sig')

        else:
            # 데이터프레임을 CSV 파일로 저장
            new_df.to_csv(save_path, encoding='utf-8-sig')


    
    def get_count_data_from_csv(self, fileName : str, daysNum = 365*2):
            """ 
            fileName: {fileName}_Counts.csv 
            """
            # ------------ nyse -------------------
            data_path = os.path.join(metadata_folder, f"{fileName}_Counts.csv")
            data = pd.read_csv(data_path)

            # 문자열을 datetime 객체로 변경
            data['Date'] = pd.to_datetime(data['Date'])

            # Date 행을 인덱스로 설정
            data.set_index('Date', inplace=True)

            # 미국 주식시장의 거래일 가져오기
            trading_days = nyse.schedule(start_date=dt.date.today() - dt.timedelta(days=daysNum), end_date=dt.date.today()).index
            startDay = trading_days[0].date()
            endDay = min(trading_days[-1], data.index[-1]).date()

            # 시작일부터 종료일까지 가져오기
            cnt_data = data[startDay:endDay]
            return cnt_data

    def downloadStockDatasFromWeb(self, daysNum = 365 * 5, bExcludeNotInLocalCsv = True):
        print("-------------------_downloadStockDatasFromWeb-----------------\n ")

        inputRes = get_yes_no_input('It will override all your local .csv files. \n Are you sure to execute this? (y/n)')
        if inputRes == False:
            return
        

        if bExcludeNotInLocalCsv == False:
            nyse_list = self.get_fdr_stock_list('NYSE', daysNum, False)
            nasdaq_list = self.get_fdr_stock_list('NASDAQ', daysNum, False)
            all_list = pd.concat([nyse_list, nasdaq_list])
        else:
            all_list = self.getStockListFromLocalCsv()
        
        sync_data_dic = {}

        stock_data_dic_fromWeb = self._getAllDatasFromWeb(daysNum, all_list)
        tickers = stock_data_dic_fromWeb.keys()

        i = 0
        tickerNum = len(tickers)
        for ticker in tickers:
            webData = stock_data_dic_fromWeb.get(ticker, pd.DataFrame())

            if webData.empty:
                sync_fail_ticker_list.append(ticker)
                continue

            # concatenate the two dataframes
            df = self._CookStockData(webData)
            sync_data_dic[ticker] = df

            i = i+1
            if i > stockIterateLimit:
                break
            print(ticker, ' download Done. {}/{}'.format(i, tickerNum))

        self._ExportDatasToCsv(sync_data_dic)

        with open('download_fail_list.txt', 'w') as f:
            outputTexts = str()
            for ticker in sync_fail_ticker_list:
                outputTexts += str(ticker) + '\n'
            f.write(outputTexts)


    # cooking 공식이 변하는 경우 로컬 데이터를 업데이트하기 위해 호출
    def cookLocalStockData(self, bUseLocalCache = False):
        print("-------------------cookLocalStockData-----------------\n ") 

        all_list = self.getStockListFromLocalCsv()


        tickers = []
        stock_datas_fromCsv = {}
        self.getStockDatasFromCsv(all_list, tickers, stock_datas_fromCsv, 365*6, bUseLocalCache)

        cooked_data_dic = {}

        for ticker in tickers:
            csvData = stock_datas_fromCsv.get(ticker, pd.DataFrame())

            if csvData.empty:
                sync_fail_ticker_list.append(ticker)
                continue

            cookedData = self._CookStockData(csvData)
            cooked_data_dic[ticker] = cookedData

            print(ticker, ' cooked!')

        self._ExportDatasToCsv(cooked_data_dic)

    def syncCsvFromWeb(self, daysNum = 14):
        self._SyncStockDatas(daysNum)

    def getUpDownDataFromCsv(self, daysNum = 365*2):
        updown_nyse = pd.DataFrame()
        updown_nasdaq = pd.DataFrame()
        updown_sp500 = pd.DataFrame()

        # ------------ nyse -------------------
        nyse_file_path = os.path.join(metadata_folder, "up_down_nyse.csv")
        data = pd.read_csv(nyse_file_path)

        # 문자열을 datetime 객체로 변경
        data['Date'] = pd.to_datetime(data['Date'])

        # Date 행을 인덱스로 설정
        data.set_index('Date', inplace=True)


        # 미국 주식시장의 거래일 가져오기
        trading_days = nyse.schedule(start_date=dt.date.today() - dt.timedelta(days=daysNum), end_date=dt.date.today()).index
        startDay = trading_days[0]
        endDay = min(trading_days[-1], data.index[-1])

        # 시작일부터 종료일까지 가져오기
        data = data[startDay:endDay]
        updown_nyse = data


        # ------------ nasdaq -------------------
        nasdaq_file_path = os.path.join(metadata_folder, "up_down_nasdaq.csv")
        data = pd.read_csv(nasdaq_file_path)

        # 문자열을 datetime 객체로 변경
        data['Date'] = pd.to_datetime(data['Date'])

        # Date 행을 인덱스로 설정
        data.set_index('Date', inplace=True)
        # 시작일부터 종료일까지 가져오기
        data = data[startDay:endDay]
        updown_nasdaq = data

        # ------------ sp500 -------------------
        sp500_file_path = os.path.join(metadata_folder, "up_down_sp500.csv")
        data = pd.read_csv(sp500_file_path)

        # 문자열을 datetime 객체로 변경
        data['Date'] = pd.to_datetime(data['Date'])

        # Date 행을 인덱스로 설정
        data.set_index('Date', inplace=True)
        # 시작일부터 종료일까지 가져오기
        data = data[startDay:endDay]
        updown_sp500 = data

        return updown_nyse, updown_nasdaq, updown_sp500

    def getStockDatasFromCsv(self, stock_list, out_tickers : list[str], out_stock_datas_dic : dict[str, pd.DataFrame], daysNum = 365*5, bUseCacheData = False):
        """
        - Caution! : if bUseCacheData is true, just return last funciton result no matter what other parameter it is. 
        - It mean that your daysNum param will affect nothing if you use cache data.
        """
        # out data must be set by extend()/update() method.
        out_tickers.clear()
        out_stock_datas_dic.clear()

        if bUseCacheData:    
            try:
                # [Optimize]
                if self.cache_getStockDatasFromCsv_out_tickers != None:
                    out_tickers.extend(self.cache_getStockDatasFromCsv_out_tickers)
                else:
                    with open('cache_getStockDatasFromCsv_out_tickers', "rb") as f:
                        cache_data = pickle.load(f)
                        out_tickers.extend(cache_data)
                        self.cache_getStockDatasFromCsv_out_tickers = out_tickers

                # [Optimize]
                if self.cache_getStockDatasFromCsv_out_stock_datas_dic != None:
                    out_stock_datas_dic.update(self.cache_getStockDatasFromCsv_out_stock_datas_dic)
                else:
                    with open('cache_getStockDatasFromCsv_out_stock_datas_dic', "rb") as f:
                        cache_data = pickle.load(f)
                        out_stock_datas_dic.update(cache_data)
                        self.cache_getStockDatasFromCsv_out_stock_datas_dic = out_stock_datas_dic

                return

            except FileNotFoundError as e:
                print('Fail to get local cache data in getStockDatasFromCsv(). Normal loading process will be excuted \n', e)

        # # [Optimize] no local cache
        else:
            self.cache_getStockDatasFromCsv_out_tickers = None
            self.cache_getStockDatasFromCsv_out_stock_datas_dic = None

        print("--- getStockDatasFromCsv ---")
        i = 0
        stockNums = len(stock_list)
        for ticker in stock_list['Symbol']:
            try:               
                csv_path = os.path.join('StockData', f"{ticker}.csv")
                data = pd.read_csv(csv_path)

                # 문자열을 datetime 객체로 변경
                data['Date'] = pd.to_datetime(data['Date'])

                # Date 행을 인덱스로 설정
                data.set_index('Date', inplace=True)
                startDay = dt.date.today() - dt.timedelta(days=daysNum)
                endDay = dt.date.today()
                # 시작일부터 종료일까지 가져오기
                data = data[startDay:endDay]
                
                out_stock_datas_dic[ticker] = data
                out_tickers.append(ticker)

                i = i+1
                print(f"{i/stockNums*100:.2f}% Done")

            except Exception as e:
                print(f"An error occurred: {e}")
                name = stock_list.loc[stock_list['Symbol'] == ticker, 'Name'].values[0]
                exception_ticker_list[ticker] = name


        # cache the result
        with open('cache_getStockDatasFromCsv_out_tickers', "wb") as f:
            pickle.dump(out_tickers, f)
        with open('cache_getStockDatasFromCsv_out_stock_datas_dic', "wb") as f:
            pickle.dump(out_stock_datas_dic, f)

    def remove_acquisition_tickers(self):
        all_list = self.getStockListFromLocalCsv()

        tickers = []
        stock_datas_fromCsv = {}
        self.getStockDatasFromCsv(all_list, tickers, stock_datas_fromCsv)


        removeTargetTickers = []

        for ticker in tickers:
            data = stock_datas_fromCsv[ticker]
            name = data['Name'].iloc[-1].lower()
            try:
                industry = data['Industry'].iloc[-1].lower()
            except Exception as e:
                removeTargetTickers.append(ticker)
                print(e)
                continue
                  
            if pd.isna(name) or pd.isna(industry):
                removeTargetTickers.append(ticker)
                continue
            if 'acquisition' in name or '기타 금융업' in industry:
                removeTargetTickers.append(ticker)
            if 'acquisition' in name or '투자 지주 회사' in industry:
                removeTargetTickers.append(ticker)


        for ticker in removeTargetTickers:
            file_path = os.path.join(data_folder, ticker + '.csv')
            if os.path.exists(file_path):
                os.remove(file_path)
                print(file_path, 'is removed from local directory!')

    def cook_Nday_ATRS150_exp(self, N=150):
        all_list = self.getStockListFromLocalCsv()

        propertyName = 'ATRS150_Exp'

        tickers = []
        stock_datas_fromCsv = {}
        self.getStockDatasFromCsv(all_list, tickers, stock_datas_fromCsv)

        date_list = None  # 변수 초기화

        atrs_dict = {}
        for ticker in tickers:
            data = stock_datas_fromCsv[ticker]
            atrs_list = data[propertyName].iloc[-N:].tolist() # 최근 N일 동안의 ATRS150 값만 가져오기=
            atrs_list = [x if not math.isnan(x) else -1 for x in atrs_list] # NaN의 경우 -1로 대체
            while len(atrs_list) < N:
                atrs_list.insert(0, -1) # 리스트앞에 -1을 추가하여 과거 NaN 데이터를 -1로 치환
            if pd.notna(atrs_list).all() and len(atrs_list) == N: # ATRS150 값이 모두 유효한 경우에만 추가
                atrs_dict[ticker] = atrs_list
                if date_list is None:  # 처음으로 유효한 atrs_list를 발견하면 날짜 정보를 가져옴
                    date_list = data.index[-N:].strftime('%Y-%m-%d').tolist()

        
        atrs_df = pd.DataFrame.from_dict(atrs_dict)
        atrs_df['Date'] = date_list
        atrs_df = atrs_df.set_index('Date')
        atrs_df = atrs_df.T # transpose

        save_path = os.path.join(metadata_folder, f'{N}day_{propertyName}.csv')
        atrs_df.to_csv(save_path, encoding='utf-8-sig', index_label='Symbol')

        return atrs_df

    def cook_ATRS150_exp_Ranks(self, N = 150):

        propertyName = 'ATRS150_Exp'

        csv_path = os.path.join(metadata_folder, f'{N}day_{propertyName}.csv')
        data = pd.read_csv(csv_path)
        data = data.set_index('Symbol')

        rank_df = data.rank(axis=0, ascending=False, method='dense')

        rank_df = rank_df

        save_path = os.path.join(metadata_folder, f'{propertyName}_Ranking.csv')
        rank_df.to_csv(save_path, encoding='utf-8-sig', index_label='Symbol')

    def get_ATRS150_exp_Ranks_Normalized(self, Symbol):

        propertyName = 'ATRS150_Exp'

        try:
            rank_df = self.get_ATRS_Ranking_df()
            serise_rankChanges = rank_df.loc[Symbol]

            max_value = len(rank_df)

            # normalize ranking [0, 1] so that value '1' is most strong ranking.
            serise_rankChanges = max_value - serise_rankChanges
            serise_rankChanges = serise_rankChanges / max_value
            
            serise_rankChanges.name = f'Rank_{propertyName}'

            return serise_rankChanges
        
        except Exception as e:
            return pd.Series()
        
    def get_ATRS150_exp_Ranks(self, Symbol):

        propertyName = 'ATRS150_Exp'

        try:
            rank_df = self.get_ATRS_Ranking_df()
            serise_rankChanges = rank_df.loc[Symbol]
            serise_rankChanges.name = f'Rank_{propertyName}'

            return serise_rankChanges
        
        except Exception as e:
            return pd.Series()
        
    def get_ATRS_Ranking_df(self):

        propertyName = 'ATRS150_Exp'

        try:
            if self.atrs_ranking_df.empty == True:
                csv_path = os.path.join(metadata_folder, f'{propertyName}_Ranking.csv')
                rank_df = pd.read_csv(csv_path)
                rank_df = rank_df.set_index('Symbol')
                self.atrs_ranking_df = rank_df
                return rank_df
            else:
                return self.atrs_ranking_df
        
        except Exception as e:
            return pd.DataFrame()
        
    def cook_Stock_GICS_df(self):
        """
        If you got the HTTP 404 error, always check  yf library version first.
        cmd: pip install --upgrade yahooquery

        sometimes there's invalid sector and industry data in the csv file because of the yf's worng database.
        in that case, just modify file in the excel editor.

        """
        all_list = self.getStockListFromLocalCsv()
        symbols = all_list['Symbol'].tolist()
        df_list = []
        requestQueue = []
        symbolsNum = len(symbols)
        errorTickers = []
        for i in range(0, symbolsNum):
            requestQueue.append(symbols[i])
            if len(requestQueue) >= 10:
                try:
                    tickers = Ticker(requestQueue)
                    profiles = tickers.get_modules("summaryProfile")
                    df = pd.DataFrame.from_dict(profiles).T
                    sector =  df['sector']
                    industry = df['industry']
                    df = pd.concat([sector, industry], axis=1)
                    df.index.name = 'Symbol'
                    df_list.append(df)
                    requestQueue.clear()
                    print(f"{i/symbolsNum*100:.2f}% Done")
                except Exception as e:
                    print(e)
                    print(requestQueue)
                    errorTickers.extend(requestQueue)
                    requestQueue.clear()
                    

        if len(requestQueue) > 0:
            try:
                tickers = Ticker(requestQueue)
                profiles = tickers.get_modules("summaryProfile")
                df = pd.DataFrame.from_dict(profiles).T
                sector =  df['sector']
                industry = df['industry']
                df = pd.concat([sector, industry], axis=1)
                df.index.name = 'Symbol'
                df_list.append(df)
                requestQueue.clear()
            except Exception as e:
                print(e)
                print(requestQueue)
                errorTickers.extend(requestQueue)
                requestQueue.clear()


        errorSymbolNum = len(errorTickers)
        symbols = errorTickers.copy()
        errorTickers.clear()
        for i in range(0, errorSymbolNum):
            requestQueue.append(symbols[i])
            if len(requestQueue) >= 1:
                try:
                    tickers = Ticker(requestQueue)
                    profiles = tickers.get_modules("summaryProfile")
                    df = pd.DataFrame.from_dict(profiles).T
                    sector =  df['sector']
                    industry = df['industry']
                    df = pd.concat([sector, industry], axis=1)
                    df.index.name = 'Symbol'
                    df_list.append(df)
                    requestQueue.clear()
                    print(f"{i/symbolsNum*100:.2f}% Done")
                except Exception as e:
                    print(e)
                    print(requestQueue)
                    errorTickers.extend(requestQueue)
                    requestQueue.clear()
                time.sleep(1)



        print('error tickers can\'t get the info')
        print(errorTickers)

        print("Complete!")

        result_df = pd.concat(df_list)
        # remove duplicated df
        result_df = result_df.loc[~result_df.index.duplicated()]
        
        result_df.index.name = 'Symbol'
        result_df.to_csv(os.path.join(metadata_folder, 'Stock_GICS.csv'), encoding='utf-8-sig')

        print('Error Tickers: ', errorTickers)


        return result_df
    

    def get_GICS_df(self):
        if self.stock_GICS_df.empty:
            # TODO: 예외처리 추가
            csv_path = os.path.join(metadata_folder, "Stock_GICS.csv")
            self.stock_GICS_df = pd.read_csv(csv_path)
            self.stock_GICS_df = self.stock_GICS_df.set_index('Symbol')
        
        return self.stock_GICS_df

   
    # method: industry or sector
    # N_day_before: rank will be calculated at the [:, -N_day_before].
    def get_industry_atrs150_ranks_mean(self, ATRS_Ranks_df, stock_GICS_df, N_day_before = 1, method : str = 'industry'):
        category = method

        tickers_by_category = stock_GICS_df.groupby(category)['Symbol'].apply(list).to_dict()

        # ex) -1, -5, -20, -60, -120, -240 (오늘, 일주전, 한달전, 3개월전, 6개월전, 1년전)
        last_ranks = ATRS_Ranks_df.iloc[:, -N_day_before]
        sectorTotalScoreMap = {}
        for category, tickers in tickers_by_category.items():
            scores = []
            # collect each sector's scores and dump lower 50%
            for ticker in tickers:          
                score = last_ranks.get(ticker) 
                if score is not None:
                    scores.append(score)

            scores = [score for score in scores if not np.isnan(score)]
            scores = sorted(scores)
            half_index = len(scores) // 2 # to dump half of scores
            half_dump_sector_scores = scores[:half_index] # dump half of scores. use 0 to halt_index (bigger numbers, bad ranks)
            length = len(half_dump_sector_scores)
            if length != 0:
                sectorTotalScoreMap[category] = sum(half_dump_sector_scores) / length

        # sort sector scores. the lower, the better
        sorted_sector_scores = dict(sorted(sectorTotalScoreMap.items(), key=lambda x: x[1]))
        return sorted_sector_scores
    

    def get_ranks_in_industries(self, ATRS_Ranks_df, stock_GICS_df):
        tickers_by_category = stock_GICS_df.groupby('industry')['Symbol'].apply(list).to_dict()

        # get last ranks
        last_ranks = ATRS_Ranks_df.iloc[:, -1]
        sorted_industries_ranks_dic = {}
        for category, tickers in tickers_by_category.items():
            # generate sorted ticker-rank dictionary
            industry_ranks = {ticker: last_ranks.get(ticker) for ticker in tickers if not pd.isna(last_ranks.get(ticker))}
            sorted_ranks = sorted(industry_ranks.items(), key=lambda x: x[1])
            sorted_industries_ranks_dic[category] = sorted_ranks

        return sorted_industries_ranks_dic

    # cook industry ranks according to the ATRS150_Exp ranks.
    def cook_long_term_industry_rank_scores(self):
        print('cook_long_term_industry_rank_scores')
        ATRS_Ranks_df = self.get_ATRS_Ranking_df()
        csv_path = os.path.join(metadata_folder, "Stock_GICS.csv")
        stock_GICS_df = pd.read_csv(csv_path)


        dn_list = []
        columnNames = []
        nDay = 365
        for i in range(1, nDay+1):
            dn = self.get_industry_atrs150_ranks_mean(ATRS_Ranks_df, stock_GICS_df, i)
            dn_list.append(dn)
            columnNames.append(f'{i}d ago')

        # reverse the list so that lastest day can be located at the last column in dataframe.
        dn_list.reverse()
        columnNames.reverse()

        d1 = dn_list[0]
        industryNames = list(d1.keys())
        industryNum = len(industryNames)
        industry_rank_history_dic = {}

        # generate industryName-list dictionary.
        for name in industryNames:
            industry_rank_history_dic[name] = []

        for i in range(1, nDay+1):
            for key, value in dn_list[i-1].items():
                try:
                    industry_rank_history_dic[key].append(value)
                except Exception as e:
                    print(e)
        rank_history_df = pd.DataFrame.from_dict(industry_rank_history_dic).transpose()
        rank_history_df = rank_history_df.rank(axis=0, ascending=True, method='dense')
        rank_history_df = (1 - rank_history_df.div(industryNum))*100
        rank_history_df = rank_history_df.round(2)
        rank_history_df.columns = columnNames
        rank_history_df.index.name = 'industry'


        # sort ranks by lastest score.
        last_col_name = rank_history_df.columns[-1]
        rank_history_df = rank_history_df.sort_values(by=last_col_name, ascending=False)


        save_path = os.path.join(metadata_folder, "long_term_industry_rank_scores.csv")
        rank_history_df.index.name = 'industry'
        rank_history_df.to_csv(save_path, encoding='utf-8-sig')
        print('long_term_industry_rank_scores.csv cooked!')

    def get_industry_atrs14_mean(self, stock_data_dic, stock_GICS_df, N_day_before = 1, method : str = 'industry'):
            category = method

            tickers_by_category = stock_GICS_df.groupby(category)['Symbol'].apply(list).to_dict()

            # ex) -1, -5, -20, -60, -120, -240 (오늘, 일주전, 한달전, 3개월전, 6개월전, 1년전)    
            sectorTotalScoreMap = {}
            for category, tickers in tickers_by_category.items():
                scores = []
                # collect each sector's atrs and dump lower 50%
                for ticker in tickers:     
                    df = stock_data_dic.get(ticker)           
                    if df is not None and (len(df) > N_day_before):
                        scores.append(df['ATRS_Exp'].iloc[-N_day_before])
                # ascending order sort
                scores = [score for score in scores if not np.isnan(score)]
                scores = sorted(scores)
                half_index = len(scores) // 2 # to dump half of scores get the index of middle
                half_dump_sector_scores = scores[half_index:] #  dump half of scores. Use from half_index to last (lower numbers)
                length = len(half_dump_sector_scores)
                if length != 0:
                    sectorTotalScoreMap[category] = sum(half_dump_sector_scores) / length

            # sort sector scores. The bigger atrs14, the better 
            sorted_sector_scores = dict(sorted(sectorTotalScoreMap.items(), key=lambda x: x[1], reverse=True))
            return sorted_sector_scores
    

    def get_percentage_AtoB(self, priceA : float, priceB : float):
        res = ((priceB - priceA)/priceA) * 100
        return res


    def get_DCR_normalized(self, inStockData: pd.DataFrame, n_day_before = -1):      
        # [DCR](%)

        d0_close = inStockData['Close'].iloc[n_day_before]
        d0_low = inStockData['Low'].iloc[n_day_before]
        d0_high = inStockData['High'].iloc[n_day_before]

        if d0_high - d0_low > 0:
            DCR = (d0_close - d0_low) / (d0_high - d0_low)
        else:
            DCR = 0.0

        return DCR

        

    # cook industry ranks according to the atrs14_exp.
    def cook_short_term_industry_rank_scores(self):
        print('cook_short_term_industry_rank_scores')

        all_list = self.getStockListFromLocalCsv()
        out_tickers = [] 
        out_stock_data_dic = {}
        self.getStockDatasFromCsv(all_list, out_tickers, out_stock_data_dic, 365, True)

        csv_path = os.path.join(metadata_folder, "Stock_GICS.csv")
        stock_GICS_df = pd.read_csv(csv_path)


        dn_list = []
        columnNames = []
        nDay = 30
        for i in range(1, nDay+1):
            dn = self.get_industry_atrs14_mean(out_stock_data_dic, stock_GICS_df, i, 'industry')
            dn_list.append(dn)
            columnNames.append(f'{i}d ago')


        # reverse the list so that lastest day can be located at the last column in dataframe.
        dn_list.reverse()
        columnNames.reverse()

        d1 = dn_list[0]
        industryNames = list(d1.keys())
        industryNum = len(industryNames)
        industry_atrs14_rank_history_dic = {}

        # generate industryName-list dictionary.
        for name in industryNames:
            industry_atrs14_rank_history_dic[name] = []

        for i in range(1, nDay+1):
            for key, value in dn_list[i-1].items():
                try:
                    industry_atrs14_rank_history_dic[key].append(value)
                except Exception as e:
                    print(e)

        rank_history_df = pd.DataFrame.from_dict(industry_atrs14_rank_history_dic).transpose()
        rank_history_df = rank_history_df.rank(axis=0, ascending=False, method='dense')
        rank_history_df = (1 - rank_history_df.div(industryNum))*100
        rank_history_df = rank_history_df.round(2)
        rank_history_df.columns = columnNames
        rank_history_df.index.name = 'industry'

        # sort ranks by lastest score.
        last_col_name = rank_history_df.columns[-1]
        rank_history_df = rank_history_df.sort_values(by=last_col_name, ascending=False)

        save_path = os.path.join(metadata_folder, "short_term_industry_rank_scores.csv")
        rank_history_df.to_csv(save_path, encoding='utf-8-sig')
        print('short_term_industry_rank_scores.csv cooked!')

    def check_NR_with_TrueRange(self, inStockData : pd.DataFrame , maxDepth = 20):
        # Calcaute NR(x) using True Range.
        last_tr = inStockData['TR'].iloc[-1]
        trueRange_NR_x = 0
        for i in range(2, maxDepth):
            tr_rangeN = inStockData['TR'][-i:]
            min_value = tr_rangeN.min()
            if min_value == last_tr:
                trueRange_NR_x = i
        return trueRange_NR_x
    
    def check_insideBar(self, inStockData: pd.DataFrame):
        d2_ago_high, d2_ago_low = inStockData['High'].iloc[-2], inStockData['Low'].iloc[-2]
        d1_ago_high, d1_ago_low = inStockData['High'].iloc[-1], inStockData['Low'].iloc[-1]
        bIsInsideBar = False
        bDoubleInsideBar = False
        if d2_ago_high > d1_ago_high and d2_ago_low < d1_ago_low:
            bIsInsideBar = True

        if bIsInsideBar:
            d3_ago_high, d3_ago_low = inStockData['High'].iloc[-3], inStockData['Low'].iloc[-3]
            if d3_ago_high > d2_ago_high and d3_ago_low < d2_ago_low:
                bDoubleInsideBar = True



        
        return bIsInsideBar, bDoubleInsideBar
    
    def get_moving_average_data(self, inStockData: pd.DataFrame, Num):
        return inStockData['Close'].rolling(window=Num).mean()

    # return boolean tuple (bIsConverging, bIsPower3)
    def check_ma_converging(self, inStockData: pd.DataFrame):

        bIsConverging = False
        bIsPower3 = False
        bIsPower2 = False

        ma10_datas = inStockData['Close'].rolling(window=10).mean()
        ma20_datas = inStockData['Close'].rolling(window=20).mean()
        ma50_datas = inStockData['Close'].rolling(window=50).mean()

        ma10 = ma10_datas.iloc[-1]
        ma20 = ma20_datas.iloc[-1]
        ma50 = ma50_datas.iloc[-1]

        dist_10_20 =  abs((ma10 - ma20)/ma20) * 100
        dist_10_50 =  abs((ma10 - ma50)/ma50) * 100
        dist_20_50 =  abs((ma50 - ma20)/ma20) * 100

        if dist_10_20 < 1.5 and dist_10_50 < 1.5 and dist_20_50 < 1.5:
            bIsConverging = True

        if bIsConverging:
            low = inStockData['Low'].iloc[-1]
            close = inStockData['Close'].iloc[-1]

            ma_list = [ma10, ma20]
            ma_min = min(ma_list)
            ma_max = max(ma_list)

            if low < ma_min and close > ma_max:
                bIsPower2 = True
           
            ma_list = [ma10, ma20, ma50]
            ma_min = min(ma_list)
            ma_max = max(ma_list)

            if low < ma_min and close > ma_max:
                bIsPower3 = True

        return (bIsConverging, bIsPower3, bIsPower2)

    
    # check the low or close was closed to the moving average
    def check_near_ma(self, inStockData: pd.DataFrame, MA_Num = 10, max_error_pct = 1.5, bUseEMA = False):

        if bUseEMA:
            ma_datas = inStockData['Close'].ewm(span=MA_Num, adjust=False).mean()
        else:
            ma_datas = inStockData['Close'].rolling(window=MA_Num).mean()

        low = inStockData['Low'].iloc[-1]
        close = inStockData['Close'].iloc[-1]
        ma = ma_datas.iloc[-1]

        

        ma_dist_from_close = abs(self.get_percentage_AtoB(close, ma))
        ma_dist_from_low = abs(self.get_percentage_AtoB(low, ma))

        return ma_dist_from_close < max_error_pct or ma_dist_from_low < max_error_pct
   

    def check_supported_by_ma(self, inStockData: pd.DataFrame, MA_Num = 10, max_error_pct = 1.5, bUseEMA = False):
        # print('low > ma and low and ma dist < max_dist_pct')
        # print('low < ma and close > ma')

        if bUseEMA:
            ma_datas = inStockData['Close'].ewm(span=MA_Num, adjust=False).mean()
        else:
            ma_datas = inStockData['Close'].rolling(window=MA_Num).mean()

        
        low = inStockData['Low'].iloc[-1]
        close = inStockData['Close'].iloc[-1]
        ma = ma_datas.iloc[-1]

        dist_from_close = abs(self.get_percentage_AtoB(close, ma))
        dist_from_low = abs(self.get_percentage_AtoB(low, ma))


        if low > ma and dist_from_low < max_error_pct:
            return True
        if low < ma and close > ma:
            return True

        return False
    

    def check_ma_touch(self, inStockData: pd.DataFrame, MA_Num = 10, bUseEMA = False, n_day_before = -1):
        """
        check if the ma price is in the day's range.
        """
        if bUseEMA:
            ma_datas = inStockData['Close'].ewm(span=MA_Num, adjust=False).mean()
        else:
            ma_datas = inStockData['Close'].rolling(window=MA_Num).mean()
     
        low = inStockData['Low'].iloc[n_day_before]
        high = inStockData['High'].iloc[n_day_before]
        ma = ma_datas.iloc[n_day_before]

        if ma <= high and ma >= low:
            return True
        
        return False


    def check_undercut_price(self, inStockData: pd.DataFrame, inPrice : float, n_day_before = -1):
        """
        check if the day's low price undercut the {price}.
        """
        low = inStockData['Low'].iloc[n_day_before]
        if low < inPrice:
            return True
        
        return False


    def check_wickplay(self, inStockData: pd.DataFrame):
        bWickPlay = False
        d2_ago_open, d2_ago_high, d2_ago_low, d2_ago_close = inStockData['Open'].iloc[-2], inStockData['High'].iloc[-2], inStockData['Low'].iloc[-2], inStockData['Close'].iloc[-2]
        d1_ago_open, d1_ago_high, d1_ago_low, d1_ago_close = inStockData['Open'].iloc[-1], inStockData['High'].iloc[-1], inStockData['Low'].iloc[-1], inStockData['Close'].iloc[-1]

        # bullish candle
        if d2_ago_open <= d2_ago_close:
            if d1_ago_high <= d2_ago_high and d1_ago_low >= d2_ago_close:
                bWickPlay = True
        # bearish candle
        else:
            if d1_ago_high <= d2_ago_high and d1_ago_low >= d2_ago_open:
                bWickPlay = True

        return bWickPlay
    

    # open equal low
    def check_OEL(self, inStockData: pd.DataFrame, n_day_before = -1):
        d1_ago_open, d1_ago_low = inStockData['Open'].iloc[n_day_before], inStockData['Low'].iloc[n_day_before]
        bOEL = d1_ago_open == d1_ago_low
        return bOEL
    
    def check_OEH(self, inStockData: pd.DataFrame, n_day_before = -1):
        d1_ago_open, d1_ago_high = inStockData['Open'].iloc[n_day_before], inStockData['High'].iloc[n_day_before]
        bOEH = d1_ago_open == d1_ago_high
        return bOEH
    



    
    


    # --------------- C/V Check list --------------
    def check_lower_lows_3(self, inStockData: pd.DataFrame, days=15):
        ll_cnt = 0
        ticker = inStockData['Symbol'].iloc[-1]
        lows = inStockData['Low'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()

        for i in range(0, days-1):
            # lower low
            prev_low = lows[i]
            today_close = closes[i+1]
            if prev_low > today_close:
                ll_cnt += 1
            else:
                ll_cnt = 0

            if ll_cnt >= 3:
                return True
        return False
    

    def check_higher_highs_3(self, inStockData: pd.DataFrame, days=15):
        hh_cnt = 0
        ticker = inStockData['Symbol'].iloc[-1]
        highs = inStockData['High'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()

        for i in range(0, days-1):
            prev_high = highs[i]
            today_close = closes[i+1]
            # higher high
            if prev_high < today_close:
                hh_cnt += 1
            else:
                hh_cnt = 0
            if hh_cnt >= 3:
                return True

        return False
    

    def _check_below_N_ma_closed_cnt(self, inStockData: pd.DataFrame, MA_Num, days):

        ticker = inStockData['Symbol'].iloc[-1]
        ma_datas = inStockData['Close'].rolling(window=MA_Num).mean().iloc[-days:].tolist()
        opens = inStockData['Open'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()

        ma_below_closed_cnt = 0
        for i in range(0, days):
            ma = ma_datas[i]
            open = opens[i]
            close = closes[i]
            if open >= ma and close < ma:
                ma_below_closed_cnt += 1
            

        return ma_below_closed_cnt


    def check_below_20ma_closed_cnt(self, inStockData: pd.DataFrame, days=15):
        return self._check_below_N_ma_closed_cnt(inStockData, 20, days)

    def check_below_50ma_closed_cnt(self, inStockData: pd.DataFrame, days=15):
        return self._check_below_N_ma_closed_cnt(inStockData, 50, days)
    
    def check_up_more_than_adr_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        closes = inStockData['Close'].iloc[-days:].tolist()
        ADRs = inStockData['ADR'].iloc[-days:].tolist()

        cnt = 0
        for i in range(0, days-1):
            prev_close = closes[i]
            close = closes[i+1]
            dist_percentage = self.get_percentage_AtoB(prev_close, close)
            if dist_percentage > 0 and abs(dist_percentage) >= ADRs[i]:
                cnt += 1
            
        return cnt
    
    def check_down_more_than_adr_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        closes = inStockData['Close'].iloc[-days:].tolist()
        ADRs = inStockData['ADR'].iloc[-days:].tolist()

        cnt = 0
        for i in range(0, days-1):
            prev_close = closes[i]
            close = closes[i+1]
            dist_percentage = self.get_percentage_AtoB(prev_close, close)
            if dist_percentage < 0 and abs(dist_percentage) >= ADRs[i]:
                cnt += 1
            
        return cnt
    
    def check_bullish_bearish_candle_count_in_n_days(self, inStockData: pd.DataFrame, days=15):
        opens = inStockData['Open'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()

        bullishCandleCnt = 0
        bearlishCandleCnt = 0

        for i in range(0, days):
            open = opens[i]
            close = closes[i]

            if open < close:
                bullishCandleCnt += 1
            else:
                bearlishCandleCnt += 1


        return bullishCandleCnt, bearlishCandleCnt
    

    def check_20ma_dist_more_than_20ptg_cnt(self, inStockData: pd.DataFrame, days=15):
        closes = inStockData['Close'].iloc[-days:].tolist()
        ma20_prices = inStockData['Close'].rolling(window=20).mean().iloc[-days:].tolist()

        big_dist_cnt = 0
        for i in range(0, days):   
            close = closes[i]
            ma20 = ma20_prices[i]

            if close > ma20:
                dist_from_20ma =  abs((ma20 - close)/close) * 100
                if dist_from_20ma > 20:
                    big_dist_cnt += 1

        return big_dist_cnt
    

    def check_close_equal_high_or_low_cnt(self, inStockData: pd.DataFrame, days=15):
        closes = inStockData['Close'].iloc[-days:].tolist()
        highs = inStockData['High'].iloc[-days:].tolist()
        lows = inStockData['Low'].iloc[-days:].tolist()
  
        close_equal_high_cnt = 0
        close_equal_low_cnt = 0


        for i in range(0, days):
            close = closes[i]
            high = highs[i]
            low = lows[i]

            if low != high:
                if close == high:
                    close_equal_high_cnt += 1
                if close == low:
                    close_equal_low_cnt += 1


        return close_equal_high_cnt, close_equal_low_cnt
    

    # gap + OEL
    def check_GOEL(self, inStockData: pd.DataFrame, n_day_before = -1):
        
        d1_ago_close = inStockData['Close'].iloc[n_day_before - 1]
        d1_ago_high = inStockData['High'].iloc[n_day_before - 1]
        d0_open =  inStockData['Open'].iloc[n_day_before]

        # Gap Start
        if d1_ago_high < d0_open:
            # more than 3% up
            if self.get_percentage_AtoB(d1_ago_close, d0_open) >= 3.0:
                # and OEL
                if self.check_OEL(inStockData, n_day_before):
                    return True
    
        return False

    def check_failed_downside_wick_BO(self, inStockData: pd.DataFrame):
        d2_ago_open, d2_ago_high, d2_ago_low, d2_ago_close = inStockData['Open'].iloc[-2], inStockData['High'].iloc[-2], inStockData['Low'].iloc[-2], inStockData['Close'].iloc[-2]
        d1_ago_open, d1_ago_high, d1_ago_low, d1_ago_close = inStockData['Open'].iloc[-1], inStockData['High'].iloc[-1], inStockData['Low'].iloc[-1], inStockData['Close'].iloc[-1]

        # bullish candle
        if d2_ago_open <= d2_ago_close:
            # 전일 심지 범위 안에서 출발
            if d1_ago_open < d2_ago_open and d1_ago_open > d2_ago_low:
                # 장중에 downside wick BO 발생
                if d1_ago_low < d2_ago_low:
                    # But 이말올 하여 종가는 전날 심지(open) 위에서 마감
                    if d1_ago_close > d2_ago_open:
                        return True
                    
        # bearish candle
        else:
            # 전일 심지 아래에서 출발
            if d1_ago_open < d2_ago_close and d1_ago_open > d2_ago_low:
                # 장중에 downside wick BO 발생
                if d1_ago_low < d2_ago_low:
                    # But 이말올하여 종가는 전날 심지(종가) 위에서 마감
                    if d1_ago_close > d2_ago_close:
                        return True
                

        return False
    


    def check_oops_up_reversal(self, inStockData: pd.DataFrame):
        low_d1_ago = inStockData['Low'].iloc[-2]
        open = inStockData['Open'].iloc[-1]
        close = inStockData['Close'].iloc[-1]

        if open < low_d1_ago and close > low_d1_ago:
            return True
        
        return False


    def check_oops_up_reversal_cnt(self, inStockData: pd.DataFrame, days=15):
        opens = inStockData['Open'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()
        lows = inStockData['Low'].iloc[-days:].tolist()

        oops_up_cnt = 0

        for i in range(0, days-1):
            prev_low = lows[i]
            open = opens[i+1]
            close = closes[i+1]

            if open < prev_low and close > prev_low:
                oops_up_cnt += 1

        return oops_up_cnt


    def check_OEL_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        opens = inStockData['Open'].iloc[-days:].tolist()
        lows = inStockData['Low'].iloc[-days:].tolist()

        condition_cnt = 0
        for i in range(0, days):
            open = opens[i]
            low = lows[i]
            if open == low:
                condition_cnt += 1
    
        return condition_cnt
    

    def check_OEH_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        opens = inStockData['Open'].iloc[-days:].tolist()
        highs = inStockData['High'].iloc[-days:].tolist()

        condition_cnt = 0
        for i in range(0, days):
            open = opens[i]
            high = highs[i]
            if open == high:
                condition_cnt += 1

        return condition_cnt
    

    # 전일 고가보다 위로 1 ADR% 만큼 주가가 상승했으나 전일 고가 아래에서 마감
    def check_squat_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        highs = inStockData['High'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()
        ADRs = inStockData['ADR'].iloc[-days:].tolist()

        squat_cnt = 0
        for i in range(0, days-1):
            prevHigh = highs[i]
            high = highs[i+1]
            close = closes[i+1]
            adr = ADRs[i]

            if high > prevHigh:
                dist_pct = self.get_percentage_AtoB(prevHigh, high)
                if dist_pct >= adr:
                    if close < prevHigh:
                        squat_cnt += 1

        return squat_cnt
    
    # 스쿼트 발생 이후 3일 안에 회복 (스쿼트가 먼저 발생해야 한다.)
    def check_squat_recovery_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        highs = inStockData['High'].iloc[-days:].tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()
        ADRs = inStockData['ADR'].iloc[-days:].tolist()

        recovery_success_cnt = 0
        for i in range(0, days-1):
            prevHigh = highs[i]
            high = highs[i+1]
            close = closes[i+1]
            adr = ADRs[i]

            if high > prevHigh:
                dist_pct = self.get_percentage_AtoB(prevHigh, high)
                if dist_pct >= adr:
                    # 스쿼트 발생!
                    if close < prevHigh:
                        # 스쿼트 발생후 3일 이내 회복 확인
                        # 1d after squat
                        if i+2 < days:
                            close_after_squat_1d = closes[i+2]
                            if high < close_after_squat_1d:
                                recovery_success_cnt += 1
                                continue

                        # 2d after squat
                        if i+3 < days:
                            close_after_squat_2d = closes[i+3]
                            if high < close_after_squat_2d:
                                recovery_success_cnt += 1
                                continue

                        # 3d after squat
                        if i+4 < days:
                            close_after_squat_3d = closes[i+4]
                            if high < close_after_squat_3d:
                                recovery_success_cnt += 1
                                continue
                    

        return recovery_success_cnt
    

    # rs_check_range: Check if RS is maximum within N days.
    # days: Check if the day with the maximum RS occurred among n days.
    def check_rs_N_day_new_high_in_n_days(self, inStockData: pd.DataFrame, atrs_ranking_df : pd.DataFrame, rs_check_range = 50, days = 15):
        ticker = inStockData['Symbol'].iloc[-1]
        rs_of_ticker = atrs_ranking_df.loc[ticker]
        rs_ranks_in_n_days = rs_of_ticker.iloc[-days:].tolist()

        for i in range(0, days):
            # [-days ~ 0]
            n_day_before_index = i - days
            range_day_before_from_index = n_day_before_index - rs_check_range
            today_rs_rank = rs_of_ticker.iloc[n_day_before_index]
            # get last 'rs_check_range' days RS ranks
            last_rs_ranks_in_range = rs_of_ticker.iloc[range_day_before_from_index : n_day_before_index]
            # get top rs rank in last 'rs_check_range' days.
            top_rank_in_last_range_days = last_rs_ranks_in_range.min()

            # rs rank new high.
            if today_rs_rank < top_rank_in_last_range_days:
                # print(ticker)
                # print(n_day_before_index, ' days before rs rank: ', today_rs_rank)
                # print('last ', rs_check_range, 'days top rs rank from ', n_day_before_index, 'days: ', top_rank_in_last_range_days)
                # print(today_rs_rank, ' < ', top_rank_in_last_range_days)
                return True

        return False
    
    def check_rs_N_day_new_low_in_n_days(self, inStockData: pd.DataFrame, atrs_ranking_df : pd.DataFrame, rs_check_range = 50, days = 15):
        ticker = inStockData['Symbol'].iloc[-1]
        rs_of_ticker = atrs_ranking_df.loc[ticker]
        rs_ranks_in_n_days = rs_of_ticker.iloc[-days:].tolist()

        for i in range(0, days):
            # [-days ~ 0]
            n_day_before_index = i - days
            range_day_before_from_index = n_day_before_index - rs_check_range
            today_rs_rank = rs_of_ticker.iloc[n_day_before_index]
            # get last 'rs_check_range' days RS ranks
            last_rs_ranks_in_range = rs_of_ticker.iloc[range_day_before_from_index : n_day_before_index]
            # get top rs rank in last 'rs_check_range' days.
            lowest_rank_in_last_range_days = last_rs_ranks_in_range.max()

            # rs rank new low.
            if today_rs_rank > lowest_rank_in_last_range_days:
                # print(ticker)
                # print(n_day_before_index, ' days before rs rank: ', today_rs_rank)
                # print('last ', rs_check_range, 'days lowest rs rank from ', n_day_before_index, 'days: ', lowest_rank_in_last_range_days)
                # print(today_rs_rank, ' > ', lowest_rank_in_last_range_days)
                return True

        return False
    



    def check_pocket_pivot(self, inStockData: pd.DataFrame):
        # Check Pocket pivot
        bIsPocketPivot = False

        # only bullish day
        bBullishDay = inStockData['Close'].iloc[-1] > inStockData['Open'].iloc[-1]
        if bBullishDay:
            # get the last 10 days from the last day. considering shift operaion below.
            recent_10_days_volumes = inStockData[-12:-1]
            last_day_volume = inStockData['Volume'].iloc[-1]

            # select price down days's volume
            price_drop_condition = recent_10_days_volumes['Close'] < recent_10_days_volumes['Close'].shift(1)
            volume_on_price_drop_days = recent_10_days_volumes.loc[price_drop_condition, 'Volume']

            # today's volume is bigger than last 10 down day's volume.
            if volume_on_price_drop_days.max() < last_day_volume:
                ma10 = self.get_moving_average_data(inStockData, 10).iloc[-1]
                low = inStockData['Low'].iloc[-1]
                ma10_to_low = self.get_percentage_AtoB(ma10, low)

                # stock price shouldn't above the ma10 more than 0.1%
                if ma10_to_low <= 0.1:
                    bIsPocketPivot = True

        return bIsPocketPivot


    def check_pocket_pivot_cnt(self, inStockData: pd.DataFrame, days=15):
        ticker = inStockData['Symbol'].iloc[-1]
        #dates = inStockData.iloc[-days:].index.tolist()
        closes = inStockData['Close'].iloc[-days:].tolist()
        opens = inStockData['Open'].iloc[-days:].tolist()
        volumes = inStockData['Volume'].iloc[-days:].tolist()
        lows = inStockData['Low'].iloc[-days:].tolist()
        ma10 = self.get_moving_average_data(inStockData, 10).iloc[-days:].tolist()
        pocket_pivot_cnt = 0

        for i in range(0, days):
            #today_date = dates[i]
            today_close = closes[i]
            today_open = opens[i]

            today_volume = volumes[i]
            today_low = lows[i]
            today_ma10 = ma10[i]

            bBullishDay = today_close > today_open
            # bullish day
            if bBullishDay:
                ma10_to_low_pcg = self.get_percentage_AtoB(today_ma10, today_low)
                # stock price shouldn't above the ma10 more than 0.1%
                if ma10_to_low_pcg <= 0.1:
                    n_day_before_index = i - days
                    last_10_days_volumes_range_index = n_day_before_index - 10
                    # [days -10, days]
                    last_10_days_volumes = inStockData[last_10_days_volumes_range_index : n_day_before_index]

                    # select price down day's volume
                    price_drop_condition = last_10_days_volumes['Close'] < last_10_days_volumes['Close'].shift(1)
                    volumes_of_price_drop_days = last_10_days_volumes.loc[price_drop_condition, 'Volume']

                    # today's day volume is bigger than last 10 down day's volume.
                    if volumes_of_price_drop_days.max() < today_volume:
                        pocket_pivot_cnt += 1

        return pocket_pivot_cnt

    # 현 주가가 150MA 및 200MA 위에 있는가?
    # 주가가 50일 MA위에 있는가?
    # 200MA, 150MA 기울기가 0보다 큰가?
    # 50MA가 150MA, 200MA 위로 상승하였는가?
    # 거래량이 거래하기 충분한가?

    # basically It's like a MMT. But ease the MA alignment condition.
    # + exclude low volume stocks
    def check_stage2(self, inStockData: pd.DataFrame, bOnly200MACheck = False):
            close = inStockData['Close'].iloc[-1]
            ma150 = inStockData['150MA'].iloc[-1]
            ma200 = inStockData['200MA'].iloc[-1]
            bIsUpperMA_150_200 = close > ma150 and close > ma200

            # early rejection for optimization
            if bOnly200MACheck == False:
                if bIsUpperMA_150_200 == False:
                    return False
            
            ma150_slope = inStockData['MA150_Slope'].iloc[-1]
            ma200_slope = inStockData['MA200_Slope'].iloc[-1]
            ma50 = inStockData['50MA'].iloc[-1]
            last_volume = inStockData['Volume'].iloc[-1]
            volume_ma50 = inStockData['Volume'].rolling(window=50).mean().iloc[-1]
            # (거래량평균 20만이상 + 10불이상 or 하루거래량 100억 이상) AND 마지막 거래량 10만주 이상
            bIsVolumeEnough = (volume_ma50 >= 200000 and close >= 10 ) or volume_ma50*close > 10000000
            bIsVolumeEnough = bIsVolumeEnough and last_volume >= 100000

            if bIsVolumeEnough == False:
                return False

            if bOnly200MACheck:
                bIsUpperMA = close > ma200

                filterMatchNum = 0
                if bIsUpperMA:
                    filterMatchNum = filterMatchNum + 1

                return filterMatchNum >= 1

            else:
                bIsUpperMA = close > bIsUpperMA_150_200 and close > ma50
                b_150ma_upper_than_200ma = ma150 > ma200
                b_50ma_biggerThan_150ma_200ma = ma50 > ma150 and ma50 > ma200
                bMA_Slope_Plus = ma150_slope > 0 and ma200_slope > 0

                filterMatchNum = 0

                if bIsUpperMA:
                    filterMatchNum = filterMatchNum + 1
                if b_150ma_upper_than_200ma or True:
                    filterMatchNum = filterMatchNum + 1
                if bMA_Slope_Plus:
                    filterMatchNum = filterMatchNum + 1
                if b_50ma_biggerThan_150ma_200ma:
                    filterMatchNum = filterMatchNum + 1

                return filterMatchNum >= 4



    def cook_top10_in_industries(self):
        ATRS_Ranks_df = self.get_ATRS_Ranking_df()
        csv_path = os.path.join(metadata_folder, "Stock_GICS.csv")
        stock_GICS_df = pd.read_csv(csv_path)
        ranks_in_industries = self.get_ranks_in_industries(ATRS_Ranks_df, stock_GICS_df)
        out_tickers = []
        out_stock_datas_dic = {}
        daysNum = 365
        stock_list = self.getStockListFromLocalCsv()
        self.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, False)

        top10_in_industries = {}
        for industry, datas in ranks_in_industries.items():
            top10_in_industry = []
            for data in datas:
                ticker, rank = data
                stock_data = out_stock_datas_dic.get(ticker, pd.DataFrame())
                if not stock_data.empty:
                    # just 200 MA check or MMT criteria
                    # bIsStage2 = self.check_stage2(stock_data, True)
                    # if bIsStage2:
                    top10_in_industry.append(data)

                    if len(top10_in_industry) == 10:
                        break

            top10_in_industries[industry] = top10_in_industry

        save_to_json(top10_in_industries, 'top10_in_industries')


    def get_long_term_industry_rank_scores_df(self):
        if self.long_term_industry_rank_df.empty:
            csv_path = os.path.join(metadata_folder, "long_term_industry_rank_scores.csv")
            self.long_term_industry_rank_df = pd.read_csv(csv_path)
            self.long_term_industry_rank_df = self.long_term_industry_rank_df.set_index('industry')

        return self.long_term_industry_rank_df
    
    def get_long_term_industry_rank_scores(self, industryName):
        df = self.get_long_term_industry_rank_scores_df()
        try:
            result = df.loc[industryName]
        except Exception as e:
            result = pd.Series()
            print(e)
            print(f"{industryName} does not exist in the DataFrame. Returning pd.Series().")

        return result

    def get_short_term_industry_rank_scores_df(self):
        if self.short_term_industry_rank_df.empty:
            csv_path = os.path.join(metadata_folder, "short_term_industry_rank_scores.csv")
            self.short_term_industry_rank_df = pd.read_csv(csv_path)
            self.short_term_industry_rank_df = self.short_term_industry_rank_df.set_index('industry')

        return self.short_term_industry_rank_df


        
        
    def get_top10_in_industries(self):
        dic = load_from_json('top10_in_industries')
        return dic
    
    def cook_stock_info_from_tickers(self, inTickers : list, fileName : str, bUseDataCache = True):
        cook_start_time = time.time()

        stock_list = self.getStockListFromLocalCsv()
        out_tickers = []
        out_stock_datas_dic = {}
        daysNum = 365
        self.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, bUseDataCache)
        atrs_ranking_df = self.get_ATRS_Ranking_df()

        nyse_list = self.get_fdr_stock_list('NYSE')
        nyse_list = nyse_list['Symbol'].tolist()

        nasdaq_list = self.get_fdr_stock_list('NASDAQ')
        nasdaq_list = nasdaq_list['Symbol'].tolist()

        stock_info_dic = {}

        for ticker in inTickers:
            gisc_df = self.get_GICS_df()
            market = ''

            if ticker in nyse_list:
                market = 'NYSE'
            if ticker in nasdaq_list:
                market = 'NASDAQ'

            if market == '':
                print('can not find ticker {0} in any nyse or nasdaq market.', ticker)
            
            try:
                industry = gisc_df.loc[ticker]['industry']
                scores = self.get_long_term_industry_rank_scores(industry)
            except:
                industry = 'None'
                scores = 'None'
                
            
            industry_score = 0
            if len(scores) != 0:
                s_rank_scores : pd.Series = self.get_long_term_industry_rank_scores(industry)
                if not s_rank_scores.empty:
                    industry_score = s_rank_scores.iloc[-1]
            else:
                print(f'can not find industry rank score from ticker: {ticker}, industry: {industry}')

            try:
                stockData = out_stock_datas_dic.get(ticker)
                atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
            except:
                continue

            lower_low_3 = -1 if self.check_lower_lows_3(stockData) else 0 # bad
            higher_high_3 = 1 if self.check_higher_highs_3(stockData) else 0 # good

            below_20ma_closed = self.check_below_20ma_closed_cnt(stockData) * -1 # bad
            below_50ma_closed = self.check_below_50ma_closed_cnt(stockData) * -1 # bad

            up_more_than_adr = self.check_up_more_than_adr_cnt(stockData) # good
            down_more_than_adr = self.check_down_more_than_adr_cnt(stockData) * -1 # bad

            bullish_candle_cnt, bearish_candle_cnt = self.check_bullish_bearish_candle_count_in_n_days(stockData) # good vs bad
            more_bullish_candle = 1 if bullish_candle_cnt > bearish_candle_cnt else -1

            ma20_disparity_more_than_20ptg = self.check_20ma_dist_more_than_20ptg_cnt(stockData) * -1 # bad

            close_equal_high, close_equal_low = self.check_close_equal_high_or_low_cnt(stockData) # bad, good
            close_equal_low *= -1

            open_equal_high = self.check_OEH_cnt(stockData) * -1 # bad
            open_equal_low = self.check_OEL_cnt(stockData) # good

            oops_up_reversal = self.check_oops_up_reversal_cnt(stockData) # good

            squat = self.check_squat_cnt(stockData) * -1 # bad
            squat_recovery = self.check_squat_recovery_cnt(stockData) # good

            rs_new_high = 1 if self.check_rs_N_day_new_high_in_n_days(stockData, atrs_ranking_df, 50, 15) else 0 # good
            rs_new_low = -1 if self.check_rs_N_day_new_low_in_n_days(stockData, atrs_ranking_df, 50, 15) else 0 # bad

            pocket_pivot_cnt = self.check_pocket_pivot_cnt(stockData)


            CV_total_cnt = (lower_low_3 + higher_high_3 + below_20ma_closed + below_50ma_closed + up_more_than_adr + down_more_than_adr + more_bullish_candle +
            ma20_disparity_more_than_20ptg + close_equal_low + close_equal_high + open_equal_high + open_equal_low + oops_up_reversal + squat + squat_recovery + 
            rs_new_high + rs_new_low + pocket_pivot_cnt)


            bPocketPivot = self.check_pocket_pivot(stockData)
            bInsideBar, bDoubleInsideBar = self.check_insideBar(stockData)
            NR_x = self.check_NR_with_TrueRange(stockData)
            bWickPlay = self.check_wickplay(stockData)
            bOEL = self.check_OEL(stockData)
            bGOEL = self.check_GOEL(stockData)

            bOopsUpReversal = self.check_oops_up_reversal(stockData)
            bFailedDownsideWickBO = self.check_failed_downside_wick_BO(stockData)
            bConverging, bPower3, bPower2 = self.check_ma_converging(stockData)

            bNearEma10 = self.check_near_ma(stockData, 10, 1.5, True)
            bNearEma21 = self.check_near_ma(stockData, 21, 1.5, True)
            bNearMa50 = self.check_near_ma(stockData, 50, 1.5)

            near_ma_list = []
            if bNearEma10:
                near_ma_list.append(10)
            if bNearEma21:
                near_ma_list.append(21)
            if bNearMa50:
                near_ma_list.append(50)
            



            bNearMA = self.check_near_ma(stockData)
            ADR = stockData['ADR'].iloc[-1]

            trandingViewFormat = market + ':' + ticker + ','

            stock_info_dic[ticker] = [market, industry, industry_score, int(atrsRank), ADR, near_ma_list, bPower3, bPocketPivot,
                                      # Volatility Contraction
                                      bInsideBar, bDoubleInsideBar, NR_x, bWickPlay,
                                      # Demand
                                      bOEL, bGOEL, bOopsUpReversal, bFailedDownsideWickBO,
                                      # C/V factors
                                      lower_low_3, higher_high_3, below_20ma_closed, below_50ma_closed, up_more_than_adr, down_more_than_adr, more_bullish_candle,
                                      ma20_disparity_more_than_20ptg, close_equal_low, close_equal_high, open_equal_high, open_equal_low, oops_up_reversal,
                                      squat, squat_recovery, rs_new_high, rs_new_low, pocket_pivot_cnt, CV_total_cnt,
                                      trandingViewFormat]


        df = pd.DataFrame.from_dict(stock_info_dic).transpose()
        columns = ['Market', 'Industry', 'Industry Score', 'RS Rank','ADR(%)', 'Near MA list(1.5%)', 'Power of 3', 'Pocket Pivot',
                   'Inside bar', 'Double Inside bar', 'NR(x)', 'Wick Play',
                   'OEL', 'bGOEL', 'Oops up reversal', 'Failed downside wick BO',
                   'lower_low_3', 'higher_high_3', 'below_20ma_closed', 'below_50ma_closed', 'up_more_than_adr', 'down_more_than_adr', 'more_bullish_candle',
                    'ma20_disparity_more_than_20ptg', 'close_equal_low', 'close_equal_high', 'open_equal_high', 'open_equal_low', 'oops_up_reversal',
                    'squat', 'squat_recovery', 'rs_50d_new_high', 'rs_50d_new_low', 'pocket_pivot_cnt' ,'CV_total_cnt',
                    'TrandingViewFormat']
        df.columns = columns
        df.index.name = 'Symbol'


        save_path = os.path.join(filteredStocks_folder, f'{fileName}.xlsx')
        
        df.to_excel(save_path, index_label='Symbol')
        print(f'{fileName}.xlsx', 'is saved!')


        # 엑셀 조건수 서식 적용
        wb = openpyxl.load_workbook(save_path)
        sheet = wb['Sheet1']
        column_range = 'H:AG'

        red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')  # 연한 빨강
        green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')  # 연한 녹색

        for column in sheet.iter_cols(min_row=2, max_row=sheet.max_row, min_col=8, max_col=len(columns)):
            for cell in column:
                if cell.value == None:
                    continue

                # NR(x)
                if cell.column == 12:
                    if cell.value > 3:
                        cell.fill = green_fill
                        continue

                if cell.value == 'TRUE':
                    cell.fill = green_fill
                elif cell.value == 'FALSE':
                    cell.fill = red_fill


                if cell.value > 0:
                    cell.fill = green_fill
                elif cell.value < 0:
                    cell.fill = red_fill

        wb.save(save_path)


        cook_end_time = time.time()
        elapsedTime = cook_end_time - cook_start_time
        print('cook elapsed time: ', elapsedTime)

    
    def date_to_index(self, df : pd.DataFrame, date_str):
        """
        - return index from end. (-N)
        """
        index_position = df.index.get_loc(pd.to_datetime(date_str))
        n_day_before_index = len(df) - index_position
        return -n_day_before_index
