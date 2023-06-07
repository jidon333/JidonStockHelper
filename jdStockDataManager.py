

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


nyse = mcal.get_calendar('NYSE')


exception_ticker_list = {}
sync_fail_ticker_list = []

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

if not os.path.exists(metadata_folder):
    os.makedirs(metadata_folder)

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
        self.csv_names = [os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith('.csv')]

        self.stock_GICS_df = pd.DataFrame()

        self.long_term_industry_rank_df = pd.DataFrame()
        self.short_term_industry_rank_df = pd.DataFrame()

# ------------------- private -----------------------------------------------
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

    def _CookStockData(self, stock_data):

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
                                                 'TR', 'ATR', 'TC', 'ATC', 'TRS', 'ATRS', 'ATRS_Exp', 'ATRS150', 'ATRS150_Exp',
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
            nyse_list = fdr.StockListing('NYSE')
            nasdaq_list = fdr.StockListing('NASDAQ')
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
        try:
            with open('cache_StockListFromLocalCsv', "rb") as f:
                all_list = pickle.load(f)
                return all_list

        except FileNotFoundError:
            nyse_list = fdr.StockListing('NYSE')
            nasdaq_list = fdr.StockListing('NASDAQ')
            all_list = pd.concat([nyse_list, nasdaq_list])

            # all_list에서 Symbol이 csv_names에 있는 경우만 추려냄
            all_list = all_list[all_list['Symbol'].isin(self.csv_names)]

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
            webData_copy.fillna(method='ffill', inplace=True)
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


    def cookUpDownDatas(self, daysNum = 365*5):
        # S&P 500 지수의 모든 종목에 대해 매일 상승/하락한 종목 수 계산
        nyse_list = fdr.StockListing('NYSE')
        nyse_list = nyse_list[nyse_list['Symbol'].isin(self.csv_names)]

        nasdaq_list = fdr.StockListing('NASDAQ')
        nasdaq_list = nasdaq_list[nasdaq_list['Symbol'].isin(self.csv_names)]

        sp500_list = fdr.StockListing('S&P500')
        sp500_list = sp500_list[sp500_list['Symbol'].isin(self.csv_names)]


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

    def downloadStockDatasFromWeb(self, daysNum = 365 * 5, bExcludeNotInLocalCsv = True):
        print("-------------------_downloadStockDatasFromWeb-----------------\n ")

        inputRes = get_yes_no_input('It will override all your local .csv files. \n Are you sure to execute this? (y/n)')
        if inputRes == False:
            return
        

        if bExcludeNotInLocalCsv == False:
            nyse_list = fdr.StockListing('NYSE')
            nasdaq_list = fdr.StockListing('NASDAQ')
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

        if bUseCacheData:
            try:
                with open('cache_getStockDatasFromCsv_out_tickers', "rb") as f:
                    cache_data = pickle.load(f)
                    out_tickers.extend(cache_data)

                    
                with open('cache_getStockDatasFromCsv_out_stock_datas_dic', "rb") as f:
                    cache_data = pickle.load(f)
                    out_stock_datas_dic.update(cache_data)

                return

            except FileNotFoundError as e:
                print('Fail to get local cache data in getStockDatasFromCsv(). Normal loading process will be excuted \n', e)


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
            name = data['Name'].iloc[-1]
            industry = data['Industry'].iloc[-1]
            if pd.isna(name) or pd.isna(industry):
                removeTargetTickers.append(ticker)
                continue
            if 'acquisition' in name or '기타 금융업' in industry:
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
            csv_path = os.path.join(metadata_folder, f'{propertyName}_Ranking.csv')
            rank_df = pd.read_csv(csv_path)
            rank_df = rank_df.set_index('Symbol')
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
            csv_path = os.path.join(metadata_folder, f'{propertyName}_Ranking.csv')
            rank_df = pd.read_csv(csv_path)
            rank_df = rank_df.set_index('Symbol')
            serise_rankChanges = rank_df.loc[Symbol]
            
            serise_rankChanges.name = f'Rank_{propertyName}'

            return serise_rankChanges
        
        except Exception as e:
            return pd.Series()
        
    def get_ATRS_Ranking_df(self):

        propertyName = 'ATRS150_Exp'

        try:
            csv_path = os.path.join(metadata_folder, f'{propertyName}_Ranking.csv')
            rank_df = pd.read_csv(csv_path)
            rank_df = rank_df.set_index('Symbol')
            return rank_df
        
        except Exception as e:
            return pd.DataFrame()
        
    def cook_Stock_GICS_df(self):
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

        print("Complete!")      
        result_df = pd.concat(df_list)
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
                industry_rank_history_dic[key].append(value)

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
                industry_atrs14_rank_history_dic[key].append(value)

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
        if d2_ago_high > d1_ago_high and d2_ago_low < d1_ago_low:
            bIsInsideBar = True
        
        return bIsInsideBar
    
    def check_pocket_pivot(self, inStockData: pd.DataFrame):
        # Check Pocket pivot
        # get the last 10 days from the last day.
        bIsPocketPivot = False

        # if last day close is plus, check the pocket pivot.
        bIsLastDayCloseUp = inStockData['Close'].iloc[-1] > inStockData['Close'].iloc[-2]
        if bIsLastDayCloseUp:
            recent_10_days_volumes = inStockData[-12:-1]
            last_day_volume = inStockData['Volume'].iloc[-1]
            # select if 
            price_drop_condition = recent_10_days_volumes['Close'] < recent_10_days_volumes['Close'].shift(1)
            volume_on_price_drop_days = recent_10_days_volumes.loc[price_drop_condition, 'Volume']
            if volume_on_price_drop_days.max() < last_day_volume:
                bIsPocketPivot = True

        return bIsPocketPivot

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

        gap_10_20 =  abs((ma10 - ma20)/ma20) * 100
        gap_10_50 =  abs((ma10 - ma50)/ma50) * 100
        gap_20_50 =  abs((ma50 - ma20)/ma20) * 100

        if gap_10_20 < 1.5 and gap_10_50 < 1.5 and gap_20_50 < 1.5:
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

    
    # check the low and close gap from the ma
    def check_near_ma(self, inStockData: pd.DataFrame, MA_Num = 10, max_gap_pct = 1.5):
        ma_datas = inStockData['Close'].rolling(window=MA_Num).mean()

        low = inStockData['Low'].iloc[-1]
        close = inStockData['Close'].iloc[-1]
        ma = ma_datas.iloc[-1]

        ma_gap_from_close = abs((ma - close)/close) * 100
        ma_gap_from_low = abs((ma - low)/low) * 100

        return ma_gap_from_close < max_gap_pct or ma_gap_from_low < max_gap_pct
   

    def check_supported_by_ma(self, inStockData: pd.DataFrame, MA_Num = 10, max_gap_pct = 1.5):
        # print('low > ma and low and ma gap < max_gap_pct')
        # print('low < ma and close > ma')

        ma_datas = inStockData['Close'].rolling(window=MA_Num).mean()
        low = inStockData['Low'].iloc[-1]
        close = inStockData['Close'].iloc[-1]
        ma = ma_datas.iloc[-1]

        gap_from_close = abs((ma - close)/close) * 100
        gap_from_low = abs((ma - low)/low) * 100


        if low > ma and gap_from_low < max_gap_pct:
            return True
        if low < ma and close > ma:
            return True

        return False


    def cook_stock_info_from_tickers(self, inTickers : list, fileName : str, bUseDataCache = True):
        stock_list = self.getStockListFromLocalCsv()
        out_tickers = []
        out_stock_datas_dic = {}
        daysNum = 365
        self.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, bUseDataCache)
        atrs_ranking_df = self.get_ATRS_Ranking_df()

        stock_info_dic = {}

        for ticker in inTickers:
            gisc_df = self.get_GICS_df()
            industry = gisc_df.loc[ticker]['industry']
            scores = self.get_long_term_industry_rank_scores(industry)
            if not scores.empty:
                industry_score = self.get_long_term_industry_rank_scores(industry).iloc[-1]
            else:
                print(f'can not find industry rank score from ticker: {ticker}, industry: {industry}')

            stockData = out_stock_datas_dic.get(ticker)
            atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
            
            bPocketPivot = self.check_pocket_pivot(stockData)
            bInsideBar = self.check_insideBar(stockData)
            NR_x = self.check_NR_with_TrueRange(stockData)
            bConverging, bPower3, bPower2 = self.check_ma_converging(stockData)
            bNearMA = self.check_near_ma(stockData)

            stock_info_dic[ticker] = [industry, industry_score, int(atrsRank), bConverging, bPocketPivot, bInsideBar, NR_x]


        df = pd.DataFrame.from_dict(stock_info_dic).transpose()
        columns = ['Industry', 'Industry Score', 'RS Rank', 'MA Converging', 'Pocket Pivot', 'Inside bar', 'NR(x)']
        df.columns = columns
        df.index.name = 'Industry'

        save_path = os.path.join(metadata_folder, f'{fileName}.csv')
        df.to_csv(save_path, encoding='utf-8-sig', index_label='Symbol')
        print(f'{fileName}.csv', 'is saved!')

           
        

    # 현 주가가 150MA 및 200MA 위에 있는가?
    # 주가가 50일 MA위에 있는가?
    # 200MA, 150MA 기울기가 0보다 큰가?
    # 50MA가 150MA, 200MA 위로 상승하였는가?
    # 거래량이 거래하기 충분한가?

    # basically It's like a MMT. But ease the MA alignment condition.
    # + exclude low volume stocks
    def check_stage2(self, inStockData: pd.DataFrame):
            close = inStockData['Close'].iloc[-1]
            ma150 = inStockData['150MA'].iloc[-1]
            ma200 = inStockData['200MA'].iloc[-1]
            bIsUpperMA_150_200 = close > ma150 and close > ma200

            # early rejection for optimization
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
                    bIsStage2 = self.check_stage2(stock_data)
                    if bIsStage2:
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
