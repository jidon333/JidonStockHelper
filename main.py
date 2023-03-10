
import datetime as dt
import glob
import json
import os
import sys
import time

import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from matplotlib.dates import num2date

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import pickle

plt.switch_backend('Qt5Agg')

# ------------------- global ----------------------
### 증권거래소 종목들 가져오기
#sp500_list = fdr.StockListing('S&P500')
#nyse_list = fdr.StockListing('NYSE')
#nasdaq_list = fdr.StockListing('NASDAQ')

### 지수 데이터 가져오기
# nasdaq = fdr.DataReader('IXIC', '2020-01-01', '2023-02-25')
# sp500 = fdr.DataReader('US500', '2020-01-01', '2023-02-25')
# dowjones = fdr.DataReader('DJI', '2020-01-01', '2023-02-25')

nyse = mcal.get_calendar('NYSE')


exception_ticker_list = {}
sync_fail_ticker_list = []
data_folder = os.path.join(os.getcwd(), 'StockData')

stockIterateLimit = 99999

markedTickerList = []


class Chart:
    def __init__(self, stockData = None, updown_nyse = None, updown_nasdaq = None, updown_sp500 = None):
        self.stockData = stockData
        self.retVal = 0
        self.annotateObj = None
        self.text_box = None

        self.updown_nyse = updown_nyse
        self.updown_nasdaq = updown_nasdaq
        self.updown_sp500 = updown_sp500

        if stockData is not None:
            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]}, figsize=(20, 10))
        else:
            self.fig, (self.ax1) = plt.subplots(figsize=(20, 10))


        plt.ion()

    def reset(self, stockData):
        self.stockData = stockData
        self.retVal = 0
        self.annotateObj = None
        self.text_box = None


    def on_close(self, event):
        print(markedTickerList)
        sys.exit()
        print("on_close")

    def on_move(self, event):
        x, y = event.xdata, event.ydata
        self.fig.format_coord = lambda x, y: f'x={x:.2f}, y={y:.2f}'
        drawAxis = None

        if event.inaxes == self.ax1:
            drawAxis = self.ax1

        if drawAxis is None:
            return

        if self.text_box is None:
            self.text_box = drawAxis.text(0.5, 0.95,
                                        f'x = {num2date(x).strftime("%Y-%m-%d")}, y = {y:.2f}',
                                        transform=drawAxis.transAxes,
                                        ha='center', va='top')
        else:
            self.text_box.set_text(
                f'x = {num2date(x).strftime("%Y-%m-%d")}, y = {y:.2f}')

        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'escape':
            plt.close()
            sys.exit()
        if event.key == 'left':
            print('left')
            #plt.close()
            self.retVal = -1
        if event.key == 'right':
            print('right')
            #plt.close()
            self.retVal = 1
        if event.key == 'enter':
            ticker = self.stockData['Symbol'].iloc[-1]
            if ticker not in markedTickerList:
                markedTickerList.append(ticker)
            print(ticker, 'is marked!')
            #plt.close()
            

    def Show(self, titleName):
        # 차트 그리기
        self.text_box = None

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)


        self.ax1.cla()

        self.ax1.plot(self.stockData['Close'], label='Close')
        self.ax1.plot(self.stockData['150MA'], label='MA150', color='blue')
        self.ax1.plot(self.stockData['50MA'], label='MA50', color='orange')

        self.ax1.legend(loc='best')
        self.ax1.grid()
        self.ax1.set_title(titleName, fontsize=30)

        self.ax2.cla()
        self.ax2.bar(self.stockData.index,
                     self.stockData['Volume'], alpha=0.3, color='blue', width=0.7)
        self.ax2.set_ylabel('Volume')

        self.ax3.cla()
        self.ax3.plot(self.stockData['RS'], label='RS')
        self.ax3.legend(loc='lower right')
        self.ax3.fill_between(self.stockData.index, self.stockData['RS'], 0, where=self.stockData['RS'] < 0, color='red', alpha=0.3)
        self.ax3.fill_between(self.stockData.index, self.stockData['RS'], 0, where=self.stockData['RS'] >= 0, color='green', alpha=0.3)
        self.ax3.axhline(y=0, color='black', linestyle='--')

        self.ax4.cla()
        self.ax4.plot(self.stockData['ATRS'], label='ATRS')
        self.ax4.legend(loc='lower right')
        self.ax4.fill_between(self.stockData.index, self.stockData['ATRS'], 0, where=self.stockData['ATRS'] < 0, color='red', alpha=0.3)
        self.ax4.fill_between(self.stockData.index, self.stockData['ATRS'], 0, where=self.stockData['ATRS'] >= 0, color='green', alpha=0.3)
        self.ax4.axhline(y=0, color='black', linestyle='--')


        plt.draw()

        while True:
            plt.pause(0.01)
            if self.retVal != 0:
                break
            if not plt.fignum_exists(self.fig.number):
                break

    def draw_updown_chart(self):

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)


        # Plot sum and MA150 changes
        self.ax1.plot(self.updown_nyse.index, self.updown_nyse['ma150_changes'], label='NYSE 150 MI')
        self.ax1.plot(self.updown_nasdaq.index, self.updown_nasdaq['ma150_changes'], label='NASDAQ 150 MI')
        self.ax1.plot(self.updown_sp500.index, self.updown_sp500['ma150_changes'], label='SP500 150 MI')

        # Add legend, title, and grid
        self.ax1.legend()
        self.ax1.set_title('NYSE, NASDAQ, SP500 150 MI')
        self.ax1.grid()

        # Add a horizontal line at y=0
        self.ax1.axhline(y=0, color='black', linestyle='--')

        # Show the chart and pause the execution
        plt.draw()

        while True:
            plt.pause(0.01)
            if self.retVal != 0:
                break
            if not plt.fignum_exists(self.fig.number):
                break

class StockDataManager:
    def __init__(self):
        self.index_data = fdr.DataReader('US500')

        self.csv_names = [os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith('.csv')]

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

        # MRS 계산
        try:
            n = 20
            rs = (stock_data['Close'] / self.index_data['Close']) * 100
            rs_ma = rs.rolling(n).mean()
            mrs = ((rs / rs_ma) - 1) * 100

            # MRS를 주식 데이터에 추가
            new_data['RS'] = mrs

            # 150MA
            ma150 = stock_data['Close'].rolling(window=150).mean()
            new_data['150MA'] = ma150

            # 150MA Slope
            ma_diff = stock_data['150MA'].diff()
            new_data['MA150_Slope'] = ma_diff / 2

            # 50MA
            ma50 = stock_data['Close'].rolling(window=50).mean()
            new_data['50MA'] = ma50

            # TR 계산
            high = stock_data['High']
            low = stock_data['Low']
            prev_close = stock_data['Close'].shift(1)

            d1 = high - low
            d2 = np.abs(high - prev_close)
            d3 = np.abs(low - prev_close)
            
            tr = np.maximum(d1, d2)
            tr = np.maximum(tr, d3)

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

            # ATRS(Average True Relative Strength)
            atrs = trs.rolling(n).mean()
            new_data['ATRS'] = atrs

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
        nyse_list = fdr.StockListing('NYSE')
        nasdaq_list = fdr.StockListing('NASDAQ')
        all_list = pd.concat([nyse_list, nasdaq_list])

        # all_list에서 Symbol이 csv_names에 있는 경우만 추려냄
        all_list = all_list[all_list['Symbol'].isin(self.csv_names)]

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

            # remove duplicate index from df2
            webData = webData[~webData.index.isin(csvData.index)]

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
        daily_changes_nyse_df.to_csv(os.path.join(data_folder, 'up_down_nyse.csv'))

        daily_changes_nasdaq_df = self._getUpDownChanges_df(nasdaq_list, valid_start_date, valid_end_date)
        daily_changes_nasdaq_df.to_csv(os.path.join(data_folder, 'up_down_nasdaq.csv'))

        daily_changes_sp500_df = self._getUpDownChanges_df(sp500_list, valid_start_date, valid_end_date)
        daily_changes_sp500_df.to_csv(os.path.join(data_folder, 'up_down_sp500.csv'))

        with open("up_down_exception.json", "w") as outfile:
                json.dump(exception_ticker_list, outfile, indent = 4)

        return daily_changes_nyse_df, daily_changes_nasdaq_df, daily_changes_sp500_df

    def downloadStockDatasFromWeb(self, daysNum = 365 * 5):
        print("-------------------_downloadStockDatasFromWeb-----------------\n ")

        while(True):
            print('It will override all your local .csv files. \n Are you sure to execute this? (y/n)')
            k = input()
            if(k == 'y'):
                break
            elif(k == 'n'):
                return
            else:
                print('input \'y\' or \'n\' to continue...')



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

        with open('download_fail_list.txt', 'wb') as f:
            outputTexts = str()
            for ticker in sync_fail_ticker_list:
                outputTexts += str(ticker) + '\n'
            f.write(outputTexts)


    # cooking 공식이 변하는 경우 로컬 데이터를 업데이트하기 위해 호출
    def cookLocalStockData(self):
        print("-------------------cookLocalStockData-----------------\n ") 

        all_list = self.getStockListFromLocalCsv()


        tickers = []
        stock_datas_fromCsv = {}
        self.getStockDatasFromCsv(all_list, tickers, stock_datas_fromCsv)

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
        nyse_file_path = os.path.join(data_folder, "up_down_nyse.csv")
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
        nasdaq_file_path = os.path.join(data_folder, "up_down_nasdaq.csv")
        data = pd.read_csv(nasdaq_file_path)

        # 문자열을 datetime 객체로 변경
        data['Date'] = pd.to_datetime(data['Date'])

        # Date 행을 인덱스로 설정
        data.set_index('Date', inplace=True)
        # 시작일부터 종료일까지 가져오기
        data = data[startDay:endDay]
        updown_nasdaq = data

        # ------------ sp500 -------------------
        sp500_file_path = os.path.join(data_folder, "up_down_sp500.csv")
        data = pd.read_csv(sp500_file_path)

        # 문자열을 datetime 객체로 변경
        data['Date'] = pd.to_datetime(data['Date'])

        # Date 행을 인덱스로 설정
        data.set_index('Date', inplace=True)
        # 시작일부터 종료일까지 가져오기
        data = data[startDay:endDay]
        updown_sp500 = data

        return updown_nyse, updown_nasdaq, updown_sp500

    def getStockDatasFromCsv(self, stock_list, out_tickers, out_stock_datas_dic, daysNum = 365*5):
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


def DrawStockDatas(stock_datas_dic, tickers, maxCnt = -1):
    stock_data = stock_datas_dic[tickers[0]]
    chart = Chart(stock_data)
    chart.Show(tickers[0])
    if maxCnt > 0:
        num = maxCnt
    else:
        num = len(tickers)

    index = 1

    while(index < num):
        ticker = tickers[index]
        print("print ", ticker)

        stock_data = stock_datas_dic[ticker]
        chart.reset(stock_data)
        chart.Show(ticker)
        index = index + chart.retVal

        if index < 0:
            index = 0

        if index >= num:
            index = num-1

        print("while loop is running")

def DrawMomentumIndex(updown_nyse, updown_nasdaq, updown_sp500):
    # show first element
    chart = Chart(None, updown_nyse, updown_nasdaq, updown_sp500)
    chart.draw_updown_chart()

def remove_outdated_tickers():
    with open("exception.json", "r") as outfile:
        data = json.load(outfile)
        keys = data.keys()

        for key in keys:
            file_path = os.path.join(data_folder, key + '.csv')
            if os.path.exists(file_path):
                os.remove(file_path)



print("Select the chart type. \n \
      1: Stock Data Chart \n \
      2: Momentum Index Chart \n \
      3: Sync local .csv datas from web \n \
      4: cook up-down datas using local csv files. \n \
      5: cook local stock data. \n \
      6: Download stock data from web and overwrite local files. (It will takes so long...)\n")
index = int(input())

sd = StockDataManager()
#stock_list = fdr.StockListing('S&P500')
stock_list = sd.getStockListFromLocalCsv()

out_tickers = []
out_stock_datas_dic = {}

if index == 1:
    try:
        with open('temp_tickers', "rb") as f:
            out_tickers = pickle.load(f)

        with open('temp_stock_datas_dic', 'rb') as f:
            out_stock_datas_dic = pickle.load(f)
            

    except FileNotFoundError:
        sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, 365*3)


    # 데이터를 파일에 저장
    with open('temp_tickers', "wb") as f:
        pickle.dump(out_tickers, f)

    # 파일에서 데이터를 불러옴
    with open('temp_stock_datas_dic', "wb") as f:
        pickle.dump(out_stock_datas_dic, f)

    ##---------------- 조건식 -----------------------------------------------------

    filtered_data = pd.DataFrame(columns=['RS', 'MA150_Slope', 'Close', '150MA'])

    # 각 종목들의 'RS'와 'MA150_Slope' 값을 추출하여 DataFrame으로 저장합니다.
    for ticker, stock_data in out_stock_datas_dic.items():
        rs = stock_data['RS'].iloc[-1]
        ma150_slope = stock_data['MA150_Slope'].iloc[-1]
        close = stock_data['Close'].iloc[-1]
        ma150 = stock_data['150MA'].iloc[-1]


        filtered_data.loc[ticker] = [rs, ma150_slope, close, ma150]

        # 150ma 이격도 검사
        diff_ma_close = abs(close - ma150)
        diffRatio = diff_ma_close / close
        if diffRatio < 0.1:
            filtered_data.loc[ticker] = [rs, ma150_slope, close, ma150]

    

    # 조건에 맞게 필터링하여 선택된 종목들을 리스트로 저장합니다.
    print(filtered_data)
    selected_tickers = filtered_data[(filtered_data['RS'] > -2) & (filtered_data['MA150_Slope'] > 0)].index.tolist()
    print('selected by technical data: \n', selected_tickers)



    data = pd.read_csv('ScreenerList.csv')
    quantTickers = data['종목코드'].tolist()
    quantTickers = [s.split(':')[1] for s in quantTickers]

    selected_tickers = list(set(selected_tickers) & set(quantTickers))
    selected_tickers.sort()


    # '종목명'에 '애퀴지션'이 들어가는 종목 제외
    new_selected_tickers = []
    for ticker in selected_tickers:
        name = data.loc[data['종목코드'].str.contains(f":{ticker}"), '종목명'].values[0]
        if '애퀴지션' not in name:
            new_selected_tickers.append(ticker)
    selected_tickers = new_selected_tickers

    print('filtered by quant data: \n', selected_tickers)


    print('selected tickers num: ', len(selected_tickers))
    DrawStockDatas(out_stock_datas_dic, selected_tickers)

    # -----------------------------------------------------------------------------------------------------



elif index == 2:
    updown_nyse, updown_nasdaq, updown_sp500 = sd.getUpDownDataFromCsv(365*3)
    DrawMomentumIndex(updown_nyse, updown_nasdaq, updown_sp500)
elif index == 3:
    sd.syncCsvFromWeb(7)
elif index == 4:
    sd.cookUpDownDatas()
elif index == 5:
    sd.cookLocalStockData()
elif index == 6:
    sd.downloadStockDatasFromWeb()


#TODO: RS Slope
#TODO: ATRS or RS가 양전으로 바뀐 종목 찾기?
#TODO: 20MA와 150MA 사이가 좁은거 찾는것도 괜춘할듯??