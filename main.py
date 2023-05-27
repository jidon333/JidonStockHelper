
import datetime as dt
import glob
import json
import os
import sys
import time

import FinanceDataReader as fdr

import yahooquery as yq 
from yahooquery import Ticker


import matplotlib.pyplot as plt
from matplotlib.dates import num2date
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import TextBox, Button

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import pickle
import math

import requests
from bs4 import BeautifulSoup
import re

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
metadata_folder = os.path.join(data_folder, 'MetaData')

stockIterateLimit = 99999

markedTickerList = []

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




class Chart:
    def __init__(self, stockData = None, updown_nyse = None, updown_nasdaq = None, updown_sp500 = None):
        
        self.stockManager : StockDataManager = None

        self.stockData = stockData
        self.retVal = 0
        self.annotateObj = None
        self.text_box_coordinate = None
        self.text_box_info = None

        self.updown_nyse = updown_nyse
        self.updown_nasdaq = updown_nasdaq
        self.updown_sp500 = updown_sp500

        self.top10_in_industries : dict = sd.get_top10_in_industries()

        if stockData is not None:
            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]}, figsize=(20, 10))
            self.fig.subplots_adjust(left=0.24, right=0.9)

        else:
            self.fig, (self.ax1) = plt.subplots(figsize=(20, 10))


        plt.ion()

    def set_stock_manager(self, inStockManager):
        self.stockManager = inStockManager

    def get_sector(self, ticker):
        GICS_df = self.stockManager.get_GICS_df()
        return GICS_df.loc[ticker]['sector']
    def get_industry(self, ticker):
        GICS_df = self.stockManager.get_GICS_df()
        return GICS_df.loc[ticker]['industry']
    
    def get_long_term_industry_rank_scores(self, industryName):
        return sd.get_long_term_industry_rank_scores(industryName)
    
    def reset(self, stockData):
        self.stockData = stockData
        self.retVal = 0
        self.annotateObj = None
        self.text_box_coordinate = None


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

        if self.text_box_coordinate is None:
            self.text_box_coordinate = drawAxis.text(0.5, 0.95,
                                        f'x = {num2date(x).strftime("%Y-%m-%d")}, y = {y:.2f}',
                                        transform=drawAxis.transAxes,
                                        ha='center', va='top')
        else:
            self.text_box_coordinate.set_text(
                f'x = {num2date(x).strftime("%Y-%m-%d")}, y = {y:.2f}')

        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'escape':
            plt.close()
            sys.exit()
        if event.key == 'left':
            print('left')
            self.retVal = -1
        if event.key == 'right':
            print('right')
            self.retVal = 1
        if event.key == 'enter':
            ticker = self.stockData['Symbol'].iloc[-1]
            if ticker not in markedTickerList:
                markedTickerList.append(ticker)
            print(ticker, 'is marked!')
            

    def Show(self, titleName, ranks_atrs, curr_rank):
        # 차트 그리기

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        name = self.stockData['Name'][0]
        ticker = titleName
        font_path = 'NanumGothic.ttf'
        fontprop = fm.FontProperties(fname=font_path, size=30)
        industryKor = self.stockData['Industry'][0]
        sectorText = self.get_sector(ticker)
        industryText =  self.get_industry(ticker)
        titleStr = f"{ticker} ({name}) \n {industryKor},  ATRS Rank: {int(curr_rank)}th"
        trs = self.stockData['TRS'].iloc[-1]
        tc = self.stockData['TC'].iloc[-1]

        top10 = self.top10_in_industries.get(industryText)
        top10_len = 0
        if top10 != None:
            top10_len = len(top10)


        trueRange_NR_x = self.stockManager.check_NR_with_TrueRange(self.stockData)
        bIsInsideBar = self.stockManager.check_insideBar(self.stockData)
        bIsPocketPivot = self.stockManager.check_pocket_pivot(self.stockData)
        bIsMaConverging, bIsPower3 = self.stockManager.check_ma_converging(self.stockData)
      
        # 좌측 info text box 설정
        if self.text_box_info != None:
            self.text_box_info.remove()
            self.text_box_info = None

        if not pd.isna(industryText):
            industryRanks_long = self.get_long_term_industry_rank_scores(industryText)

            msg = (
            f"========================\n"
            f"Ticker: {ticker}\n"
            f"Sector: {sectorText}\n"
            f"Industry: {industryText}\n"
            f"TRS: {trs}\n"
            f"TC: {tc}\n"
            f"NR(x): {trueRange_NR_x}\n"
            f"Inside bar: {bIsInsideBar}\n"
            f"Pocket Pivot: {bIsPocketPivot}\n"
            f"MA Converging: {bIsMaConverging}\n"
            f"Power of 3: {bIsPower3}\n"
            f"Industry RS Score : {int(industryRanks_long['1d ago'])} \n\n"
            
            )

            msg += f"Top10 in \n'{industryText}' group \n"
            for i in range(0, top10_len):
                top10_ticker = top10[i][0]
                top10_rank = top10[i][1]
                msg += f"{i+1}st: {top10_ticker}, {int(top10_rank)}\n"
            
            msg += f"\n========================"


            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            self.text_box_info = self.fig.text(0.01, 0.9,
                                        msg,
                                        transform=self.fig.transFigure, fontsize=14,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                        verticalalignment='top', horizontalalignment='left')


        self.ax1.cla()

        # 그래프 설정
        self.ax1.plot(self.stockData['Close'], label='Close')
        self.ax1.plot(self.stockData['200MA'], label='MA200', color='green')
        self.ax1.plot(self.stockData['150MA'], label='MA150', color='blue')
        self.ax1.plot(self.stockData['50MA'], label='MA50', color='orange')

        self.ax1.legend(loc='lower left')
        self.ax1.grid()
        self.ax1.set_title(titleStr, fontproperties=fontprop)



        # Draw industry ranks to the ax2 instead of volumes
        stockDataLen = len(self.stockData['Close'].index)
        # Can not draw chart if the stock data len is longer than industry rank data.
        if stockDataLen < len(industryRanks_long):
            long_industry_rank_datas = industryRanks_long.values[-stockDataLen:]
            long_industry_rank_reindexed = pd.Series(long_industry_rank_datas, index=self.stockData['Close'].index)

            self.ax2.cla()
            self.ax2.plot(long_industry_rank_reindexed, label ='Industry RS score', color='green')
            self.ax2.set_ylim([0, 100])
            self.ax2.legend(loc='lower left')
            self.ax2.axhline(y=50, color='black', linestyle='--')
        else:
            # Draw volume chart to the ax2
            self.ax2.cla()
            self.ax2.bar(self.stockData.index,
                        self.stockData['Volume'], alpha=0.3, color='blue', width=0.7)
            self.ax2.set_ylabel('Volume')

        ############## Rank data를 그래프에 추가하기 ###############
        ranks_atrs_exp_df = ranks_atrs.to_frame()
        ranks_atrs_exp_df.index = pd.to_datetime(ranks_atrs_exp_df.index) # 문자열을 datetime 객체로 변경


        # self.stockData와 rank_df를 합치기 위해 index 기준으로 join
        self.stockData = self.stockData.join(ranks_atrs_exp_df, how='left')

        # NaN 값을 0으로 대체
        self.stockData.fillna(0, inplace=True)

        # ax3에 그래프 그리기
        self.ax3.cla()

        self.ax3.set_ylim([0, 1])
        if len(ranks_atrs_exp_df) != 0:
            self.ax3.plot(self.stockData['Rank_ATRS150_Exp'], label='Rank_ATRS150_Exp', color='red', alpha=0.5)
        self.ax3.legend(loc='lower left')
        self.ax3.axhline(y=0.5, color='black', linestyle='--')
 
        self.ax4.cla()
        self.ax4.set_ylim([-0.5, 0.5])
        self.ax4.plot(self.stockData['ATRS_Exp'], label='ATRS_Exp')
        self.ax4.legend(loc='lower left')
        self.ax4.fill_between(self.stockData.index, self.stockData['ATRS_Exp'], 0, where=self.stockData['ATRS_Exp'] < 0, color='red', alpha=0.3)
        self.ax4.fill_between(self.stockData.index, self.stockData['ATRS_Exp'], 0, where=self.stockData['ATRS_Exp'] >= 0, color='green', alpha=0.3)
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

            # ATRS150 (150 days Average True Relative Strength)
            atrs150 = trs.rolling(150).mean()
            new_data['ATRS150'] = atrs150

            atrs150_exp = trs.ewm(span=150, adjust=False).mean()
            new_data['ATRS150_Exp'] = atrs150_exp

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
        ATRS_Ranks_df = sd.get_ATRS_Ranking_df()
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

        all_list = sd.getStockListFromLocalCsv()
        out_tickers = [] 
        out_stock_data_dic = {}
        sd.getStockDatasFromCsv(all_list, out_tickers, out_stock_data_dic, 365, True)

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
            
            ma_list = [ma10, ma20, ma50]
            ma_min = min(ma_list)
            ma_max = max(ma_list)

            if low < ma_min and close > ma_max:
                bIsPower3 = True

        return (bIsConverging, bIsPower3)

    
    # check the low and close gap from the ma
    def check_near_ma(self, inStockData: pd.DataFrame, MA_Num = 10, max_gap_pct = 1.5):
        ma_datas = inStockData['Close'].rolling(window=MA_Num).mean()

        low = inStockData['Low'].iloc[-1]
        close = inStockData['Close'].iloc[-1]
        ma = ma_datas.iloc[-1]

        gap_from_close = abs((ma - close)/close) * 100
        gap_from_low = abs((ma - low)/low) * 100

        return gap_from_close < max_gap_pct or gap_from_low < max_gap_pct
   

    def check_supported_by_ma(self, inStockData: pd.DataFrame, MA_Num = 10, max_gap_pct = 1.5):
        print('low > ma and low and ma gap < max_gap_pct')
        # OR
        print('low < ma and close > ma')


    def cook_stock_info_from_tickers(self, inTickers : list, fileName : str, bUseDataCache = True):
        stock_list = self.getStockListFromLocalCsv()
        out_tickers = []
        out_stock_datas_dic = {}
        daysNum = 365
        sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, bUseDataCache)
        atrs_ranking_df = sd.get_ATRS_Ranking_df()

        stock_info_dic = {}

        for ticker in inTickers:
            gisc_df = sd.get_GICS_df()
            industry = gisc_df.loc[ticker]['industry']
            scores = sd.get_long_term_industry_rank_scores(industry)
            if not scores.empty:
                industry_score = sd.get_long_term_industry_rank_scores(industry).iloc[-1]
            else:
                print(f'can not find industry rank score from ticker: {ticker}, industry: {industry}')

            stockData = out_stock_datas_dic.get(ticker)
            atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
            
            bPocketPivot = sd.check_pocket_pivot(stockData)
            bInsideBar = sd.check_insideBar(stockData)
            NR_x = sd.check_NR_with_TrueRange(stockData)
            bConverging, bPower3 = sd.check_ma_converging(stockData)
            bNearMA = sd.check_near_ma(stockData)

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
        ATRS_Ranks_df = sd.get_ATRS_Ranking_df()
        csv_path = os.path.join(metadata_folder, "Stock_GICS.csv")
        stock_GICS_df = pd.read_csv(csv_path)
        ranks_in_industries = sd.get_ranks_in_industries(ATRS_Ranks_df, stock_GICS_df)
        out_tickers = []
        out_stock_datas_dic = {}
        daysNum = 365
        stock_list = sd.getStockListFromLocalCsv()
        sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, False)

        top10_in_industries = {}
        for industry, datas in ranks_in_industries.items():
            top10_in_industry = []
            for data in datas:
                ticker, rank = data
                stock_data = out_stock_datas_dic.get(ticker, pd.DataFrame())
                if not stock_data.empty:
                    bIsStage2 = sd.check_stage2(stock_data)
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

def DrawStockDatas(stock_datas_dic, tickers, inStockManager, maxCnt = -1):
    stock_data = stock_datas_dic[tickers[0]]
    chart = Chart(stock_data)
    chart.set_stock_manager(inStockManager)
    ranksATRS = inStockManager.get_ATRS150_exp_Ranks_Normalized(tickers[0])
    currRank = inStockManager.get_ATRS150_exp_Ranks(tickers[0]).iloc[-1]
    chart.Show(tickers[0], ranksATRS, currRank)
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
        ranksATRS = inStockManager.get_ATRS150_exp_Ranks_Normalized(ticker)
        currRank = inStockManager.get_ATRS150_exp_Ranks(ticker).iloc[-1]

        chart.Show(ticker, ranksATRS, currRank)
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
    with open("DataReader_exception.json", "r") as outfile:
        data = json.load(outfile)
        keys = data.keys()

        for key in keys:
            file_path = os.path.join(data_folder, key + '.csv')
            if os.path.exists(file_path):
                os.remove(file_path)
                print(file_path, 'is removed!')


def remove_local_caches():
    local_dir = os.path.join(os.getcwd())
    for filename in os.listdir(local_dir):
        if filename.startswith('cache_'):
            os.remove(os.path.join(local_dir, filename))


def get_yes_no_input(qustionString):
    while(True):
        print(qustionString)
        k = input()
        if(k == 'y' or k == 'Y'):
            return True
        elif(k == 'n' or k == 'N'):
            return False
        else:
            print('input \'y\' or \'n\' to continue...')


def cook_infos_from_last_searched_tickers(inStockManager : StockDataManager, inFileName):
    tickers = []
    try:
        with open('cache_tickers', "rb") as f:
            tickers = pickle.load(f)
        
    except FileNotFoundError:
        print(f'Can not find your last searched ticker list file \'cache_tickers\' ')

    if len(tickers) != 0:
        sd.cook_stock_info_from_tickers(tickers, inFileName)



print("Select the chart type. \n \
      1: Stock Data Chart \n \
      2: Momentum Index Chart \n \
      3: Sync local .csv datas from web \n \
      4: cook up-down datas using local csv files. \n \
      5: cook local stock data. \n \
      6: Download stock data from web and overwrite local files. (It will takes so long...) \n \
      7: cook ATRS Ranking \n \
      8: cook industry Ranking \n \
      9: cook stock infos from the last searched tickers \n")

index = int(input())

sd = StockDataManager()
out_tickers = []
out_stock_datas_dic = {}

if index == 1:

    bUseLocalCache = get_yes_no_input('Do you want to see last chart data? \n It will use cached local data and it will save loading time. \n (y/n)')
    daysNum = int(365)

    if bUseLocalCache:
        try:
            with open('cache_tickers', "rb") as f:
                out_tickers = pickle.load(f)

            with open('cache_stock_datas_dic', 'rb') as f:
                out_stock_datas_dic = pickle.load(f)
            
        except FileNotFoundError:
            print('Can not find your last stock chart data in local.\n The chart data will be re-generated. ')
            bUseLocalCache = False
            stock_list = sd.getStockListFromLocalCsv()
            sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum)
    else:
        stock_list = sd.getStockListFromLocalCsv()
        sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, True)


    ##---------------- 조건식 -----------------------------------------------------

    # Collect Technical data for screening.
    if not bUseLocalCache:

        search_start_time = time.time()

        selected_tickers = []
    
        atrs_ranking_df = sd.get_ATRS_Ranking_df()

        for ticker, inStockData in out_stock_datas_dic.items():

            close = inStockData['Close'].iloc[-1]
            ma150 = inStockData['150MA'].iloc[-1]
            ma200 = inStockData['200MA'].iloc[-1]
            bIsUpperMA_150_200 = close > ma150 and close > ma200

            # early rejection for optimization
            if bIsUpperMA_150_200 == False:
                continue
            

            rs = inStockData['RS'].iloc[-1]
            ma150_slope = inStockData['MA150_Slope'].iloc[-1]
            ma200_slope = inStockData['MA200_Slope'].iloc[-1]
       
            ma50 = inStockData['50MA'].iloc[-1]
            ma20 = inStockData['Close'].rolling(window=20).mean().iloc[-1]
            ma10 = inStockData['Close'].rolling(window=10).mean().iloc[-1]
            tr = inStockData['TR'].iloc[-1]
            tc = inStockData['TC'].iloc[-1]
            atr = inStockData['ATR'].iloc[-1]
            last_volume = inStockData['Volume'].iloc[-1]
            volume_ma50 = inStockData['Volume'].rolling(window=50).mean().iloc[-1]
            # (거래량평균 20만이상 + 10불이상 or 하루거래량 100억 이상) AND 마지막 거래량 10만주 이상
            bIsVolumeEnough = (volume_ma50 >= 200000 and close >= 10 ) or volume_ma50*close > 10000000
            bIsVolumeEnough = bIsVolumeEnough and last_volume >= 100000
            bIsUpperMA = close > bIsUpperMA_150_200 and close > ma50
            b_150ma_upper_than_200ma = ma150 > ma200
            bMA_Slope_Plus = ma150_slope > 0 and ma200_slope > 0
            b_50ma_biggerThan_150ma_200ma = ma50 > ma150 and ma50 > ma200

            bIsRSGood = rs > 0

            gap_from_10ma =  abs((ma10 - close)/close) * 100
            gap_from_20ma =  abs((ma20 - close)/close) * 100
            gap_from_50ma =  abs((ma50 - close)/close) * 100
            gap_from_200ma =  abs((ma200 - close)/close) * 100
            bIsPriceNearMA = (gap_from_10ma < 5) or (gap_from_20ma < 5)

            bIsVolatilityLow = tc < 1 and tr < atr

            bIsATRS_Ranking_Good = False
            try:
                atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
                bIsATRS_Ranking_Good = atrsRank < 1000
            except Exception as e:
                print(e)
                bIsATRS_Ranking_Good = False

            filterMatchNum = 0

            if bIsUpperMA:
                filterMatchNum = filterMatchNum + 1
            if b_150ma_upper_than_200ma:
                filterMatchNum = filterMatchNum + 1
            if bMA_Slope_Plus:
                filterMatchNum = filterMatchNum + 1
            if b_50ma_biggerThan_150ma_200ma:
                filterMatchNum = filterMatchNum + 1
            if bIsATRS_Ranking_Good:
                filterMatchNum = filterMatchNum + 1

            # # 기본 MMT 만족 종목에 대해서만 계산하도록 하자.
            # bPocketPivot = sd.check_pocket_pivot(inStockData)
            # bInsideBar = sd.check_insideBar(inStockData)
            # NR_x = sd.check_NR_with_TrueRange(inStockData)
            # bConverging, bPower3 = sd.check_ma_converging(inStockData)
            #bNearMA = sd.check_near_ma(inStockData)


            # 거래량, VCP, RS는 포기 못함
            if filterMatchNum >= 5 and bIsVolumeEnough:
                selected_tickers.append(ticker)

   
    
        search_end_time = time.time()
        execution_time = search_end_time - search_start_time
        print(f"Search tiem elapsed: {execution_time}sec")




    elif bUseLocalCache:
        selected_tickers = out_tickers


    # 데이터를 파일에 저장
    if not bUseLocalCache:
        with open('cache_tickers', "wb") as f:
            pickle.dump(selected_tickers, f)

        with open('cache_stock_datas_dic', "wb") as f:
            pickle.dump(out_stock_datas_dic, f)


    # pocket pivot
    #quantTickers = ['LTH']
    #selected_tickers = list(set(selected_tickers) & set(quantTickers))
    selected_tickers.sort()


    print('filtered by quant data: \n', selected_tickers)
    print('selected tickers num: ', len(selected_tickers))
    DrawStockDatas(out_stock_datas_dic, selected_tickers, sd)
    # -----------------------------------------------------------------------------------------------------


elif index == 2:
    updown_nyse, updown_nasdaq, updown_sp500 = sd.getUpDownDataFromCsv(365*3)
    DrawMomentumIndex(updown_nyse, updown_nasdaq, updown_sp500)
elif index == 3:
    remove_local_caches()
    sd.syncCsvFromWeb(3)
    sd.cookUpDownDatas()
    sd.cook_Nday_ATRS150_exp(365*2)
    sd.cook_ATRS150_exp_Ranks(365*2)
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_top10_in_industries()
elif index == 4:
    sd.cookUpDownDatas()
elif index == 5:
    sd.cookLocalStockData()
elif index == 6:
    sd.downloadStockDatasFromWeb(365*6, False)
elif index == 7:
    sd.cook_Nday_ATRS150_exp(365*2)
    sd.cook_ATRS150_exp_Ranks(365*2)
elif index == 8:
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_top10_in_industries()
elif index == 9:
    cook_infos_from_last_searched_tickers(sd, 'US_MMT_0528')

# --------------------------------------------------------------------