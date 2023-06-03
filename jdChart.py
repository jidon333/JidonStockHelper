


import numpy as np
import pandas as pd
import sys


import os
import json



import matplotlib.pyplot as plt
from matplotlib.dates import num2date
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import TextBox, Button
import matplotlib.dates as mdates

from jdStockDataManager import JdStockDataManager 

plt.switch_backend('Qt5Agg')

class JdChart:
    def __init__(self, stockData = None, updown_nyse = None, updown_nasdaq = None, updown_sp500 = None):

        self.bDrawBarChart = True
        self.bDrawingUpDownChart = False
        
        self.stockManager : JdStockDataManager = None

        self.stockData = stockData
        self.retVal = 0
        self.annotateObj = None
        self.text_box_coordinate = None
        self.text_box_info = None

        self.updown_nyse = updown_nyse
        self.updown_nasdaq = updown_nasdaq
        self.updown_sp500 = updown_sp500

        self.top10_in_industries : dict

        if stockData is not None:
            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]}, figsize=(20, 10))
            self.fig.subplots_adjust(left=0.24, right=0.9)

        else:
            self.fig, (self.ax1) = plt.subplots(figsize=(20, 10))

        self.markedTickerList = []

        plt.ion()

    def set_stock_manager(self, inStockManager):
        self.stockManager = inStockManager
        self.top10_in_industries = self.stockManager.get_top10_in_industries()


    def get_sector(self, ticker):
        GICS_df = self.stockManager.get_GICS_df()
        return GICS_df.loc[ticker]['sector']
    def get_industry(self, ticker):
        GICS_df = self.stockManager.get_GICS_df()
        return GICS_df.loc[ticker]['industry']
    
    def get_long_term_industry_rank_scores(self, industryName):
        return self.stockManager.get_long_term_industry_rank_scores(industryName)
    
    def reset(self, stockData):
        self.stockData = stockData
        self.retVal = 0
        self.annotateObj = None
        self.text_box_coordinate = None


    def on_close(self, event):
        print(self.markedTickerList)
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
                if self.bDrawBarChart and not self.bDrawingUpDownChart:
                    x = int(x)
                    if x < 0 : x=0
                    if x >= len(self.stockData): x = len(self.stockData) - 1
                    index = self.stockData.index[x]
                    x_axis_str = pd.to_datetime(index).strftime('%Y-%m-%d')
                    self.text_box_coordinate.set_text(
                    f'x = {x_axis_str}, y = {y:.2f}')
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
            if ticker not in self.markedTickerList:
                self.markedTickerList.append(ticker)
            print(ticker, 'is marked!')
            
    def _draw_bar_chart_ax1(self, in_stock_date):
        ### 바 차트 ###
        # 선을 그리기위해 기존의 yy-mm-dd 형식의 Date 인덱스를 0~N 사이의 정수로 변경
        temp_df = in_stock_date
        temp_df = temp_df.reset_index()
        horizontal_OHLC_length = 0.2
        x = np.arange(0,len(temp_df))
        for idx, val in temp_df.iterrows():
            color = 'green'
            if val['Open'] > val['Close']:
                color = 'red'

            # 바 차트를 위한 선 그리기
            self.ax1.plot([x[idx], x[idx]], [val['Low'], val['High']], color = color)
            self.ax1.plot([x[idx], x[idx]+horizontal_OHLC_length], 
                    [val['Close'], val['Close']], 
                    color=color)

        # 그래프 설정
        self.ax1.plot(temp_df['200MA'], label='MA200', color='green')
        self.ax1.plot(temp_df['150MA'], label='MA150', color='blue')
        self.ax1.plot(temp_df['50MA'], label='MA50', color='orange')

        self.ax1.set_xticks(x[::40])
        self.ax1.set_xticklabels(self.stockData.index[::40].date)

    def _draw_line_chart_ax1(self, in_stock_date):
        self.ax1.plot(in_stock_date['Close'], label='Close')
        self.ax1.plot(in_stock_date['200MA'], label='MA200', color='green')
        self.ax1.plot(in_stock_date['150MA'], label='MA150', color='blue')
        self.ax1.plot(in_stock_date['50MA'], label='MA50', color='orange')

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
        
        bNearMa10 = self.stockManager.check_near_ma(self.stockData, 10, 1.5)
        bNearMa20 = self.stockManager.check_near_ma(self.stockData, 20, 1.5)
        bNearMa50 = self.stockManager.check_near_ma(self.stockData, 50, 1.5)

        bSupported_by_ma10 = self.stockManager.check_supported_by_ma(self.stockData, 10, 1.5)
        bSupported_by_ma20 = self.stockManager.check_supported_by_ma(self.stockData, 20, 1.5)
        bSupported_by_ma50 = self.stockManager.check_supported_by_ma(self.stockData, 50, 1.5)

        near_ma_list = []
        if bNearMa10:
            near_ma_list.append(10)
        if bNearMa20:
            near_ma_list.append(20)
        if bNearMa50:
            near_ma_list.append(50)

        supported_by_ma_list = []
        if bSupported_by_ma10:
            supported_by_ma_list.append(10)
        if bSupported_by_ma20:
            supported_by_ma_list.append(20)
        if bSupported_by_ma50:
            supported_by_ma_list.append(50)

      
        # 좌측 info text box 설정
        if self.text_box_info != None:
            self.text_box_info.remove()
            self.text_box_info = None

        industryRanks_long = pd.Series(dtype='float64')
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
            f"Near MA : {near_ma_list}\n"
            f"MA Supported by: {supported_by_ma_list}\n"
            f"MA Converging: {bIsMaConverging}\n"

            f"Power of 3: {bIsPower3}\n"
            f"Industry RS Score : {int(industryRanks_long.get('1d ago', 0))} \n\n"
            
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


        if self.bDrawBarChart:
            self._draw_bar_chart_ax1(self.stockData)
        else:
            self._draw_line_chart_ax1(self.stockData)

        self.ax1.legend(loc='lower left')
        self.ax1.grid()
        self.ax1.set_title(titleStr, fontproperties=fontprop)


        # Draw industry ranks to the ax2 instead of volumes
        stockDataLen = len(self.stockData['Close'].index)
        # Can not draw chart if the stock data len is longer than industry rank data.
        if not industryRanks_long.empty and stockDataLen < len(industryRanks_long):
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

        self.bDrawingUpDownChart = True


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