
import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib.dates import num2date
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as tk

from jdStockDataManager import JdStockDataManager 
from jdGlobal import SCREENSHOT_FOLDER
from jdGlobal import UI_FOLDER

plt.switch_backend('Qt5Agg')

class JdChart:
    def __init__(self, stock_manager=None):

        self.is_only_for_screenshot = False

        self.is_bar_chart_mode = True
        self.is_drawing_stock_chart = True

        self.is_show_ma10 = False
        self.is_show_ma20 = False
        self.is_show_ema10 = False
        self.is_show_ema21 = False
        
        self.stock_manager: JdStockDataManager = stock_manager

        self.annotation = None
        self.text_box_coordinate = None
        self.text_box_info = None


        # for stock data chart
        self.stock_datas_dic: dict[str, pd.DataFrame]
        self.curr_ticker_index = 0
        self.selected_tickers: list

        # for up-down chart
        self.updown_nyse: pd.DataFrame
        self.updown_nasdaq: pd.DataFrame
        self.updown_sp500: pd.DataFrame

        self.updown_atr: pd.DataFrame

        self.mtt_count_df: pd.DataFrame

        self.top10_in_industries: dict
        self.top10_in_industries = self.stock_manager.get_top10_in_industries()

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
 
        self.marked_ticker_list = []

        plt.ion()
    
    # must be called before to show stock chart
    def init_plots_for_stock(self, stock_datas_dic: dict[str, pd.DataFrame], selected_tickers):

        self.fig = Figure(figsize=(20,10))
        #self.fig.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        self.fig.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 2]})

        self.ax1 = self.fig.axes[0]
        self.ax2 = self.fig.axes[1]
        self.ax3 = self.fig.axes[2]
        #self.ax4 = self.fig.axes[3]
        self.fig.subplots_adjust(left=0.24, right=0.95)


        self.stock_datas_dic = stock_datas_dic
        self.selected_tickers = selected_tickers
        self.curr_ticker_index = 0

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('close_event', self.on_close)


    # must be called before to show up-down chart
    def init_plots_for_up_down(self, updown_nyse : pd.DataFrame , updown_nasdaq : pd.DataFrame, updown_sp500 : pd.DataFrame):
            self.updown_nyse = updown_nyse
            self.updown_nasdaq = updown_nasdaq
            self.updown_sp500 = updown_sp500
            self.fig, (self.ax1) = plt.subplots(figsize=(20, 10))

            self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
            self.fig.canvas.mpl_connect('close_event', self.on_close)

    def init_plots_for_atr_up_down(self, updown_atr : pd.DataFrame):
        self.updown_atr = updown_atr
        self.fig, (self.ax1) = plt.subplots(figsize=(20, 10))

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def init_plots_for_count_data(self, mtt_count_df : pd.DataFrame, type = "line"):
        """
        type: line or bar
        """
        self.mtt_count_df = mtt_count_df

        if type == "line":
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})  # 2개의 서브플롯을 생성
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
            self.fig.canvas.mpl_connect('close_event', self.on_close)
        elif type == "bar":
            #self.fig, (self.ax1) = plt.subplots(figsize=(20, 10))  # 1개의 서브플롯을 생성
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})  # 2개의 서브플롯을 생성
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
            self.fig.canvas.mpl_connect('close_event', self.on_close)
        
    def get_curr_ticker(self):
        return self.selected_tickers[self.curr_ticker_index]
    
    def get_curr_stock_data(self):
        ticker = self.get_curr_ticker()
        return self.stock_datas_dic.get(ticker, pd.DataFrame())
     

    def get_sector(self, ticker):
        GICS_df = self.stock_manager.get_GICS_df()
        return GICS_df.loc[ticker]['sector']
    def get_industry(self, ticker):
        GICS_df = self.stock_manager.get_GICS_df()
        return GICS_df.loc[ticker]['industry']
    
    def get_long_term_industry_rank_scores(self, industry_name):
        return self.stock_manager.get_long_term_industry_rank_scores(industry_name)
    
    
    def move_to_next_stock(self, step_num=1):
        self.curr_ticker_index = min(self.curr_ticker_index + step_num, len(self.selected_tickers) - 1)
        self.annotation = None
        self.text_box_coordinate = None

    def move_to_prev_stock(self, step_num=1):
        self.curr_ticker_index = max(self.curr_ticker_index - step_num, 0)
        self.annotation = None
        self.text_box_coordinate = None

    def move_to_ticker_stock(self, ticker: str):
        try:
            ticker = ticker.lower()
            lowercase_list = [item.lower() for item in self.selected_tickers]
            index = lowercase_list.index(ticker)
            self.curr_ticker_index = index
            self.annotation = None
            self.text_box_coordinate = None
            return True
        except Exception as e:
            print('can not find ticker ', ticker)
            return False
      
    def set_ma_visibility(self, is_show_ma10, is_show_ma20, is_show_ema10, is_show_ema21):
        self.is_show_ma10 = is_show_ma10
        self.is_show_ma20 = is_show_ma20
        self.is_show_ema10 = is_show_ema10
        self.is_show_ema21 = is_show_ema21


    def on_close(self, event):
        print(self.marked_ticker_list)
        plt.close()
        if self.is_only_for_screenshot is False:
            sys.exit()

    def on_move(self, event):
        x, y = event.xdata, event.ydata
        self.fig.format_coord = lambda x, y: f'x={x:.2f}, y={y:.2f}'
        draw_axis = None

        if event.inaxes == self.ax1:
            draw_axis = self.ax1

        if draw_axis is None:
            return

        if self.text_box_coordinate is None:
            self.text_box_coordinate = draw_axis.text(0.5, 0.95,
                                        f'x = {num2date(x).strftime("%Y-%m-%d")}, y = {y:.2f}',
                                        transform=draw_axis.transAxes,
                                        ha='center', va='top')
        else:
                if self.is_bar_chart_mode and self.is_drawing_stock_chart:
                    curr_stock_data = self.get_curr_stock_data()
                    x = int(x)
                    if x < 0:
                        x = 0
                    if x >= len(curr_stock_data):
                        x = len(curr_stock_data) - 1
                    index = curr_stock_data.index[x]
                    x_axis_str = pd.to_datetime(index).strftime('%Y-%m-%d')
                    self.text_box_coordinate.set_text(
                    f'x = {x_axis_str}, y = {y:.2f}')
                else:
                    self.text_box_coordinate.set_text(
                    f'x = {num2date(x).strftime("%Y-%m-%d")}, y = {y:.2f}')

        self.fig.canvas.draw()

    def mark_ticker(self):
            ticker = self.get_curr_stock_data()['Symbol'].iloc[-1]
            if ticker not in self.marked_ticker_list:
                self.marked_ticker_list.append(ticker)
                print(ticker, ' is marked!')
            
    def _draw_bar_chart_ax1(self, in_stock_date):
        ### 바 차트 ###
        # 선을 그리기위해 기존의 yy-mm-dd 형식의 Date 인덱스를 0~N 사이의 정수로 변경
        temp_df = in_stock_date
        temp_df = temp_df.reset_index()

        # Draw ma first for the drawing order
        self.ax1.plot(temp_df['200MA'], label='MA200', color='green')
        self.ax1.plot(temp_df['150MA'], label='MA150', color='blue')
        self.ax1.plot(temp_df['50MA'], label='MA50', color='orange')


        if self.is_show_ma20:
            temp_df['20MA'] = temp_df['Close'].rolling(window=20).mean()
            self.ax1.plot(temp_df['20MA'], label='MA20', color='red', alpha=0.5)

        if self.is_show_ma10:
                temp_df['10MA'] = temp_df['Close'].rolling(window=10).mean()
                self.ax1.plot(temp_df['10MA'], label='MA10', color='black', alpha=0.5)

        if self.is_show_ema10:
            temp_df['10EMA'] = temp_df['Close'].ewm(span=10, adjust=False).mean()
            self.ax1.plot(temp_df['10EMA'], label='EMA10', color='black', alpha=0.5)

        if self.is_show_ema21:
            temp_df['21EMA'] = temp_df['Close'].ewm(span=21, adjust=False).mean()
            self.ax1.plot(temp_df['21EMA'], label='EMA21', color='red', alpha=0.5)


        # Draw bar chart
        horizontal_OHLC_length = 0.2
        x = np.arange(0,len(temp_df))
        for idx, val in temp_df.iterrows():
            color = 'green'
            if val['Open'] > val['Close']:
                color = 'red'

            self.ax1.plot([x[idx], x[idx]], [val['Low'], val['High']], color = color)
            self.ax1.plot([x[idx], x[idx]+horizontal_OHLC_length], 
                    [val['Close'], val['Close']], 
                    color=color)



        self.ax1.set_xticks(x[::40])
        self.ax1.set_xticklabels(in_stock_date.index[::40].date)

    def _draw_line_chart_ax1(self, in_stock_date):
        # Draw ma first for the drawing order
        self.ax1.plot(in_stock_date['200MA'], label='MA200', color='green')
        self.ax1.plot(in_stock_date['150MA'], label='MA150', color='blue')
        self.ax1.plot(in_stock_date['50MA'], label='MA50', color='orange')

        if self.is_show_ma20:
            in_stock_date['20MA'] = in_stock_date['Close'].rolling(window=20).mean()
            self.ax1.plot(in_stock_date['20MA'], label='MA20', color='red', alpha=0.5)

        if self.is_show_ma10:
            in_stock_date['10MA'] = in_stock_date['Close'].rolling(window=10).mean()
            self.ax1.plot(in_stock_date['10MA'], label='MA10', color='black', alpha=0.5)

        if self.is_show_ema10:
            in_stock_date['10EMA'] = in_stock_date['Close'].ewm(span=10, adjust=False).mean()
            self.ax1.plot(in_stock_date['10EMA'], label='EMA10', color='black', alpha=0.5)

        if self.is_show_ema21:
            in_stock_date['21EMA'] = in_stock_date['Close'].ewm(span=21, adjust=False).mean()
            self.ax1.plot(in_stock_date['21EMA'], label='EMA21', color='red', alpha=0.5)

        self.ax1.plot(in_stock_date['Close'], label='Close')

   





    def draw_stock_chart(self):

        # 차트 그리기
        ticker = self.get_curr_ticker()
        curr_stock_data = self.get_curr_stock_data()
        curr_stock_data = curr_stock_data[['Name', 'Industry', 'ADR', 'TRS', 'TC', 'TR', 'High', 'Low', 'Open', 'Close', 'Volume',
                                           'ATRS_Exp', 'ATRS150_Exp', '50MA', '150MA', '200MA']]
        
        print('ticker: ', ticker)

        ranks_atrs = self.stock_manager.get_ATRS150_exp_Ranks_Normalized(ticker)
        curr_rank = self.stock_manager.get_ATRS150_exp_Ranks(ticker).iloc[-1]

        name = curr_stock_data['Name'].iloc[0]
        font_path = os.path.join(UI_FOLDER, 'NanumGothic.ttf')
        fontprop = fm.FontProperties(fname=font_path, size=30)
        industry_kor = curr_stock_data['Industry'].iloc[0]
        try:
            sector_text = self.get_sector(ticker)
            industry_text = self.get_industry(ticker)
        except Exception as e:
            sector_text = 'None'
            industry_text = 'None'
        
        title_str = f"{ticker} ({name}) \n {industry_kor},  ATRS Rank: {int(curr_rank)}th"
        trs = curr_stock_data['TRS'].iloc[-1]
        tc = curr_stock_data['TC'].iloc[-1]
        adr = curr_stock_data['ADR'].iloc[-1]

        try:
            top10 = self.top10_in_industries.get(industry_text, None)
            top10_len = 0
            if top10 != None:
                top10_len = len(top10)
        except Exception as e:
            print(e)
            top10 = None
            top10_len = 0


        true_range_nr_x = self.stock_manager.check_NR_with_TrueRange(curr_stock_data)
        is_inside_bar, is_double_inside_bar = self.stock_manager.check_insideBar(curr_stock_data)
        is_pocket_pivot = self.stock_manager.check_pocket_pivot(curr_stock_data)
        is_wick_play = self.stock_manager.check_wickplay(curr_stock_data)
        is_oel = self.stock_manager.check_OEL(curr_stock_data)
        is_ma_converging, is_power3, is_power2 = self.stock_manager.check_ma_converging(curr_stock_data)
        
        is_near_ema10 = self.stock_manager.check_near_ma(curr_stock_data, 10, 1.5, True)
        is_near_ema21 = self.stock_manager.check_near_ma(curr_stock_data, 21, 1.5, True)
        is_near_ma50 = self.stock_manager.check_near_ma(curr_stock_data, 50, 1.5)

        is_supported_by_ema10 = self.stock_manager.check_supported_by_ma(curr_stock_data, 10, 1.5, True)
        is_supported_by_ema21 = self.stock_manager.check_supported_by_ma(curr_stock_data, 21, 1.5, True)
        is_supported_by_ma50 = self.stock_manager.check_supported_by_ma(curr_stock_data, 50, 1.5)

        near_ma_list = []
        if is_near_ema10:
            near_ma_list.append(10)
        if is_near_ema21:
            near_ma_list.append(21)
        if is_near_ma50:
            near_ma_list.append(50)

        supported_by_ma_list = []
        if is_supported_by_ema10:
            supported_by_ma_list.append(10)
        if is_supported_by_ema21:
            supported_by_ma_list.append(21)
        if is_supported_by_ma50:
            supported_by_ma_list.append(50)

      
        # 좌측 info text box 설정
        if self.text_box_info is None:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            self.text_box_info = self.fig.text(0.01, 0.9, "", transform=self.fig.transFigure, fontsize=14, bbox=props, verticalalignment='top', horizontalalignment='left')

        industry_ranks_long = pd.Series(dtype='float64')
        if not pd.isna(industry_text):
            industry_ranks_long = self.get_long_term_industry_rank_scores(industry_text)

            msg = (
            f"========================\n"
            f"Ticker: {ticker}\n"
            f"Sector: {sector_text}\n"
            f"Industry: {industry_text}\n"
            f"TRS: {trs}\n"
            f"TC: {tc}\n"
            f"ADR(%): {adr}\n"
            f"NR(x): {true_range_nr_x}\n"
            f"Inside bar: {is_inside_bar}\n"
            f"Double Inside bar: {is_double_inside_bar}\n"
            f"Wick Play: {is_wick_play}\n"
            f"OEL: {is_oel}\n"
            f"Pocket Pivot: {is_pocket_pivot}\n"
            f"MA Converging: {is_ma_converging}\n"
            f"Near MA : {near_ma_list}\n"
            f"MA Supported by: {supported_by_ma_list}\n"

            f"Power of 3: {is_power3}\n"
            f"Power of 2: {is_power2}\n"
            f"Industry RS Score : {int(industry_ranks_long.get('1d ago', 0))} \n\n"
            
            )

            msg += f"Top10 in \n'{industry_text}' group \n"
            for i in range(0, top10_len):
                top10_ticker = top10[i][0]
                top10_rank = top10[i][1]
                msg += f"No.{i+1}: {top10_ticker}, {int(top10_rank)}\n"
            
            msg += f"\n========================"


            self.text_box_info.set_text(msg)

        self.ax1.cla()


        if self.is_bar_chart_mode:
            self._draw_bar_chart_ax1(curr_stock_data)
        else:
            self._draw_line_chart_ax1(curr_stock_data)

        self.ax1.legend(loc='lower left')
        self.ax1.grid()
        self.ax1.set_title(title_str, fontproperties=fontprop)
        self.ax1.set_yscale('log')
        # 메이저 눈금선 포매터 설정
        self.ax1.yaxis.set_major_formatter(tk.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        # 마이너 눈금선 포매터 설정
        self.ax1.yaxis.set_minor_formatter(tk.FuncFormatter(lambda y, _: '{:g}'.format(y)))



        # Draw industry ranks to the ax2 instead of volumes
        stock_data_len = len(curr_stock_data['Close'].index)
        # Can not draw chart if the stock data len is longer than industry rank data.
        if not industry_ranks_long.empty and stock_data_len < len(industry_ranks_long):
            long_industry_rank_datas = industry_ranks_long.values[-stock_data_len:]
            long_industry_rank_reindexed = pd.Series(long_industry_rank_datas, index=curr_stock_data['Close'].index)

            self.ax2.cla()
            self.ax2.plot(long_industry_rank_reindexed, label ='Industry RS score', color='green')
            self.ax2.set_ylim([0, 100])
            self.ax2.legend(loc='lower left')
            self.ax2.axhline(y=50, color='black', linestyle='--')
        else:
            # Draw volume chart to the ax2
            self.ax2.cla()
            self.ax2.bar(curr_stock_data.index,
                        curr_stock_data['Volume'], alpha=0.3, color='blue', width=0.7)
            self.ax2.set_ylabel('Volume')

        ############## Rank data를 그래프에 추가하기 ###############
        ranks_atrs_exp_df = ranks_atrs.to_frame()
        ranks_atrs_exp_df.index = pd.to_datetime(ranks_atrs_exp_df.index) # 문자열을 datetime 객체로 변경


        # self.stockData와 rank_df를 합치기 위해 index 기준으로 join
        curr_stock_data = curr_stock_data.join(ranks_atrs_exp_df, how='left')

        # NaN 값을 0으로 대체
        curr_stock_data.fillna(0, inplace=True)

        # ax3에 그래프 그리기
        self.ax3.cla()

        self.ax3.set_ylim([0, 1.2])
        if len(ranks_atrs_exp_df) != 0:
            self.ax3.plot(curr_stock_data['Rank_ATRS150_Exp'], label='RS Rank Score', color='red', alpha=0.5)
        self.ax3.legend(loc='lower left')
        #self.ax3.axhline(y=0.5, color='black', linestyle='--')
        self.ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)

 
        # self.ax4.cla()
        # self.ax4.set_ylim([-0.5, 0.5])
        # self.ax4.plot(currStockData['ATRS_Exp'], label='ATRS_Exp')

        # self.ax4.fill_between(currStockData.index, currStockData['ATRS_Exp'], 0, where=currStockData['ATRS_Exp'] < 0, color='red', alpha=0.3)
        # self.ax4.fill_between(currStockData.index, currStockData['ATRS_Exp'], 0, where=currStockData['ATRS_Exp'] >= 0, color='green', alpha=0.3)

        # self.ax4.axhline(y=0, color='black', linestyle='--')
        # self.ax4.legend(loc='lower left')


        return self.fig


    def draw_updown_chart(self):

        self.is_drawing_stock_chart = False

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

        if self.is_only_for_screenshot:
            lastday = str(self.updown_sp500.index[-1].date())
            file_path = os.path.join(SCREENSHOT_FOLDER, f'MI_Index_chart_{lastday}.png')
            plt.savefig(file_path)
            time.sleep(1)
            plt.close()

        while True:
            if not plt.get_fignums():
                break
            plt.pause(0.01)


    def draw_updown_ATR_chart(self):

        self.is_drawing_stock_chart = False

        # Plot sum and MA150 changes
        self.ax1.plot(self.updown_atr.index, self.updown_atr['ma200_changes'], label='ma200_changes')
        self.ax1.plot(self.updown_atr.index, self.updown_atr['ma50_changes'], label='ma50_changes')
        self.ax1.plot(self.updown_atr.index, self.updown_atr['ma20_changes'], label='ma20_changes')

        # Add legend, title, and grid
        self.ax1.legend()
        self.ax1.set_title('ATR Expansion')
        self.ax1.grid()

        # Add a horizontal line at y=0
        self.ax1.axhline(y=0, color='black', linestyle='--')

        # Show the chart and pause the execution
        plt.draw()

        if self.is_only_for_screenshot:
            lastday = str(self.updown_atr.index[-1].date())
            file_path = os.path.join(SCREENSHOT_FOLDER, f'ATR_Expansion{lastday}.png')
            plt.savefig(file_path)
            time.sleep(1)
            plt.close()

        while True:
            if not plt.get_fignums():
                break
            plt.pause(0.01)


    def draw_count_data_chart(self, name : str, chart_type = "line"):
        """
        name : {name} Count chart, {name} Count Moving Average ..
        chart_type : line, bar
        """

        self.is_drawing_stock_chart = False

        temp_df = self.mtt_count_df


        if chart_type == "line":
            temp_df['150MA'] = temp_df['Count'].rolling(window=150).mean()
            temp_df['50MA'] = temp_df['Count'].rolling(window=50).mean()
            temp_df['10MA'] = temp_df['Count'].rolling(window=10).mean()

            # Draw ma first for the drawing order
            self.ax1.plot(temp_df['150MA'], label='MA150', color='blue')
            self.ax1.plot(temp_df['50MA'], label='MA50', color='orange')
            self.ax1.plot(temp_df['10MA'], label='MA10', color='black', alpha=0.5)
            # Add legend, title, and grid
            self.ax1.legend()
            self.ax1.set_title(f'{name} Count Moving Averages')
            self.ax1.grid()

            self.ax2.plot(temp_df.index, temp_df['Count'], label='Count', alpha=0.3, color='blue')
            self.ax2.axhline(y=400, color='black', linestyle='--')
            self.ax2.legend()
            self.ax2.set_title(f'{name} Count Chart')
            self.ax2.grid()
        elif chart_type == "bar":

           
            temp_df['10MA'] = temp_df['Count'].rolling(window=10).mean()
            self.ax1.plot(temp_df['10MA'], label='MA10', color='black', alpha=0.5)
            self.ax1.legend()
            self.ax1.set_title(f'{name} Count Moving Average')
            self.ax1.grid()

            colors = ['green' if val > 100 else 'red' for val in temp_df['Count']]
            self.ax2.bar(temp_df.index, temp_df['Count'], label='Count', alpha=0.5, color=colors)
            self.ax2.axhline(y=100, color='black', linestyle='--')
            self.ax2.legend()
            self.ax2.set_title(f'{name} Count Bar Chart')
            self.ax2.grid(axis='y')


        # Show the chart and pause the execution
        plt.draw()

        if self.is_only_for_screenshot:
            lastday = str(temp_df.index[-1].date())
            file_path = os.path.join(SCREENSHOT_FOLDER, f'{name}_chart_{lastday}.png')
            plt.savefig(file_path)
            time.sleep(1)
            plt.close()

        while True:
            if not plt.get_fignums():
                break
            plt.pause(0.01)

