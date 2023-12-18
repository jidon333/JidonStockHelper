
import datetime as dt
import glob
import json
import os
import sys
import time
import pickle


# ------------------- global ----------------------
### 증권거래소 종목들 가져오기
#sp500_list = fdr.StockListing('S&P500')
#nyse_list = fdr.StockListing('NYSE')
#nasdaq_list = fdr.StockListing('NASDAQ')

### 지수 데이터 가져오기
# nasdaq = fdr.DataReader('IXIC', '2020-01-01', '2023-02-25')
# sp500 = fdr.DataReader('US500', '2020-01-01', '2023-02-25')
# dowjones = fdr.DataReader('DJI', '2020-01-01', '2023-02-25')


from jdStockDataManager import JdStockDataManager 
from jdChart import JdChart
from jdGlobal import get_yes_no_input
from jdGlobal import data_folder
from jdGlobal import metadata_folder

from qtWindow import JdWindowClass

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pandas as pd

import jdStockFilteringManager


sd = JdStockDataManager()
sf = jdStockFilteringManager.JdStockFilteringManager(sd)




def DrawStockDatas(stock_datas_dic, selected_tickers, inStockManager : JdStockDataManager, maxCnt = -1):
    

    if __name__ == "__main__" :
        #QApplication : 프로그램을 실행시켜주는 클래스
        app = QApplication(sys.argv) 

        #WindowClass의 인스턴스 생성
        myWindow = JdWindowClass() 

        chart = JdChart(inStockManager)
        chart.init_plots_for_stock(stock_datas_dic, selected_tickers)
        myWindow.set_chart_class(chart)

        #프로그램 화면을 보여주는 코드
        myWindow.show()

        #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
        app.exec_()

def DrawMomentumIndex(updown_nyse, updown_nasdaq, updown_sp500, bOnlyForScreenShot = False):
    # show first element
    chart = JdChart(sd)
    chart.bOnlyForScreenShot = bOnlyForScreenShot
    chart.init_plots_for_up_down(updown_nyse, updown_nasdaq, updown_sp500)
    chart.draw_updown_chart()


def draw_count_data_Index(mtt_cnt_df, name : str, chart_type : str, bOnlyForScreenShot = False):
    """
    name : {name} Count chart, {name} Count Moving Average ..
    chart_type : line or bar
    """
    chart = JdChart(sd)
    chart.bOnlyForScreenShot = bOnlyForScreenShot
    chart.init_plots_for_count_data(mtt_cnt_df, chart_type)
    chart.draw_count_data_chart(name, chart_type)

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
    
    sd.reset_caches()




def screen_stocks_and_show_chart(filter_function, bUseLocalLoadedStockDataForScreening, bSortByRS):
    bUseLocalCache = get_yes_no_input('Do you want to see last chart data? \n It will just show your last chart data without screening. \n (y/n)')
    if bUseLocalCache:
        try:
            with open('cache_tickers', "rb") as f:
                tickers = pickle.load(f)

            with open('cache_stock_datas_dic', 'rb') as f:
                stock_data = pickle.load(f)
            
        except FileNotFoundError:
            print('Can not find your last stock chart data in local.\n The chart data will be re-generated. ')
            bUseLocalCache = False
            stock_data, tickers = sf.screening_stocks_by_func(filter_function, bUseLocalLoadedStockDataForScreening, bSortByRS)

    else:
        stock_data, tickers = sf.screening_stocks_by_func(filter_function, bUseLocalLoadedStockDataForScreening, bSortByRS)

    # 데이터를 파일에 저장
    if not bUseLocalCache:
        with open('cache_tickers', "wb") as f:
            pickle.dump(tickers, f)

        with open('cache_stock_datas_dic', "wb") as f:
            pickle.dump(stock_data, f)

    print(tickers)
    DrawStockDatas(stock_data, tickers, sd)

print("Select the chart type. \n \
      1: Stock Data Chart \n \
      2: Momentum Index Chart \n \
      3: Sync local .csv datas from web and gernerate other meta datas.(up_down, RS, industry, mtt count, etc ..) \n \
      4: cook up-down datas using local csv files. \n \
      5: cook local stock data. \n \
      6: Download stock data from web and overwrite local files. (It will takes so long...) \n \
      7: cook ATRS Ranking \n \
      8: cook industry Ranking \n \
      9: cook screening result as xlsx file. \n \
      10: MTT Index chart \n \
      11: FA50 Index chart \n \
      12: Generate All indicators and screening result \n \
      13: Power gap histroy screen \n ")

index = int(input())

if index == 1:
    sf.MTT_ADR_minimum = 3
    #screen_stocks_and_show_chart(sf.filter_stocks_MTT, True, True)
    sf.MTT_ADR_minimum = 1
    #screen_stocks_and_show_chart(filter_stock_hope_from_bottom, True, True)
    screen_stocks_and_show_chart(sf.filter_stock_ALL, True, False)
    #screen_stocks_and_show_chart(filter_stock_Good_RS, True, True)
    #screen_stocks_and_show_chart(filter_stocks_high_ADR_swing, True, True)
    #screen_stocks_and_show_chart(filter_stock_power_gap, True, True)

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
    sd.cook_filter_count_data(sf.filter_stocks_MTT, "MTT_Counts", 10, True)
    sd.cook_filter_count_data(sf.filter_stock_FA50, "FA50_Counts", 10, True)
    sd.cook_top10_in_industries()
elif index == 4:
    sd.cookUpDownDatas()
elif index == 5:
    remove_local_caches()
    sd.cookLocalStockData()
elif index == 6:
    remove_local_caches()
    sd.downloadStockDatasFromWeb(365*6, False) # you have 6 year data.... 
    remove_outdated_tickers()
    sd.remove_acquisition_tickers()
    sd.cook_Stock_GICS_df()
    sd.cook_Nday_ATRS150_exp(365*2)
    sd.cook_ATRS150_exp_Ranks(365*2)
    sd.cook_top10_in_industries()
    sd.cook_filter_count_data(sf.filter_stocks_MTT, "MTT_Counts", 365*3, False)
    sd.cook_filter_count_data(sf.filter_stock_FA50, "FA50_Counts", 365*3, False)

elif index == 7:
    sd.cook_Nday_ATRS150_exp(365*2)
    sd.cook_ATRS150_exp_Ranks(365*2)
elif index == 8:
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_top10_in_industries()
elif index == 9:
    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stock_hope_from_bottom, True, True)
    sd.cook_stock_info_from_tickers(tickers, 'US_hope_from_bottom_2023-11-03')

    #stock_data, tickers = screening_stocks_by_func(filter_stock_hope_from_bottom, True)
    #sd.cook_stock_info_from_tickers(tickers, 'US_hope_from_bottom_1030')

    #stock_data, tickers = screening_stocks_by_func(filter_stock_FA50, True, True)
    # sd.cook_stock_info_from_tickers(tickers, 'US_FA50_1030')
elif index == 10:
    df = sd.get_count_data_from_csv("MTT")
    draw_count_data_Index(df, "MTT", "line")
elif index == 11:
    df = sd.get_count_data_from_csv("FA50")
    draw_count_data_Index(df, "FA50", "bar")

elif index == 12:
    # auto generate indicator screenshot and MTT xlsx file. #

    ### MI index chart
    updown_nyse, updown_nasdaq, updown_sp500 = sd.getUpDownDataFromCsv(365*3)
    DrawMomentumIndex(updown_nyse, updown_nasdaq, updown_sp500, True)

    ### MTT count chart
    df = sd.get_count_data_from_csv("MTT")
    draw_count_data_Index(df, "MTT", "line", True)

    ### FA50 count chart
    df = sd.get_count_data_from_csv("FA50")
    draw_count_data_Index(df, "FA50", "bar", True)

    ### cook MTT stock list as xlsx.

    # get mtt screen list
    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stocks_MTT, True, True)
    
    # Hack: get last date string from first stock data and use it for filename.
    first_stock_data : pd.DataFrame = stock_data[tickers[0]]
    lastday = str(first_stock_data.index[-1].date())

    # cook file
    sd.cook_stock_info_from_tickers(tickers, f'US_MTT_{lastday}')

    ### High ADR Swing
    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stocks_high_ADR_swing, True, True)
    first_stock_data : pd.DataFrame = stock_data[tickers[0]]
    lastday = str(first_stock_data.index[-1].date())
    sd.cook_stock_info_from_tickers(tickers, f'US_HighAdrSwing_{lastday}')

    stock_data_dic, tickers = sf.screening_stocks_by_func(sf.filter_stock_power_gap, True, True, -1)
    s = str.format(f"[{lastday}] power gap tickers: ") + str(tickers)
    print(s)

elif index == 13:

    sf.cook_power_gap_profiles(20*12*1, 20, 20)
    sf.cook_open_gap_profiles(20*12*1, 20, 20)


    
# --------------------------------------------------------------------



