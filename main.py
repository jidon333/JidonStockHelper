
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

def DrawMomentumIndex(updown_nyse, updown_nasdaq, updown_sp500):
    # show first element
    chart = JdChart(sd)
    chart.init_plots_for_up_down(updown_nyse, updown_nasdaq, updown_sp500)
    chart.draw_updown_chart()


def draw_MTT_count_Index(mtt_cnt_df):
    chart = JdChart(sd)
    chart.init_plots_for_mtt_count(mtt_cnt_df)
    chart.draw_mtt_count_chart()

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





def cook_infos_from_last_searched_tickers(inStockManager : JdStockDataManager, inFileName):
    tickers = []
    try:
        with open('cache_tickers', "rb") as f:
            tickers = pickle.load(f)
        
    except FileNotFoundError:
        print(f'Can not find your last searched ticker list file \'cache_tickers\' ')

    if len(tickers) != 0:
        sd.cook_stock_info_from_tickers(tickers, inFileName)



# return filtered tickers
def filter_stocks_MTT(stock_datas_dic : dict, n_day_before = -1):
    filtered_tickers = []
    atrs_ranking_df = sd.get_ATRS_Ranking_df()
    gisc_df = sd.get_GICS_df()

    for ticker, inStockData in stock_datas_dic.items():

        try:
            close = inStockData['Close'].iloc[n_day_before]
            ma150 = inStockData['150MA'].iloc[n_day_before]
            ma200 = inStockData['200MA'].iloc[n_day_before]
        except Exception as e:
            continue

        bIsUpperMA_150_200 = close > ma150 and close > ma200
        # early rejection for optimization
        if bIsUpperMA_150_200 == False:
            continue        

        try:
            rs = inStockData['RS'].iloc[n_day_before]
            ma150_slope = inStockData['MA150_Slope'].iloc[n_day_before]
            ma200_slope = inStockData['MA200_Slope'].iloc[n_day_before]
            ma50 = inStockData['50MA'].iloc[n_day_before]
            volume_ma50 = inStockData['Volume'].rolling(window=50).mean().iloc[n_day_before]
            ADR = inStockData['ADR'].iloc[n_day_before]
        except Exception as e:
            continue

        # (거래량 50일 평균 20만이상 + 10불이상 or 하루거래량 100억 이상)
        bIsVolumeEnough = (volume_ma50 >= 200000 and close >= 10 ) or volume_ma50*close > 10000000
        # 마지막날 거래량 100,000주 이상
        #bIsVolumeEnough = bIsVolumeEnough and last_volume >= 100000
        bIsUpperMA = close > bIsUpperMA_150_200 and close > ma50
        b_150ma_upper_than_200ma = ma150 > ma200
        bMA_Slope_Plus = ma150_slope > 0 and ma200_slope > 0
        b_50ma_biggerThan_150ma_200ma = ma50 > ma150 and ma50 > ma200

        bIsATRS_Ranking_Good = False
        try:
            atrsRank = atrs_ranking_df.loc[ticker].iloc[n_day_before]
            bIsATRS_Ranking_Good = atrsRank < 1000
        except Exception as e:
            print(e)
            bIsATRS_Ranking_Good = False
  
        filterMatchNum = 0

        if ADR < 1:
            continue

        if bIsUpperMA:
            filterMatchNum = filterMatchNum + 1
        if b_150ma_upper_than_200ma or True: # 150, 200 정배열 조건 삭제
            filterMatchNum = filterMatchNum + 1
        if bMA_Slope_Plus:
            filterMatchNum = filterMatchNum + 1
        if b_50ma_biggerThan_150ma_200ma:
            filterMatchNum = filterMatchNum + 1
        if bIsATRS_Ranking_Good:
            filterMatchNum = filterMatchNum + 1

        #거래량, VCP, RS는 포기 못함
        if filterMatchNum >= 5 and bIsVolumeEnough:
            filtered_tickers.append(ticker)

    return filtered_tickers

def filter_stocks_high_ADR_swing(stock_datas_dic : dict):
    filtered_tickers = []
    atrs_ranking_df = sd.get_ATRS_Ranking_df()
    gisc_df = sd.get_GICS_df()

    for ticker, inStockData in stock_datas_dic.items():

        close = inStockData['Close'].iloc[-1]
        ma150 = inStockData['150MA'].iloc[-1]
        ma200 = inStockData['200MA'].iloc[-1]
        bIsUpperMA_150_200 = close > ma150 and close > ma200

        # early rejection for optimization
        if bIsUpperMA_150_200 == False:
            continue
    
        ma50 = inStockData['50MA'].iloc[-1]
        volume_ma50 = inStockData['Volume'].rolling(window=50).mean().iloc[-1]
        ADR = inStockData['ADR'].iloc[-1]
        bIsUpperMA = close > bIsUpperMA_150_200 and close > ma50

        # 설명 : 스테이지1을 포함, 단기 트레이딩을 위한 고ADR 단기 모멘텀 스크리너
        # 1. $5 이상
        # 2. ETF/ETN 제외
        # 3. 20일 ADR 4% 이상
        # 4. 50일 평균 거래량 100만주 이상
        # 5. 가격 > 50SMA
        # 6. 가격 > 200SMA
        # 7. 섹터 제외 : Health Technology

        bIsVolumeEnough = (volume_ma50 >= 1000000 and close >= 5 )
        bADRMoreThan4 = ADR >=4.0
        bIsUpperMA
        bIsNotHealthCare = False
        try:
            sector = gisc_df.loc[ticker]['sector']
            bIsNotHealthCare = sector != 'Healthcare'
        except Exception as e:
            print(e)

        if bIsVolumeEnough and bADRMoreThan4 and bIsUpperMA and bIsNotHealthCare:
            filtered_tickers.append(ticker)

    return filtered_tickers

# just return all stock's tickers
def filter_stock_ALL(stock_datas_dic : dict):
    filtered_tickers = []
    for ticker, inStockData in stock_datas_dic.items():
        filtered_tickers.append(ticker)

    return filtered_tickers


def screening_stocks_by_func(filter_func, bForceNotUseCache = True):
    out_tickers = []
    out_stock_datas_dic = {}

    if not bForceNotUseCache:
        bUseLocalCache = get_yes_no_input('Do you want to see last chart data? \n It will use cached local data and it will save loading time. \n (y/n)')
    else:
        bUseLocalCache = False

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
    if not bUseLocalCache:

        search_start_time = time.time()
        selected_tickers = []
        selected_tickers = filter_func(out_stock_datas_dic)
        selected_tickers.sort()
   
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

    print('filtered by quant data: \n', selected_tickers)
    print('selected tickers num: ', len(selected_tickers))

    return out_stock_datas_dic, selected_tickers



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
      10: MTT Index chart ")

index = int(input())
sd = JdStockDataManager()

if index == 1:
    stock_data, tickers = screening_stocks_by_func(filter_stocks_MTT, False)
    #stock_data, tickers = screening_stocks_by_func(filter_stocks_high_ADR_swing, False)
    #stock_data, tickers = screening_stocks_by_func(filter_stock_ALL, False)

    # mask_tickers = ['FSLY', 'VRT', 'CLS', 'CELH', 'BLDR', 'KNF', 'FRSH', 'SOFI', 'AEHR',
    #   'CVNA', 'DLO', 'ACLS', 'DKNG', 'LSCC', 'CFLT', 'NVDA', 'TSLA', 'SMCI',
    #     'PLTR', 'GLBE', 'UPWK', 'MBD', 'NFLX', 'META', 'ROKU', 'UBER', 'MNDY', 'STRL', 'TEAM', 'KD', 'IONQ']
    
    #tickers = list(set(mask_tickers) & set(tickers))
    #tickers.sort()
    print('filtered by quant data: \n', tickers)
    print('selected tickers num: ', len(tickers))

    DrawStockDatas(stock_data, tickers, sd)

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
    sd.cook_MTT_count_data(filter_stocks_MTT, 10, True)
elif index == 4:
    sd.cookUpDownDatas()
elif index == 5:
    remove_local_caches()
    sd.cookLocalStockData()
elif index == 6:
    remove_local_caches()
    sd.downloadStockDatasFromWeb(365*6, False)
    remove_outdated_tickers()
    sd.remove_acquisition_tickers()
    sd.cook_Stock_GICS_df()

elif index == 7:
    sd.cook_Nday_ATRS150_exp(365*2)
    sd.cook_ATRS150_exp_Ranks(365*2)
elif index == 8:
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_top10_in_industries()
elif index == 9:
    screening_stocks_by_func(filter_stocks_MTT)
    cook_infos_from_last_searched_tickers(sd, 'US_MTT_0909')
elif index == 10:
    df = sd.get_MTT_count_data_from_csv()
    draw_MTT_count_Index(df)


# --------------------------------------------------------------------