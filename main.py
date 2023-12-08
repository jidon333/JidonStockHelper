
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



sd = JdStockDataManager()
MTT_ADR_minimum = 1


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




# return filtered tickers
def filter_stocks_MTT(stock_datas_dic : dict, n_day_before = -1):
    filtered_tickers = []
    atrs_ranking_df = sd.get_ATRS_Ranking_df()
    gisc_df = sd.get_GICS_df()
    rs_ranks = []

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

        # (거래량 50일 평균 20만이상 + 10불이상 or 평균거래대금 1000만불 이상)
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

        if ADR < MTT_ADR_minimum:
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
            print('Can\'t find GISC for ',e)

        if bIsVolumeEnough and bADRMoreThan4 and bIsUpperMA and bIsNotHealthCare:
            filtered_tickers.append(ticker)

    return filtered_tickers

# just return all stock's tickers
def filter_stock_ALL(stock_datas_dic : dict):
    filtered_tickers = []
    for ticker, inStockData in stock_datas_dic.items():
        filtered_tickers.append(ticker)

    return filtered_tickers


def filter_stock_Custom(stock_datas_dic : dict):
    # filter stock good RS 
    filtered_tickers = []
    Mtt_tickers = filter_stocks_MTT(stock_datas_dic)
    atrs_ranking_df = sd.get_ATRS_Ranking_df()
    gisc_df = sd.get_GICS_df()
    bIsATRS_Ranking_Good = False
    bIsNotHealthCare = False
    bIsVolumeEnough = False
    for ticker in Mtt_tickers:
        stockData = stock_datas_dic[ticker]
        try:
            ADR = stockData['ADR'].iloc[-1]
            volume_ma50 = stockData['Volume'].rolling(window=50).mean().iloc[-1]
            close = stockData['Close'].iloc[-1]
            ma150 = stockData['150MA'].iloc[-1]
            ma200 = stockData['200MA'].iloc[-1]
            bNear150or200 = False

            if abs(sd.get_percentage_AtoB(close, ma150)) < ADR*2 or abs(sd.get_percentage_AtoB(close, ma200)) < ADR*2:
                bNear150or200 = True

            # atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
            # bIsATRS_Ranking_Good = atrsRank < 2000
            sector = gisc_df.loc[ticker]['sector']
            # bIsNotHealthCare = sector != 'Healthcare'
            # bIsNotEnergy = sector == 'Energy'

            bIsVolumeEnough = (volume_ma50 >= 1000000 and close >= 5 )

            if sector == 'Energy':
                filtered_tickers.append(ticker)

        except Exception as e:
            continue
    
    return filtered_tickers


def filter_stock_hope_from_bottom(stock_datas_dic : dict):
    """
    - ADR 2이상
    - RS 랭킹 상위 30%
    - 헬스케어, 에너지 섹터 제외
    - Volume 50MA 100만 이상 and 5불 이상 주식
    - 150SMA > 200SMA (바닥으로 추락하기전 2단계였던 주식을 보고 싶었음)
    - 150 or 200SMA 이격도 2 ADR 미만 (장기 이평선 근처에서 횡보하는 것을 찾기 위함)
    """
    filtered_tickers = []
    Mtt_tickers = filter_stock_ALL(stock_datas_dic)
    atrs_ranking_df = sd.get_ATRS_Ranking_df()
    gisc_df = sd.get_GICS_df()
    bIsATRS_Ranking_Good = False
    bIsNotHealthCare = False
    bIsVolumeEnough = False
    for ticker in Mtt_tickers:
        stockData = stock_datas_dic[ticker]
        try:
            ADR = stockData['ADR'].iloc[-1]
            volume_ma50 = stockData['Volume'].rolling(window=50).mean().iloc[-1]
            close = stockData['Close'].iloc[-1]
            ma150 = stockData['150MA'].iloc[-1]
            ma200 = stockData['200MA'].iloc[-1]
            bNear150or200 = False

            if abs(sd.get_percentage_AtoB(close, ma150)) < ADR*2 or abs(sd.get_percentage_AtoB(close, ma200)) < ADR*2:
                bNear150or200 = True

            atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
            bIsATRS_Ranking_Good = atrsRank < 2000
            sector = gisc_df.loc[ticker]['sector']
            bIsNotHealthCare = sector != 'Healthcare'
            bIsNotEnergy = sector != 'Energy'

            bIsVolumeEnough = (volume_ma50 >= 1000000 and close >= 5 )

            if ADR > 2 and bIsATRS_Ranking_Good and bIsNotHealthCare and bIsNotEnergy and bIsVolumeEnough and bNear150or200:
                if ma150 > ma200:
                    filtered_tickers.append(ticker)

        except Exception as e:
            continue
    
    return filtered_tickers


def filter_stock_power_gap(stock_datas_dic : dict, n_day_before = -1):
    """
    - ADR 2이상
    - 갭으로 주가 상승
    - 최소 10% 이상 주가 상승
        - 2 ADR(%) 이상 상승 
        - ADR은 갭상한 날의 거래량이 포함 안되도록 하루 전날을 기준으로 한다.
    - 주가는 200MA 위에 있어야 한다.
    - 50일 평균 거래량의 200% 이상의 거래량 증가
        - 거래량 50일 평균 20만이상 + 5불이상 or 평균거래대금 500만불 이상
        - (원래는 10불이상, 거래대금 1000만불 이상을 거래 기준으로 넣지만 파워 갭은 주식의 성격을 변화시키므로 조건을 완화한다.)
        - ADR과 마찬가지로 갭상 전날 거래량이 기준에 만족하지 못하는 것은 제외한다.
    - DCR 50% 이상
    - 헬스케어 섹터 제외(바이오 무빙 혼란하다.)

    """
    filtered_tickers = []
    all_tickers = filter_stock_ALL(stock_datas_dic)
    gisc_df = sd.get_GICS_df()
    bIsNotHealthCare = False
    bIsVolumeEnough = False
    for ticker in all_tickers:
        stockData = stock_datas_dic[ticker]
        try:
            ADR_1d_ago = stockData['ADR'].iloc[n_day_before-1]
            volume_ma50_1d_ago = stockData['Volume'].rolling(window=50).mean().iloc[n_day_before-1]
            volume = stockData['Volume'].iloc[n_day_before]

            close = stockData['Close'].iloc[n_day_before]
            open = stockData['Open'].iloc[n_day_before]
            high = stockData['High'].iloc[n_day_before]
            low = stockData['Low'].iloc[n_day_before]

            close_1d_ago = stockData['Close'].iloc[n_day_before -1]
            high_1d_ago = stockData['High'].iloc[n_day_before - 1]
            ma200 = stockData['200MA'].iloc[n_day_before]

            sector = gisc_df.loc[ticker]['sector']
            bIsVolumeEnough = (volume_ma50_1d_ago >= 200000 and close >= 5 ) or volume_ma50_1d_ago*close > 5000000

            change = sd.get_percentage_AtoB(close_1d_ago, close)

            if high - low > 0:
                DCR = (close - low) / (high - low)
            else:
                continue

            if sector == 'Healthcare':
                continue
            
            # 순서대로
            # ADR > 2, Gap Open, 10% 이상 상승, 
            #if ADR > 2 and open > high_1d_ago and change > 10 and change > ADR*2 and close > ma200 and volume > volume_ma50*2 and bIsVolumeEnough and DCR >= 0.5:
            if ADR_1d_ago > 2 and open > high_1d_ago and change > 10 and change > ADR_1d_ago*2 and close > ma200 and volume > volume_ma50_1d_ago*2 and bIsVolumeEnough:
                filtered_tickers.append(ticker)

        except Exception as e:
            continue
    
    return filtered_tickers

# ADR 2이상
# RS 랭킹 상위 15%
# 헬스케어, 에너지 섹터 제외
# Volume 50MA 100만 이상 and 5불 이상 주식

def filter_stock_Good_RS(stock_datas_dic : dict):
    # filter stock good RS 
    filtered_tickers = []
    Mtt_tickers = filter_stock_ALL(stock_datas_dic)
    atrs_ranking_df = sd.get_ATRS_Ranking_df()
    gisc_df = sd.get_GICS_df()
    bIsATRS_Ranking_Good = False
    bIsNotHealthCare = False
    bIsVolumeEnough = False
    for ticker in Mtt_tickers:
        stockData = stock_datas_dic[ticker]
        try:
            ADR = stockData['ADR'].iloc[-1]
            volume_ma50 = stockData['Volume'].rolling(window=50).mean().iloc[-1]
            close = stockData['Close'].iloc[-1]
            
            atrsRank = atrs_ranking_df.loc[ticker].iloc[-1]
            bIsATRS_Ranking_Good = atrsRank < 1000
            sector = gisc_df.loc[ticker]['sector']
            bIsNotHealthCare = sector != 'Healthcare'
            bIsNotEnergy = sector != 'Energy'

            bIsVolumeEnough = (volume_ma50 >= 1000000 and close >= 5 )

            if ADR > 2 and bIsATRS_Ranking_Good and bIsNotHealthCare and bIsNotEnergy and bIsVolumeEnough:
                filtered_tickers.append(ticker)

        except Exception as e:
            continue
    
    return filtered_tickers


def filter_stock_FA50(stock_datas_dic : dict, n_day_before = -1):
    filtered_tickers = []
    Mtt_tickers = filter_stock_ALL(stock_datas_dic)
    gisc_df = sd.get_GICS_df()
    bIsVolumeEnough = False
    for ticker in Mtt_tickers:
        stockData = stock_datas_dic[ticker]
        try:
            ADR = stockData['ADR'].iloc[n_day_before]
            volume_ma50 = stockData['Volume'].rolling(window=50).mean().iloc[n_day_before]
            close = stockData['Close'].iloc[n_day_before]
            ma50 = stockData['50MA'].iloc[n_day_before]

            bIsVolumeEnough = (volume_ma50 >= 1000000 and close >= 8 )
            bUpperThan50MA = close >= ma50

            if ADR > 3 and bIsVolumeEnough and bUpperThan50MA:
                filtered_tickers.append(ticker)

        except Exception as e:
            continue
    
    return filtered_tickers


def screening_stocks_by_func(filter_func, bUseLoadedStockData = True, bSortByRS = False, n_day_before = -1):
    out_tickers = []
    out_stock_datas_dic = {}

    daysNum = int(365)
    stock_list = sd.getStockListFromLocalCsv()
    sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, bUseLoadedStockData)


    ##---------------- 조건식 -----------------------------------------------------
    search_start_time = time.time()
    selected_tickers = []

    # 마지막 날 기준이 아닌 과거를 기준으로 데이터를 뽑고 싶은 경우 n_day_before를 사용.
    if n_day_before == -1:
        selected_tickers = filter_func(out_stock_datas_dic)
    else:
        selected_tickers = filter_func(out_stock_datas_dic, n_day_before)


    # sort
    if bSortByRS:
        rs_ranks = []
        for ticker in selected_tickers:
            try:
                rank = sd.get_ATRS150_exp_Ranks(ticker).iloc[-1]
                rs_ranks.append((ticker, rank))
            except Exception as e:
                print(e)
        
        rs_ranks.sort(key=lambda x : x[1])
        keys = [x[0] for x in rs_ranks]
        selected_tickers = keys
    else:
        selected_tickers.sort()
    
    search_end_time = time.time()
    execution_time = search_end_time - search_start_time
    print(f"Search tiem elapsed: {execution_time}sec")
    print('filtered by quant data: \n', selected_tickers)
    print('selected tickers num: ', len(selected_tickers))

    return out_stock_datas_dic, selected_tickers

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
            stock_data, tickers = screening_stocks_by_func(filter_function, bUseLocalLoadedStockDataForScreening, bSortByRS)

    else:
        stock_data, tickers = screening_stocks_by_func(filter_function, bUseLocalLoadedStockDataForScreening, bSortByRS)

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
    MTT_ADR_minimum = 3
    #screen_stocks_and_show_chart(filter_stocks_MTT, True, True)
    MTT_ADR_minimum = 1
    #screen_stocks_and_show_chart(filter_stock_hope_from_bottom, True, True)
    screen_stocks_and_show_chart(filter_stock_ALL, True, False)
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
    sd.cook_filter_count_data(filter_stocks_MTT, "MTT_Counts", 10, True)
    sd.cook_filter_count_data(filter_stock_FA50, "FA50_Counts", 10, True)
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
    sd.cook_filter_count_data(filter_stocks_MTT, "MTT_Counts", 365*3, False)
    sd.cook_filter_count_data(filter_stock_FA50, "FA50_Counts", 365*3, False)

elif index == 7:
    sd.cook_Nday_ATRS150_exp(365*2)
    sd.cook_ATRS150_exp_Ranks(365*2)
elif index == 8:
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_top10_in_industries()
elif index == 9:
    stock_data, tickers = screening_stocks_by_func(filter_stock_hope_from_bottom, True, True)
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
    stock_data, tickers = screening_stocks_by_func(filter_stocks_MTT, True, True)
    
    # Hack: get last date string from first stock data and use it for filename.
    first_stock_data : pd.DataFrame = stock_data[tickers[0]]
    lastday = str(first_stock_data.index[-1].date())

    # cook file
    sd.cook_stock_info_from_tickers(tickers, f'US_MTT_{lastday}')

    ### High ADR Swing
    stock_data, tickers = screening_stocks_by_func(filter_stocks_high_ADR_swing, True, True)
    first_stock_data : pd.DataFrame = stock_data[tickers[0]]
    lastday = str(first_stock_data.index[-1].date())
    sd.cook_stock_info_from_tickers(tickers, f'US_HighAdrSwing_{lastday}')
elif index == 13:
    daysNum = int(365)
    stock_list = sd.getStockListFromLocalCsv()
    out_tickers = []
    out_stock_datas_dic = {}
    sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, True)
    # To get trade day easily.
    appleData = out_stock_datas_dic['AAPL']

    print("start power gap screening!")
    power_gap_screen_list = []
    for i in range(1, 40):
        stock_data_dic, tickers = screening_stocks_by_func(filter_stock_power_gap, True, True, -i)
        tradeDay = str(appleData.index[-i].date())
        s = str.format(f"[{tradeDay}] power gap tickers: ") + str(tickers)
        print(s)
        power_gap_screen_list.append(s)

    print("Done. print power gap screen list")
    for s in power_gap_screen_list:
        print(s)
# --------------------------------------------------------------------


