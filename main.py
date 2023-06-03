
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




def DrawStockDatas(stock_datas_dic, tickers, inStockManager : JdStockDataManager, maxCnt = -1):
    stock_data = stock_datas_dic[tickers[0]]
    chart = JdChart(stock_data)
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
    chart = JdChart(None, updown_nyse, updown_nasdaq, updown_sp500)
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





def cook_infos_from_last_searched_tickers(inStockManager : JdStockDataManager, inFileName):
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
#index = 0

sd = JdStockDataManager()
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
        sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, False)


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
            # (거래량 50일 평균 20만이상 + 10불이상 or 하루거래량 100억 이상)
            bIsVolumeEnough = (volume_ma50 >= 200000 and close >= 10 ) or volume_ma50*close > 10000000
            # 마지막날 거래량 100,000주 이상
            #bIsVolumeEnough = bIsVolumeEnough and last_volume >= 100000
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
            #bPocketPivot = sd.check_pocket_pivot(inStockData)
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
    #quantTickers = ['SYM']
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
    cook_infos_from_last_searched_tickers(sd, 'US_MMT_0603_2')

# --------------------------------------------------------------------