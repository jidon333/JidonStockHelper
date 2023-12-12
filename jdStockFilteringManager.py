

from jdStockDataManager import JdStockDataManager
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import time
import os
import pickle

from jdGlobal import profiles_folder

if not os.path.exists(profiles_folder):
    os.makedirs(profiles_folder)

class JdStockFilteringManager:
    def __init__(self, inStockDataManager : JdStockDataManager):
        print("Hello JdScreenStockManager!")
        self.sd = inStockDataManager
        self.MTT_ADR_minimum = 1


    

    def screening_stocks_by_func(self, filter_func, bUseLoadedStockData = True, bSortByRS = False, n_day_before = -1):
        out_tickers = []
        out_stock_datas_dic = {}

        daysNum = int(365)
        stock_list = self.sd.getStockListFromLocalCsv()
        self.sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, bUseLoadedStockData)


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
                    rank = self.sd.get_ATRS150_exp_Ranks(ticker).iloc[-1]
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
        print(f"Search time elapsed: {execution_time}sec")
        print('filtered by quant data: \n', selected_tickers)
        print('selected tickers num: ', len(selected_tickers))

        return out_stock_datas_dic, selected_tickers


    # return filtered tickers
    def filter_stocks_MTT(self, stock_datas_dic : dict, n_day_before = -1):
        filtered_tickers = []
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()
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

            if ADR < self.MTT_ADR_minimum:
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

    def filter_stocks_high_ADR_swing(self, stock_datas_dic : dict, n_day_before = -1):
        filtered_tickers = []
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()

        for ticker, inStockData in stock_datas_dic.items():

            close = inStockData['Close'].iloc[n_day_before]
            ma150 = inStockData['150MA'].iloc[n_day_before]
            ma200 = inStockData['200MA'].iloc[n_day_before]
            bIsUpperMA_150_200 = close > ma150 and close > ma200

            # early rejection for optimization
            if bIsUpperMA_150_200 == False:
                continue
        
            ma50 = inStockData['50MA'].iloc[n_day_before]
            volume_ma50 = inStockData['Volume'].rolling(window=50).mean().iloc[n_day_before]
            ADR = inStockData['ADR'].iloc[n_day_before]
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
    def filter_stock_ALL(self, stock_datas_dic : dict):
        filtered_tickers = []
        for ticker, inStockData in stock_datas_dic.items():
            filtered_tickers.append(ticker)

        return filtered_tickers


    def filter_stock_Custom(self, stock_datas_dic : dict):
        # filter stock good RS 
        filtered_tickers = []
        Mtt_tickers = self.filter_stocks_MTT(stock_datas_dic)
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()
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

                if abs(self.sd.get_percentage_AtoB(close, ma150)) < ADR*2 or abs(self.sd.get_percentage_AtoB(close, ma200)) < ADR*2:
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


    def filter_stock_FA50(self, stock_datas_dic : dict, n_day_before = -1):
        filtered_tickers = []
        Mtt_tickers = self.filter_stock_ALL(stock_datas_dic)
        gisc_df = self.sd.get_GICS_df()
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

    def filter_stock_Good_RS(self, stock_datas_dic : dict):
        """
        - ADR 2이상
        - RS 랭킹 상위 15%
        - 헬스케어, 에너지 섹터 제외
        - Volume 50MA 100만 이상 and 5불 이상 주식
        """
        # filter stock good RS 
        filtered_tickers = []
        Mtt_tickers = self.filter_stock_ALL(stock_datas_dic)
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()
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
    def filter_stock_hope_from_bottom(self, stock_datas_dic : dict):
        """
        - ADR 2이상
        - RS 랭킹 상위 30%
        - 헬스케어, 에너지 섹터 제외
        - Volume 50MA 100만 이상 and 5불 이상 주식
        - 150SMA > 200SMA (바닥으로 추락하기전 2단계였던 주식을 보고 싶었음)
        - 150 or 200SMA 이격도 2 ADR 미만 (장기 이평선 근처에서 횡보하는 것을 찾기 위함)
        """
        filtered_tickers = []
        Mtt_tickers = self.filter_stock_ALL(stock_datas_dic)
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()
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

                if abs(self.sd.get_percentage_AtoB(close, ma150)) < ADR*2 or abs(self.sd.get_percentage_AtoB(close, ma200)) < ADR*2:
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


    def filter_stock_power_gap(self, stock_datas_dic : dict, n_day_before = -1):
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
        - 헬스케어 섹터 제외(바이오 무빙 혼란하다.)

        -- 이후 데이터 재가공 과정에서 갭 이후 ADR 1% 미만 주식은 제외(아마도 인수합병)

        """
        filtered_tickers = []
        all_tickers = self.filter_stock_ALL(stock_datas_dic)
        gisc_df = self.sd.get_GICS_df()
        bIsNotHealthCare = False
        bIsVolumeEnough = False
        for ticker in all_tickers:
            stockData = stock_datas_dic[ticker]
            try:
                # [Optimize] Early rejection 

                # sector check
                sector = gisc_df.loc[ticker]['sector']
                if sector == 'Healthcare':
                    continue

                # ADR check
                ADR_1d_ago = stockData['ADR'].iloc[n_day_before-1]
                if ADR_1d_ago < 2:
                    continue

                # gap check
                open = stockData['Open'].iloc[n_day_before]
                high_1d_ago = stockData['High'].iloc[n_day_before - 1]
                if open <= high_1d_ago:
                    continue 


                # change(%) check
                close_1d_ago = stockData['Close'].iloc[n_day_before -1]
                close = stockData['Close'].iloc[n_day_before]
                change = self.sd.get_percentage_AtoB(close_1d_ago, close)
                if change < 10.0:
                    continue
                if change < ADR_1d_ago * 2.0:
                    continue

                # above ma200 check
                ma200 = stockData['200MA'].iloc[n_day_before]
                if close < ma200:
                    continue 

                volume_ma50_1d_ago = stockData['Volume'].rolling(window=50).mean().iloc[n_day_before-1]
                volume = stockData['Volume'].iloc[n_day_before]

                high = stockData['High'].iloc[n_day_before]
                low = stockData['Low'].iloc[n_day_before]

                bIsVolumeEnough = (volume_ma50_1d_ago >= 200000 and close >= 5 ) or volume_ma50_1d_ago*close > 5000000

                # 순서대로
                # ADR > 2, Gap Open, 10% 이상 상승, 
                #if ADR > 2 and open > high_1d_ago and change > 10 and change > ADR*2 and close > ma200 and volume > volume_ma50*2 and bIsVolumeEnough and DCR >= 0.5:
                if  volume > volume_ma50_1d_ago*2 and bIsVolumeEnough:
                    filtered_tickers.append(ticker)

            except Exception as e:
                continue
        
        return filtered_tickers

    

    def get_power_gap_stocks_in_range(self, range_from : int , range_to : int):
        """
        return date-tickers dictionary
        range_from :  Screening will be started from this 'param trading day ago' ex) 40 mean that searching process start from 40 trading day ago.
        range_to :  Screening will be ended at this 'param trading day ago' ex) 10 mean that searching process will be ended at the 10 trading day ago.

        range_from must be bigger than range_to

        """
        range_from = abs(range_from)
        range_to = abs(range_to)
        if range_from <= range_to:
            print('error. range_from must be bigger than range_to')
            return

        years = float(range_from) / 240.0
        daysNum = int(365) + int(years * 365.0)
        stock_list = self.sd.getStockListFromLocalCsv()
        out_tickers = []
        out_stock_datas_dic = {}
        self.sd.getStockDatasFromCsv(stock_list, out_tickers, out_stock_datas_dic, daysNum, True)
        # To get trade day easily.
        appleData = out_stock_datas_dic['AAPL']

        print("start power gap screening!")

        date_tickers_dic = {}

        power_gap_screen_list = []
        for i in range(range_to, range_from):
            stock_data_dic, tickers = self.screening_stocks_by_func(self.filter_stock_power_gap, True, True, -i)
            tradeDay = str(appleData.index[-i].date())
            s = str.format(f"[{tradeDay}] power gap tickers: ") + str(tickers)
            print(s)
            power_gap_screen_list.append(s)
            date_tickers_dic[appleData.index[-i].date()] = tickers

        print("Done. print power gap screen list")
        for s in power_gap_screen_list:
            print(s)

        return date_tickers_dic
    
    def cook_power_gap_profiles(self, range_from : int , range_to : int , performance_period : int, profile_period : int):
        """
        - range_from :          Screening will be started from this 'param trading day ago' ex) 40 mean that searching process start from 40 trading day ago.
        - range_to :            Screening will be ended at this 'param trading day ago' ex) 10 mean that searching process will be ended at the 10 trading day ago.
        - performance_period:   Measures the return of the stock from the gap day to the {performance_period} trading day.
        - profile_period        profile check period for C/V check, ma touch, highest price etc ... 

        range_from must be bigger than range_to

        """

        sd = self.sd

        gap_date_tickers_dic = {}

        bUseGapDataCache = True
        if bUseGapDataCache:
            try: 
                with open('cache_gap_date_tickers_dic', "rb") as f:
                    gap_date_tickers_dic = pickle.load(f)
            except Exception as e:
                print('no cache_gap_date_tickers_dic in local')

        if not gap_date_tickers_dic:
            gap_date_tickers_dic = self.get_power_gap_stocks_in_range(range_from, range_to)
            try:
                with open('cache_gap_date_tickers_dic', "wb") as f:
                    pickle.dump(gap_date_tickers_dic, f)
            except Exception as e:
                print(e)
    
        all_tickers = []
        all_stock_datas_dic = {}
        sd.getStockDatasFromCsv(sd.getStockListFromLocalCsv(), all_tickers, all_stock_datas_dic, 365, True)
        gisc_df = sd.get_GICS_df()

        gap_profile_dic = {}

        for gap_date, gap_tickers in gap_date_tickers_dic.items():
            for ticker in gap_tickers:
                stockData : pd.DataFrame = all_stock_datas_dic[ticker]
                d0_index = sd.date_to_index(stockData, gap_date)

                performance_measure_day : int = d0_index + performance_period -1
                profile_end_day : int = d0_index + profile_period -1

                if performance_measure_day >= 0:
                    print("Error: performance_period is longer than last data.")
                    continue

                
                d0_open = stockData['Open'].iloc[d0_index]
                d0_close = stockData['Close'].iloc[d0_index]
                d0_low = stockData['Low'].iloc[d0_index]
                d0_high = stockData['High'].iloc[d0_index]

                close_1d_ago = stockData['Close'].iloc[d0_index - 1]
                ADR_1d_ago = stockData['ADR'].iloc[d0_index - 1]
                volume_ma50_1d_ago = stockData['Volume'].rolling(window=50).mean().iloc[d0_index-1]
                d0_volume = stockData['Volume'].iloc[d0_index]

                n_day_later_close = stockData['Close'].iloc[performance_measure_day]

                # 입수합병 필터링
                # 갭당일부터 성과측정일까지 ADR이 1% 미만으로 줄어들면 인수합병으로 본다.
                # 애초에 ADR 2 이상의 주식이 Power gap 이후 ADR이 1로 줄어들었다면 뭔가 잘못된 것이다. 굼뱅이 주식은 필요 없다!

                # DR% (Daily Range)
                daily_range_percentages = stockData['High'] / stockData['Low']

                # ADR(%) since gap day close
                n = performance_period
                ADRs_since_gap = daily_range_percentages.rolling(n).mean()
                ADRs_since_gap = 100 * (ADRs_since_gap - 1)
                adr_since_gap = ADRs_since_gap.iloc[performance_measure_day]
                if adr_since_gap < 1:
                    print("It's probabily M&A. reject this ticker from the profiles, ticker: ",  ticker)
                    continue

                # [gap_date]
                gap_date = gap_date

                # [performance] N day 이후 성과
                performance = sd.get_percentage_AtoB(d0_close, n_day_later_close)

                # [d0_close] 갭 종가($)
                d0_close = d0_close

                # [d0_open_change] 갭 상승폭 Open(%)
                d0_open_change = sd.get_percentage_AtoB(close_1d_ago, d0_open)

                # [d0_close_change] 종가 상승폭 (%)
                d0_close_change = sd.get_percentage_AtoB(close_1d_ago, d0_close)

                # [d0_daily_range] Daily Range(%)
                d0_daily_range = sd.get_percentage_AtoB(d0_low, d0_high)

                # [d0_adr_multiple] (nADR%)
                d0_adr_multiple = d0_close_change / ADR_1d_ago

                
                # [DCR](%)
                if d0_high - d0_low > 0:
                    DCR = (d0_close - d0_low) / (d0_high - d0_low)
                else:
                    DCR = 0

                # [d0_volume] 거래량
                d0_volume = d0_volume

                # [d0_volume_vs_50Avg] 거래량(50일 평균 대비)
                d0_volume_vs_50Avg = d0_volume / volume_ma50_1d_ago

                # [bOEL]
                bOEL = d0_open == d0_low
                
                # ------------ d0 이후에 알 수 있는 것들 ------------

                # [first_ma_touch_day]
                first_ma_touch_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_ma_touch(stockData, 10, True, d0_index + i):
                        first_ma_touch_day = i
                        break


                # [d0_open_violation_day]
                d0_open_violation_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stockData, d0_open, d0_index + i):
                        d0_open_violation_day = i
                        break

                # [d0_low_violation_day]
                d0_low_violation_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stockData, d0_low, d0_index + i):
                        d0_low_violation_day = i
                        break

                # [HVC_violation_first_day]
                HVC_violation_first_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stockData, d0_close, d0_index + i):
                        HVC_violation_first_day = i
                        break


                # [HVC_violation_last_day] (저가가 HVC를 마지막으로 침범한 날)
                # [HVC_violation_cnt] (HVC 아래에 주가가 위치했던 날의 수)
                HVC_violation_last_day = profile_period
                HVC_violation_cnt = 0
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stockData, d0_close, d0_index + i):
                        HVC_violation_last_day = i
                        HVC_violation_cnt = HVC_violation_cnt + 1
                        


                # [alpha_window_lowest_day]
                # Alpha Window: day1 ~ day5 of Power gap. day0 is gap day.)

                # start_pos는 시작 위치입니다. 예를 들어, 0은 데이터프레임의 첫 번째 행입니다.
                start_pos = d0_index + 1
                # 5일간의 가장 낮은 가격의 위치를 구합니다.
                lowest_price_pos = stockData['Low'].iloc[start_pos:start_pos + 5].idxmin()
                # 실제 위치를 얻기 위해 데이터프레임의 인덱스와 비교합니다.
                lowest_index_pos = stockData.index.get_loc(lowest_price_pos)
                # 계속해서 사용하는 인덱스는 마이너스(-) 인덱스임.
                lowest_index_pos = lowest_index_pos - len(stockData)
                alpha_window_lowest_day = lowest_index_pos - d0_index

                # [alpha_window_lowest_pct_from_HVC]
                alpha_window_lowest_price = stockData['Low'].iloc[lowest_index_pos]
                alpha_window_lowest_pct_from_HVC = sd.get_percentage_AtoB(d0_close, alpha_window_lowest_price)

                # [alpha_window_highest_day]
                start_pos = d0_index + 1
                highest_price_pos = stockData['High'].iloc[start_pos:start_pos + 5].idxmax()
                highest_index_pos = stockData.index.get_loc(highest_price_pos)
                highest_index_pos = highest_index_pos - len(stockData)
                alpha_window_highest_day = highest_index_pos - d0_index

                # [alpha_window_highest_pct_from_HVC]
                alpha_window_highest_price = stockData['High'].iloc[highest_index_pos]
                alpha_window_highest_pct_from_HVC = sd.get_percentage_AtoB(d0_close, alpha_window_highest_price)

                # [HVC_recovery_day_from_alpha_window_lowest]
                # HVC violation이 발생할때만 유효한 프로퍼티
                # d1 ~ d5 영역에서 HVC 를 침범한 저가가 며칠만에 회복되었는지? (회복: 종가가 HVC 위에서 다시 마감)
                # alpha_window_lowest_day로부터 카운팅하며 종가가 HVC 위에 있는지 확인 해야 한다.

                HVC_recovery_day_from_alpha_window_lowest = profile_period
                for i in range(lowest_index_pos, profile_end_day + 1):
                    cnt_from_lowest_day = i - lowest_index_pos
                    c = stockData['Close'].iloc[i]
                    if c > d0_close:
                        HVC_recovery_day_from_alpha_window_lowest = cnt_from_lowest_day
                        break

                
                gap_profile_dic[ticker] = [gap_date, performance_period, performance, d0_close, d0_open_change, d0_close_change,
                                            d0_daily_range, d0_adr_multiple, DCR, d0_volume, d0_volume_vs_50Avg, bOEL,
                                            first_ma_touch_day, d0_open_violation_day, d0_low_violation_day, HVC_violation_first_day, HVC_violation_last_day, HVC_violation_cnt,
                                            alpha_window_lowest_day, alpha_window_lowest_pct_from_HVC, alpha_window_highest_day, alpha_window_highest_pct_from_HVC,
                                              HVC_recovery_day_from_alpha_window_lowest]


        df = pd.DataFrame.from_dict(gap_profile_dic).transpose()
        columns = ['gap_date', 'performance_period', 'performance', 'd0_close', 'd0_open_change', 'd0_close_change',
                                            'd0_daily_range', 'd0_adr_multiple', 'DCR', 'd0_volume', 'd0_volume_vs_50Avg', 'bOEL',
                                            'first_ma_touch_day', 'd0_open_violation_day', 'd0_low_violation_day', 'HVC_violation_first_day', 'HVC_violation_last_day', 'HVC_violation_cnt',
                                            'alpha_window_lowest_day', 'alpha_window_lowest_pct_from_HVC', 'alpha_window_highest_day', 'alpha_window_highest_pct_from_HVC',
                                              'HVC_recovery_day_from_alpha_window_lowest']
        df.columns = columns
        df.index.name = 'Symbol'

        # fix object types to numeric.
        # 자료형이 통합되지 않은 리스트를 value로 갖는 딕셔너리를 DataFrame으로 변환하는 과정에서 모든 column이 object로 변환되는 문제가 있음.
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        df = df.round(2)

        try:
            #save_path = os.path.join(profiles_folder, f'power_gap.csv')
            #df.to_csv(save_path, encoding='utf-8-sig')
            save_path = os.path.join(profiles_folder, f'power_gap_{range_from}_{range_to}_{performance_period}_{profile_period}.xlsx')
            df.to_excel(save_path, index_label='Symbol')

            print(f"{save_path}", "is saved!")
        except Exception as e:
            print(f"An error occurred: {e}")

    