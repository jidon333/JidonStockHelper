

from jdStockDataManager import JdStockDataManager 

class JdStockFilteringManager:
    def __init__(self, inStockDataManager : JdStockDataManager):
        print("Hello JdScreenStockManager!")
        self.sd = inStockDataManager
        self.MTT_ADR_minimum = 1


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

    def filter_stocks_high_ADR_swing(self, stock_datas_dic : dict):
        filtered_tickers = []
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()

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
        - DCR 50% 이상
        - 헬스케어 섹터 제외(바이오 무빙 혼란하다.)

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


                if high - low > 0:
                    DCR = (close - low) / (high - low)
                else:
                    continue


                # 순서대로
                # ADR > 2, Gap Open, 10% 이상 상승, 
                #if ADR > 2 and open > high_1d_ago and change > 10 and change > ADR*2 and close > ma200 and volume > volume_ma50*2 and bIsVolumeEnough and DCR >= 0.5:
                if  volume > volume_ma50_1d_ago*2 and bIsVolumeEnough:
                    filtered_tickers.append(ticker)

            except Exception as e:
                continue
        
        return filtered_tickers

    # ADR 2이상
    # RS 랭킹 상위 15%
    # 헬스케어, 에너지 섹터 제외
    # Volume 50MA 100만 이상 and 5불 이상 주식

    def filter_stock_Good_RS(self, stock_datas_dic : dict):
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
