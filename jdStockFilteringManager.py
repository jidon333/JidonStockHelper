

from jdStockDataManager import JdStockDataManager
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import time
import os
import pickle

import logging

from jdGlobal import PROFILES_FOLDER

from jd_filter_utils import precheck 

if not os.path.exists(PROFILES_FOLDER):
    os.makedirs(PROFILES_FOLDER)

# -------------------------------------------------------------------------
# Helper function for 50-day volume & close check
# -------------------------------------------------------------------------

def check_50day_volume_and_close_price(
    stock_data: pd.DataFrame,
    day_before: int,
    min_avg_volume: float,
    min_close_price: float
) -> bool:
    """
    Checks whether the given stock data meets the minimum 50-day average volume
    and close price requirements.

    :param stock_data: DataFrame for a single ticker (containing 'Volume', 'Close', etc.)
    :param day_before: Index offset for n_day_before
    :param min_avg_volume: Minimum 50-day average volume required
    :param min_close_price: Minimum close price required
    :return: True if both conditions (volume & price) are satisfied, otherwise False
    """
    try:
        volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[day_before]
        close_price = stock_data['Close'].iloc[day_before]
        return (volume_ma50 >= min_avg_volume) and (close_price >= min_close_price)
    except (KeyError, IndexError):
        return False



def has_required_data(stock_data: pd.DataFrame, day_before: int, columns: list) -> bool:
    """
    Checks if the specified columns in stock_data have non-NaN values at the given index.
    Returns True if all exist and are not NaN, otherwise False.
    """
    for col in columns:
        try:
            val = stock_data[col].iloc[day_before]
            if pd.isna(val):
                return False
        except Exception:
            return False
    return True


# -------------------------------------------------------------------------
# Filtering Manager
# -------------------------------------------------------------------------
class JdStockFilteringManager:
    def __init__(self, inStockDataManager : JdStockDataManager):
        print("Hello JdScreenStockManager!")
        self.sd = inStockDataManager
        self.MTT_ADR_minimum = 1
        self.LastDayMinimumVolume = 0


    

    def screening_stocks_by_func(self, filter_func, bUseLoadedStockData = True, bSortByRS = False, n_day_before = -1):
        """
        Loads stock data, applies a given filter function, and optionally sorts by RS rank.
        """

        daysNum = int(365)
        stock_list = self.sd.get_local_stock_list()
        out_stock_datas_dic = self.sd.get_stock_datas_from_csv(stock_list, daysNum, bUseLoadedStockData)
        out_tickers = list(out_stock_datas_dic.keys())


        search_start_time = time.time()
        selected_tickers = []

        # 마지막 날 기준이 아닌 과거를 기준으로 데이터를 뽑고 싶은 경우 n_day_before를 사용.
        if n_day_before == -1:
            selected_tickers = filter_func(out_stock_datas_dic)
        else:
            selected_tickers = filter_func(out_stock_datas_dic, n_day_before)


        # Optional: sort by RS
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

        return out_stock_datas_dic, selected_tickers



    # -------------------------------------------------------------------------
    # 1) filter_stocks_MTT
    # -------------------------------------------------------------------------
    @precheck(
        cols=["Close", "150MA", "200MA", "MA150_Slope", "MA200_Slope", "ADR"],
        min_volume=200_000,
        min_price=10,
        vol_window=50
    )
    def filter_stocks_MTT(self, stock_datas_dic: dict, n_day_before=-1):
        """
        Filters tickers that meet the following conditions (MTT criteria):

        1) close > 150SMA and close > 200SMA
        2) close > 50SMA
        3) 50SMA > 150SMA > 200SMA
        4) 50-day avg volume >= 200,000
        5) close >= 10
        6) 50-day avg dollar volume >= 10,000,000
        7) ATRS rank < 1000
        8) 150SMA slope > 0 and 200SMA slope > 0
        9) last day's volume > self.LastDayMinimumVolume 
        10) ADR > self.MTT_ADR_minimum

        :param stock_datas_dic: {ticker: DataFrame} for each ticker
        :param n_day_before: day offset to check (default: -1 means last day)
        :return: List of tickers passing all conditions
        """
        filtered_tickers = []
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()

        for ticker, stock_data in stock_datas_dic.items():

            # (1) precheck를 통해 데코레이터가 통과시킨 dict 만 오므로 NaN 검사 진행 x
            close  = stock_data['Close'].iloc[n_day_before]
            ma50   = stock_data['50MA'].iloc[n_day_before]
            ma150  = stock_data['150MA'].iloc[n_day_before]
            ma200  = stock_data['200MA'].iloc[n_day_before]
            ma150_slope = stock_data['MA150_Slope'].iloc[n_day_before]
            ma200_slope = stock_data['MA200_Slope'].iloc[n_day_before]
            volume_ma50 = stock_data['Volume'].rolling(50).mean().iloc[n_day_before]
            ADR    = stock_data['ADR'].iloc[n_day_before]
            last_volume = stock_data['Volume'].iloc[n_day_before]


            if not (close > ma150 and close > ma200):
                continue
            if not (close > ma50):
                continue
            if not (ma50 > ma150 > ma200):
                continue
            if volume_ma50 * close < 10_000_000:         # 6)
                continue
            if ma150_slope <= 0 or ma200_slope <= 0:     # 8)
                continue
            try:
                atrs_rank = atrs_ranking_df.loc[ticker].iloc[n_day_before]
            except KeyError:
                logging.debug(f"[filter_stocks_MTT] {ticker}: not in ATRS df")
                continue
            except IndexError as e:
                logging.debug(f"[filter_stocks_MTT] {ticker}: IndexErr {e}")
                continue
            if atrs_rank >= 1000:                        # 7)
                continue
            if last_volume < self.LastDayMinimumVolume:  # 9)
                continue
            if ADR < self.MTT_ADR_minimum:               # 10)
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers


    # -------------------------------------------------------------------------
    # 2) filter_stocks_high_ADR_swing
    # -------------------------------------------------------------------------
    @precheck(
        cols=["Close", "50MA", "200MA", "ADR"],
        min_volume=1_000_000,
        min_price=5,
        vol_window=50
    )
    def filter_stocks_high_ADR_swing(self, stock_datas_dic : dict, n_day_before = -1):
        """
        High ADR short-term momentum filter:

        1) Price >= 5
        2) 20-day ADR >= 4.0
        3) 50-day average volume >= 1,000,000
        4) Price > 50SMA
        5) Price > 200SMA
        6) Exclude Healthcare sector
        7) (Optional) last day's volume >= self.LastDayMinimumVolume
        """


        filtered_tickers = []
        gisc_df = self.sd.get_GICS_df()

        for ticker, stock_data in stock_datas_dic.items():

            close = stock_data['Close'].iloc[n_day_before]
            ma200 = stock_data['200MA'].iloc[n_day_before]
            ma50 = stock_data['50MA'].iloc[n_day_before]
            last_volume = stock_data['Volume'].iloc[n_day_before]
            ADR = stock_data['ADR'].iloc[n_day_before]


            # 2) 20-day ADR >= 4.0
            # code uses ADR from 20-day by default if inStockData['ADR'] is 20day. 
            if ADR < 4.0:
                continue

            if last_volume < self.LastDayMinimumVolume:
                continue


            # 4) Price > 50SMA
            if not (close > ma50):
                continue

            # 5) Price > 200SMA
            if not (close > ma200):
                continue

             # 6) Exclude Healthcare sector
            is_healthcare = False
            try:
                sector = gisc_df.loc[ticker]['sector']
                if sector == 'Healthcare':
                    is_healthcare = True
            except Exception:
                pass

            if is_healthcare:
                continue
                        
                        
            filtered_tickers.append(ticker)


        return filtered_tickers


    # -------------------------------------------------------------------------
    # 3) filter_stocks_Bull_Snort
    # -------------------------------------------------------------------------
    def filter_stocks_Bull_Snort(self, stock_datas_dic : dict, n_day_before = -1):

        """
        Bull Snort Filter:

        1) Price >= 8
        2) DCR >= 0.5 (last candle closes in upper 50% range)
        3) Volume >= 3x of 20-day average volume
        4) Price up >= 3% from previous close
        5) 20-day ADR >= 2.0
        6) 50-day average volume >= 5,000,000
        """

        filtered_tickers = []

        for ticker, stock_data in stock_datas_dic.items():

            if not has_required_data(stock_data, n_day_before, ['Close','Volume']):
                continue

            try:
                close = stock_data['Close'].iloc[n_day_before]
                volume = stock_data['Volume'].iloc[n_day_before]
                volume_ma20 = stock_data['Volume'].rolling(window=20).mean().iloc[n_day_before]
                volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before]
            except Exception as e:
                print(e)
                continue

            # 1) Price >= 8
            if close < 8:
                continue

            # 2) DCR >= 0.5
            DCR = self.sd.get_DCR_normalized(stock_data, n_day_before)  # n_day_before might be used
            if DCR < 0.5:
                continue


            # 3) Volume >= 3x of 20-day average volume
            if volume <= volume_ma20 * 3.0:
                continue

            # 4) Price up >= 3% from previous close
            try:
                prev_close = stock_data['Close'].iloc[n_day_before - 1]
                change_pct = self.sd.get_percentage_AtoB(prev_close, close)
                if change_pct < 3:
                    continue
            except Exception:
                continue


            # 5) 20-day ADR >= 2.0
            try:
                ADR = stock_data['ADR'].iloc[n_day_before]
                if ADR < 2.0:
                    continue
            except Exception:
                continue

            # 6) 50-day average volume >= 5,000,000
            if volume_ma50 < 5_000_000:
                continue

                       
            filtered_tickers.append(ticker)

        return filtered_tickers

    # -------------------------------------------------------------------------
    # 4) filter_stocks_rs_8_10
    # -------------------------------------------------------------------------
    def filter_stocks_rs_8_10(self, stock_datas_dic : dict, n_day_before = -1):
        """
        Filters tickers where:
         1) Price >= 8
         2) 50-day average volume >= 3,000,000
         3) ADR >= 2.5
         4) close > 200SMA
         5) In last 10 days, ticker outperformed index on 8 days
        """

        filtered_tickers = []

        for ticker, stock_data in stock_datas_dic.items():

            if not has_required_data(stock_data, n_day_before, ['200MA','ADR']):
                continue

            try:
                close = stock_data['Close'].iloc[n_day_before]
                ma200 = stock_data['200MA'].iloc[n_day_before]
                volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before]
                ADR = stock_data['ADR'].iloc[n_day_before]

            except Exception:
                continue

            # 1) Price >= 8
            if close < 8:
                continue

          
            # 2) 50-day average volume >= 3,000,000
            if volume_ma50 < 3_000_000:
                continue

            # 3) ADR >= 2.5
            if ADR < 2.5:
                continue

            # 4) close > 200SMA
            if close < ma200:
                continue

            # 5) In last 10 days, ticker outperformed index on 8 days
            try:
                index_data = self.sd.us500_data
                changes_index = (index_data['Close'] - index_data['Close'].shift(1)) / index_data['Close'].shift(1)
                changes_ticker = (stock_data['Close'] - stock_data['Close'].shift(1)) / stock_data['Close'].shift(1)
                rs_strong_cnt = 0
                for i in range(10):
                    day_idx = n_day_before - i
                    if changes_ticker.iloc[day_idx] > changes_index.iloc[day_idx]:
                        rs_strong_cnt += 1

                if rs_strong_cnt < 8:
                    continue
            except Exception:
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers


    # -------------------------------------------------------------------------
    # 5) filter_stocks_young
    # -------------------------------------------------------------------------
    def filter_stocks_young(self, stock_datas_dic : dict, n_day_before = -1):
        """
        1) Stock IPO < 200 days (ma200 not present => no 200MA)
        2) Volume >= 1,000,000
        3) close >= 10
        4) ATRS rank < 1000
        """
        filtered_tickers = []
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()


        for ticker, stock_data in stock_datas_dic.items():

            if not has_required_data(stock_data, n_day_before, ['Close','Volume']):
                continue

            try:
                close = stock_data['Close'].iloc[n_day_before]
                ma200 = stock_data['200MA'].iloc[n_day_before]
                volume = stock_data['Volume'].iloc[n_day_before]
            except Exception:
                continue

            bNo_ma200 = pd.isna(ma200)

            # 4) ATRS rank < 1000
            is_ATRS_good = False
            try:
                rank = atrs_ranking_df.loc[ticker].iloc[n_day_before]
                is_ATRS_good = (rank < 1000)
            except Exception:
                continue


            # 1) no 200MA => IPO < 200 days
            if not bNo_ma200:
                continue

            # 2) volume >= 1,000,000
            if volume < 1_000_000:
                continue

            # 3) close >= 10
            if close < 10:
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    # -------------------------------------------------------------------------
    # 6) filter_stock_ALL
    # -------------------------------------------------------------------------
    def filter_stock_ALL(self, stock_datas_dic : dict):
        """
        Simply returns all tickers.
        """
        
        return list(stock_datas_dic.keys())


    # -------------------------------------------------------------------------
    # 7) filter_stock_Custom
    # -------------------------------------------------------------------------
    def filter_stock_Custom(self, stock_datas_dic : dict):
        """
        Example custom filter: intersect with a predefined mylist
        """
        filtered_tickers = []
        all_tickers = self.filter_stock_ALL(stock_datas_dic)

        # Example list
        mylist = [
            'MHO','BLDR','GFF','BVN','DRCT','ANF','STRL','TPG','PTVE','COIN','AFRM','SNAP',
            'DFH','LSEA','RKT','MARA','AAOI','S','GPS','MOD','XMTR','ASPN','CUBI','GIII',
            'PDD','STNE','CLSK','ASTL','OSW','ESTC','WD','WIX','AMD','FROG','RCKT','GTX',
            'PLAY','ELF','JBI','TEAM'
        ]

        filtered_tickers = set(all_tickers) & set(mylist)
        return list(filtered_tickers)



    # -------------------------------------------------------------------------
    # 8) filter_stock_FA50
    # -------------------------------------------------------------------------
    def filter_stock_FA50(self, stock_datas_dic : dict, n_day_before = -1):
        """
        1) ADR > 3
        2) 50-day avg volume >= 1,000,000
        3) close >= 8
        4) close >= 50MA
        """
        filtered_tickers = []
        all_tickers = self.filter_stock_ALL(stock_datas_dic)

        for ticker in all_tickers:
            stock_data = stock_datas_dic[ticker]


            if not has_required_data(stock_data, n_day_before, ['ADR','50MA']):
                continue


            stock_data = stock_datas_dic[ticker]
            try:
                ADR = stock_data['ADR'].iloc[n_day_before]
                volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before]
                close = stock_data['Close'].iloc[n_day_before]
                ma50 = stock_data['50MA'].iloc[n_day_before]
            except Exception:
                continue

            if ADR <= 3:
                continue
            if volume_ma50 < 1_000_000:
                continue
            if close < 8:
                continue
            if close < ma50:
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers
    

    # -------------------------------------------------------------------------
    # 9) filter_stock_ATR_plus_150
    # -------------------------------------------------------------------------
    def filter_stock_ATR_plus_150(self, stock_datas_dic : dict, n_day_before = -1):
        """
         - 50-day avg volume >= 2,000,000
         - close >= 10
         - (close - open) > ATR * 1.5
        """

        filtered_tickers = []
        all_tickers = self.filter_stock_ALL(stock_datas_dic)

        for ticker in all_tickers:
            stock_data = stock_datas_dic[ticker]


            if not has_required_data(stock_data, n_day_before, ['ATR']):
                continue


            try:
                ATR = stock_data['ATR'].iloc[n_day_before]
                volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before]
                open_ = stock_data['Open'].iloc[n_day_before]
                close = stock_data['Close'].iloc[n_day_before]
            except Exception:
                continue

            if volume_ma50 < 2_000_000 or close < 10:
                continue

            diff = close - open_
            if diff <= (ATR * 1.5):
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers
  

    # -------------------------------------------------------------------------
    # 10) filter_stock_Good_RS
    # -------------------------------------------------------------------------
    def filter_stock_Good_RS(self, stock_datas_dic : dict, n_day_before = -1):
        """
        1) ADR >= 2
        2) RS rank top 1000
        3) Exclude Healthcare, Energy
        4) 50-day avg volume >= 1,000,000 and close >= 5
        5) last day's volume >= self.LastDayMinimumVolume
        6) close > 200MA
        7) close > 21EMA
        """

        filtered_tickers = []
        Mtt_tickers = self.filter_stock_ALL(stock_datas_dic)
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()

        for ticker in Mtt_tickers:
            stock_data = stock_datas_dic[ticker]

            if not has_required_data(stock_data, n_day_before, ['ADR','200MA']):
                continue


            try:
                ADR = stock_data['ADR'].iloc[n_day_before]
                volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before]
                last_volume = stock_data['Volume'].iloc[n_day_before]
                close = stock_data['Close'].iloc[n_day_before]
                ma200 = stock_data['200MA'].iloc[n_day_before]
                ema21 = stock_data['Close'].ewm(span=20, adjust=False).mean().iloc[n_day_before]
            except Exception:
                continue

            # 1) ADR >= 2
            if ADR < 2:
                continue

            # 2) RS rank top 1000
            is_ATRS_Ranking_Good = False
            try:
                atrsRank = atrs_ranking_df.loc[ticker].iloc[n_day_before]
                is_ATRS_Ranking_Good = (atrsRank < 1000)
            except Exception:
                continue

            if not is_ATRS_Ranking_Good:
                continue

            # 3) Exclude Healthcare, Energy
            try:
                sector = gisc_df.loc[ticker]['sector']
                if sector in ['Healthcare', 'Energy']:
                    continue
            except Exception:
                continue

            # 4) 50-day avg volume >= 1,000,000 and close >= 5
            if volume_ma50 < 1_000_000 or close < 5:
                continue

            # 5) last day's volume >= self.LastDayMinimumVolume
            if last_volume < self.LastDayMinimumVolume:
                continue

            # 6) close > 200MA
            if close <= ma200:
                continue

            # 7) close > 21EMA
            if close <= ema21:
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers
    

    # -------------------------------------------------------------------------
    # 11) filter_stock_hope_from_bottom
    # -------------------------------------------------------------------------
    def filter_stock_hope_from_bottom(self, stock_datas_dic : dict, n_day_before = -1):
        """
        1) ADR >= 2
        2) RS rank < 2000
        3) Exclude Healthcare, Energy
        4) volume_ma50 >= 1,000,000 and close >= 5
        5) 150SMA > 200SMA
        6) near 150 or 200SMA (within 2 * ADR)
        """
        filtered_tickers = []
        Mtt_tickers = self.filter_stock_ALL(stock_datas_dic)
        atrs_ranking_df = self.sd.get_ATRS_Ranking_df()
        gisc_df = self.sd.get_GICS_df()

        for ticker in Mtt_tickers:
            stock_data = stock_datas_dic[ticker]

            if not has_required_data(stock_data, n_day_before, ['ADR','150MA','200MA']):
                continue

            try:
                ADR = stock_data['ADR'].iloc[n_day_before]
                volume_ma50 = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before]
                close = stock_data['Close'].iloc[n_day_before]
                ma150 = stock_data['150MA'].iloc[n_day_before]
                ma200 = stock_data['200MA'].iloc[n_day_before]
            except Exception:
                continue

            # 1) ADR >= 2
            if ADR < 2:
                continue

            # 2) RS rank < 2000
            bIsATRS_Ranking_Good = False
            try:
                rank = atrs_ranking_df.loc[ticker].iloc[n_day_before]
                bIsATRS_Ranking_Good = (rank < 2000)
            except Exception:
                continue

            if not bIsATRS_Ranking_Good:
                continue

            # 3) Exclude Healthcare, Energy
            try:
                sector = gisc_df.loc[ticker]['sector']
                if sector in ['Healthcare', 'Energy']:
                    continue
            except Exception:
                continue

            # 4) volume_ma50 >= 1,000,000 and close >= 5
            if volume_ma50 < 1_000_000 or close < 5:
                continue

            # 5) 150SMA > 200SMA
            if not (ma150 > ma200):
                continue

            # 6) near 150 or 200SMA (within 2 * ADR)
            dist_150 = abs(self.sd.get_percentage_AtoB(close, ma150))
            dist_200 = abs(self.sd.get_percentage_AtoB(close, ma200))
            near_150_or_200 = (dist_150 < ADR*2) or (dist_200 < ADR*2)
            if not near_150_or_200:
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    # -------------------------------------------------------------------------
    # 12) filter_stock_power_gap
    # -------------------------------------------------------------------------
    def filter_stock_power_gap(self, stock_datas_dic : dict, n_day_before = -1):
        """
        1) ADR(1 day ago) >= 2
        2) Gap up: open > previous day's high
        3) Price up >= 10% from previous close
           and also >= 2 * previous day's ADR
        4) 50-day average volume >= 200,000 and close >= 5
           or 50-day avg dollar volume > 5,000,000
        5) Healthcare sector excluded
        """

        filtered_tickers = []
        all_tickers = self.filter_stock_ALL(stock_datas_dic)
        gisc_df = self.sd.get_GICS_df()

        for ticker in all_tickers:
            stock_data = stock_datas_dic[ticker]

            if not has_required_data(stock_data, n_day_before - 1, ['ADR']):
                continue


            try:
                sector = gisc_df.loc[ticker]['sector']
                if sector == 'Healthcare':
                    continue

                ADR_1d_ago = stock_data['ADR'].iloc[n_day_before - 1]
                open_ = stock_data['Open'].iloc[n_day_before]
                high_1d_ago = stock_data['High'].iloc[n_day_before - 1]
                close_1d_ago = stock_data['Close'].iloc[n_day_before - 1]
                close_ = stock_data['Close'].iloc[n_day_before]
            except Exception:
                continue

            # 1) ADR(1 day ago) >= 2
            if ADR_1d_ago < 2:
                continue

            # 2) Gap up
            if not (open_ > high_1d_ago):
                continue

            # 3) Price up >= 10% from previous close
            change_pct = self.sd.get_percentage_AtoB(close_1d_ago, close_)
            if change_pct < 10:
                continue
            if change_pct < ADR_1d_ago * 2:
                continue

            try:
                volume_ma50_1d_ago = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before - 1]
                volume_ = stock_data['Volume'].iloc[n_day_before]
            except Exception:
                continue

            # 4) volume check
            # e.g. volume_ma50_1d_ago >= 200000 & close >= 5 OR dollar volume > 5,000,000
            is_volume_enough = False
            if (volume_ma50_1d_ago >= 200000 and close_ >= 5) or (volume_ma50_1d_ago * close_ > 5_000_000):
                is_volume_enough = True

            if not (volume_ > volume_ma50_1d_ago * 2 and is_volume_enough):
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    
    
    # -------------------------------------------------------------------------
    # 13) filter_stock_open_gap
    # -------------------------------------------------------------------------
    def filter_stock_open_gap(self, stock_datas_dic: dict, n_day_before=-1):
        """
        1) ADR(1 day ago) >= 2
        2) Gap up: open > previous day's high
        3) Price up >= 3% from previous close
        4) Volume check: 50-day avg volume >= 200,000 & close >= 5, or avg dollar vol > 5,000,000
        5) Exclude Healthcare
        """
        filtered_tickers = []
        all_tickers = self.filter_stock_ALL(stock_datas_dic)
        gisc_df = self.sd.get_GICS_df()

        for ticker in all_tickers:
            stock_data = stock_datas_dic[ticker]


            if not has_required_data(stock_data, n_day_before - 1, ['ADR']):
                continue

            try:
                sector = gisc_df.loc[ticker]['sector']
                if sector == 'Healthcare':
                    continue

                ADR_1d_ago = stock_data['ADR'].iloc[n_day_before - 1]
                open_ = stock_data['Open'].iloc[n_day_before]
                high_1d_ago = stock_data['High'].iloc[n_day_before - 1]
                close_1d_ago = stock_data['Close'].iloc[n_day_before - 1]
                close_ = stock_data['Close'].iloc[n_day_before]
            except Exception:
                continue

            # 1) ADR >= 2
            if ADR_1d_ago < 2:
                continue

            # 2) Gap up
            if not (open_ > high_1d_ago):
                continue

            # 3) Price up >= 3% from previous close
            open_change_pct = self.sd.get_percentage_AtoB(close_1d_ago, open_)
            if open_change_pct < 3:
                continue

            try:
                volume_ma50_1d_ago = stock_data['Volume'].rolling(window=50).mean().iloc[n_day_before - 1]
                volume_ = stock_data['Volume'].iloc[n_day_before]
            except Exception:
                continue

            # 4) volume check
            #  - 50-day avg volume >= 200,000 & close >= 5 or
            #    50-day avg dollar volume > 5,000,000
            is_volume_ok = False
            if (volume_ma50_1d_ago >= 200000 and close_ >= 5) or (volume_ma50_1d_ago * close_ > 5_000_000):
                is_volume_ok = True

            if not is_volume_ok:
                continue

            filtered_tickers.append(ticker)

        return filtered_tickers

    
    # -------------------------------------------------------------------------
    # Gap screening helpers
    # -------------------------------------------------------------------------
    def get_filter_gap_stocks_in_range(self, range_from : int , range_to : int, filter_stock_gap_func):
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
        stock_list = self.sd.get_local_stock_list()
        out_stock_datas_dic = self.sd.get_stock_datas_from_csv(stock_list, daysNum, False)
        out_tickers = list(out_stock_datas_dic.keys())

        print(f"start {str(filter_stock_gap_func.__name__)} screening!")
        date_tickers_dic = {}

        screening_log = []
        total_count = (range_from - range_to + 1)

        for i in range(range_to, range_from):
            stock_data_dic, tickers = self.screening_stocks_by_func(filter_stock_gap_func, True, False, -i)

            for ticker in tickers:
                # tradeday can be difference for each stocks. (Trading halt maybe??)
                tradeday = stock_data_dic[ticker].index[-i].date()

                # add ticker to the [tradeday-ticker] dictionary.
                if tradeday not in date_tickers_dic:
                    # assign empty list first if don't exist any tickers.
                    date_tickers_dic[tradeday] = []
                date_tickers_dic[tradeday].append(ticker)

                msg = f"[{tradeday}] {filter_stock_gap_func.__name__} ticker: {ticker}"
                screening_log.append(msg)

            progress_pct = (i - range_to) / total_count * 100
            print(f"{filter_stock_gap_func.__name__} process {progress_pct:.2f}% Done")

        print(f"Done. print {str(filter_stock_gap_func.__name__)} screen list")
        for s in screening_log:
            print(s)

        return date_tickers_dic
    

    def cook_gap_profiles(self, range_from : int , range_to : int, profile_period : int, all_stock_datas_dic : dict, gap_date_tickers_dic : dict):
        """
        Process power/open gap profiles and return a DataFrame
        """
        sd = self.sd
        gap_profile_dic = {}

        for gap_date, gap_tickers in gap_date_tickers_dic.items():
            for ticker in gap_tickers:
                stock_data : pd.DataFrame = all_stock_datas_dic[ticker]
                d0_index = sd.date_to_index(stock_data, gap_date)
                ticker_date = f"{ticker}_{gap_date}"

                # Calculate day indices
                d5_index = max(d0_index + 5 - 1, 0)
                d10_index = max(d0_index + 10 - 1, 0)
                d20_index = max(d0_index + 20 - 1, 0)
                d30_index = max(d0_index + 30 - 1, 0)
                d40_index = max(d0_index + 40 - 1, 0)
                d50_index = max(d0_index + 50 - 1, 0)

                if d20_index == 0:
                    print("Error!. To profile power gap, stock need time at least 20 days. Ticker_Date: ", ticker_date)
                    continue

                day_n_indices = [d5_index, d10_index, d20_index, d30_index, d40_index, d50_index]
                day_n_performances = []
                profile_end_day = d0_index + profile_period -1
            
                d0_open = stock_data['Open'].iloc[d0_index]
                d0_close = stock_data['Close'].iloc[d0_index]
                d0_low = stock_data['Low'].iloc[d0_index]
                d0_high = stock_data['High'].iloc[d0_index]
                d0_ma200 = stock_data['200MA'].iloc[d0_index]

                close_1d_ago = stock_data['Close'].iloc[d0_index - 1]
                ADR_1d_ago = stock_data['ADR'].iloc[d0_index - 1]
                volume_ma50_1d_ago = stock_data['Volume'].rolling(window=50).mean().iloc[d0_index-1]
                d0_volume = stock_data['Volume'].iloc[d0_index]

                
                # DR% (Daily Range)
                daily_range_percentages = stock_data['High'] / stock_data['Low']

                # ADR(%) 20 day later since gap
                # 입수합병 필터링
                # 갭당일부터 성과측정일까지 ADR이 1% 미만으로 줄어들면 인수합병으로 본다.
                # 애초에 ADR 2 이상의 주식이 Power gap 이후 ADR이 1로 줄어들었다면 뭔가 잘못된 것이다. 굼뱅이 주식은 필요 없다!

                n = 20
                ADRs_since_gap = daily_range_percentages.rolling(n).mean()
                ADRs_since_gap = 100 * (ADRs_since_gap - 1)
                adr_since_gap = ADRs_since_gap.iloc[d20_index]
                if adr_since_gap < 1:
                    print("It's probabily M&A. reject this ticker from the profiles, ticker: ",  ticker_date)
                    continue

                # [day_n_performances] N day 이후 성과[
                for day_n_index in day_n_indices:
                    if day_n_index != 0:
                        day_n_close = stock_data['Close'].iloc[day_n_index]
                        day_n_performance = sd.get_percentage_AtoB(d0_close, day_n_close)
                        day_n_performances.append(day_n_performance)
                    else:
                        day_n_performances.append(0)

                # [d0_close] 갭 종가($)
                d0_close = d0_close

                # [d0_open_change] 갭 상승폭 Open(%)
                d0_open_change = sd.get_percentage_AtoB(close_1d_ago, d0_open)

                # [d0_close_change] 종가 상승폭 (%)
                d0_close_change = sd.get_percentage_AtoB(close_1d_ago, d0_close)

                # [d0_low_change_from_open] 시가로부터 저가까지 하락폭(%)
                d0_low_change_from_open = sd.get_percentage_AtoB(d0_open, d0_low)

                # [d0_close_change_from_open] 시가로부터 종가까지 (%)
                d0_close_change_from_open = sd.get_percentage_AtoB(d0_open, d0_close)

                # [d0_daily_range] Daily Range(%)
                d0_daily_range = sd.get_percentage_AtoB(d0_low, d0_high)

                # [d0_performance_vs_ADR] (nADR%)
                d0_performance_vs_ADR = d0_close_change / ADR_1d_ago

                
                # [DCR](%)
                if d0_high - d0_low > 0:
                    DCR = (d0_close - d0_low) / (d0_high - d0_low)
                else:
                    DCR = 0

                # [d0_volume] 거래량
                d0_volume = d0_volume

                # [d0_volume_vs_50Avg] 거래량(50일 평균 대비)
                d0_volume_vs_50Avg = d0_volume / volume_ma50_1d_ago

                d0_dollar_volume = d0_volume * d0_close

                # [bOEL]
                bOEL = d0_open == d0_low

                # [Above 200sma]
                bAbove200ma = d0_close > d0_ma200
                
                # ------------ d0 이후에 알 수 있는 것들 ------------

                # [first_ma_touch_day]
                first_ma_touch_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_ma_touch(stock_data, 10, True, d0_index + i):
                        first_ma_touch_day = i
                        break


                # [d0_open_violation_day]
                d0_open_violation_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stock_data, d0_open, d0_index + i):
                        d0_open_violation_day = i
                        break

                # [d0_low_violation_day]
                d0_low_violation_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stock_data, d0_low, d0_index + i):
                        d0_low_violation_day = i
                        break

                # [HVC_violation_first_day]
                HVC_violation_first_day = profile_period
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stock_data, d0_close, d0_index + i):
                        HVC_violation_first_day = i
                        break


                # [HVC_violation_last_day] (저가가 HVC를 마지막으로 침범한 날)
                # [HVC_violation_cnt] (HVC 아래에 주가가 위치했던 날의 수)
                HVC_violation_last_day = profile_period
                HVC_violation_cnt = 0
                for i in range(1, profile_period):
                    if sd.check_undercut_price(stock_data, d0_close, d0_index + i):
                        HVC_violation_last_day = i
                        HVC_violation_cnt = HVC_violation_cnt + 1
                        


                # [alpha_window_lowest_day]
                # Alpha Window: day1 ~ day5 of Power gap. day0 is gap day.)

                # start_pos는 시작 위치입니다. 예를 들어, 0은 데이터프레임의 첫 번째 행입니다.
                start_pos = d0_index + 1
                # 5일간의 가장 낮은 가격의 위치를 구합니다.
                lowest_price_pos = stock_data['Low'].iloc[start_pos:start_pos + 5].idxmin()
                # 실제 위치를 얻기 위해 데이터프레임의 인덱스와 비교합니다.
                lowest_index_pos = stock_data.index.get_loc(lowest_price_pos)
                # 계속해서 사용하는 인덱스는 마이너스(-) 인덱스임.
                lowest_index_pos = lowest_index_pos - len(stock_data)
                alpha_window_lowest_day = lowest_index_pos - d0_index

                # [alpha_window_lowest_pct_from_HVC]
                alpha_window_lowest_price = stock_data['Low'].iloc[lowest_index_pos]
                alpha_window_lowest_pct_from_HVC = sd.get_percentage_AtoB(d0_close, alpha_window_lowest_price)

                # [alpha_window_highest_day]
                start_pos = d0_index + 1
                highest_price_pos = stock_data['High'].iloc[start_pos:start_pos + 5].idxmax()
                highest_index_pos = stock_data.index.get_loc(highest_price_pos)
                highest_index_pos = highest_index_pos - len(stock_data)
                alpha_window_highest_day = highest_index_pos - d0_index

                # [alpha_window_highest_pct_from_HVC]
                alpha_window_highest_price = stock_data['High'].iloc[highest_index_pos]
                alpha_window_highest_pct_from_HVC = sd.get_percentage_AtoB(d0_close, alpha_window_highest_price)

                # [HVC_recovery_day_from_alpha_window_lowest]
                # HVC violation이 발생할때만 유효한 프로퍼티
                # d1 ~ d5 영역에서 HVC 를 침범한 저가가 며칠만에 회복되었는지? (회복: 종가가 HVC 위에서 다시 마감)
                # alpha_window_lowest_day로부터 카운팅하며 종가가 HVC 위에 있는지 확인 해야 한다.

                HVC_recovery_day_from_alpha_window_lowest = profile_period
                for i in range(lowest_index_pos, profile_end_day + 1):
                    cnt_from_lowest_day = i - lowest_index_pos
                    c = stock_data['Close'].iloc[i]
                    if c > d0_close:
                        HVC_recovery_day_from_alpha_window_lowest = cnt_from_lowest_day
                        break

                gap_profile_dic[ticker_date] = [ticker,
                                           # d5, d10, d20, d30, d40, d50 퍼포먼스
                                           gap_date, day_n_performances[0], day_n_performances[1], day_n_performances[2], day_n_performances[3], day_n_performances[4], day_n_performances[5],
                                            d0_close, d0_open_change, d0_close_change, d0_low_change_from_open, d0_close_change_from_open,
                                            d0_daily_range, d0_performance_vs_ADR, DCR, d0_volume, d0_volume_vs_50Avg, d0_dollar_volume, bOEL, bAbove200ma,
                                            first_ma_touch_day, d0_open_violation_day, d0_low_violation_day, HVC_violation_first_day, HVC_violation_last_day, HVC_violation_cnt,
                                            alpha_window_lowest_day, alpha_window_lowest_pct_from_HVC, alpha_window_highest_day, alpha_window_highest_pct_from_HVC,
                                              HVC_recovery_day_from_alpha_window_lowest]


        df = pd.DataFrame.from_dict(gap_profile_dic).transpose()
        columns = ['Symbol', 'gap_date', 'd5_performance', 'd10_performance', 'd20_performance', 'd30_performance', 'd40_performance', 'd50_performance',
                    'd0_close', 'd0_open_change', 'd0_close_change', 'd0_low_change_from_open', 'd0_close_change_from_open',
                    'd0_daily_range', 'd0_performance_vs_ADR', 'DCR', 'd0_volume', 'd0_volume_vs_50Avg', 'd0_dollar_volume', 'bOEL', 'bAbove200ma',
                    'first_ma_touch_day', 'd0_open_violation_day', 'd0_low_violation_day', 'HVC_violation_first_day', 'HVC_violation_last_day', 'HVC_violation_cnt',
                    'alpha_window_lowest_day', 'alpha_window_lowest_pct_from_HVC', 'alpha_window_highest_day', 'alpha_window_highest_pct_from_HVC',
                        'HVC_recovery_day_from_alpha_window_lowest']
        df.columns = columns
        df.index.name = 'Symbol_Date'

        # fix object types to numeric.
        # 자료형이 통합되지 않은 리스트를 value로 갖는 딕셔너리를 DataFrame으로 변환하는 과정에서 모든 column이 object로 변환되는 문제가 있음.
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        df = df.round(2)

        return df

    
    def cook_power_gap_profiles(self, range_from : int , range_to : int, profile_period : int):
        """
        - range_from :          Screening will be started from this 'param trading day ago' ex) 40 mean that searching process start from 40 trading day ago.
        - range_to :            Screening will be ended at this 'param trading day ago' ex) 10 mean that searching process will be ended at the 10 trading day ago.
        - profile_period        profile check period for C/V check, ma touch, highest price etc ... 

        range_from must be bigger than range_to

        """

        print("cook_power_gap_profiles!!")

        sd = self.sd
        gap_date_tickers_dic = {}
        bUseGapDataCache = True

        if bUseGapDataCache:
            try: 
                with open('cache_power_gap_date_tickers_dic', "rb") as f:
                    gap_date_tickers_dic = pickle.load(f)
            except Exception as e:
                print('[Cache] no cache_gap_date_tickers_dic in local')

        if not gap_date_tickers_dic:
            gap_date_tickers_dic = self.get_filter_gap_stocks_in_range(range_from, range_to, self.filter_stock_power_gap)
            try:
                with open('cache_power_gap_date_tickers_dic', "wb") as f:
                    pickle.dump(gap_date_tickers_dic, f)
            except Exception as e:
                print(e)
    
        years = float(range_from) / 240.0
        daysNum = int(365) + int(years * 365.0)
        # use cache made by upper scan code
        all_stock_datas_dic = sd.get_stock_datas_from_csv(sd.get_local_stock_list(), daysNum, True)

        df = self.cook_gap_profiles(range_from, range_to, profile_period, all_stock_datas_dic, gap_date_tickers_dic)

        try:
            save_path = os.path.join(PROFILES_FOLDER, f'power_gap_{range_from}_{range_to}_{profile_period}.xlsx')
            df.to_excel(save_path, index_label='Symbol_Date')
            print(f"{save_path}", "is saved!")
        except Exception as e:
            print(f"An error occurred: {e}")

    
    def cook_open_gap_profiles(self, range_from : int , range_to : int, profile_period : int):
        """
        - range_from :          Screening will be started from this 'param trading day ago' ex) 40 mean that searching process start from 40 trading day ago.
        - range_to :            Screening will be ended at this 'param trading day ago' ex) 10 mean that searching process will be ended at the 10 trading day ago.
        - profile_period        profile check period for C/V check, ma touch, highest price etc ... 

        range_from must be bigger than range_to

        """

        print("cook_open_gap_profiles!!")

        sd = self.sd
        gap_date_tickers_dic = {}

        bUseGapDataCache = True
        if bUseGapDataCache:
            try: 
                with open('cache_open_gap_date_tickers_dic', "rb") as f:
                    gap_date_tickers_dic = pickle.load(f)
            except Exception as e:
                print('[Cache] no cache_open_gap_date_tickers_dic in local')

        if not gap_date_tickers_dic:
            gap_date_tickers_dic = self.get_filter_gap_stocks_in_range(range_from, range_to, self.filter_stock_open_gap)
            try:
                with open('cache_open_gap_date_tickers_dic', "wb") as f:
                    pickle.dump(gap_date_tickers_dic, f)
            except Exception as e:
                print(e)
    
        years = float(range_from) / 240.0
        daysNum = int(365) + int(years * 365.0)
        # use cache made by upper scan code
        all_stock_datas_dic = sd.get_stock_datas_from_csv(sd.get_local_stock_list(), daysNum, True)

        df = self.cook_gap_profiles(range_from, range_to, profile_period, all_stock_datas_dic, gap_date_tickers_dic)

        try:
            save_path = os.path.join(PROFILES_FOLDER, f'open_gap_{range_from}_{range_to}_{profile_period}.xlsx')
            df.to_excel(save_path, index_label='Symbol_Date')

            print(f"{save_path}", "is saved!")
        except Exception as e:
            print(f"An error occurred: {e}")
