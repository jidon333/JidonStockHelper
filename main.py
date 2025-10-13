
"""
main.py

This script provides a CLI menu to execute various functions such as charting, data synchronization,
filtering, and indicator generation. Each menu option is mapped to a separate function with input
validation and documentation.
"""

import datetime as dt
import glob
import json
import os
import sys
import time
import pickle


from jdStockDataManager import JdStockDataManager 
from jdChart import JdChart
from jdGlobal import get_yes_no_input
from jdGlobal import DATA_FOLDER
from jdGlobal import METADATA_FOLDER

from qtWindow import JdWindowClass

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pandas as pd
import jdStockFilteringManager



import logging
import logging_conf  # 전역 로깅 설정 로드
import datetime



# Global variables (initialized in main)
sd = None
sf = None


def display_menu() -> int:
    """
    Displays the CLI menu and returns the user's choice as an integer
    
    :return: An integer representing the chosen menu option
    """
    menu_text = (
        "\n\n ===================Select the chart type: ===================\n"
        "1: Stock Data Chart\n"
        "2: Momentum Index Chart\n"
        "3: Sync local CSV data from the web and generate metadata\n"
        "4: Generate up-down data using local CSV files\n"
        "5: Process local stock data\n"
        "6: Download stock data from the web and overwrite local files\n"
        "7: Generate ATRS Ranking\n"
        "8: Generate Industry Ranking\n"
        "9: Generate screening result as XLSX file\n"
        "10: MTT Index Chart\n"
        "11: FA50 Index Chart\n"
        "12: Generate all indicators and screening results\n"
        "13: Power gap history screen\n"
        "14: ATR Expansion Chart\n"
        "15: Investigate ticker drop days (single ticker)\n"
        "\nEnter your choice: "
    )
    while True:
        try:
            choice = int(input(menu_text))
            if 1 <= choice <= 15:
                return choice
            else:
                print("Please enter a number between 1 and 15.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def draw_stock_datas(stock_datas_dic, selected_tickers, inStockManager : JdStockDataManager):
    """
    Displays a stock chart using the provided stock data.
    
    :param stock_datas_dic: Dictionary containing stock data for each ticker.
    :param selected_tickers: List of selected tickers.
    :param inStockManager: Instance of JdStockDataManager.
    """

    # Create the QApplication and initialize the main window.
    app = QApplication(sys.argv) 
    myWindow = JdWindowClass() 
    chart = JdChart(inStockManager)
    chart.init_plots_for_stock(stock_datas_dic, selected_tickers)
    myWindow.set_chart_class(chart)
    myWindow.show()
    app.exec_()

def draw_momentum_index(updown_nyse, updown_nasdaq, updown_sp500, bOnlyForScreenShot = False):
    """
    Draws the momentum index chart.
    
    :param updown_nyse: DataFrame for NYSE up-down data.
    :param updown_nasdaq: DataFrame for NASDAQ up-down data.
    :param updown_sp500: DataFrame for S&P500 up-down data.
    :param bOnlyForScreenShot: If True, the chart is saved for screenshot purposes.
    """
    chart = JdChart(sd)
    chart.bOnlyForScreenShot = bOnlyForScreenShot
    chart.init_plots_for_up_down(updown_nyse, updown_nasdaq, updown_sp500)
    chart.draw_updown_chart()


def draw_atr_expansion_chart(atr_changes_df : pd.DataFrame , bOnlyForScreenShot = False):
    """
    Draws the ATR Expansion chart.
    
    :param atr_changes_df: DataFrame containing ATR expansion data.
    :param bOnlyForScreenShot: If True, the chart is saved for screenshot purposes.
    """
    chart = JdChart(sd)
    chart.bOnlyForScreenShot = bOnlyForScreenShot
    chart.init_plots_for_atr_up_down(atr_changes_df)
    chart.draw_updown_ATR_chart()

def draw_count_data_chart(mtt_cnt_df, name : str, chart_type : str, bOnlyForScreenShot = False):
    """
    Draws a chart for count data.
    
    :param mtt_cnt_df: DataFrame containing count data.
    :param name: Chart name (e.g., "MTT" or "FA50").
    :param chart_type: Type of chart ("line" or "bar").
    :param bOnlyForScreenShot: If True, the chart is saved for screenshot purposes.
    """
    chart = JdChart(sd)
    chart.bOnlyForScreenShot = bOnlyForScreenShot
    chart.init_plots_for_count_data(mtt_cnt_df, chart_type)
    chart.draw_count_data_chart(name, chart_type)

def remove_outdated_tickers():
    """
    Removes local CSV files for tickers listed in DataReader_exception.json.
    """
    with open("DataReader_exception.json", "r") as outfile:
        data = json.load(outfile)
        for key in data.keys():
            file_path = os.path.join(DATA_FOLDER, key + '.csv')
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"{file_path} is removed!")


def remove_local_caches():
    """
    Removes all local cache files that start with 'cache_' and resets the stock manager's caches.
    """
    local_dir = METADATA_FOLDER
    for filename in os.listdir(local_dir):
        if filename.startswith('cache_'):
            os.remove(os.path.join(local_dir, filename))
    sd.reset_caches()




def screen_stocks_and_show_chart(filter_function, bUseLocalLoadedStockDataForScreening, bSortByRS):
    """
    Filters stocks using the provided filter function and displays the corresponding chart.
    
    :param filter_function: Function used to filter stock data.
    :param bUseLocalLoadedStockDataForScreening: If True, uses locally cached stock data for screening.
    :param bSortByRS: If True, sorts the resulting tickers by RS.
    """
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

    if not bUseLocalCache:
        with open('cache_tickers', "wb") as f:
            pickle.dump(tickers, f)

        with open('cache_stock_datas_dic', "wb") as f:
            pickle.dump(stock_data, f)

    print(tickers)
    print("filtered stock count: " ,len(tickers))

    if len(tickers) > 0:
        draw_stock_datas(stock_data, tickers, sd)
    else:

        print("there's no tickers to draw!")


def run_stock_data_chart():
    """Option 1: Stock Data Chart"""
    sf.MTT_ADR_minimum = 2.5
    sf.LastDayMinimumVolume = 1000000
    screen_stocks_and_show_chart(sf.filter_stocks_MTT, True, True)
    #screen_stocks_and_show_chart(sf.filter_stock_power_gap, True, True)
    #screen_stocks_and_show_chart(sf.filter_stocks_young, True, True)
    

    #screen_stocks_and_show_chart(sf.filter_stocks_high_ADR_swing, True, True)
    #screen_stocks_and_show_chart(sf.filter_stocks_young, True, True)
    #screen_stocks_and_show_chart(sf.filter_stocks_Bull_Snort, True, True)
    #screen_stocks_and_show_chart(sf.filter_stocks_rs_8_10, True, True)
    #screen_stocks_and_show_chart(sf.filter_stock_hope_from_bottom, True, True)
    #screen_stocks_and_show_chart(sf.filter_stock_ALL, True, False)
    #screen_stocks_and_show_chart(sf.filter_stock_Good_RS, True, True)3
    #screen_stocks_and_show_chart(sf.filter_stocks_high_ADR_swing, True, True)
    #screen_stocks_and_show_chart(sf.filter_stock_power_gap, True, True)

def run_momentum_index_chart():
    """Option 2: Momentum Index Chart"""
    updown_nyse, updown_nasdaq, updown_sp500 = sd.getUpDownDataFromCsv(365 * 2)
    draw_momentum_index(updown_nyse, updown_nasdaq, updown_sp500)


def run_sync_csv_and_generate_metadata():
    """3
    Option 3: Sync local CSV data from the web and generate metadata 
              (e.g., up_down, RS, industry, MTT count, etc.)
    """
    remove_local_caches()
    sd.sync_csv_from_web(10)
    sd.cookUpDownDatas()
    sd.cook_ATR_Expansion_Counts()
    sd.cook_Nday_ATRS150_exp(365 * 2)
    sd.cook_ATRS150_exp_Ranks(365 * 2)
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_filter_count_data(sf.filter_stocks_MTT, "MTT_Counts", 10, True)
    sd.cook_filter_count_data(sf.filter_stock_FA50, "FA50_Counts", 10, True)
    sd.cook_top10_in_industries()



def run_cook_updown_datas():
    """Option 4: Generate up-down data using local CSV files"""
    sd.cookUpDownDatas()
    sd.cook_ATR_Expansion_Counts()


def run_cook_local_stock_data():
    """Option 5: Process local stock data"""
    remove_local_caches()
    sd.cookLocalStockData()


def run_download_stock_data():
    """
    Option 6: Download stock data from the web and overwrite local files.
    (This may take a long time.)
    """
    remove_local_caches()
    # [TODO] 10년치 데이터가 실제로 확보되지 않는 이슈. 아래 인자로 실행시 2020년부터 데이터가 존재함. 확인 필요...!
    sd.download_stock_datas_from_web(365 * 10, False)
    remove_outdated_tickers()
    sd.remove_acquisition_tickers()
    sd.cook_Stock_GICS_df()
    sd.cook_Nday_ATRS150_exp(365 * 2)
    sd.cook_ATRS150_exp_Ranks(365 * 2)
    sd.cook_top10_in_industries()
    sd.cook_filter_count_data(sf.filter_stocks_MTT, "MTT_Counts", 365 * 3, False)
    sd.cook_filter_count_data(sf.filter_stock_FA50, "FA50_Counts", 365 * 3, False)


def run_atrs_ranking():
    """Option 7: Generate ATRS Ranking"""
    sd.cook_Nday_ATRS150_exp(365 * 2)
    sd.cook_ATRS150_exp_Ranks(365 * 2)


def run_industry_ranking():
    """Option 8: Generate Industry Ranking"""
    sd.cook_short_term_industry_rank_scores()
    sd.cook_long_term_industry_rank_scores()
    sd.cook_top10_in_industries()


def run_screening_to_xlsx():
    """Option 9: Generate screening result as an XLSX file"""
    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stock_Custom, True, True)
    first_stock_data: pd.DataFrame = stock_data[tickers[0]]
    lastday = str(first_stock_data.index[-1].date())
    sd.cook_stock_info_from_tickers(tickers, f'MTT_Leaders_{lastday}')


def run_mtt_index_chart():
    """Option 10: MTT Index Chart"""
    df = sd.get_count_data_from_csv("MTT")
    draw_count_data_chart(df, "MTT", "line")


def run_fa50_index_chart():
    """Option 11: FA50 Index Chart"""
    df = sd.get_count_data_from_csv("FA50", 365 * 3)
    draw_count_data_chart(df, "FA50", "bar")


def run_generate_all_indicators():
    """Option 12: Generate all indicators and screening results"""
    # MI Index chart
    updown_nyse, updown_nasdaq, updown_sp500 = sd.getUpDownDataFromCsv(365 * 3)
    draw_momentum_index(updown_nyse, updown_nasdaq, updown_sp500, True)

    # MTT Count chart
    df = sd.get_count_data_from_csv("MTT")
    draw_count_data_chart(df, "MTT", "line", True)

    # FA50 Count chart
    df = sd.get_count_data_from_csv("FA50")
    draw_count_data_chart(df, "FA50", "bar", True)

    # ATR Expansion chart
    df = sd.get_count_data_from_csv("ATR_Expansion", 365 * 1)
    draw_atr_expansion_chart(df, True)

    # Generate various screening XLSX outputs
    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stocks_MTT, True, True)
    first_stock_data: pd.DataFrame = stock_data[tickers[0]]
    lastday = str(first_stock_data.index[-1].date())

    if tickers:
        sd.cook_stock_info_from_tickers(tickers, f'US_MTT_{lastday}')

    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stocks_high_ADR_swing, True, True)
    if tickers:
        sd.cook_stock_info_from_tickers(tickers, f'US_HighAdrSwing_{lastday}')

    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stocks_rs_8_10, True, True)
    if tickers:
        sd.cook_stock_info_from_tickers(tickers, f'US_RS_8_10_{lastday}')

    stock_data, tickers = sf.screening_stocks_by_func(sf.filter_stock_hope_from_bottom, True, True)
    if tickers:
        sd.cook_stock_info_from_tickers(tickers, f'US_hope_from_bottom_{lastday}')

    # Print various screening results
    for func, label in [
        (sf.filter_stock_power_gap, "power gap"),
        (sf.filter_stocks_Bull_Snort, "bull snort"),
        (sf.filter_stocks_rs_8_10, "RS 8/10"),
        (sf.filter_stocks_young, "Young"),
        (sf.filter_stock_hope_from_bottom, "Hope from bottom"),
        (sf.filter_stocks_high_ADR_swing, "High ADR Swing")
    ]:
        stock_data_dic, tickers = sf.screening_stocks_by_func(func, True, True, -1)
        print(f"[{lastday}] {label} tickers: {tickers}")

def run_power_gap_history_screen():
    """Option 13: Power gap history screen"""
    sf.cook_power_gap_profiles(20 * 12 * 5, 20, 20)
    sf.cook_open_gap_profiles(20 * 12 * 5, 20, 20)
    sf.get_filter_gap_stocks_in_range(20, 0, sf.filter_stock_power_gap)


def run_atr_expansion_chart_option():
    """Option 14: ATR Expansion Chart"""
    sd.cook_ATR_Expansion_Counts()
    df = sd.get_count_data_from_csv("ATR_Expansion", 365 * 1)
    draw_atr_expansion_chart(df, True)


def run_investigate_ticker_drop():
    """Option 15: Investigate single ticker drop days (handles ETF like QQQ/SPY)."""
    ticker = input("Ticker (default: QQQ): ").strip() or "QQQ"
    try:
        threshold = float(input("하락 임계값 % (default: -3): ").strip() or "-3")
    except ValueError:
        threshold = -3.0

    # 로컬 CSV가 없으면 FinanceDataReader에서 직접 가져옵니다.
    res = sd.find_days_by_return(ticker, threshold, 365*5)
    if res is None or res.empty:
        print(f"{ticker}: 조건을 만족하는 날짜가 없습니다.")
        return

    print(f"{ticker} 조건 일치 {len(res)}건:")
    try:
        print(res[["Close","change_pct"]].tail(20))
    except Exception:
        print(res.tail(20))


def print_drop_tail(df, title):
    """Prints last 10 rows of drop result with key columns; no-op for empty input."""
    if df is not None and not df.empty:
        print(title)
        try:
            temp = df.tail(20).copy()
            # Prefer precomputed ATR_Drop; fall back to TC if present
            if 'ATR_Drop' in temp.columns:
                temp['ATR_Drop'] = temp['ATR_Drop'].round(2)
            cols = ["Open","High","Low","Close","EMA10","EMA21","ATR","change_pct"]
            if 'ATR_Drop' in temp.columns:
                cols.append("ATR_Drop")
            print(temp[cols])
        except Exception:
            print(df.tail(10))
        print("")


def run_investigate_ticker_drop_ext():
    """Investigate drop days with EMA rejection using percent and ATR thresholds."""
    ticker = input("Ticker (default: QQQ): ").strip() or "QQQ"
    try:
        pct_threshold = float(input("Drop threshold % (default: -3): ").strip() or "-3")
    except ValueError:
        pct_threshold = -3.0
    try:
        atr_multiple = float(input("ATR multiple (default: 2): ").strip() or "2")
    except ValueError:
        atr_multiple = 2.0

    res = sd.find_days_drop_and_ema_violation(
        ticker=ticker,
        pct_threshold=pct_threshold,
        atr_multiple=atr_multiple,
        days_num=365*10,
    )
    drop_pct = res.get('drop_pct')
    drop_atr = res.get('drop_atr')
    both = res.get('intersection')
    print(f"\n[{ticker}] EMA rejection days summary (last 10y):")
    print(f" - Percent drop <= {pct_threshold}%: {0 if drop_pct is None or drop_pct.empty else len(drop_pct)} days")
    print(f" - ATR drop <= -{atr_multiple} ATR: {0 if drop_atr is None or drop_atr.empty else len(drop_atr)} days")
    print(f" - Intersection (both conditions): {0 if both is None or both.empty else len(both)} days\n")
    print_drop_tail(drop_pct,  f"Recent percent-drop days (<= {pct_threshold}%):")
    print_drop_tail(drop_atr,  f"Recent ATR-drop days (<= -{atr_multiple} ATR):")
    print_drop_tail(both,      "Recent intersection days (both conditions):")


def main():
    """
    Main function:
      - Initializes the JdStockDataManager and JdStockFilteringManager,
      - Executes the corresponding function based on the user's menu selection.
    """

    logging.debug("main() -- DEBUG")
    logging.info("main() -- INFO ")
    logging.warning("main()-- WARNING")
    logging.error("main() -- ERROR")
    logging.critical("main() -- CRITICAL")


    global sd, sf
    sd = JdStockDataManager()
    sf = jdStockFilteringManager.JdStockFilteringManager(sd)

    choice = display_menu()
    
    if choice == 1:
        run_stock_data_chart()
    elif choice == 2:
        run_momentum_index_chart()
    elif choice == 3:
        run_sync_csv_and_generate_metadata()
    elif choice == 4:
        run_cook_updown_datas()
    elif choice == 5:
        run_cook_local_stock_data()
    elif choice == 6:
        run_download_stock_data()
    elif choice == 7:
        run_atrs_ranking()
    elif choice == 8:
        run_industry_ranking()
    elif choice == 9:
        run_screening_to_xlsx()
    elif choice == 10:
        run_mtt_index_chart()
    elif choice == 11:
        run_fa50_index_chart()
    elif choice == 12:
        run_generate_all_indicators()
    elif choice == 13:
        run_power_gap_history_screen()
    elif choice == 14:
        run_atr_expansion_chart_option()
    elif choice == 15:
        run_investigate_ticker_drop_ext()


# --------------------------------------------------------------------

if __name__ == "__main__":
    main()


