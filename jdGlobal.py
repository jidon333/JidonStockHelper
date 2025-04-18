
import os

DATA_FOLDER = os.path.join(os.getcwd(), 'StockData')
METADATA_FOLDER = os.path.join(DATA_FOLDER, 'MetaData')
UI_FOLDER = os.path.join(os.getcwd(), 'UI')
SCREENSHOT_FOLDER = os.path.join(METADATA_FOLDER, 'ScreenShot')
FILTERED_STOCKS_FOLDER = os.path.join(METADATA_FOLDER, "FilteredStocks")
PROFILES_FOLDER = os.path.join(METADATA_FOLDER, "Profiles")

# ----------------------------
# 전역 리스트/딕셔너리
# ----------------------------
exception_ticker_list = {}
sync_fail_ticker_list = []

def get_yes_no_input(qustionString):
    while(True):
        print(qustionString)
        k = input()
        if(k == 'y' or k == 'Y'):
            return True
        elif(k == 'n' or k == 'N'):
            return False
        else:
            print('input \'y\' or \'n\' to continue...')
