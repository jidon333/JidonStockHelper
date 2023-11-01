
import os

data_folder = os.path.join(os.getcwd(), 'StockData')
metadata_folder = os.path.join(data_folder, 'MetaData')
ui_folder = os.path.join(os.getcwd(), 'UI')
screenshot_folder = os.path.join(metadata_folder, 'ScreenShot')


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

            