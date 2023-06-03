
import os

data_folder = os.path.join(os.getcwd(), 'StockData')
metadata_folder = os.path.join(data_folder, 'MetaData')


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

            