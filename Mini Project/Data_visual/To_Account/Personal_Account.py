import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
class Personal_Account:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)
        self.setup_pandas_options()

    @staticmethod
    def setup_pandas_options():
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)



    def data_split_by_regular(self):





def main():
    xlsx_files = glob.glob('Personal/*.xlsx')

    for file_name in xlsx_files:
        if file_name.endswith('.xlsx'):
            Personal_Account.data_split_by_regular()



if __name__ == '__main__':
    main()
