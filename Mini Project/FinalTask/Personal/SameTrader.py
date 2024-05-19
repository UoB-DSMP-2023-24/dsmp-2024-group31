import os
import pandas as pd

# Load the dataset
file_path = '../RawData/simulated_transaction_2024_Processed(without salary).csv'
df = pd.read_csv(file_path)

grouped =df.groupby(['Account No','Third Party Account No'])

same_account_Third = grouped['Amount'].count()
print(same_account_Third.sort_values())
