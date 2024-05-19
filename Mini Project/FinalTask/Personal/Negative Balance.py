import os
import pandas as pd

# Load the dataset
file_path = '../RawData/simulated_transaction_2024_Processed(without salary).csv'
df = pd.read_csv(file_path)

# The data is already filtered for the year 2023 as per the given dataset.
# We will now calculate the duration of negative balance for each account.

# Parse the Date and Timestamp into a single datetime column for accurate duration calculation
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'])

# Initialize a dictionary to store the balance status for each account
account_negative_balance_duration = {}

# Sort the dataframe by account number and datetime to process in order
df_sorted = df.sort_values(by=['Account No', 'Datetime'])

# Go through each account and calculate the duration of negative balance
for account_no in df_sorted['Account No'].unique():
    account_transactions = df_sorted[df_sorted['Account No'] == account_no]
    negative_balance_start_time = None
    total_negative_duration = pd.Timedelta(0)

    for i, transaction in account_transactions.iterrows():
        # Check if balance goes negative and no start time is set
        if transaction['Balance'] < 0 and negative_balance_start_time is None:
            negative_balance_start_time = transaction['Datetime']
        # If balance is positive and start time is set, calculate the duration and reset
        elif transaction['Balance'] >= 0 and negative_balance_start_time:
            negative_duration = transaction['Datetime'] - negative_balance_start_time
            total_negative_duration += negative_duration
            negative_balance_start_time = None

    # If the last balance is negative, we consider up to the end of 2023
    if negative_balance_start_time is not None:
        end_of_year = pd.Timestamp('2023-12-31 23:59:59')
        negative_duration = end_of_year - negative_balance_start_time
        total_negative_duration += negative_duration

    # Store the total negative duration for the account
    account_negative_balance_duration[account_no] = total_negative_duration

# The total time in 2023
total_time_in_2023 = pd.Timestamp('2023-12-31 23:59:59') - pd.Timestamp('2023-01-01 00:00:00')

# Calculate the percentage of time each account had a negative balance
negative_balance_percentage = {account: (duration.total_seconds() / total_time_in_2023.total_seconds()) * 100
                               for account, duration in account_negative_balance_duration.items()}

negative_balance_percentage_df = pd.DataFrame(list(negative_balance_percentage.items()), columns=['Account_No', 'Negative_Balance_Percentage'])


negative_balance_percentage_df.to_csv(('negative_balance.csv'), index=False)


print(negative_balance_percentage_df)

