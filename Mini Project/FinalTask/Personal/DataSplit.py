import os
import pandas as pd

# Load the dataset
file_path = '../RawData/simulated_transaction_2024_Processed(without salary).csv'
df = pd.read_csv(file_path)


# Create a directory for the account CSV files if it doesn't already exist
output_directory = '../SplitedData/Personal/'
os.makedirs(output_directory, exist_ok=True)

# Group the data by 'Account No' and write separate CSV files
for account_no, group in df.groupby('Account No'):
    output_file_path = f'{output_directory}{account_no}.csv'
    group.to_csv(output_file_path, index=False)

# Verify the operation by listing the first few files created
output_files = os.listdir(output_directory)
output_files_paths = [os.path.join(output_directory, file) for file in output_files]

# Display the paths of the first few files to confirm
output_files_paths[:5]