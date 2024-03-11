import pandas as pd

csv_file_path = 'fake_transactional_data_24.csv'

# Number of rows to read
num_rows = 1000000

# Read the first 10000 rows of the CSV file
df = pd.read_csv(csv_file_path, nrows=num_rows)

# Path for the new Excel file
xlsx_file_path = f'fake_transactional_data_{num_rows}.xlsx'  # Replace or modify this path as needed

# Save the DataFrame to an Excel file
df.to_excel(xlsx_file_path, index=False)

print("Excel file has been created.")
