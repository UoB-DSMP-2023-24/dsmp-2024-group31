import pandas as pd



file_path='../RawData/Salary.csv'
db_salary= pd.read_csv(file_path)
print(db_salary.info())