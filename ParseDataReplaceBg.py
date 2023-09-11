from utils.PropertyNames import ColumnNames as Cols

import pandas as pd


def add_date_to_time_column(patient_mask, time_column_name, starting_date):
    # Convert the starting date to a datetime object
    current_date = pd.to_datetime(starting_date)

    # Initialize elapsed time to 0
    prev_hour = 0

    # Iterate through rows and update the time column
    for index, row in df[patient_mask].iterrows():
        # Convert the time to a Timedelta for arithmetic
        current_hour = row[time_column_name].hour

        # Check if the elapsed time crosses a day boundary
        if current_hour < prev_hour:
            current_date += pd.Timedelta(days=1)

        df.at[index, time_column_name] = pd.Timestamp(current_date.date()).replace(
            hour=row[time_column_name].hour,
            minute=row[time_column_name].minute,
            second=row[time_column_name].second)

        # Update the DataFrame
        prev_hour = current_hour

file_path = 'C:/Users/cakir/OneDrive/Belgeler/Datasets/REPLACE-BG/Data Tables/HDeviceCGM.txt'

delimiter = '|'

print('Starting...')
# Read the data into a DataFrame
df = pd.read_csv(file_path, sep=delimiter)
columns_to_drop = ['RecID', 'ParentHDeviceUploadsID', 'SiteID', 'DeviceDtTmDaysFromEnroll',
                   'DexInternalDtTmDaysFromEnroll', 'DexInternalTm']
df.drop(columns=columns_to_drop, inplace=True)
df.rename(columns={'PtID': Cols.patient, 'DeviceTm': Cols.date, 'GlucoseValue': Cols.value}, inplace=True)
df[Cols.date] = pd.to_datetime(df[Cols.date], format='%H:%M:%S').dt.time
patients = df[Cols.patient].unique()

print(len(patients), len(df))

"""
Delete me for final use
"""
patients = patients[:3]
df = df[df[Cols.patient].isin(patients)]
df = df.iloc[::-1].reset_index(drop=True)


print(len(patients), patients)

start_date = '2023-01-01'  # Dummy date
for i, patient in enumerate(patients):
    print(f'Starting {patient} ({i + 1}/{len(patients)})...')
    add_date_to_time_column(df[Cols.patient] == patient, Cols.date, start_date)

for patient in patients:
    print(patient)
    print(df[df[Cols.patient] == patient].head())

print(df.columns)
print('Done!')
