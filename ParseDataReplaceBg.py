from utils.PropertyNames import ColumnNames as Cols

import pandas as pd


def add_date_to_time_column(patient_mask, time_column_name, starting_date):
    # Iterate through rows and update the time column
    for index, row in df[patient_mask].iterrows():
        time_value = row[time_column_name]

        # Create a datetime object by combining the starting date with the time value
        combined_datetime = pd.to_datetime(f"{starting_date} {time_value}")

        # Check if the hour of the time value enters a new date
        if combined_datetime.hour < row[Cols.date].hour:
            row[Cols.date] += pd.DateOffset(days=1)

        # Update the time column with the combined datetime
        df.at[index, time_column_name] = combined_datetime


file_path = '/home/lumpus/Documents/deBruijnData/ReplaceBg/Data Tables/HDeviceCGM.txt'

delimiter = '|'

print('Starting...')
# Read the data into a DataFrame
df = pd.read_csv(file_path, sep=delimiter)
columns_to_drop = ['RecID', 'ParentHDeviceUploadsID', 'SiteID', 'DeviceDtTmDaysFromEnroll',
                   'DexInternalDtTmDaysFromEnroll', 'DexInternalTm', 'RecordType']
df.drop(columns=columns_to_drop, inplace=True)
df.rename(columns={'PtID': Cols.patient, 'DeviceTm': Cols.date, 'GlucoseValue': Cols.value}, inplace=True)
df[Cols.date] = pd.to_datetime(df[Cols.date], format='%H:%M:%S').dt.time
patients = df[Cols.patient].unique()
print(len(patients), len(df))
exit()

"""
Delete me for final use
"""
patients = patients[:3]
df = df[df[Cols.patient] in patients]

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
