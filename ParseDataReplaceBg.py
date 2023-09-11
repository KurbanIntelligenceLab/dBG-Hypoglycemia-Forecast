from utils.PropertyNames import ColumnNames as Cols
from multiprocessing import Pool, cpu_count

import pandas as pd
import pickle as pkl

def add_date_to_time_column(patient_mask, time_column_name, starting_date):
    # Convert the starting date to a datetime object
    current_date = pd.to_datetime(starting_date)

    # Initialize elapsed time to 0
    prev_hour = 0

    # Iterate through rows and update the time column
    for index, row in replaceBgDf[patient_mask].iterrows():
        # Convert the time to a Timedelta for arithmetic
        current_hour = row[time_column_name].hour

        # Check if the elapsed time crosses a day boundary
        if current_hour < prev_hour:
            current_date += pd.Timedelta(days=1)

        replaceBgDf.at[index, time_column_name] = pd.Timestamp(current_date.date()).replace(
            hour=row[time_column_name].hour,
            minute=row[time_column_name].minute,
            second=row[time_column_name].second)

        # Update the DataFrame
        prev_hour = current_hour


def process_patient(patient_tuple):
    i, patient_param = patient_tuple
    print(f'Starting {patient_param} ({i + 1}/{len(patients)})...')
    add_date_to_time_column(replaceBgDf[Cols.patient] == patient_param, Cols.date, start_date)
    print(f'{patient_param} ({i + 1}/{len(patients)}) Just Ended!')

file_path = 'C:/Users/cakir/OneDrive/Belgeler/Datasets/REPLACE-BG/Data Tables/HDeviceCGM.txt'

delimiter = '|'

print('Starting...')
# Read the data into a DataFrame
replaceBgDf = pd.read_csv(file_path, sep=delimiter)
columns_to_drop = ['RecID', 'ParentHDeviceUploadsID', 'SiteID', 'DeviceDtTmDaysFromEnroll',
                   'DexInternalDtTmDaysFromEnroll', 'DexInternalTm']
replaceBgDf.drop(columns=columns_to_drop, inplace=True)
replaceBgDf.rename(columns={'PtID': Cols.patient, 'DeviceTm': Cols.date, 'GlucoseValue': Cols.value}, inplace=True)
replaceBgDf[Cols.date] = pd.to_datetime(replaceBgDf[Cols.date], format='%H:%M:%S').dt.time
patients = replaceBgDf[Cols.patient].unique()

print(len(patients), len(replaceBgDf))

"""
Delete me for final use
"""
#patients = patients[:1]
#replaceBgDf = replaceBgDf[replaceBgDf[Cols.patient].isin(patients)]

replaceBgDf = replaceBgDf.iloc[::-1].reset_index(drop=True)

print(len(patients), patients)

start_date = '2023-01-01'  # Dummy date
num_cpus = cpu_count()
print('Thread count:', num_cpus)
with Pool(num_cpus) as p:
    p.map(process_patient, enumerate(patients))

print(replaceBgDf.columns)
print('Done!')

# TODO: Convert to timedelta
# Define `date` gap
replaceBgDf[Cols.date_gap] = replaceBgDf[Cols.date].diff()

# Define `diff` column
replaceBgDf[Cols.diff] = replaceBgDf[Cols.value].diff()

custom_bins = [0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 500]
replaceBgDf[Cols.disc] = pd.cut(replaceBgDf[Cols.value], bins=custom_bins)
replaceBgDf[Cols.isDangerous] = replaceBgDf[Cols.value] < 70

# Before defining the char `char` column we sort the intervals, so we can define `char` based on the `value`
intervals = sorted(replaceBgDf[Cols.disc].unique().tolist(), key=lambda x: x.left)
interval_dict = {interval: index for index, interval in enumerate(intervals)}
replaceBgDf[Cols.char] = replaceBgDf[Cols.disc].apply(lambda disc_row: interval_dict[disc_row])

with open('/Data/REPLACE_BG.dat', 'wb') as file:
    pkl.dump(replaceBgDf, file)

