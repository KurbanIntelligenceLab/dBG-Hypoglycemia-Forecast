import glob
import os
from datetime import datetime

import openpyxl
import pandas as pd

from utils.PropertyNames import ColumnNames as Cols


def _transform_tuple(sheet_row: tuple):
    """
    Parses a single row of datasheet, transforming it into a tuple with datetime and value.

    :param sheet_row: Tuple representation of the sheet row. The first item is expected to be
                      a datetime or string representing datetime. The subsequent items are searched
                      for a non-empty value to be used as the value in the resulting tuple.
    :type sheet_row: tuple

    :return: Returns a 2-length tuple. The first index is datetime (parsed from the first non-empty cell).
             The second index is the first non-empty value found in the subsequent cells. Returns `None`
             if the value is empty, the date cannot be parsed, or the first cell is `None`.
    :rtype: tuple

    :raises ValueError: If the date string cannot be parsed into a datetime object.
    :raises TypeError: If the date is not a string or datetime, or the value is not a number.
    """

    # Assign the first column as date
    if sheet_row[0] is None:
        return None

    # Find the next non-empty cell as the value
    second_index = None
    date_time_value = sheet_row[0]
    for item in sheet_row[1:]:
        if item is not None:
            second_index = item
            break

    # If the date column type is not datetime attempt to fix it
    if type(date_time_value) == str:
        date_time_value = datetime.strptime(date_time_value, "%m/%d/%y %H:%M")

    try:
        return date_time_value, float(second_index)
    except ValueError:
        return None
    except TypeError:
        return None


def parse_dataset(dataset_directory: str, custom_bins: list = None, silent: bool = True):
    """
    This function takes the dataset directory and parses the xlsx files into a dataframe.

    :param dataset_directory: The directory that contains the original dataset.
    :type dataset_directory: str

    :param custom_bins: Sorted bin ranges of the alphabet. Used for discretization.

        **Default Ranges**
        Char    Range
        0       <60: Low Hypo (dangerously low, hypoglycemia)
        1       60-70: High hypo
        2       70-80
        ...
        12      170-180
        13      180-190 Low Hyper
        14      >190: High Hyper (dangerously high, hyperglycemia)
    :type custom_bins: list

    :param silent: If false, prints parse information. Used for debugging.
    :type silent: bool

    :return: The parsed data in a dataframe. The returned dataframe contains `Docs.date`, `Docs.patient`, `Docs.value`,
    `Docs.diff`, `Docs.disc`, `Docs.char` and `Docs.char_norep` columns
    :rtype: pandas.DataFrame
    """
    if custom_bins is None:
        custom_bins = [0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 500]

    try:
        # Find all XLSX files in the directory
        file_paths = glob.glob(dataset_directory + "/*.xlsx")

        # Initialize the dataframe
        data_df = pd.DataFrame()

        # Iterate over files
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            if not silent:
                print(f"Opening file {(i + 1)}-{file_name}:")
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active

            # List of all rows in the currently opened sheet
            sheet_data = list()

            # Iterate over rows
            for row in sheet.iter_rows(values_only=True):
                processed_row = _transform_tuple(row)
                # Parse and append row if the row does not contain any None values
                if processed_row is not None and processed_row[0] is not None and processed_row[1] is not None:
                    sheet_data.append(processed_row)
                elif not silent:
                    print(f"Unable to parse row. Skipping... {row}")

            # Convert file content into a dataframe
            sheet_df = pd.DataFrame(sheet_data, columns=[Cols.date, Cols.value])

            # Sort by `date` just in case
            sheet_df.sort_values(Cols.date, ascending=True, inplace=True)

            # Define `date` gap
            sheet_df[Cols.date_gap] = sheet_df[Cols.date].diff()

            # Define `diff` column
            sheet_df[Cols.diff] = sheet_df[Cols.value].diff()
            sheet_df[Cols.disc] = pd.cut(sheet_df[Cols.value], bins=custom_bins)

            # Extract the patient name from the file name
            patient_id = file_name.split(' ')[0]
            sheet_df[Cols.patient] = patient_id

            # Concatenate `data_df` by `sheet_df`
            data_df = pd.concat([data_df, sheet_df], ignore_index=True)

        # Define `isDangerous` column
        data_df[Cols.isDangerous] = data_df[Cols.value] < 70

        # Before defining the char `char` column we sort the intervals, so we can define `char` based on the `value`
        intervals = sorted(data_df[Cols.disc].unique().tolist(), key=lambda x: x.left)
        interval_dict = {interval: index for index, interval in enumerate(intervals)}
        data_df[Cols.char] = data_df[Cols.disc].apply(lambda disc_row: interval_dict[disc_row])

        # Remove duplicates from `char` column to define `char_norep`
        data_df[Cols.char_norep] = data_df[Cols.char].where(data_df[Cols.char].ne(data_df[Cols.char].shift()))
        if not silent:
            print("Data Loaded!")
        return data_df

    except FileNotFoundError:
        print("Directory not found.")
