from utils.PropertyNames import ColumnNames as Cols
import pandas as pd


def compute_threshold_alarm(df: pd.DataFrame, x: float):
    """
    Compute the naive threshold alarm based on a given threshold value. The function performs the operations separately
    for each patient, generating a new column for the output. This gives an alert if the current value is at least `x`
    units close to dangerous char, and it is getting close to it.

    :param df: The input DataFrame.
    :type df: pandas.DataFrame

    :param x: The threshold to apply for determining alerts. This value is used to adjust the areas where we consider
    'too close to dangerous areas'
    :type x: float

    :return: The DataFrame with added 'Naive Alert' column.
    :rtype: pandas.DataFrame
    """

    # Sort the dataframe
    df.sort_values(by=[Cols.patient, Cols.date], inplace=True)

    # Convert date to datetime object for date operations
    df[Cols.date] = pd.to_datetime(df[Cols.date])

    # Initialize output dataframe
    df_out = pd.DataFrame()
    patients = df[Cols.patient].unique()

    # Perform operations separately for each patient
    for p in patients:
        patient_df = df[df[Cols.patient] == p].copy()  # Create a copy to avoid SettingWithCopyWarning

        # Define ThresholdAlarm column
        patient_df[Cols.naive_alert] = patient_df.apply(
            lambda row: (row[Cols.value] > 180 - x and row[Cols.diff] > 0)
                        or ((row[Cols.value] < 70 + x and row[Cols.diff] < 0)
                            or (row[Cols.value] > 180) or (row[Cols.value] < 70)),
            axis=1
        )

        # Append to the output dataframe
        df_out = pd.concat([df_out, patient_df], ignore_index=True)

    return df_out


