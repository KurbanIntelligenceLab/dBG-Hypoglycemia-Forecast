import numpy as np

from utils.PropertyNames import ColumnNames as Cols
import pandas as pd


def compute_threshold_alarm(df: pd.DataFrame, x: float, th: float=150):
    """
    This method computes the supervision mask for the model. Creates Cols.naive_alert column for the results.

    :param df: The input DataFrame.
    :type df: pandas.DataFrame

    :param x: The min diff value to override the alerts. Sets the row False if the valıue difference is over this value.
    :type x: float

    :param th: The min threshold to override alerts. Sets the row False if the value is over this value.
    :type th: float

    :return: The DataFrame with added 'Naive Alert' column.
    :rtype: pandas.DataFrame
    """

    df[Cols.naive_alert] = np.where((df[Cols.diff] > x) | (df[Cols.value] > th), False, True)
    return df
