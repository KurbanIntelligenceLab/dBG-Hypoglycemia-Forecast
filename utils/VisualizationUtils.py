import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils.PropertyNames import ColumnNames as Cols


def draw_histogram(values: list, title: str, x_label: str, y_label: str, **kwargs):
    """
    Draws and displays a histogram from a list of numbers.

    :param values: List of values to draw a histogram with
    :type values: list

    :param title: Title of the histogram
    :type title: str

    :param x_label: X-axis label of the histogram
    :type x_label: str

    :param y_label: Y-axis label of the histogram
    :type y_label: str

    :return:
    """
    # create histogram
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(values, alpha=0.7, rwidth=0.85, **kwargs)

    # label x-axis and y-axis
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # title the histogram
    plt.title(title)

    # show the values at the top of each bar
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        plt.annotate(str(int(count)), xy=(x, 0), xycoords=('data', 'axes fraction'),
                     xytext=(0, -40), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        plt.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                     xytext=(0, -54), textcoords='offset points', va='top', ha='center')

    # show the histogram
    plt.show()


def draw_timeline(df: pd.DataFrame, patient: str, column: str):
    """
    Draws and displays the timeline of alerts. Highlights the datapoints with alert with red.

    :param df: Dataframe with the alert labels
    :type df: pandas.DataFrame

    :param patient: The patient to use
    :type patient: str

    :param column: Name of the column to use it as alert
    :type column: str

    :return:
    """
    df = df.dropna()
    # Assume df is your DataFrame, and it already has datetime, value and flag columns
    df[Cols.date] = pd.to_datetime(df[Cols.date])  # convert to datetime if not already
    df = df.sort_values(Cols.date)  # sort the data based on datetime

    plt.figure(figsize=(30, 10))  # set the size of the figure

    # Create color array
    colors = np.where(df[column], 'red', 'blue')

    # Plot segments in different colors
    for i in range(len(df[Cols.date]) - 1):
        plt.plot_date(df[Cols.date].iloc[i:i + 2], df[Cols.value].iloc[i:i + 2], color=colors[i], linestyle='solid')

    # Add horizontal lines
    plt.axhline(y=70, color='green', linestyle='dotted')
    plt.axhline(y=180, color='green', linestyle='dotted')

    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', lw=2, linestyle='dotted')]
    plt.legend(custom_lines, ['No Alert', 'Alert', 'Hypo/Hyper-glycine Limits'])

    plt.gcf().autofmt_xdate()  # auto-format the x-axis dates

    plt.title(f'{patient} Timeline of Blood Sugar')  # set the title of the plot
    plt.xlabel('Time')  # set the x-axis label
    plt.ylabel('Blood Sugar')  # set the y-axis label

    plt.show()
