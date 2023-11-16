import matplotlib.pyplot as plt
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
    plt.figure(figsize=(15, 12))

    counts, bins, patches = plt.hist(values, alpha=0.7, rwidth=0.85, **kwargs)

    # label x-axis and y-axis
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # title the histogram
    plt.title(title)
    plt.xticks(bins, rotation=45)

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


def draw_timeline(df: pd.DataFrame, patient: str, column: str, include_already_dangerous: bool = True):
    """
    Draws and displays the timeline of alerts. Highlights the datapoints with alert with red. Skipped datapoints are
    grey.

    :param df: Dataframe with the alert labels
    :type df: pandas.DataFrame

    :param patient: The patient to use
    :type patient: str

    :param column: Name of the column to use it as alert
    :type column: str

    :return:
    """
    # Assume df is your DataFrame, and it already has datetime, value and flag columns
    df[Cols.date] = pd.to_datetime(df[Cols.date])  # convert to datetime if not already
    df = df.sort_values(Cols.date)  # sort the data based on datetime

    plt.figure(figsize=(30, 10))  # set the size of the figure

    colors = list()

    # Create color array
    for _, row in df.iterrows():
        column_bool = None if pd.isna(row[Cols.prob_alert]) or pd.isna(Cols.naive_alert) or pd.isna(
            Cols.combined_alert_or) or pd.isna(Cols.combined_alert_and) else bool(row[column])
        target_bool = None if pd.isna(row[Cols.target]) else bool(row[Cols.target])
        is_dangerous_bool = None if pd.isna(row[Cols.isDangerous]) else bool(row[Cols.isDangerous])
        if column_bool is None or target_bool is None or (not include_already_dangerous and is_dangerous_bool):
            colors.append('grey')
        elif column_bool is True and target_bool is True:
            colors.append('green')
        elif column_bool is True and target_bool is False:
            colors.append('orange')
        elif column_bool is False and target_bool is True:
            colors.append('red')
        elif column_bool is False and target_bool is False:
            colors.append('blue')
        else:
            raise ValueError('Unexpected Value!')

    # Plot segments in different colors
    for i in range(len(df[Cols.date]) - 1):
        plt.plot_date(df[Cols.date].iloc[i:i + 2], df[Cols.value].iloc[i:i + 2], color=colors[i], linestyle='solid')

    # Add horizontal lines
    plt.axhline(y=70, color='black', linestyle='dotted')
    # plt.axhline(y=180, color='black', linestyle='dotted')

    custom_lines = [Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='orange', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='grey', lw=2),
                    Line2D([0], [0], color='black', lw=2, linestyle='dotted')]
    plt.legend(custom_lines,
               ['True Positive', 'False Negative', 'False Positive', 'True Negative', 'Not Used in Evaluation',
                'Hypoglycemia Limit'])

    plt.gcf().autofmt_xdate()  # auto-format the x-axis dates

    plt.title(f'{patient} Timeline of Blood Glucose')  # set the title of the plot
    plt.xlabel('Time')  # set the x-axis label
    plt.ylabel('Blood Glucose (mg/dL)')  # set the y-axis label

    plt.show()

def draw_timeline_no_colors(df: pd.DataFrame, patient: str, colname:str, hypo_line=70):
    """
    Draws and displays the timeline of alerts. Highlights the datapoints with alert with red. Skipped datapoints are
    grey.

    :param df: Dataframe with the alert labels
    :type df: pandas.DataFrame

    :param patient: The patient to use
    :type patient: str

    :param column: Name of the column to use it as alert
    :type column: str

    :return:
    """
    # Assume df is your DataFrame, and it already has datetime, value and flag columns
    df[Cols.date] = pd.to_datetime(df[Cols.date])  # convert to datetime if not already
    df = df.sort_values(Cols.date)  # sort the data based on datetime

    plt.figure(figsize=(5, 10))  # set the size of the figure

    # Plot segments in different colors
    for i in range(len(df[Cols.date]) - 1):
        plt.plot_date(df[Cols.date].iloc[i:i + 2], df[colname].iloc[i:i + 2], color='blue', linestyle='solid')

    # Add horizontal lines
    # plt.axhline(y=hypo_line, color='black', linestyle='dotted')
    # plt.axhline(y=180, color='black', linestyle='dotted')

    plt.gcf().autofmt_xdate()  # auto-format the x-axis dates

    plt.title(f'{patient} Timeline of Blood Glucose')  # set the title of the plot
    plt.xlabel('Time')  # set the x-axis label
    plt.ylabel('Blood Glucose (mg/dL)')  # set the y-axis label

    plt.show()
