import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score

import utils.IterationUtils as Iter
from deBruijn.ProbabilityGraph import ProbabilityGraph
from models.Supervise import compute_threshold_alarm
from utils.PropertyNames import ColumnNames as Cols


def add_target_column(df: pd.DataFrame,
                      time_range_start: pd.Timedelta = pd.Timedelta(hours=0),
                      time_range_end: pd.Timedelta = pd.Timedelta(hours=1)):
    """
    Generate the target column for the data

    :param df: Dataframe containing the patient data.
    :type df: pandas.DataFrame

    :param time_range_start: The start of the time range to check for dangerous values, default is 0.5 hours.
    :type time_range_start: pandas.Timedelta

    :param time_range_end: The end of the time range to check for dangerous values, default is 1 hour.
    :type time_range_end: pandas.Timedelta

    :return: A dictionary of confusion matrices for each patient and the counter for the rows in the DataFrame.

    True Positive: The alert is on, and any of the data points in the range from that point are at a dangerous value.
    False Positive: The alert is on, but none of the data points in the range from that point are at a dangerous value.
    False Negative: The alert is off, but any of the data points in the range from that point are at a dangerous value.
    True Negative: The alert is off, and none of the data points in the range from that point are at a dangerous value.

    :rtype: pandas.DataFrame
    """

    patients = df[Cols.patient].unique()

    df[Cols.target] = np.nan

    # Loop over each unique patient
    for patient_label_value in patients:
        # Filter the DataFrame for the current patient
        df_patient = df[df[Cols.patient] == patient_label_value].sort_values(Cols.date, ascending=True)

        for index, row in df_patient.iterrows():
            # Calculate the range of dates to check for dangerous values
            start_date = row[Cols.date] + time_range_start
            end_date = start_date + time_range_end

            # Get the subset of the DataFrame within the range
            df_range = df_patient[(df_patient[Cols.date] > start_date) & (df_patient[Cols.date] < end_date)]

            if len(df_range) == 0:
                # No such point exists to evaluate
                continue

            # Check if there are any dangerous values in the range
            dangerous_in_range = (df_range[Cols.isDangerous] == True).any()

            # Assign to dataframe columns
            df.loc[index, Cols.target] = (bool(dangerous_in_range) or bool(row[Cols.isDangerous]))

    return df

def calculate_metrics(df: pd.DataFrame, alarm_column: str, include_already_dangerous: bool = False):
    """
    Calculates confusion matrix, accuracy, balanced accuracy, precision, sensitivity, specificity, and F1 score for a
    given prediction column.

    :param df: The input confusion matrix.
    :type df: pandas.DataFrame

    :param alarm_column: Name of the alarm column to be evaluated
    :type alarm_column: str

    :param include_already_dangerous: Include dangerous rows in evaluation. Note: Setting this parameter as True will
    bloat the metrics.
    :type include_already_dangerous: bool

    :return: A dictionary of the calculated metrics.
    :rtype: dict
    """
    # Creating a copy of the original dataframe
    df_clean = df.copy()

    # Removing None values from the x and y columns
    df_clean = df_clean.dropna(subset=Cols.alert_columns)

    # If the parameter is set False only include non-dangerous columns
    if not include_already_dangerous:
        df_clean = df_clean[df_clean[Cols.isDangerous] == False]

    # Defining true and predicted values
    true_values = df_clean[Cols.target].tolist()
    predicted_values = df_clean[alarm_column].tolist()

    # Calculating metrics
    accuracy = accuracy_score(true_values, predicted_values)
    bal_accuracy = balanced_accuracy_score(true_values, predicted_values)
    precision = precision_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values)  # Sensitivity
    f1 = f1_score(true_values, predicted_values)
    auc_roc = roc_auc_score(true_values, predicted_values)

    # Calculating confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_values, predicted_values).ravel()

    # Calculating specificity
    specificity = tn / (tn + fp)

    # Creating the confusion matrix dataframe
    total_confusion_matrix = pd.DataFrame({
        'Actual Positive': [tp, fn],
        'Actual Negative': [fp, tn]
    }, index=['Predicted Positive', 'Predicted Negative'])

    return {
        'Accuracy': accuracy,
        'Balanced Accuracy': bal_accuracy,
        'AUC-ROC': auc_roc,
        'Precision': precision,
        'Sensitivity': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'Confusion Matrix': total_confusion_matrix
    }, true_values, predicted_values


def loo_validation(sequences: list, k: int, risky_chars: set = None, **kwargs):
    """
    Performs leave-one-out cross validation.

    :param sequences: A list of sequences to use for validation.
    :type sequences: list

    :param k: The size of the k-mer.
    :type k: int

    :param risky_chars: A set of risky characters.
    :type risky_chars: set

    :param kwargs: Parameters for the model.

    :return: A list of probabilistic alerts.
    :rtype: list
    """
    probabilistic_alert = list()
    probabilistic_prob = list()
    for train_data, test_data in Iter.loo_partition(sequences):
        list_patient_result = list()
        list_prob_result = list()
        # Convert 3D list into 2D
        train_data = [item for sublist in train_data for item in sublist]
        probability_graph = ProbabilityGraph(k, train_data, risky_chars)
        probability_model = probability_graph.get_probability_model(**kwargs)
        for test_sequence in test_data:
            alerts, probs = probability_model.get_alerts(test_sequence)
            list_patient_result.append(alerts)
            list_prob_result.append(probs)
        probabilistic_alert.append(list_patient_result)
        probabilistic_prob.append(list_prob_result)
    return probabilistic_alert, probabilistic_prob


def add_alerts(dataframe: pd.DataFrame, naive_threshold: float, **kwargs):
    """
    Adds all alert outputs from all models to the DataFrame. Split sequences if the date gap is more than 20 minutes

    :param dataframe: The input DataFrame.
    :type dataframe: pandas.DataFrame

    :param naive_threshold: The threshold for the naive model.
    :type naive_threshold: float

    :param kwargs: Parameters for the probabilistic model.

    :return: The DataFrame with added alerts.
    :rtype: pandas.DataFrame
    """
    patients = dataframe[Cols.patient].unique()
    sequences = []
    sequence_indexes = []

    for p in patients:
        patient_sequences = []
        patient_sequence_indexes = []
        float_seq = dataframe[dataframe[Cols.patient] == p]
        float_seq = float_seq.sort_values(Cols.date, ascending=True)[[Cols.date_gap, Cols.char]]

        sequence = []
        indexes = []

        for index, row in float_seq.iterrows():
            date_gap = row[Cols.date_gap]
            seq_char = row[Cols.char]
            if pd.isna(date_gap) or (date_gap < pd.Timedelta(minutes=20)):
                # If the date gap is smaller than 20 minutes, append the sequence
                sequence.append(seq_char)
                indexes.append(index)
            elif len(sequence) > kwargs['k']:
                # Create a new sequence if the gap is more than 20 minutes
                patient_sequences.append(sequence)
                patient_sequence_indexes.append(indexes)
                sequence = []
                indexes = []

        # Ignore if the sequence is shorter than k
        if len(sequence) > kwargs['k']:
            patient_sequences.append(sequence)
            patient_sequence_indexes.append(indexes)

        sequences.append(patient_sequences)
        sequence_indexes.append(patient_sequence_indexes)
    probabilistic_alert, probabilistic_prob = loo_validation(sequences, **kwargs)

    for sequence_index, prob_alerts, probabilities in zip(sequence_indexes, probabilistic_alert, probabilistic_prob):
        for indexes, alerts, probs in zip(sequence_index, prob_alerts, probabilities):
            dataframe.loc[indexes, Cols.prob_alert] = alerts
            dataframe.loc[indexes, Cols.prob_alert+'_Prob'] = probs

    dataframe = compute_threshold_alarm(dataframe, naive_threshold)

    dataframe[Cols.combined_alert_or] = dataframe[Cols.prob_alert] | dataframe[Cols.naive_alert]
    dataframe[Cols.combined_alert_and] = dataframe[Cols.prob_alert] & dataframe[Cols.naive_alert]

    return dataframe


def benchmark(dataframe: pd.DataFrame, end_time_range_hours: float, start_time_range_hours: float = 0,
              include_already_dangerous=False, **kwargs):
    """
    Benchmarks the all the model and displays all the scores in a table.

    :param dataframe: The input DataFrame.
    :type dataframe: pandas.DataFrame

    :param end_time_range_hours: The end time range for the alerts.
    :type end_time_range_hours: float

    :param start_time_range_hours: The start time range for the alerts.
    :type start_time_range_hours: float

    :param include_already_dangerous: Include dangerous rows in evaluation. Note: Setting this parameter as True will
    bloat the metrics.
    :type include_already_dangerous: bool

    :param kwargs: Parameters for the models.

    :return:
    """
    dataframe = add_alerts(dataframe, **kwargs)

    dataframe = add_target_column(dataframe, time_range_start=pd.Timedelta(hours=start_time_range_hours),
                                  time_range_end=pd.Timedelta(hours=end_time_range_hours))

    df_list = []  # List to store all the score DataFrames
    roc_dict = dict()

    for alert_column in [Cols.prob_alert, Cols.naive_alert, Cols.combined_alert_or, Cols.combined_alert_and]:
        metrics_dict, y_true, y_pred = calculate_metrics(dataframe, alert_column, include_already_dangerous)
        confusion_matrix_dataframe = metrics_dict.pop('Confusion Matrix')
        roc_dict[alert_column] = (y_true, y_pred)
        # Convert the dictionary into a DataFrame and add a column for the alert type
        alert_type = 'Alert Type'
        df = pd.DataFrame([metrics_dict])
        df[alert_type] = str(alert_column)
        df.insert(0, alert_type, df.pop(alert_type))

        # Append the DataFrame to the list of DataFrames
        df_list.append(df)

        print(f'Confusion matrix for {alert_column}')
        display(confusion_matrix_dataframe)

    # Concatenate all DataFrames in the list and print the result
    result_df = pd.concat(df_list, ignore_index=True)

    display(result_df)
    return result_df, roc_dict, dataframe
