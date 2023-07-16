import pandas as pd
from IPython.display import display

import utils.IterationUtils as Iter
from deBruijn.ProbabilityGraph import ProbabilityGraph
from models.NaiveModel import compute_threshold_alarm
from utils.PropertyNames import ColumnNames as Cols


def generate_confusion_matrix(df: pd.DataFrame, boolean_column: str,
                              time_range_start: pd.Timedelta = pd.Timedelta(hours=0.5),
                              time_range_end: pd.Timedelta = pd.Timedelta(hours=1),
                              max_safe: int = 180, min_safe: int = 70):
    """
    Generate confusion matrices for each patient in the dataset.

    :param df: Dataframe containing the patient data.
    :type df: pandas.DataFrame

    :param boolean_column: The column of the DataFrame used to classify the data.
    :type boolean_column: str

    :param time_range_start: The start of the time range to check for dangerous values, default is 0.5 hours.
    :type time_range_start: pandas.Timedelta

    :param time_range_end: The end of the time range to check for dangerous values, default is 1 hour.
    :type time_range_end: pandas.Timedelta

    :param max_safe: The maximum safe value, values above are considered dangerous, default is 180.
    :type max_safe: int

    :param min_safe: The minimum safe value, values below are considered dangerous, default is 70.
    :type min_safe: int

    :return: A dictionary of confusion matrices for each patient and the counter for the rows in the DataFrame.

    True Positive: The alert is on, and any of the data points in the range from that point are at a dangerous value.
    False Positive: The alert is on, but none of the data points in the range from that point are at a dangerous value.
    False Negative: The alert is off, but any of the data points in the range from that point are at a dangerous value.
    True Negative: The alert is off, and none of the data points in the range from that point are at a dangerous value.
    :rtype: tuple
    """

    # Initialize a dictionary to store the confusion matrices for each patient
    confusion_matrices = {}
    counter = 0
    patients = df[Cols.patient].unique()

    # Loop over each unique patient
    for patient_label_value in patients:
        # Filter the DataFrame for the current patient
        df_patient = df[df[Cols.patient] == patient_label_value].sort_values(Cols.date, ascending=True)

        # Initialize confusion matrix components
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for index, row in df_patient.iterrows():
            # Calculate the range of dates to check for dangerous values
            start_date = row[Cols.date] + time_range_start
            end_date = start_date + time_range_end

            # Get the subset of the DataFrame within the range
            df_range = df_patient[(df_patient[Cols.date] > start_date) & (df_patient[Cols.date] < end_date)]

            # Skip if there are no data points in the range
            if df_range.empty or pd.isna(row[boolean_column]) or row[boolean_column] is None:
                continue

            # Check if there are any dangerous values in the range
            dangerous_in_range = ((df_range[Cols.value] < min_safe) | (df_range[Cols.value] > max_safe)).any()
            self_dangerous = (row[Cols.value] < min_safe) | (row[Cols.value] > max_safe)

            counter += 1

            if self_dangerous:
                continue

            # Classify the row based on the boolean column and the presence of dangerous values
            if (row[boolean_column] and dangerous_in_range) or (row[boolean_column] and self_dangerous):
                TP += 1
            elif row[boolean_column] and not dangerous_in_range:
                FP += 1
            elif not row[boolean_column] and dangerous_in_range:
                FN += 1
            elif not row[boolean_column] and not dangerous_in_range:
                TN += 1

        # Create a confusion matrix
        confusion_matrix = pd.DataFrame({
            'Actual Positive': [TP, FN],
            'Actual Negative': [FP, TN]
        }, index=['Predicted Positive', 'Predicted Negative'])

        # Store the confusion matrix for the current patient
        confusion_matrices[patient_label_value] = confusion_matrix

    return confusion_matrices, counter


def calculate_metrics(confusion_matrix: pd.DataFrame):
    """
    Calculates accuracy, precision, sensitivity, specificity, and F1 score.

    :param confusion_matrix: The input confusion matrix.
    :type confusion_matrix: pandas.DataFrame

    :return: A dictionary of the calculated metrics.
    :rtype: dict
    """
    # Initialize a dictionary to store the metrics for each patient

    # Compute the metrics
    TP, FN = confusion_matrix['Actual Positive']
    FP, TN = confusion_matrix['Actual Negative']

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    # Store the metrics for the current patient
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1_score,
    }

    return metrics


def combine_dicts(dicts: list):
    """
    Combines the list dictionaries by taking the mean of the values

    :param dicts: A list of dictionaries to combine.
    :type dicts: list

    :return: A combined dictionary.
    :rtype: dict
    """
    result = {}
    count = len(dicts)
    for d in dicts:
        for key, dict_value in d.items():
            if key not in result:
                result[key] = dict_value
            else:
                result[key] += dict_value

    for key in result:
        result[key] /= count

    return result


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
    for train_data, test_data in Iter.loo_partition(sequences):
        probability_graph = ProbabilityGraph(k, train_data, risky_chars)
        probability_model = probability_graph.get_probability_model(**kwargs)
        probabilistic_alert.append(probability_model.get_alerts(test_data))
    return probabilistic_alert


def add_alerts(dataframe: pd.DataFrame, naive_threshold: float, **kwargs):
    """
    Adds all alert outputs from all models to the DataFrame.

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
    for p in patients:
        float_seq = dataframe[dataframe[Cols.patient] == p]
        float_seq = float_seq.sort_values(Cols.date, ascending=True)[Cols.char]
        sequences.append(float_seq)

    probabilistic_alert = loo_validation(sequences, **kwargs)

    for i, p in enumerate(patients):
        # get the indices of the rows for the patient p
        idx = dataframe[dataframe[Cols.patient] == p].sort_values(Cols.date, ascending=True)[Cols.char].index

        if len(idx) != len(probabilistic_alert[i]):
            raise Exception('Invalid Format!')

        # set the 'Alert' column for these rows
        dataframe.loc[idx, Cols.prob_alert] = probabilistic_alert[i]

    dataframe = compute_threshold_alarm(dataframe, naive_threshold)
    dataframe[Cols.combined_alert] = dataframe[Cols.prob_alert] & dataframe[Cols.naive_alert]

    return dataframe


def benchmark(dataframe: pd.DataFrame, end_time_range_hours: int, start_time_range_hours: int = 0, **kwargs):
    """
    Benchmarks the all the model and displays all the scores in a table.

    :param dataframe: The input DataFrame.
    :type dataframe: pandas.DataFrame

    :param end_time_range_hours: The end time range for the alerts.
    :type end_time_range_hours: int

    :param start_time_range_hours: The start time range for the alerts.
    :type start_time_range_hours: int

    :param kwargs: Parameters for the models.

    :return:
    """
    dataframe = add_alerts(dataframe, **kwargs)

    df_list = []  # List to store all the score DataFrames

    for alert_column in [Cols.prob_alert, Cols.naive_alert, Cols.combined_alert]:
        confusion_dict, scored_data_point_counter = generate_confusion_matrix(dataframe, alert_column,
                                                                              time_range_start=pd.Timedelta(
                                                                                  hours=start_time_range_hours),
                                                                              time_range_end=pd.Timedelta(
                                                                                  hours=end_time_range_hours))

        # Calculate used and skipped data
        skipped_data = len(dataframe) - scored_data_point_counter

        # Generate scores for each patient key
        score_list = []
        for patient_key, confusion in confusion_dict.items():
            score_list.append(calculate_metrics(confusion))

        # Combine all scores into a single dictionary
        combined_dict = combine_dicts(score_list)

        # Convert the dictionary into a DataFrame and add a column for the alert type
        alert_type = 'Alert Type'
        df = pd.DataFrame([combined_dict])
        df[alert_type] = str(alert_column)
        df.insert(0, alert_type, df.pop(alert_type))

        df['Skipped Data'] = f"{skipped_data}/{len(dataframe)}"

        # Append the DataFrame to the list of DataFrames
        df_list.append(df)

    # Concatenate all DataFrames in the list and print the result
    result_df = pd.concat(df_list, ignore_index=True)
    display(result_df)
