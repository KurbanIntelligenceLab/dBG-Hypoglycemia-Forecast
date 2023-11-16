import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import networkx as nx
import utils.IterationUtils as Iter

from Benchmark import add_target_column
from ParseData import parse_dataset
from deBruijn.ProbabilityGraph import ProbabilityGraph
from utils.PropertyNames import ColumnNames as Cols
from utils.PropertyNames import MethodOptions as Opts
import warnings

warnings.filterwarnings("ignore")
params = {
    "k": 4,
    "risky_chars": {0, 1},
    "risk_threshold": 0.2,
    "prune": True,
    "prune_method": Opts.adaptive,
    "prune_threshold": 1,
    "weight_thresholds": [1, 2, 2],
    "value_ranges": [(0, 2), (2, 3), (3, float('inf'))],
    "max_steps": 6,
    "naive_threshold": 15
}


def split_list(lst, n):
    '''Split a list into n parts of approximately equal size.'''
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


patient_data = parse_dataset("/home/meocakir/Documents/Datasets/Diabetes", silent=False)
patients = patient_data[Cols.patient].unique()

patient_data = add_target_column(patient_data)

patients = patient_data[Cols.patient].unique()
sequences = dict()
sequence_indexes = dict()

for p in patients:
    print('P', p)
    patient_sequences = []
    patient_sequence_indexes = []
    float_seq = patient_data[patient_data[Cols.patient] == p]
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
        elif len(sequence) > params['k']:
            # Create a new sequence if the gap is more than 20 minutes
            patient_sequences.append(sequence)
            patient_sequence_indexes.append(indexes)
            sequence = []
            indexes = []

    # Ignore if the sequence is shorter than k
    if len(sequence) > params['k']:
        patient_sequences.append(sequence)
        patient_sequence_indexes.append(indexes)

    sequences[p] = patient_sequences
    sequence_indexes[p] = patient_sequence_indexes

for train_data, test_data in Iter.loo_partition(patients):
    print(test_data)
    splitted_patients = split_list(train_data, 3)
    graphs = list()
    models = list()
    print('Eigs')
    for group_patient in splitted_patients:
        graph_sequences = list()
        for patient in group_patient:
            graph_sequences.extend(sequences[patient])
        graph = ProbabilityGraph(params['k'], graph_sequences)
        graphs.append(graph)

        undirected_graph = graph.graph.copy().to_undirected()
        lap_mat = nx.laplacian_matrix(undirected_graph).toarray()

        # Calculate eigenvalues of the adjacency matrix
        eigenvalues = np.linalg.eigvals(lap_mat)

        # Sort the eigenvalues by magnitude
        sorted_eigenvalues = sorted(eigenvalues, reverse=True)

        lambda1 = sorted_eigenvalues[0]
        lambda2 = sorted_eigenvalues[1]
        print(int(lambda1), int(lambda2))
        models.append(graph.get_probability_model(risk_threshold=params['risk_threshold'],
                                                  prune_method=params['prune_method'],
                                                  prune_threshold=params['prune_threshold'],
                                                  weight_thresholds=params['weight_thresholds'],
                                                  max_steps=params['max_steps'],
                                                  prune=params['prune'],
                                                  value_ranges=params['value_ranges']
                                                  ))
    test_sequence = [item for sublist in sequences[test_data] for item in sublist]
    test_indexes = [item for sublist in sequence_indexes[test_data] for item in sublist]
    splitted_test_sequence = split_list(test_sequence, 2)
    splitted_test_index = split_list(test_indexes, 2)

    test_graph = ProbabilityGraph(3, [splitted_test_sequence[0]])
    undirected_test_graph = test_graph.graph.copy().to_undirected()
    lap_mat = nx.laplacian_matrix(undirected_test_graph).toarray()
    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(lap_mat)
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)

    # Print the eigenvalues
    lambda1 = sorted_eigenvalues[0]
    lambda2 = sorted_eigenvalues[1]
    print('>> ', int(lambda1), int(lambda2))

    results = list()
    print('Evals')
    for model in models:
        y_pred = list()
        y_real = list()
        alert_sequence = model.get_alerts(splitted_test_sequence[0])
        for index, alert in zip(splitted_test_index[0], alert_sequence):
            row = patient_data.loc[index]
            if not pd.isna(alert) and not pd.isna(row[Cols.target]):
                y_real.append(row[Cols.target])
                if row[Cols.diff] > params['naive_threshold'] or row[Cols.value] > 150:
                    y_pred.append(False)
                else:
                    y_pred.append(alert)

        bacc = balanced_accuracy_score(y_real, y_pred)
        results.append(bacc)
        print(bacc)

    best_model_index = results.index(max(results))
    best_graph = graphs[best_model_index]
    best_graph.update_graph([splitted_test_sequence[1]], 1)
    best_model = best_graph.get_probability_model(risk_threshold=params['risk_threshold'],
                                                  prune_method=params['prune_method'],
                                                  prune_threshold=params['prune_threshold'],
                                                  weight_thresholds=params['weight_thresholds'],
                                                  max_steps=params['max_steps'],
                                                  prune=params['prune'],
                                                  value_ranges=params['value_ranges']
                                                  )
    y_pred = list()
    y_real = list()
    alert_sequence = best_model.get_alerts(splitted_test_sequence[1])
    for index, alert in zip(splitted_test_index[1], alert_sequence):
        row = patient_data.loc[index]
        if not pd.isna(alert) and not pd.isna(row[Cols.target]):
            y_real.append(row[Cols.target])
            if row[Cols.diff] > params['naive_threshold'] or row[Cols.value] > 150:
                y_pred.append(False)
            else:
                y_pred.append(alert)
    print('')
    bacc = balanced_accuracy_score(y_real, y_pred)
    results.append(bacc)
    print(bacc)
    print('===')

print(split_list(patients, 3))