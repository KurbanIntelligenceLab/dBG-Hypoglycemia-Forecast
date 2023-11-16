import pickle
import random
import time

import Benchmark
from deBruijn.ProbabilityGraph import ProbabilityGraph
from utils.PropertyNames import ColumnNames as Cols
from utils.PropertyNames import MethodOptions as Opts
from sklearn.metrics import balanced_accuracy_score

def split_on_gaps(sequence):
    """Splits a list of numbers on gaps"""
    if not sequence:
        return []

    clusters = []
    cluster = [sequence[0]]

    for i in range(1, len(sequence)):
        if sequence[i] - sequence[i - 1] == 1:  # contiguous
            cluster.append(sequence[i])
        else:  # gap found
            clusters.append(cluster)
            cluster = [sequence[i]]

    if cluster:
        clusters.append(cluster)

    return clusters


def split_dict(data_dict, train_ratio=0.8, seed=None):
    if seed is not None:
        random.seed(seed)

    keys = list(data_dict.keys())
    random.shuffle(keys)

    train_size = int(len(keys) * train_ratio)
    train_keys = keys[:train_size]
    test_keys = keys[train_size:]

    train_dict = {key: data_dict[key] for key in train_keys}
    test_dict = {key: data_dict[key] for key in test_keys}

    return train_dict, test_dict


start = time.time()

with open('../Data/REPLACE_BG.dat', 'rb') as f:
    replaceBg = pickle.load(f)

print(len(replaceBg))
patients = replaceBg[Cols.patient].unique()

included_indexes = dict()
included_values = dict()

for patient_label_value in patients:
    df_patient = replaceBg[replaceBg[Cols.patient] == patient_label_value]
    true_indices = df_patient[df_patient[Cols.isDangerous] == True].index.tolist()
    patient_include_indices = set()

    for idx in true_indices:
        patient_include_indices.update(
            list(range(max(df_patient.index.min(), idx - 12), min(df_patient.index.max(), idx + 13))))

    sorted_indices = sorted(list(patient_include_indices))
    split_indices = split_on_gaps(sorted_indices)

    values_at_indices = replaceBg.loc[replaceBg.index.isin(sorted_indices)][Cols.value]

    # Extracting the exact values corresponding to each sub-array of indices
    split_values = [[values_at_indices.loc[idx] for idx in sublist] for sublist in split_indices]

    included_indexes[patient_label_value] = split_indices
    included_values[patient_label_value] = split_values

    print(
        f'Patient: {patient_label_value}, '
        f'Included Row Count : {len(patient_include_indices)} '
        f'({(len(patient_include_indices) / len(df_patient) * 100):.2f} %)')

params = {
    "k": 6,
    "risky_chars": {0, 1},
    "risk_threshold": 0.05,
    "prune": True,
    "prune_method": Opts.filter,
    "prune_threshold": 1,
    "weight_thresholds": [1, 10, 20],
    "value_ranges": [(0, 1), (2, 3), (3, float('inf'))],
    "max_steps": 3,
    "naive_threshold": 15
}

# excluded, included = split_dict(sequences, train_ratio=0.8, seed=0)
train_data, test_data = split_dict(included_values, train_ratio=0.8, seed=0)
print(f'Train len:{len(train_data)}, Test len:{len(test_data)}')

print('Generating a graph...')

train_data_2d_list = [item for sublist in train_data.values() for item in sublist]

probability_graph = ProbabilityGraph(k=params['k'],
                                     sequences=train_data_2d_list,
                                     risky_chars=params['risky_chars'])

print(probability_graph)
probability_model = probability_graph.get_probability_model(risk_threshold=params['risk_threshold'],
                                                            prune_method=params['prune_method'],
                                                            prune_threshold=params['prune_threshold'],
                                                            weight_thresholds=params['weight_thresholds'],
                                                            max_steps=params['max_steps'],
                                                            prune=params['prune'],
                                                            value_ranges=params['value_ranges']
                                                            )

print('Making Predictions...')

replaceBg = Benchmark.add_target_column(replaceBg)
included_alerts = dict()
for test_patient in test_data.keys():
    print(test_patient)
    patient_values = included_values[test_patient]
    for sequence in patient_values:
        included_alerts[test_patient] = probability_model.get_alerts(sequence)

print(included_alerts)
target = list()
pred = list()
for test_patient in test_data.keys():
    for seq_index, seq_alert in zip(included_indexes[test_patient], included_alerts[test_patient]):
        for index, alert in zip(seq_index, seq_alert):
            if not replaceBg.loc[index][Cols.isDangerous]:
                target.append(replaceBg.loc[index][Cols.target])
                pred.append(alert)


balanced_acc = balanced_accuracy_score(target, pred)
print(balanced_acc)
end = time.time()
print(end - start)
