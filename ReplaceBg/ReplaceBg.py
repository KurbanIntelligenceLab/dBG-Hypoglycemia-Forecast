import pickle as pkl

import pandas as pd

from Benchmark import benchmark
from utils.PropertyNames import MethodOptions as Opts

with open('../Data/REPLACE_BG.dat', 'rb') as f:
    replaceBg = pkl.load(f)
print(len(replaceBg))

# patient_data = parse_dataset("/home/meocakir/Documents/Datasets/Diabetes", silent=False)


params = {
    "k": 6,
    "risky_chars": {0, 1},
    "risk_threshold": 0.2,
    "prune": True,
    "prune_method": Opts.adaptive,
    "prune_threshold": 1,
    "weight_thresholds": [1, 4, 10],
    "value_ranges": [(0, 2), (2, 3), (3, float('inf'))],
    "max_steps": 6,
    "naive_threshold": 15
}

pd.set_option('display.float_format', '{:.4f}'.format)

results = benchmark(replaceBg, start_time_range_hours=0, end_time_range_hours=1, **params)

print('Writing Binary...')
with open('Data/Results_REPLACE_BG.dat', 'wb') as file:
    pkl.dump(results, file)

print('Writing Csv...')
results.to_csv('Data/Results_REPLACE_BG.csv')

print('Writing Binary...')
with open('Data/Out_REPLACE_BG.dat', 'wb') as file:
    pkl.dump(replaceBg, file)

print('Writing Csv...')
replaceBg.to_csv('Data/Out_REPLACE_BG.csv')
