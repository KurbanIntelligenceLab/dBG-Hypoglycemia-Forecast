import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from ParseData import parse_dataset
from deBruijn.ProbabilityGraph import ProbabilityGraph
from utils.PropertyNames import ColumnNames as Cols
from utils.VisualizationUtils import draw_timeline_no_colors

patient_data = parse_dataset("/home/meocakir/Documents/Datasets/Diabetes", silent=False)
patients = patient_data[Cols.patient].unique()

selected_patient = 'P9'

selected_timeline = patient_data[patient_data[Cols.patient] == selected_patient]
timeline_subset = selected_timeline[50:75].copy()

draw_timeline_no_colors(timeline_subset, selected_patient, Cols.value)
draw_timeline_no_colors(timeline_subset, selected_patient, Cols.char, hypo_line=1)

G = ProbabilityGraph(4, [timeline_subset[Cols.char].tolist()]).graph
# Drawing the graph without labels
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_color='skyblue')

# Adjusting label positions
label_pos = {k: [v[0], v[1] - 0.1] for k, v in pos.items()}  # Adjust 0.05 to your needs


# Drawing edge weights (labels)
edge_labels = nx.get_edge_attributes(G, 'weight')
# Drawing labels with offset and increased font size
nx.draw_networkx_labels(G, pos=label_pos, font_size=12)  # Adjust font_size to your needs
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='blue')

# Adjusting self-loop label positions for visibility
for node, data in pos.items():
    if G.has_edge(node, node):  # Check if there's a self-loop on this node
        loop_label_pos = (data[0], data[1] + 0.24)  # Adjust the Y offset for clarity
        plt.text(loop_label_pos[0], loop_label_pos[1], s=G[node][node]['weight'], horizontalalignment='center', verticalalignment='center', color='blue')
plt.figure(figsize=(10, 10))
plt.show()
print(len(timeline_subset))
