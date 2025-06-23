# === Cell 1 ===# Install required packages
# code must run on python 3.12
print("=== notebook started ===")

# === Cell 2 ===import warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"outdated"
)


# === Cell 3 ===# Import required modules

import ogb; print('ogb version {}'.format(ogb.__version__)) # make sure the version is =>1.1.1.
from ogb.graphproppred import PygGraphPropPredDataset
from WEGL.WEGL import WEGL

# === Cell 4 ===# Set the random seed

random_seed = 55

# === Cell 5 ===# Load the dataset
# We can try different datasets like 'ogbg-molpcba', 'ogbg-molhiv', etc.
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

print('# of graphs = {0}\n# of classes = {1}\n# of node features = {2}\n# of edge features = {3}'.\
         format(len(dataset), dataset.num_classes, dataset.num_node_features, dataset.num_edge_features))

if isinstance(dataset, PygGraphPropPredDataset): # OGB datasets
    print('# of tasks = {}'.format(dataset.num_tasks))

# === Cell 6 ===# Specify the parameters

# num_hidden_layers = range(3, 9)
num_hidden_layers = [4]

# node_embedding_sizes = [100, 300, 500]
node_embedding_sizes = [300]

# dim_reduction_method = ['PCA', 'kernelPCA', 'Isomap', 'LLE']
dim_reduction_methods = ['PCA', 'KernelPCA', 'Isomap', 'LLE']

n_landmarks=2000

# option are LOT/GME
graph_embedding_tools = ['LOT', 'GME']

# final_node_embeddings = ['concat', 'avg', 'final']
final_node_embeddings = ['final']

num_pca_components = 20
num_experiments = 5
classifiers = ['RF']
device = 'cpu'

# === Cell 7 ===# Run the algorithm
results = []  # List to store all result_table outputs
for dim_reduction_method in dim_reduction_methods:
    for graph_embedding_tool in graph_embedding_tools:
        result_table = WEGL(
            dataset=dataset,
            num_hidden_layers=num_hidden_layers,
            node_embedding_sizes=node_embedding_sizes,
            final_node_embedding='final',
            num_pca_components=num_pca_components,
            num_experiments=num_experiments,
            dim_reduction_method=dim_reduction_method,
            n_landmarks=n_landmarks,
            classifiers=classifiers,
            graph_embedding_tool=graph_embedding_tool,
            random_seed=random_seed,
            device=device
        )
        results.append(result_table)

# === Cell 8 ===# After running your experiments and collecting results in the 'results' list
from prettytable import PrettyTable\

dataset_name = dataset.name if hasattr(dataset, 'name') else str(dataset)

final_table = PrettyTable()
final_table.title = f"Final Results ({dataset_name})"
final_table.field_names = results[0].field_names

# Define a separator row (e.g., a row of dashes)
separator_row = ['-' * 8 for _ in final_table.field_names]

first = True
for result_table in results:
    for row in result_table.rows:
        if not first:
            final_table.add_row(separator_row)
        final_table.add_row(row)
        first = False

print(final_table)

# === Cell 9 ===import matplotlib.pyplot as plt
import numpy as np

methods = []
train_means, train_stds = [], []
val_means, val_stds = [], []
test_means, test_stds = [], []

for row in final_table._rows:
    # Skip separator rows (all dashes)
    if all(str(cell).strip('-') == '' for cell in row):
        continue
    label = f"{row[0]}-{row[1]}"
    methods.append(label)
    # Each result is like '85.23 $\pm$ 2.34'
    for idx, arr, stds in zip(
        [5, 6, 7],
        [train_means, val_means, test_means],
        [train_stds, val_stds, test_stds]
    ):
        cell = row[idx]
        # Split by '$\pm$' or '+/-' or similar
        if '±' in cell:
            mean_str, std_str = cell.split('±')
        elif '+/-' in cell:
            mean_str, std_str = cell.split('+/-')
        elif '$\\pm$' in cell:
            mean_str, std_str = cell.split('$\\pm$')
        else:
            mean_str, std_str = cell.split()
        mean = float(mean_str.strip())
        std = float(std_str.strip())
        arr.append(mean)
        stds.append(std)

x = np.arange(len(methods))
width = 0.25

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - width, train_means, width, yerr=train_stds, label='Train')
bars2 = plt.bar(x, val_means, width, yerr=val_stds, label='Val')
bars3 = plt.bar(x + width, test_means, width, yerr=test_stds, label='Test')
plt.xticks(x, methods, rotation=45)
plt.ylabel('ROC-AUC (%)')
plt.title(f'Final Results on {dataset_name}: Train, Val, Test ROC-AUC Comparison')
plt.legend()
plt.tight_layout()

# Add value numbers on the bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90
        )

plt.show()

