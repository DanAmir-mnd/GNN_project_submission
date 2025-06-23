from prettytable import PrettyTable
from collections import defaultdict
from tqdm.notebook import tqdm
import itertools
import matplotlib.pyplot as plt
import random

import numpy as np
import torch
from torch_geometric.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred import Evaluator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import ot

from .onehot_bondencoder import BondEncoderOneHot
from .diffusion import Diffusion


def select_landmarks(X, n_landmarks, state=0):
    """
    Select landmarks from the dataset X using random sampling.

    Inputs:
        - X: dataset of shape (num_samples, num_features)
        - n_landmarks: number of landmarks to select
        - random_seed: random seed for reproducibility

    Outputs:
        - landmarks: selected landmarks of shape (n_landmarks, num_features)
    """
    rng = np.random.RandomState(state)
    indices = rng.choice(X.shape[0], n_landmarks, replace=False)
    landmarks = X[indices]
    return landmarks


def batched_transform(model, X_matrix, batch_size=10000):
    """Apply mode transform to X_matrix in batches."""
    results = []
    for start in range(0, len(X_matrix), batch_size):
        end = start + batch_size
        batch = X_matrix[start:end]
        results.append(model.transform(batch))
    return np.concatenate(results, axis=0)


def WEGL(dataset,
         num_hidden_layers,
         node_embedding_sizes,
         final_node_embedding,
         dim_reduction_method='PCA',
         n_landmarks=None,
         num_pca_components=20,
         num_experiments=10,
         graph_embedding_tool='GME',
         classifiers=['RF'],
         random_seed=0,
         device='cpu'
         ):
    """
    # The WEGL pipeline
    
    Inputs:
        - dataset: dataset object
        - num_hidden_layers: number of diffusion layers
        - node_embedding_sizes: node embedding dimensionality created by the AtomEncoder module
        - final_node_embedding: final node embedding type $\in$ {'concat', 'avg', 'final'}
        - num_pca_components: number of PCA components applied on node embeddings. -1 means no PCA.
        - num_experiments: number of experiments with different random seeds
        - classifiers: list of downstream classifiers
        (currently random forest ('RF') only; other classifiers, e.g., SVM, can be added if desired
        - random_seed: the random seed
        - device # the device to run the diffusion over ('cpu'/'cuda')
        
    Outputs:
        - A table containing the classification results
        
    """

    # Set the random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create data loaders
    split_idx = dataset.get_idx_split()  # train/val/test split
    loader_dict = {}
    for phase in split_idx:
        batch_size = 32
        loader_dict[phase] = DataLoader(dataset[split_idx[phase]], batch_size=batch_size, shuffle=False)

    # prepare the output table
    results_table = PrettyTable()
    results_table.title = (
        'Final ROC-AUC(%) results for the {0} dataset with \'{1}\' node embedding, '
        'one-hot 13-dim edge embedding, dim reduction: {2}, graph embedding: {3}'
        .format(dataset.name, final_node_embedding, dim_reduction_method, graph_embedding_tool)
    )

    # Add two new columns before 'Classifier'
    results_table.field_names = [
        'Dim Reduction Method', 'Graph Embedding Tool', 'Classifier', 
        '# Diffusion Layers', 'Node Embedding Size', 'Train.', 'Val.', 'Test'
    ]

    n_jobs = 14
    verbose = 0

    for L, F in itertools.product(num_hidden_layers, node_embedding_sizes):
        print('*' * 100)
        print('# diffusion layers = {0}, node embedding size = {1}, node embedding mode: {2}\n'. \
              format(L, F, final_node_embedding))

        # create an instance of the diffusion object
        diffusion = Diffusion(num_hidden_layers=L,
                              final_node_embedding=final_node_embedding).to(device)
        diffusion.eval()

        # create the node encoder
        node_feature_encoder = AtomEncoder(F).to(device)
        node_feature_encoder.eval()

        phases = list(loader_dict.keys())  # determine different partitions of data ('train', 'valid' and 'test')

        # pass the all the graphs in the data through the GNN
        X = defaultdict(list)
        Y = defaultdict(list)

        for phase in phases:
            print('Now diffusing the ' + phase + ' data ...')
            for i, batch in enumerate(tqdm(loader_dict[phase])):
                batch = batch.to(device)

                # encode node features
                batch.x = node_feature_encoder(batch.x)

                # encode edge features
                batch.edge_attr = BondEncoderOneHot(batch.edge_attr)

                # add virtual nodes
                batch_size = len(batch.y)
                num_original_nodes = batch.x.size(0)
                batch.batch = torch.cat((batch.batch, torch.Tensor(range(batch_size)).to(batch.batch.dtype)), dim=0)

                # make the initial features of all virtual nodes zero
                batch.x = torch.cat((batch.x, batch.x.new_zeros(batch_size, batch.x.size(1))), dim=0)

                # add edges between all nodes in each graph and the virtual node for that graph
                for g in range(batch_size):
                    node_indices = np.where(batch.batch == g)[0][:-1]  # last node is the virtual node
                    virtual_edges_one_way = np.array(
                        [node_indices, (num_original_nodes + g) * np.ones_like(node_indices)])
                    virtual_edges_two_ways = np.concatenate((virtual_edges_one_way,
                                                             np.take(virtual_edges_one_way, [1, 0], axis=0)),
                                                            axis=1)

                    batch.edge_index = torch.cat((batch.edge_index,
                                                  torch.Tensor(virtual_edges_two_ways).to(batch.edge_index.dtype)),
                                                 dim=1)

                    # make the initial edge features of all edges to/from virtual nodes all 1 / number of graph nodes
                    batch.edge_attr = torch.cat((batch.edge_attr, batch.edge_attr.new_ones(2 * len(node_indices),
                                                                                           batch.edge_attr.size(
                                                                                               1)) / len(node_indices)),
                                                dim=0)

                # pass the data through the diffusion process
                z = diffusion(batch)
                batch_indices = batch.batch.cpu()
                for b in range(batch_size):
                    node_indices = np.where(batch_indices == b)[0]
                    X[phase].append(z[node_indices].detach().cpu().numpy())

                Y[phase].extend(batch.y.detach().cpu().numpy().flatten().tolist())
                
        for phase in phases:
            print(f"[LOG] Total graphs in '{phase}': {len(X[phase])}")
            if len(X[phase]) > 0:
                print(f"[LOG] Example graph embedding shape after diffusion for '{phase}': {X[phase][0].shape}")

        # standardize the features based on mean and std of the training data
        ss = StandardScaler()
        ss.fit(np.concatenate(X['train'], 0))
        for phase in phases:
            for i in range(len(X[phase])):
                X[phase][i] = ss.transform(X[phase][i])
        print(f"[LOG] After standardization, example graph embedding shape: {X['train'][0].shape}")

        # apply PCA if needed
        if num_pca_components > 0:
            # define dimensionality reduction method which stated in the input parameters
            # pca = PCA(n_components=num_pca_components, random_state=random_seed)
            if dim_reduction_method == 'PCA':
                print('Now applying PCA for dimensionality reduction ...')
                dim_reduction = PCA(n_components=num_pca_components, random_state=random_seed)
            elif dim_reduction_method == 'KernelPCA':
                print('Now applying KernelPCA for dimensionality reduction ...')
                dim_reduction = KernelPCA(n_components=num_pca_components, kernel='rbf', n_jobs=-1,
                                          random_state=random_seed, eigen_solver='auto')

            elif dim_reduction_method == 'Isomap':
                print('Now applying Isomap for dimensionality reduction ...')
                dim_reduction = Isomap(n_components=num_pca_components, n_jobs=-1)
            elif dim_reduction_method == 'LLE':
                print('Now applying Locally Linear Embedding (LLE) for dimensionality reduction ...')
                dim_reduction = LocallyLinearEmbedding(n_components=num_pca_components, random_state=random_seed,
                                                       n_neighbors=5, n_jobs=-1)
            else:
                raise ValueError('Unknown dimensionality reduction method: {}'.format(dim_reduction_method))
            if n_landmarks is not None and dim_reduction_method != 'PCA':
                X_landmarks = select_landmarks(np.concatenate(X['train'], 0), n_landmarks, random_seed)
                dim_reduction.fit(X_landmarks)
            else:
                dim_reduction.fit(np.concatenate(X['train'], 0))
            for phase in phases:
                print('Now transforming the ' + phase + ' data ...')
                for i in tqdm(range(len(X[phase]))):
                    X[phase][i] = dim_reduction.transform(X[phase][i])
                # all_nodes = np.concatenate(X[phase], 0)
                # node_count = [len(graph) for graph in X[phase]]
                # all_nodes_transformed = dim_reduction.transform(all_nodes)
                # X[phase] = np.split(all_nodes_transformed, np.cumsum(node_count[:-1]), axis=0)


            # plot the variance % explained by PCA components
            # plt.plot(np.arange(1, num_pca_components + 1), pca.explained_variance_ratio_, 'o--')
            # plt.grid(True)
            # plt.xlabel('Principal component')
            # plt.ylabel('Eigenvalue')
            # plt.xticks(np.arange(1, num_pca_components + 1, step=2))
            # plt.show()

        if graph_embedding_tool == 'GME':
            print('Now deriving the final graph embeddings using Gaussian-Moment Embedding (GME) ...')
            print('GME will output a vector of size 2 * F, where F is the node embedding size.')
            # === Gaussian-Moment Embedding (GME) ===
            V = defaultdict(list)
            for phase in phases:
                for x in X[phase]:                    # x: (num_nodes, F)
                    mu = x.mean(axis=0)               # (F,)
                    var = x.var(axis=0)               # (F,)  – element-wise variance
                    gme = np.concatenate([mu, var])   # (2F,)
                    V[phase].append(gme)
                V[phase] = np.stack(V[phase])
                print(f"[LOG] GME output shape for phase '{phase}': {V[phase].shape}", flush=True)
        
        else:
            print('Now deriving the final graph embeddings using LOT ...')
            # number of samples in the template distribution
            N = int(round(np.asarray([x.shape[0] for x in X['train']]).mean()))

            # derive the template distribution using K-means
            print('Now running k-means for deriving the template ...\n')
            kmeans = KMeans(n_clusters=N, verbose=verbose, random_state=random_seed)
            kmeans.fit(np.concatenate(X['train'], 0))
            template = kmeans.cluster_centers_
            
            # calculate the final graph embeddings based on LOT
            V = defaultdict(list)
            for phase in phases:
                print('Now deriving the final graph embeddings for the ' + phase + ' data ...')
                for x in tqdm(X[phase]):
                    M = x.shape[0]
                    C = ot.dist(x, template)
                    b = np.ones((N,)) / float(N)
                    a = np.ones((M,)) / float(M)
                    p = ot.emd(a,b,C) # exact linear program
                    V[phase].append(np.matmul((N * p).T, x) - template)
                V[phase] = np.stack(V[phase])
                
        # create the parameter grid for random forest
        param_grid_RF = {
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 5],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [25, 50, 100, 150, 200]
        }

        param_grid_all = {'RF': param_grid_RF}

        # load the ROC-AUC evaluator
        evaluator = Evaluator(name=dataset.name)

        # run the classifier
        print('Now running the classifiers ...')
        for classifier in classifiers:
            if classifier not in param_grid_all:
                print('Classifier {} not supported! Skipping ...'.format(classifier))
                continue

            param_grid = param_grid_all[classifier]

            # determine train and validation index split for grid search
            test_fold = [-1] * len(V['train']) + [0] * len(V['valid'])
            ps = PredefinedSplit(test_fold)

            # concatenate train and validation datasets
            X_grid_search = np.concatenate((V['train'], V['valid']), axis=0)
            X_grid_search = X_grid_search.reshape(X_grid_search.shape[0], -1)
            Y_grid_search = np.concatenate((Y['train'], Y['valid']), axis=0)
            print(f"[LOG] X_grid_search shape: {X_grid_search.shape}")
            print(f"[LOG] Y_grid_search shape: {Y_grid_search.shape}")

            results = defaultdict(list)
            for experiment in range(num_experiments):
                # Create a base model
                if classifier == 'RF':
                    model = RandomForestClassifier(n_jobs=n_jobs, class_weight='balanced',
                                                   random_state=random_seed + experiment)

                # Instantiate the grid search model
                print("[RF] Running GridSearchCV for hyperparameter tuning...")
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                          cv=ps, n_jobs=n_jobs, verbose=verbose, refit=False)

                # Fit the grid search to the data
                grid_search.fit(X_grid_search, Y_grid_search)
                print(f"[RF] GridSearchCV completed. Best params: {grid_search.best_params_}")

                # Fit the model with best parameters on the training data (again)
                for param in grid_search.best_params_:
                    setattr(model, param, grid_search.best_params_[param])
                print("[RF] Training RandomForestClassifier with best parameters on training data...")
                model.fit(V['train'].reshape(V['train'].shape[0], -1), Y['train'])
                print("[RF] Training completed.")

                # Evaluate the performance
                for phase in phases:
                    print(f"[RF] Predicting and evaluating on '{phase}' set...")
                    pred_probs = model.predict_proba(V[phase].reshape(V[phase].shape[0], -1))
                    input_dict = {'y_true': np.array(Y[phase]).reshape(-1, 1),
                                  'y_pred': pred_probs[:, 1].reshape(-1, 1)}
                    result_dict = evaluator.eval(input_dict)
                    results[phase].append(result_dict['rocauc'])
                    print(f"[RF] {phase} ROC-AUC: {result_dict['rocauc']:.4f}")

                print('experiment {0}/{1} for {2} completed ...'.format \
                          (experiment + 1, num_experiments, classifier))

            # Add the new columns to the row
            results_table.add_row([
                dim_reduction_method, graph_embedding_tool, classifier, str(L), str(F)
            ] + ['{0:.2f} $\pm$ {1:.2f}'.format(100 * np.mean(results[phase]), 100 * np.std(results[phase])) for phase in phases])

    print('\n\n' + results_table.title)
    print(results_table)
    return results_table
