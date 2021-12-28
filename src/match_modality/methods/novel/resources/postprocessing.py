import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm

from networkx.algorithms import bipartite
from scipy import sparse



def calculate_rank(mask, logits):
    soft_logits = F.softmax(logits).detach().cpu().numpy()
    indexes_target = np.argmax(mask, axis=1)
    rank = soft_logits.argsort()
    
    list_indexes_rank = []
    for i in range(len(rank)):
        list_indexes_rank.append(np.where(indexes_target[i] == rank[i])[0][0])
        
    return list_indexes_rank

    
    
def topN_logits(logits, topn, ):
    argsort_logits = logits.argsort()
    ind = argsort_logits[:,-topn:]
    n_logits = np.zeros(logits.shape)
    n_logits[:] = -100000
    n_logits = torch.tensor(n_logits)
    
    #TODO: Vectorize!?
    for i in tqdm(range(len(ind))):
        for j in range(len(ind[0])):
            ii = ind[i][j]
            n_logits[i][ii.item()] = logits[i][ii.item()]
            
    return n_logits

def get_bipartite_matching_adjacency_matrix(raw_logits, threshold_quantile=0.995):
    #getting rid of unpromising graph connections
    weights = raw_logits.copy()
    quantile_row = np.quantile(weights, threshold_quantile, axis=0, keepdims=True)
    quantile_col = np.quantile(weights, threshold_quantile, axis=1, keepdims=True)
    quantile_minimum = np.minimum(quantile_row, quantile_col)
    weights[weights<quantile_minimum] = 0
    weights_sparse = sparse.csr_matrix(-weights)
    graph = bipartite.matrix.from_biadjacency_matrix(weights_sparse)
    #explicitly combining top nodes in once component or networkx freaks tf out
    u = [n for n in graph.nodes if graph.nodes[n]['bipartite'] == 0]
    matches = bipartite.matching.minimum_weight_full_matching(graph, top_nodes=u)
    best_matches = np.array([matches[x]-len(u) for x in u])
    bipartite_matching_adjacency = np.zeros(raw_logits.shape)
    bipartite_matching_adjacency[np.arange(raw_logits.shape[0]), best_matches]=1
    return bipartite_matching_adjacency