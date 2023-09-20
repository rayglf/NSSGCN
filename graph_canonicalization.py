import numpy as np
import networkx as nx
import breadth_first_search as bfs
import feature_maps_d as fm
from scipy.sparse import save_npz
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix


def compute_centrality(adj):#特征向量中心性
    n = adj.shape[0]
    adj = adj + np.eye(n)
    cen = np.zeros(n)
    G = nx.from_numpy_matrix(adj)
    nodes = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-4)
    for i in range(len(nodes)):
        cen[i] = nodes[i]
    cen =cen/sum(cen)
    return cen
    
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def compute_similarity(adj,ft):#语义重要性
    n = adj.shape[0]

    cen = np.zeros(n)
    #print(type(ft))
    #print(ft.shape)
    s1=np.sum(ft,0)
    for i in range(n):
        cen[i]=cosine_similarity(s1,ft[i])
    
    
    
    #G = nx.from_numpy_matrix(adj)
    #nodes = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-4)
    
    
    #for i in range(len(nodes)):
    #    cen[i] = nodes[i]
    
    #print(cen)
    
    #ccc=input('pauseccc')
    cen=cen/sum(cen)

    return cen


def canonicalization(z,ds_name, graph_data, hasnl, filter_size, feature_type, graphlet_size, max_h,feature_size):
    depth = 10
    graphs = {}
    labels = {}
    attributes = {}
    num_graphs = len(graph_data[0])
    centrality_vector = {}

    num_sample = 0

    for gidx in range(num_graphs):
        #adj = graph_data[0][gidx]['am'].toarray()
        adj = graph_data[0][gidx]['am']
        n = adj.shape[0]
        #print(n)
        if n >= num_sample:
            num_sample = n

        graphs[gidx] = adj
        #v = compute_centrality(adj)
        #centrality_vector[gidx] = v

        degree = np.sum(adj, axis=1)
        if hasnl == 0:
            labels[gidx] = degree
        else:
            label = graph_data[0][gidx]['nl'].T
            labels[gidx] = label[0]


    if  feature_type == 3:
        features = fm.wl_subtree_feature_map(num_graphs, graphs, labels, max_h,feature_size)

    else:
        raise Exception("Unknown feature type!")
    
    
    
    for gidx in range(num_graphs):
        #adj = graph_data[0][gidx]['am'].toarray()
        adj = graph_data[0][gidx]['am']
        n = adj.shape[0]
        if n >= num_sample:
            num_sample = n

        graphs[gidx] = adj
        v1 = compute_similarity(adj,features[gidx])#语义重要性
        v2=compute_centrality(adj)#特征向量中心性

        centrality_vector[gidx] = (v1*z)+(v2*(1-z))
        #centrality_vector[gidx] = v1








    for gidx in range(num_graphs):
        path_feature = features[gidx]
        attributes[gidx] = path_feature

    all_samples = {}

    for gidx in range(num_graphs):
        adj = graphs[gidx]
        nx_G = nx.from_numpy_matrix(adj)
        label = labels[gidx]
        nodetrees = []
        n = adj.shape[0]
        cen = centrality_vector[gidx]

        sorting_vertex = -1 * np.ones(num_sample)
        cen_v = np.zeros(n)
        vertex = np.zeros(n)
        for i in range(n):
            vertex[i] = i
            cen_v[i] = cen[i]
        sub = np.argsort(-cen_v)
        vertex = vertex[sub]
        if gidx==1:
            print(cen)
            print(sub)
            print(vertex)
            
        #sorted by vertex centrality

        if num_sample <= n:
            for i in range(num_sample):
                sorting_vertex[i] = vertex[i]

        else:
            for i in range(n):
                sorting_vertex[i] = vertex[i]
        if gidx==1:
            print(sorting_vertex)

        sample = []
        for node in sorting_vertex:

            if node != -1:
                edges = list(bfs.bfs_edges(nx_G, cen, source=int(node), depth_limit=depth))
                truncated_edges = edges[:filter_size - 1]
                # if gidx==1:
                #     print(edges)
                #     print(truncated_edges)
                #     print()
                
                #if not truncated_edges or len(truncated_edges) != filter_size - 1:
                #    if gidx==1:
                #        print("1111")
                #    continue
                #else:
                if 1==1:
                    tmp = []
                    tmp_cen = []
                    tmp.append(int(node))
                    tmp_cen.append(cen[int(node)])
                    for u, v in truncated_edges:
                        tmp.append(int(v))
                        tmp_cen.append(cen[int(v)])
                    tmp_cen = np.array(tmp_cen)
                    tmp_cen = -1 * tmp_cen
                    sub = np.argsort(tmp_cen)
                    tmp = np.array(tmp)
                    tmp = tmp[sub]
                    for v in tmp:
                        sample.append(v)
                    lt=len(sample)
                    for ls in range(filter_size-lt):
                        sample.append(-1)
            else:
                for i in range(filter_size):
                    sample.append(-1)

        all_samples[gidx] = sample

    #type(all_samples):dict
    #type(all_samples[0]):list, filter sequence of graph 0 padding with -1
    
    att = attributes[0]
    #print(type(att))
    feature_size = att.shape[1]
    #feature_size = len(att[1])
    

    #type(att[node,:]) csr matrix for node i
    '''
    eg. (0,0) 1.0
        (0,1) 1.0
        (0,2) 1.0
    '''
    #att.shape graph_size, feature_size
    #att[i,:] the i-th line of att(feature of node i)
    #att[0,0] the value of att[0,0]
    
    
    graph_tensor = []
    for gidx in range(num_graphs):
        sample = all_samples[gidx]
        
        att = attributes[gidx]
        #feature_matrix = csr_matrix((num_sample * filter_size, feature_size), dtype=np.float32)
        feature_matrix = np.zeros((num_sample * filter_size, feature_size), dtype=np.float32)
        pointer = 0
        for node in sample:
            if node != -1:
                feature_matrix[pointer, :] = att[node, :]

            pointer += 1

        graph_tensor.append(feature_matrix)

    print("feature_size:",feature_size)
    return graph_tensor, feature_size, num_sample,features
