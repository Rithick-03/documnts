import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load graph from file (TXT, CSV, GML)
def load_graph(file_path, file_type="gml"):
    if file_type == "gml":
        G = nx.read_gml(file_path)
    elif file_type == "txt":
        G = nx.read_edgelist(file_path)
    elif file_type == "csv":
        edges = pd.read_csv(file_path)
        G = nx.from_pandas_edgelist(edges, source='source', target='target')
    else:
        raise ValueError("Unsupported file type")
    return G

# 2. Generate random graphs
def generate_er_graph(n, p):
    return nx.erdos_renyi_graph(n, p)

def generate_scale_free_graph(n, m):
    return nx.barabasi_albert_graph(n, m)

# 3. Degree Distribution
def plot_degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=10, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()
    return degrees

# 4. Compute metrics
def compute_metrics(G):
    clustering_coeff = nx.average_clustering(G)
    avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else "Graph not connected"
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    degree_sequence = [d for n, d in G.degree()]
    degree_distribution = np.bincount(degree_sequence)
    
    # Critical threshold formula: <k^2>/<k>
    k = np.array(degree_sequence)
    critical_threshold = np.mean(k**2) / np.mean(k)
    
    print("Clustering Coefficient:", clustering_coeff)
    print("Average Path Length:", avg_path_length)
    print("Average Degree:", avg_degree)
    print("Degree Distribution:", degree_distribution)
    print("Critical Threshold:", critical_threshold)

# 5. Six Degrees of Separation
def six_degrees(G):
    if nx.is_connected(G):
        return nx.diameter(G)
    else:
        return "Graph not connected"

# Example Usage:
if __name__ == "__main__":
    #file_path = "graph_data.gml"
    #G_loaded = load_graph(file_path, file_type="gml")
    
    #compute_metrics(G_loaded)
    #plot_degree_distribution(G_loaded)
    #print("Six Degrees (Loaded Graph):", six_degrees(G_loaded))
    
    G_er = generate_er_graph(50, 0.05)
    G_sf = generate_scale_free_graph(50, 2)
    
    # Plot degree distribution
    print("Erdős–Rényi Graph Metrics:")
    compute_metrics(G_er)
    plot_degree_distribution(G_er)

    print("Scale-Free Graph Metrics:")
    compute_metrics(G_sf)
    plot_degree_distribution(G_sf)

    # Check six degrees of separation
    print("Six Degrees (ER):", six_degrees(G_er))
    print("Six Degrees (SF):", six_degrees(G_sf))
