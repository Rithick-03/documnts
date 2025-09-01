import networkx as nx
import matplotlib.pyplot as plt
import csv
import math


def generate_erdos_renyi(n, p):
    """
    Generates a random Erdos-Renyi (G(n,p)) graph.
    n: The number of nodes in the graph.
    p: The probability of an edge between any two nodes.
    """
    print(f"Generating an Erdos-Renyi graph with n={n} nodes and p={p} probability.")
    return nx.erdos_renyi_graph(n, p)

def generate_barabasi_albert(n, m):
    """
    Generates a scale-free graph using the Barabasi-Albert model.
    n: The number of nodes in the graph.
    m: The number of edges to attach from a new node to existing nodes.
    """
    print(f"Generating a scale-free (Barabasi-Albert) graph with n={n} nodes and m={m} edges.")
    return nx.barabasi_albert_graph(n, m)

def create_graph_from_file(filepath):
    """
    Builds a network graph from a specified file (assumes an edge list format).
    The function tries to guess the file format based on the extension.
    Supported formats: .gml (GML), .csv (CSV with 'source,target' columns).
    """
    print(f"\nAttempting to load graph from file: {filepath}")
    graph = None
    try:
        if filepath.lower().endswith('.gml'):
            # GML files are typically used for network data
            graph = nx.read_gml(filepath)
            print("Successfully loaded graph from GML file.")
        elif filepath.lower().endswith('.csv'):
            # Assuming a simple CSV edge list with two columns
            graph = nx.Graph()
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Assuming row has at least two elements for a basic edge
                    if len(row) >= 2:
                        u, v = row[0].strip(), row[1].strip()
                        graph.add_edge(u, v)
            print("Successfully loaded graph from CSV file.")
        else:
            print("Unsupported file format. Please use a .gml or .csv file.")
            return None
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    return graph

# --- 2. Network Analysis and Measures ---

def analyze_network(graph):
    """
    Calculates and prints key measures for a given network graph.
    This function covers the requirements mentioned in the notes.
    """
    if not graph:
        print("Cannot analyze an empty graph.")
        return

    print("\n--- Network Measures Analysis ---")

    # Number of nodes and edges
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"1. Number of Nodes (N): {num_nodes}")
    print(f"2. Number of Edges (E): {num_edges}")

    # Average Degree
    average_degree = sum(dict(graph.degree()).values()) / num_nodes
    print(f"3. Average Degree (<k>): {average_degree:.2f}")

    # Degree Distribution
    # This involves calculating the frequency of each degree and visualizing it.
    print("\n4. Degree Distribution:")
    degree_counts = nx.degree_histogram(graph)
    degrees = range(len(degree_counts))
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, degree_counts)
    plt.title("Degree Distribution")
    plt.xlabel("Degree (k)")
    plt.ylabel("Number of Nodes")
    plt.show()
    print("   (A plot of the degree distribution has been generated.)")

    # Clustering Coefficient
    # Note: `nx.average_clustering` calculates the average over all nodes.
    try:
        avg_clustering_coeff = nx.average_clustering(graph)
        print(f"\n5. Average Clustering Coefficient (C): {avg_clustering_coeff:.4f}")
    except nx.NetworkXPointlessConcept as e:
        print(f"\n5. Average Clustering Coefficient: {e}")

    # Average Path Length (related to Six Degrees of Separation)
    # Note: This is only valid for a connected graph.
    print("\n6. Average Path Length:")
    if nx.is_connected(graph):
        avg_path_length = nx.average_shortest_path_length(graph)
        print(f"   Average Shortest Path Length (L): {avg_path_length:.4f}")
    else:
        print("   Graph is not connected. Average path length is calculated for each component.")
        avg_path_lengths = []
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            if subgraph.number_of_nodes() > 1:
                path_length = nx.average_shortest_path_length(subgraph)
                avg_path_lengths.append(path_length)
        if avg_path_lengths:
            print(f"   Average path lengths for components: {avg_path_lengths}")
            print(f"   Overall average: {sum(avg_path_lengths) / len(avg_path_lengths):.4f}")
        else:
            print("   The graph has no connected components with more than one node.")

    # Finding 'e' and 'c' as per the notes (Edges and Connected Components)
    num_edges = graph.number_of_edges()
    num_components = nx.number_connected_components(graph)
    print(f"\n7. Number of edges (e): {num_edges}")
    print(f"8. Number of connected components (c): {num_components}")

# --- 3. Critical Threshold for Erdos-Renyi Graph ---

def find_critical_threshold(n):
    """
    Calculates the theoretical critical threshold for an Erdos-Renyi graph.
    The critical threshold for a random graph to transition from a disjoint
    set of nodes to a single giant connected component is when the average
    degree (<k>) is approximately 1. This corresponds to a probability p = 1/n.
    """
    critical_p = 1.0 / n
    print(f"\n--- Critical Threshold Analysis ---")
    print(f"For a graph with N={n} nodes, the theoretical critical probability (p_c) is approximately 1/N.")
    print(f"Therefore, the critical threshold for p is: {critical_p:.4f}")
    return critical_p

# --- Main execution block ---
if __name__ == "__main__":
    print("Welcome to the Network Science Lab Test Preparation Script.")

    # --- Q1: Erdos-Renyi / Scale-Free Network Analysis ---
    print("\n--- Question 1: Erdos-Renyi and Scale-Free Networks ---")
    
    # 1a. Example of an Erdos-Renyi graph
    n_er = 100
    p_er = 0.05
    er_graph = generate_erdos_renyi(n_er, p_er)
    analyze_network(er_graph)

    # 1b. Example of a Scale-Free network
    n_sf = 100
    m_sf = 2
    sf_graph = generate_barabasi_albert(n_sf, m_sf)
    analyze_network(sf_graph)

    # --- Q2: Build Network from Data and Find Measures ---
    print("\n--- Question 2: Build Network from File and Analyze ---")
    
    # Create a dummy CSV file for demonstration
    dummy_csv_file = "sample_data.csv"
    print(f"Creating a dummy data file '{dummy_csv_file}' for demonstration...")
    with open(dummy_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["NodeA", "NodeB"])
        writer.writerow(["1", "2"])
        writer.writerow(["1", "3"])
        writer.writerow(["2", "3"])
        writer.writerow(["4", "5"])
        writer.writerow(["6", "7"])
    print("Dummy file created.")

    # Use the function to load the dummy file
    graph_from_file = create_graph_from_file(dummy_csv_file)
    if graph_from_file:
        print("Graph loaded successfully from file. Now analyzing...")
        analyze_network(graph_from_file)

    # --- Q3: Critical Threshold Question ---
    print("\n--- Question 3: Critical Threshold ---")
    find_critical_threshold(1000) # Example for a graph of 1000 nodes

    # --- Q4: Six Degrees of Separation ---
    # The 'analyze_network' function already calculates the average path length,
    # which is the formal definition of 'six degrees of separation'.
    print("\n--- Question 4: Six Degrees of Separation ---")
    print("The 'Six Degrees of Separation' concept is a colloquial term for the average shortest path length.")
    print("The average path length for the last analyzed graph is already printed above.")

    # --- Q5: Find e and c from a generated network ---
    # This is also covered in the 'analyze_network' function.
    print("\n--- Question 5: Finding 'e' and 'c' ---")
    print("The number of edges (e) and connected components (c) for the generated graphs")
    print("are included in the network analysis output above.")
    

    if graph_from_file:
        e = graph_from_file.number_of_edges()
        c = nx.number_connected_components(graph_from_file)
        print(f"For the 'sample_data.csv' graph, edges (e) = {e} and components (c) = {c}.")