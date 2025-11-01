def er_hardcode(n,p):
  G=nx.Graph()
  G.add_nodes_from(range(n))
  for i in range(n):
    for j in range(i+1,n):
      if random.random()<p:
        G.add_edge(i,j)
  return G

G=er_hardcode(n,p_er)
import networkx as nx
G = nx.erdos_renyi_graph(10, 0.5)
print(dir(nx))
print(help(nx))
print(help(nx.DiGraph()))
print(dir(G))  # list of methods/attributes
help(G.add_edge)

#ba hardcoded
m_ba=4
flag=1
if flag==1:
        BA = nx.DiGraph()
        BA.add_nodes_from(range(n))
        # Start with a small clique (m_ba nodes, bidirectional)
        for i in range(m_ba):
            for j in range(i + 1, m_ba):
                BA.add_edge(i, j)
                BA.add_edge(j, i)
        # Add remaining nodes with preferential attachment
        for new_node in range(m_ba, n):
            degrees = [BA.out_degree(i) for i in range(new_node)]
            probs = [d / sum(degrees) if sum(degrees) > 0 else 1 / new_node for d in degrees]
            targets = random.choices(range(new_node), weights=probs, k=m_ba)
            for target in targets:
                BA.add_edge(new_node, target)
              
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import random

N = 500
p = 0.1
k = 6
m = 3

ba_g = nx.barabasi_albert_graph(N,m)
er_g = nx.erdos_renyi_graph(N,p)
ws_g = nx.watts_strogatz_graph(N,k,p)

deg = []
for _,d in ba_g.degree():
  deg.append(d)
node_sizes=[]
for _,d in ba_g.degree():
  node_sizes.append(d)

plt.figure(figsize=(10,10))
pos = nx.spring_layout(ba_g)
plt.title("Original graph")
nx.draw(ba_g,pos,node_size=node_sizes,node_color="skyblue",edge_color="grey",alpha = 0.6)
plt.show()

plt.figure(figsize=(6,6))
plt.title("Deg dist ")
plt.hist(deg, bins=30, density = True, log = True, color = "skyblue", alpha=0.6)
plt.show()

avg_d = np.mean(deg)
min_d = min(deg)
max_d = max(deg)
diameter = nx.diameter(ba_g)
density = nx.density(ba_g)
avg_cc = nx.average_clustering(ba_g)
asp = nx.average_shortest_path_length(ba_g)
assortativity = nx.degree_assortativity_coefficient(ba_g)

#centrality

between = nx.betweenness_centrality(ba_g)
page_rank = nx.pagerank(ba_g)
eig_vec = nx.eigenvector_centrality(ba_g)
closeness = nx.closeness_centrality(ba_g)
top_hubs = sorted(between, key = between.get, reverse= True)[:5]
top_hubs


plt.figure(figsize=(6,6))
deg_count = Counter(deg)
x,y = zip(*deg_count.items())
plt.scatter(x,y,alpha=0.6)
plt.title("Log-Log Degree Plot")
plt.show()

er_deg = []
ba_deg = []
for _,d in er_g.degree():
  er_deg.append(d)
for _,d in ba_g.degree():
  ba_deg.append(d)
plt.figure(figsize=(6,6))
plt.title("comparison graph")
sns.kdeplot(er_deg)
sns.kdeplot(ba_deg)
plt.show()

mean_d = np.mean(deg)
std_d = np.std(deg)
norm = np.random.normal(mean_d,std_d,N)
pois = np.random.poisson(mean_d,N)
uni = np.random.uniform(mean_d-std_d,mean_d+std_d,N)
plt.figure(figsize=(6,6))
plt.title("Comparison of basic dist with the graph")
plt.hist(deg, bins=30, density = True, log = True, color = "skyblue", alpha=0.6)
plt.hist(norm, bins=30, density = True, log = True, color = "pink", alpha=0.6)
plt.hist(pois, bins=30, density = True, log = True, color = "grey", alpha=0.6)
plt.hist(uni, bins=30, density = True, log = True, color = "purple", alpha=0.6)
plt.show()

print(ba_g.nodes())

#random nodes selection
random_nodes = list(ba_g.nodes())
random.shuffle(random_nodes)
random_nodes = random_nodes[:50]

#target attack node selection
top_hub = sorted(between, key= between.get, reverse = True)[:50]

G_random = ba_g.copy()
for i in random_nodes:
  G_random.remove_node(i)

G_target = ba_g.copy()
for i in top_hub:
  G_target.remove_node(i)

plt.figure(figsize=(10,10))
pos = nx.spring_layout(ba_g)
plt.title("Original graph")
nx.draw(ba_g,pos,node_size=node_sizes,node_color="skyblue",edge_color="grey",alpha = 0.6)
plt.show()

node_sizes=[]
for _,d in G_random.degree():
  node_sizes.append(d)
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G_random)
plt.title("Random rem graph")
nx.draw(G_random,pos,node_size=node_sizes,node_color="skyblue",edge_color="grey",alpha = 0.6)
plt.show()

node_sizes=[]
for _,d in G_target.degree():
  node_sizes.append(d)
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G_target)
plt.title("target removal graph")
nx.draw(G_target,pos,node_size=node_sizes,node_color="skyblue",edge_color="grey",alpha = 0.6)
plt.show()

def cascading_failure(G, steps=10):
    G_copy = G.copy()
    sizes = []
    for _ in range(steps):
        if len(G_copy) == 0: break
        node_to_remove = max(G_copy.degree(), key=lambda x: x[1])[0]
        G_copy.remove_node(node_to_remove)
        largest_cc = len(max(nx.connected_components(G_copy), key=len))
        sizes.append(largest_cc)
    steps=[0,1,2,3,4,5,6,7,8,9]
    plt.figure(figsize=(6,4))
    plt.plot(steps, sizes, '-o')
    plt.title("Cascading Failure Simulation (BA Network)")
    plt.xlabel("Step"); plt.ylabel("Largest Component Size")
    plt.show()

cascading_failure(ba_g, steps=10)

pagerank = nx.pagerank(ba_g)
page_rank
top_nodes = sorted(page_rank.items(), key = lambda x:x[1], reverse = True)[:10]
top_nodes

k = np.mean(deg)
k2 = np.mean(np.square(deg))
kappa = k2/k
fc = 1-(1/(kappa-1))
print(kappa)
print(fc)

import networkx as nx
from networkx_robustness import networkx_robustness as netrob
import matplotlib.pyplot as plt
N = 500  # Number of nodes
# BA: m = number of edges to attach from new node
# ER: p = probability of edge creation
# WS: k = each node connects to k nearest, p = rewiring prob

# Generate graphs
G_ba = nx.barabasi_albert_graph(N, m=3)
G_er = nx.erdos_renyi_graph(N, p=0.02)  # Approx average degree ~ N*p = 10
G_ws = nx.watts_strogatz_graph(N, k=10, p=0.1)  # Average degree 2*k=20

graphs = {
    'BA (Scale-Free)': G_ba,
    'ER (Random)': G_er,
    'WS (Small-World)': G_ws
}

# Simulate robustness for random failures (kind='random')
# Returns fractions removed and corresponding giant component sizes
fig, ax = plt.subplots(figsize=(10, 6))

for label, G in graphs.items():
    frac_removed, giant_comp_frac = rb.robustness_attack(G, kind='random', percent=100)
    ax.plot(frac_removed, giant_comp_frac, marker='o', label=label, linewidth=2)

ax.set_xlabel('Fraction of Nodes Removed')
ax.set_ylabel('Fraction in Largest Connected Component')
ax.set_title('Network Robustness to Random Node Failures')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Optional: Print percolation threshold (approx where giant_comp_frac drops below 0.5)
for label, G in graphs.items():
    frac_removed, giant_comp_frac = rb.robustness_attack(G, kind='random', percent=100)
    threshold_idx = np.where(giant_comp_frac < 0.5)[0]
    threshold = frac_removed[threshold_idx[0]] if len(threshold_idx) > 0 else 1.0
    print(f"{label} Percolation Threshold (random failures): {threshold:.2f}")
G = nx.barabasi_albert_graph(50, 2)
initial, frac, apl = netrob.simulate_random_attack(G, attack_fraction=0.2)

plt.figure(figsize=(10, 6))
plt.plot(frac, apl, marker='o', linestyle='-', color='red')
plt.title('Network Robustness under Random Attack')
plt.xlabel('Fraction of Nodes Removed')
plt.ylabel('Average Path Length of Largest Component')
plt.grid(True)
plt.show()
molloy_reed = netrob.molloy_reed(G)
molloy_reed
critical_threshold = netrob.critical_threshold(G)
print(critical_threshold)
initial, frac_random, apl_random = netrob.simulate_random_attack(G, attack_fraction=0.2)
initial, frac_degree, apl_degree = netrob.simulate_degree_attack(G, attack_fraction=0.1, weight=None)
initial, frac_betweenness, apl_betweenness = netrob.simulate_betweenness_attack(G, attack_fraction=0.1, weight=None, normalized=True, k=None, seed=None, endpoints=False)
initial, frac_closeness, apl_closeness = netrob.simulate_closeness_attack(G, attack_fraction=0.1, weight=None, u=None, wf_improved=True)
initial, frac_eigenvector, apl_eigenvector = netrob.simulate_eigenvector_attack(G, attack_fraction=0.1, weight=None, tol=1e-06, max_iter=100, nstart=None)
plt.figure(figsize=(12, 8))

plt.plot(frac_random, apl_random, marker='o', linestyle='-', color='blue', label='Random Attack')
plt.plot(frac_degree, apl_degree, marker='x', linestyle='--', color='green', label='Degree Attack')
plt.plot(frac_betweenness, apl_betweenness, marker='s', linestyle='-.', color='purple', label='Betweenness Attack')
plt.plot(frac_closeness, apl_closeness, marker='^', linestyle=':', color='orange', label='Closeness Attack')
plt.plot(frac_eigenvector, apl_eigenvector, marker='d', linestyle='-', color='red', label='Eigenvector Attack')

plt.title('Network Robustness under Different Attack Strategies')
plt.xlabel('Fraction of Nodes Removed')
plt.ylabel('Average Path Length of Largest Component')
plt.grid(True)
plt.legend()
plt.show()

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit

# Parameters
N = 10000  # Final number of nodes
m = 4      # Edges per new node
N_points = [100, 1000, N]  # Intermediate points for degree distribution
tracked_nodes = {0: 'Initial node', 99: 'Added at N=100', 999: 'Added at N=1000', 4999: 'Added at N=5000'}
degree_history = {node: [] for node in tracked_nodes}

# Initialize fully connected network with 4 nodes
G = nx.complete_graph(4)

# Record initial degrees for tracked nodes present
for node in tracked_nodes:
    if node in G:
        degree_history[node].append(G.degree(node))

# Lists for clustering vs N (compute at log-spaced points to speed up)
clustering_points = np.unique(np.logspace(np.log10(10), np.log10(N), num=20, dtype=int))
clustering_values = []
current_N_list = []

# Degree distributions at intermediate N
distributions = {}
fits = {}

# Build the network
for t in range(4, N):
    # Add new node t
    G.add_node(t)

    # Preferential attachment: choose m unique targets with prob ~ degree
    degrees = np.array([G.degree(u) for u in G.nodes()])
    probs = degrees / degrees.sum()
    targets = np.random.choice(list(G.nodes())[:-1], size=m, replace=False, p=probs[:-1]/probs[:-1].sum())  # Exclude self
    for target in targets:
        G.add_edge(t, target)

    # Track degrees
    for node in tracked_nodes:
        if node in G:
            degree_history[node].append(G.degree(node))
        else:
            # If node not added yet, append 0 (pre-addition)
            degree_history[node].append(0)

    current_N = len(G)

    # Compute clustering at specified points
    if current_N in clustering_points:
        clustering_values.append(nx.average_clustering(G))
        current_N_list.append(current_N)

    # Compute degree distribution at intermediate points
    if current_N in N_points:
        degrees = [d for n, d in G.degree()]
        distributions[current_N] = degrees
        fit = Fit(degrees, discrete=True, verbose=False)
        fits[current_N] = fit
        print(f"At N={current_N}, power-law exponent γ = {fit.power_law.alpha:.3f}")

# Final clustering
if N not in current_N_list:
    clustering_values.append(nx.average_clustering(G))
    current_N_list.append(N)

# Plot degree distributions
plt.figure(figsize=(8, 6))
for n_val, degrees in distributions.items():
    hist, bins = np.histogram(degrees, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid = hist > 0
    plt.loglog(bin_centers[valid], hist[valid], '.', label=f'N={n_val}')
plt.xlabel('Degree k')
plt.ylabel('P(k)')
plt.title('Degree Distributions at Intermediate Steps')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

# Assess convergence: As N increases, the distributions should stabilize to a power-law with γ ≈ 3. If the plots overlay more closely in the tail for larger N, they "converge."

# Plot cumulative degree distributions
plt.figure(figsize=(8, 6))
for n_val, degrees in distributions.items():
    sorted_deg = np.sort(degrees)[::-1]
    cdf = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
    plt.loglog(sorted_deg, cdf, '-', label=f'N={n_val}')
plt.xlabel('Degree k')
plt.ylabel('P(K ≥ k)')
plt.title('Cumulative Degree Distributions at Intermediate Steps')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

# Plot average clustering coefficient vs N
plt.figure(figsize=(8, 6))
plt.loglog(current_N_list, clustering_values, 'o-')
plt.xlabel('N')
plt.ylabel('<C>')
plt.title('Average Clustering Coefficient vs. N')
plt.grid(True, which='both', ls='--')
plt.show()

# Plot degree dynamics
plt.figure(figsize=(8, 6))
time_steps = np.arange(1, N + 1)  # Time t ≈ N
for node, label in tracked_nodes.items():
    plt.plot(time_steps[:len(degree_history[node])], degree_history[node], label=label)
plt.xlabel('Time t (≈ N)')
plt.ylabel('Degree k')
plt.title('Degree Dynamics of Selected Nodes')
plt.legend()
plt.grid(True)
plt.show()

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

# Parameters
N = 10000  # Number of nodes
min_degree = 4  # Minimum degree for power-law with cutoff
gamma = 2.5  # Power-law exponent
lambda_ = 50  # Exponential cutoff parameter
mu = 1  # Lognormal mean (log scale)
sigma = 1  # Lognormal std (log scale)
k0 = 4  # Degree for delta distribution

# Function to generate degree sequence for power-law with exponential cutoff
def generate_powerlaw_cutoff(N, gamma, lambda_, min_degree):
    """Generate degree sequence from P(k) ~ k^-gamma * exp(-k/lambda)."""
    max_degree = int(min_degree * 100)  # Cap for practicality
    k = np.arange(min_degree, max_degree + 1)
    probs = k**(-gamma) * np.exp(-k/lambda_)
    probs /= probs.sum()
    degrees = np.random.choice(k, size=N, p=probs)
    if sum(degrees) % 2 != 0:  # Ensure sum is even for valid graph
        degrees[-1] += 1
    return degrees

# Function to generate lognormal degree sequence
def generate_lognormal(N, mu, sigma, min_degree):
    """Generate degree sequence from lognormal distribution."""
    degrees = np.random.lognormal(mu, sigma, N).astype(int)
    degrees = np.maximum(degrees, min_degree)  # Enforce min degree
    max_degree = int(min_degree * 100)
    degrees = np.minimum(degrees, max_degree)  # Cap for practicality
    if sum(degrees) % 2 != 0:
        degrees[-1] += 1
    return degrees

# Function to compute moments and theoretical fc
def compute_theoretical_fc(degrees, distribution_type, params=None):
    """Compute <k>, <k^2>, and theoretical fc."""
    degrees = np.array(degrees)  # Convert to NumPy array
    k_mean = np.mean(degrees)
    k2_mean = np.mean(degrees**2)
    kappa = k2_mean / k_mean
    fc = 1 - 1 / (kappa - 1) if kappa > 2 else 0
    return k_mean, k2_mean, fc

# Function to simulate random node removal and find empirical fc
def simulate_percolation(G, steps=50, threshold=0.01):
    """Simulate random node removal and find empirical fc."""
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []
    for f in fractions:
        G_copy = G.copy()
        nodes_to_remove = np.random.choice(list(G_copy.nodes()), size=int(f * N), replace=False)
        G_copy.remove_nodes_from(nodes_to_remove)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = list(nx.connected_components(G_copy))
            largest_components.append(max(len(c) for c in components) / N)

    # Find empirical fc where largest component < threshold
    for i, size in enumerate(largest_components):
        if size <= threshold:
            return fractions[i], fractions, largest_components
    return 1.0, fractions, largest_components

# Generate networks
# 1. Power-law with exponential cutoff
pl_degrees = generate_powerlaw_cutoff(N, gamma, lambda_, min_degree)
G_pl = nx.configuration_model(pl_degrees, create_using=nx.Graph)
G_pl = nx.Graph(G_pl)  # Remove multi-edges/self-loops

# 2. Lognormal
ln_degrees = generate_lognormal(N, mu, sigma, min_degree)
G_ln = nx.configuration_model(ln_degrees, create_using=nx.Graph)
G_ln = nx.Graph(G_ln)

# 3. Delta (k-regular graph)
G_delta = nx.random_regular_graph(k0, N)

# Compute theoretical fc
pl_k_mean, pl_k2_mean, pl_fc = compute_theoretical_fc(pl_degrees, 'powerlaw_cutoff')
ln_k_mean, ln_k2_mean, ln_fc = compute_theoretical_fc(ln_degrees, 'lognormal')
delta_k_mean, delta_k2_mean, delta_fc = compute_theoretical_fc([k0] * N, 'delta')

# Simulate empirical fc
pl_fc_emp, pl_fracs, pl_sizes = simulate_percolation(G_pl)
ln_fc_emp, ln_fracs, ln_sizes = simulate_percolation(G_ln)
delta_fc_emp, delta_fracs, delta_sizes = simulate_percolation(G_delta)

# Print results
print("=== Critical Thresholds for Random Failure ===")
print("\nPower-law with Exponential Cutoff:")
print(f"Theoretical fc: {pl_fc:.3f}, Empirical fc: {pl_fc_emp:.3f}")
print(f"<k>: {pl_k_mean:.2f}, <k^2>: {pl_k2_mean:.2f}, kappa: {pl_k2_mean/pl_k_mean:.2f}")

print("\nLognormal:")
print(f"Theoretical fc: {ln_fc:.3f}, Empirical fc: {ln_fc_emp:.3f}")
print(f"<k>: {ln_k_mean:.2f}, <k^2>: {ln_k2_mean:.2f}, kappa: {ln_k2_mean/ln_k_mean:.2f}")

print("\nDelta (k0=4):")
print(f"Theoretical fc: {delta_fc:.3f}, Empirical fc: {delta_fc_emp:.3f}")
print(f"<k>: {delta_k_mean:.2f}, <k^2>: {delta_k2_mean:.2f}, kappa: {delta_k2_mean/delta_k_mean:.2f}")

# Plot percolation curves
plt.figure(figsize=(8, 6))
plt.plot(pl_fracs, pl_sizes, 'b-', label=f'Power-law Cutoff (fc={pl_fc:.2f})')
plt.plot(ln_fracs, ln_sizes, 'g-', label=f'Lognormal (fc={ln_fc:.2f})')
plt.plot(delta_fracs, delta_sizes, 'r-', label=f'Delta k={k0} (fc={delta_fc:.2f})')
plt.axhline(y=0.01, color='k', linestyle='--', alpha=0.5, label='Threshold')
plt.xlabel('Fraction of Nodes Removed (f)')
plt.ylabel('Normalized Largest Component Size')
plt.title('Percolation for Different Degree Distributions')
plt.legend()
plt.grid(True)
plt.show()

#criticial threshold - generate 3 netwrks

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000  # Number of nodes
gamma = 2.2  # Power-law exponent
k_min = 2  # Minimum degree
rewiring_steps = 100000  # Number of rewiring steps for XB&S algorithm

# Function to generate power-law degree sequence
def generate_powerlaw_degrees(N, gamma, k_min):
    """Generate degree sequence from P(k) ~ k^-gamma."""
    max_degree = int(N**(1/(gamma-1)))  # Finite-size cutoff ~ N^(1/(gamma-1))
    k = np.arange(k_min, max_degree + 1)
    probs = k**(-gamma)
    probs /= probs.sum()
    degrees = np.random.choice(k, size=N, p=probs)
    if sum(degrees) % 2 != 0:  # Ensure sum is even
        degrees[-1] += 1
    return degrees

# Xulvi-Brunet & Sokolov algorithm for assortative/disassortative rewiring
def xbs_rewiring(G, assortative=True, steps=100000):
    """Rewire network to increase (assortative=True) or decrease (assortative=False) assortativity."""
    G_copy = G.copy()
    for _ in range(steps):
        # Select two random edges
        edges = list(G_copy.edges())
        if len(edges) < 2:
            break
        e1, e2 = np.random.choice(len(edges), size=2, replace=False)
        u, v = edges[e1]
        x, y = edges[e2]
        # Ensure distinct nodes
        if u in (x, y) or v in (x, y):
            continue
        # Degrees
        du, dv, dx, dy = G_copy.degree(u), G_copy.degree(v), G_copy.degree(x), G_copy.degree(y)
        if assortative:
            # Swap to increase assortativity: connect similar-degree pairs
            if abs(du - dy) + abs(dv - dx) < abs(du - dv) + abs(dx - dy):
                G_copy.remove_edges_from([(u, v), (x, y)])
                G_copy.add_edges_from([(u, y), (x, v)])
        else:
            # Swap to decrease assortativity: connect dissimilar-degree pairs
            if abs(du - dx) + abs(dv - dy) < abs(du - dv) + abs(dx - dy):
                G_copy.remove_edges_from([(u, v), (x, y)])
                G_copy.add_edges_from([(u, x), (v, y)])
    return G_copy

# Simulate random node removal
def simulate_percolation(G, steps=20):
    """Simulate random node removal and return f, P_infty(f)/P_infty(0)."""
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []
    for f in fractions:
        G_copy = G.copy()
        nodes_to_remove = np.random.choice(list(G_copy.nodes()), size=int(f * N), replace=False)
        G_copy.remove_nodes_from(nodes_to_remove)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = list(nx.connected_components(G_copy))
            largest_components.append(max(len(c) for c in components) / N)
    return fractions, largest_components

# Generate degree sequence
degrees = generate_powerlaw_degrees(N, gamma, k_min)

# Create networks
# 1. Neutral (configuration model)
G_neutral = nx.configuration_model(degrees, create_using=nx.Graph)
G_neutral = nx.Graph(G_neutral)  # Remove multi-edges/self-loops

# 2. Assortative
G_assortative = xbs_rewiring(G_neutral, assortative=True, steps=rewiring_steps)

# 3. Disassortative
G_disassortative = xbs_rewiring(G_neutral, assortative=False, steps=rewiring_steps)

# Compute assortativity coefficients
r_neutral = nx.degree_assortativity_coefficient(G_neutral)
r_assortative = nx.degree_assortativity_coefficient(G_assortative)
r_disassortative = nx.degree_assortativity_coefficient(G_disassortative)

# Simulate percolation for each network
fracs_neutral, sizes_neutral = simulate_percolation(G_neutral)
fracs_assort, sizes_assort = simulate_percolation(G_assortative)
fracs_disassort, sizes_disassort = simulate_percolation(G_disassortative)

# Print assortativity
print("=== Assortativity Coefficients ===")
print(f"Neutral: r = {r_neutral:.3f}")
print(f"Assortative: r = {r_assortative:.3f}")
print(f"Disassortative: r = {r_disassortative:.3f}")

# Plot percolation curves
plt.figure(figsize=(8, 6))
plt.plot(fracs_neutral, sizes_neutral, 'b-', label=f'Neutral (r={r_neutral:.2f})')
plt.plot(fracs_assort, sizes_assort, 'g-', label=f'Assortative (r={r_assortative:.2f})')
plt.plot(fracs_disassort, sizes_disassort, 'r-', label=f'Disassortative (r={r_disassortative:.2f})')
plt.xlabel('Fraction of Nodes Removed (f)')
plt.ylabel('P_infty(f) / P_infty(0)')
plt.title('Robustness of Correlated Power-Law Networks (γ=2.2)')
plt.legend()
plt.grid(True)
plt.show()

# Determine critical thresholds empirically (where P_infty drops below 0.01)
def find_empirical_fc(fractions, sizes, threshold=0.01):
    for i, size in enumerate(sizes):
        if size <= threshold:
            return fractions[i]
    return 1.0

fc_neutral = find_empirical_fc(fracs_neutral, sizes_neutral)
fc_assort = find_empirical_fc(fracs_assort, sizes_assort)
fc_disassort = find_empirical_fc(fracs_disassort, sizes_disassort)

print("\n=== Empirical Critical Thresholds ===")
print(f"Neutral: fc = {fc_neutral:.3f}")
print(f"Assortative: fc = {fc_assort:.3f}")
print(f"Disassortative: fc = {fc_disassort:.3f}")

import networkx as nx # CONSPIRACY IN SOCIAL NETWORK - BIG BROTHER
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_config = 15625  # To match hierarchical N for comparison (5**6 = 15625; close to 10^4)
gamma = 2.5  # Power-law exponent for configuration model
k_min = 3  # Minimum degree to ensure connectivity
steps = 20  # Number of fractions for removal simulation

# Function to generate power-law degree sequence for configuration model
def generate_powerlaw_degrees(N, gamma, k_min):
    max_degree = int(N ** (1 / (gamma - 1)))  # Finite-size cutoff
    k = np.arange(k_min, max_degree + 1)
    probs = k ** (-gamma)
    probs /= probs.sum()
    degrees = np.random.choice(k, size=N, p=probs)
    if sum(degrees) % 2 != 0:
        degrees[-1] += 1
    return degrees

# Function to generate hierarchical network (Ravasz-Barabási model)
def generate_hierarchical(level):
    if level == 1:
        G = nx.complete_graph(5)
        central = 0
        external = [1, 2, 3, 4]
        return G, central, external

    G_prev, central_prev, external_prev = generate_hierarchical(level - 1)
    G = G_prev.copy()
    node_offset = len(G)
    replicas_offsets = []
    new_external = []
    for _ in range(4):
        H = G_prev.copy()
        mapping = {u: u + node_offset for u in H.nodes()}
        H = nx.relabel_nodes(H, mapping)
        G = nx.union(G, H)
        replicas_offsets.append(node_offset)
        rep_external = [e + node_offset for e in external_prev]
        new_external += rep_external
        for ex in rep_external:
            G.add_edge(ex, central_prev)
        node_offset += len(G_prev)
    new_central = central_prev
    return G, new_central, new_external

# Function to simulate targeted removal
def simulate_targeted_removal(G, metric_key, steps=20):
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []
    # Compute metrics
    if metric_key == 'degree':
        metrics = {n: d for n, d in G.degree()}
    elif metric_key == 'clustering':
        metrics = nx.clustering(G)
    # Sort nodes by metric descending
    sorted_nodes = sorted(G.nodes(), key=lambda n: metrics[n], reverse=True)
    for f in fractions:
        num_remove = int(f * N)
        remove_list = sorted_nodes[:num_remove]
        G_copy = G.copy()
        G_copy.remove_nodes_from(remove_list)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = list(nx.connected_components(G_copy))
            largest_components.append(max(len(c) for c in components) / N)
    return fractions, largest_components

# Generate configuration model network
config_degrees = generate_powerlaw_degrees(N_config, gamma, k_min)
G_config = nx.configuration_model(config_degrees, create_using=nx.Graph)
G_config = nx.Graph(G_config)  # Remove multi-edges/self-loops

# Generate hierarchical network (level 6, N=15625)
G_hier, _, _ = generate_hierarchical(6)

# Simulate attacks on configuration model
config_fracs_deg, config_sizes_deg = simulate_targeted_removal(G_config, 'degree', steps)
config_fracs_clus, config_sizes_clus = simulate_targeted_removal(G_config, 'clustering', steps)

# Simulate attacks on hierarchical network
hier_fracs_deg, hier_sizes_deg = simulate_targeted_removal(G_hier, 'degree', steps)
hier_fracs_clus, hier_sizes_clus = simulate_targeted_removal(G_hier, 'clustering', steps)

# Plot results for configuration model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(config_fracs_deg, config_sizes_deg, 'r-', label='High Degree Removal')
plt.plot(config_fracs_clus, config_sizes_clus, 'b-', label='High Clustering Removal')
plt.xlabel('Fraction Removed (f)')
plt.ylabel('Giant Component Size S(f)/N')
plt.title('Configuration Model (γ=2.5)')
plt.legend()
plt.grid(True)

# Plot results for hierarchical model
plt.subplot(1, 2, 2)
plt.plot(hier_fracs_deg, hier_sizes_deg, 'r-', label='High Degree Removal')
plt.plot(hier_fracs_clus, hier_sizes_clus, 'b-', label='High Clustering Removal')
plt.xlabel('Fraction Removed (f)')
plt.ylabel('Giant Component Size S(f)/N')
plt.title('Hierarchical Model')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Analysis (as comments):
# For both networks, high degree removal (red) reduces the giant component faster than high clustering removal (blue).
# Thus, degree is more sensitive topological information; protecting high-degree individuals limits damage best, as removing them fragments the network
