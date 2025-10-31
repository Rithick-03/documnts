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
