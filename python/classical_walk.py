import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

graph = nx.Graph()
graph.add_nodes_from(range(4))
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph = nx.generators.cycle_graph(5)

def walk_me(v):
    return np.random.choice(graph[v])

visits = np.array([0] * 5)
vertex = 0
unif = np.array([1.] * 5) / 5.

data = []
nsteps = 100
for i in range(nsteps):
    visits[vertex] += 1
    vertex = walk_me(vertex)
    data.append(sum(np.abs(a - b)**2 for a, b in zip(unif, visits / sum(visits))))
plt.plot(list(range(nsteps)), data)
plt.ylabel("$||v - u||^2$")
plt.xlabel("step")
plt.savefig("plot.png")
