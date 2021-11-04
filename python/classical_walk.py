import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

n = 5
graph = nx.generators.cycle_graph(n)

mat = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        if j in graph[i]:
            mat[i, j] = 1 / 2.
print(np.linalg.eig(mat)[0])

print(graph[0])

unif = np.array([1.] * n) / n

distrib = np.array([1] + [0] *(n-1))
data = []
nsteps = 100
for k in range(nsteps):
    distrib = mat.dot(distrib)
    data.append(np.linalg.norm(unif - distrib))
distrib = np.array([1, 1] + [0] *(n-2)) / 2

data2 = []
nsteps = 100
for k in range(nsteps):
    distrib = mat.dot(distrib)
    data2.append(np.linalg.norm(unif - distrib))

plt.plot(list(range(nsteps)), data)
plt.plot(list(range(nsteps)), data2)
plt.ylabel("$||v - u||$")
plt.xlabel("step")
plt.savefig("plot_mat_1.png")

plt.plot(list(range(nsteps)), data)
plt.plot(list(range(nsteps)), data2)
plt.ylabel("$||v - u||$")
plt.xlabel("step")
plt.savefig("plot_mat_2.png")
