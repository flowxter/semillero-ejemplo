import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt # type: ignore

# ---------------------------
# Función de optimización (Sphere)
# ---------------------------
def sphere(x):
    return sum(x**2)

# ---------------------------
# Algoritmo Evolutivo Tradicional
# ---------------------------
def evolutionary_algorithm(pop_size=30, generations=50, dim=2):
    population = np.random.uniform(-5, 5, (pop_size, dim))
    best_history = []

    for _ in range(generations):
        fitness = np.array([sphere(ind) for ind in population])
        best_history.append(fitness.min())

        # Selección: los mejores
        idx = np.argsort(fitness)
        population = population[idx[:pop_size // 2]]

        # Reproducción
        children = []
        while len(children) < pop_size // 2:
            p1, p2 = random.sample(list(population), 2)
            child = (p1 + p2) / 2
            child += np.random.normal(0, 0.1, dim)  # mutación
            children.append(child)

        population = np.vstack((population, children))

    return best_history

# ---------------------------
# Algoritmo Evolutivo guiado por red libre de escala
# ---------------------------
def evolutionary_scale_free(pop_size=30, generations=50, dim=2):
    population = np.random.uniform(-5, 5, (pop_size, dim))
    graph = nx.barabasi_albert_graph(pop_size, 2)
    best_history = []

    for _ in range(generations):
        fitness = np.array([sphere(ind) for ind in population])
        best_history.append(fitness.min())

        new_population = population.copy()

        for i in range(pop_size):
            neighbors = list(graph.neighbors(i))
            if neighbors:
                best_neighbor = min(neighbors, key=lambda j: sphere(population[j]))
                parent = population[best_neighbor]
            else:
                parent = population[i]

            child = (population[i] + parent) / 2
            child += np.random.normal(0, 0.1, dim)

            if sphere(child) < sphere(population[i]):
                new_population[i] = child

        population = new_population

    return best_history

# ---------------------------
# Ejecutar experimentos
# ---------------------------
trad = evolutionary_algorithm()
sf = evolutionary_scale_free()

# ---------------------------
# Gráfica de comparación
# ---------------------------
plt.plot(trad, label="Evolutivo Tradicional")
plt.plot(sf, label="Evolutivo Red Libre de Escala")
plt.yscale("log")
plt.xlabel("Generaciones")
plt.ylabel("Fitness")
plt.legend()
plt.title("Comparación de convergencia")
plt.show()
