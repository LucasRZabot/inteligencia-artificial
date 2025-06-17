import pandas as pd
import numpy as np
import random
import time
import os

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def total_distance(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1)) + euclidean(coords[route[-1]], coords[route[0]])

class GeneticAlgorithmTSP:
    def __init__(self, coords, population_size, generations, crossover_type, mutation_rate, init_method):
        self.coords = coords
        self.pop_size = population_size
        self.generations = generations
        self.crossover_type = crossover_type
        self.mutation_rate = mutation_rate
        self.init_method = init_method
        self.num_cities = len(coords)

    def initialize_population(self):
        if self.init_method == "random":
            return [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.pop_size)]
        elif self.init_method == "nearest":
            return [self.nearest_neighbor(i) for i in range(self.pop_size)]

    def nearest_neighbor(self, _):
        unvisited = list(range(self.num_cities))
        route = [unvisited.pop(random.randint(0, len(unvisited) - 1))]
        while unvisited:
            last = route[-1]
            next_city = min(unvisited, key=lambda city: euclidean(self.coords[last], self.coords[city]))
            unvisited.remove(next_city)
            route.append(next_city)
        return route

    def crossover(self, parent1, parent2):
        if self.crossover_type == "one_point":
            pt = random.randint(1, self.num_cities - 2)
            child = parent1[:pt] + [c for c in parent2 if c not in parent1[:pt]]
        elif self.crossover_type == "two_point":
            p1, p2 = sorted(random.sample(range(self.num_cities), 2))
            middle = parent1[p1:p2]
            child = [c for c in parent2 if c not in middle]
            child = child[:p1] + middle + child[p1:]
        else:  # uniform
            child = [-1] * self.num_cities
            for i in range(self.num_cities):
                child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
            seen = set()
            for i in range(self.num_cities):
                if child[i] in seen or child[i] == -1:
                    child[i] = next(c for c in range(self.num_cities) if c not in child[:i])
                seen.add(child[i])
        return child

    def mutate(self, route):
        for i in range(self.num_cities):
            if random.random() < self.mutation_rate:
                j = random.randint(0, self.num_cities - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def run(self):
        population = self.initialize_population()
        best_solution = min(population, key=lambda x: total_distance(x, self.coords))
        best_distance = total_distance(best_solution, self.coords)

        for _ in range(self.generations):
            new_population = []
            fitness = [1 / total_distance(ind, self.coords) for ind in population]
            probs = [f / sum(fitness) for f in fitness]

            for _ in range(self.pop_size):
                parents = random.choices(population, probs, k=2)
                child = self.crossover(parents[0], parents[1])
                new_population.append(self.mutate(child))

            population = new_population
            candidate = min(population, key=lambda x: total_distance(x, self.coords))
            candidate_dist = total_distance(candidate, self.coords)
            if candidate_dist < best_distance:
                best_solution, best_distance = candidate, candidate_dist

        return best_solution, best_distance

# Configurações fixas
tsp_files = [f"tsp_{i}.csv" for i in range(1, 11)]
configurations = [
    {"crossover_type": "one_point", "mutation_rate": 0.01, "init_method": "random"},
    {"crossover_type": "two_point", "mutation_rate": 0.05, "init_method": "random"},
    {"crossover_type": "uniform", "mutation_rate": 0.1, "init_method": "nearest"},
]

def run_experiments(selected_files):
    results = []
    for file in selected_files:
        df = pd.read_csv(file)
        coords = list(zip(df["X"], df["Y"]))
        for cfg in configurations:
            start = time.time()
            ga = GeneticAlgorithmTSP(
                coords, 50, 200,
                crossover_type=cfg["crossover_type"],
                mutation_rate=cfg["mutation_rate"],
                init_method=cfg["init_method"]
            )
            _, best_dist = ga.run()
            duration = time.time() - start
            results.append({
                "Arquivo": file,
                "Crossover": cfg["crossover_type"],
                "Mutação": cfg["mutation_rate"],
                "Inicialização": cfg["init_method"],
                "Distância": round(best_dist, 2),
                "Tempo (s)": round(duration, 2)
            })
    df_result = pd.DataFrame(results)
    df_result.to_csv("resultados_parciais.csv", index=False)
    print("\nResultados salvos em 'resultados_parciais.csv':")
    print(df_result)

def menu():
    print("\n==== Menu de Execução TSP com Algoritmo Genético ====\n")
    for i, file in enumerate(tsp_files):
        print(f"{i+1}. {file}")
    print("0. Sair")

    choice = input("\nDigite os números dos arquivos desejados separados por vírgula (ex: 1,3,5): ")
    if choice.strip() == "0":
        print("Encerrando...")
        return

    try:
        indices = [int(i.strip()) - 1 for i in choice.split(",") if i.strip().isdigit()]
        selected = [tsp_files[i] for i in indices if 0 <= i < len(tsp_files)]
        if not selected:
            print("Nenhum arquivo válido selecionado.")
            return
        run_experiments(selected)
    except Exception as e:
        print(f"Erro ao interpretar entrada: {e}")

# Executar menu
if __name__ == "__main__":
    menu()
