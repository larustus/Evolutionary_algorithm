import random
import time


# DNA class for individual representation
class DNA:
    def __init__(self, num_vertex):
        self.num_vertex = num_vertex
        self.genes = []
        self.fitness = 0

    def generate_genes(self):
        for _ in range(self.num_vertex):
            self.genes.append(self.generate_random_gene())

    def generate_random_gene(self):
        return random.randint(0, 1)


def selection(population):
    # applying tournament selection
    tournament_size = 5
    tournament = []
    for _ in range(tournament_size):
        tournament.append(random.choice(population))
    tournament.sort(key=lambda x: x.fitness)
    return tournament[0]


def mutation(individual, mutation_rate):
    # applying mutation
    for i in range(individual.num_vertex):
        if random.random() < mutation_rate:
            individual.genes[i] = individual.generate_random_gene()
    return individual


def fitness(individual, graph):
    visited_count = [0] * individual.num_vertex
    visited = [False] * individual.num_vertex
    for i in range(individual.num_vertex):
        if individual.genes[i] == 1:
            visited_count[i] += 1
            visited[i] = True
            for j in graph[i]:
                visited_count[j] += 1
                visited[j] = True

    # penalizing individuals if not all vertices are visited
    all_visited = True
    for i in visited:
        all_visited = all_visited and i
    if not all_visited:
        individual.fitness += 1000

    # penalizing individuals if there are vertices with more than 1 visit
    for count in visited_count:
        if count > 1:
            individual.fitness += 10

    edges = []
    for i in range(individual.num_vertex):
        for j in graph[i]:
            edges.append((i, j))
    edges_in_individual = []
    for i in range(individual.num_vertex):
        if individual.genes[i] == 1:
            for j in graph[i]:
                edges_in_individual.append((i, j))
    # penalizing individuals if not all edges are visited
    for edge in edges:
        if edge not in edges_in_individual and edge[::-1] not in edges_in_individual:
            individual.fitness += 1000

    # minimizing the number of torches used
    individual.fitness += sum(individual.genes)


def generate_initial_population(population_size, num_vertex):
    population = []
    for _ in range(population_size):
        individual = DNA(num_vertex)
        individual.generate_genes()
        population.append(individual)
    return population


def generate_new_population(population, graph, mutation_rate):
    new_population = []
    for _ in range(len(population)):
        parent = selection(population)
        child = DNA(parent.num_vertex)
        child.genes = parent.genes.copy()
        child = mutation(child, mutation_rate)
        fitness(child, graph)
        new_population.append(child)
    return new_population


def apply_genetic_algorithm(num_vertex, graph, population_size, mutation_rate, max_generations):
    population = generate_initial_population(population_size, num_vertex)
    for individual in population:
        fitness(individual, graph)
    generation = 0
    best_individual = None
    best_fitness = float('inf')
    while generation < max_generations:
        average_fitness = 0
        for individual in population:
            if individual.fitness < best_fitness:
                best_individual = individual
                best_fitness = individual.fitness
            average_fitness += individual.fitness
        average_fitness /= population_size
        print(f"Generation: {generation}, Best fitness: {best_fitness}, Average fitness: {average_fitness}")
        generation += 1
        population = generate_new_population(population, graph, mutation_rate)
    return best_individual


def generate_graph(num_vertex, num_edges):
    graph = dict()
    for i in range(num_vertex):
        graph[i] = []
    for _ in range(num_edges):
        vertex1 = random.randint(0, num_vertex - 1)
        vertex2 = random.randint(0, num_vertex - 1)
        graph[vertex1].append(vertex2)
        graph[vertex2].append(vertex1)
    return graph


if __name__ == '__main__':
    start_time = time.time()
    num_vertex = 25
    num_edges = 30
    graph = generate_graph(num_vertex, num_edges)
    population_size = 50
    mutation_rate = 0.02
    max_generations = 50
    best_individual = apply_genetic_algorithm(num_vertex, graph, population_size, mutation_rate, max_generations)
    print("\n\nNumber of lamps: ", sum(best_individual.genes))
    print("Vertices with lighted lamps: ")

    result = []
    for i in range(best_individual.num_vertex):
        if best_individual.genes[i] == 1:
            print(i)
            result.append(i)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Checking which points have not been lighted
    new_dict = {k: v for k, v in graph.items() if k not in result}

    all_values_left = list(new_dict.values())
    # print(all_values_left)

    lighted = []
    for i in range(len(new_dict)):
        for j in range(len(list(new_dict.values())[i])):
            if list(new_dict.values())[i][j] in result:
                lighted.append(list(new_dict.values())[i][j])
    # print(lighted)

    empty = True
    for i in range(len(all_values_left)):
        for j in range(len(all_values_left[i])):
            if all_values_left[i][j] not in lighted:
                empty = False
    print(empty)

