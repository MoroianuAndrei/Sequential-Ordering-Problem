import numpy as np
import random
import time
import os
from collections import defaultdict


class SequentialOrderingProblem:
    def __init__(self, cost_matrix, precedence_constraints):
        self.n = len(cost_matrix)
        self.cost_matrix = cost_matrix
        self.precedence_constraints = precedence_constraints
        
        self.predecessors = defaultdict(set)
        self.successors = defaultdict(set)
        for i, j in precedence_constraints:
            self.predecessors[j].add(i)
            self.successors[i].add(j)

        self._compute_transitive_closure()

    def _compute_transitive_closure(self):
        self.must_precede = defaultdict(set)
        self.must_follow = defaultdict(set)
        
        for i, j in self.precedence_constraints:
            self.must_precede[i].add(j)
            self.must_follow[j].add(i)
        
        for k in range(self.n):
            for i in range(self.n):
                if k in self.must_precede[i]:
                    self.must_precede[i].update(self.must_precede[k])
            for j in range(self.n):
                if k in self.must_follow[j]:
                    self.must_follow[j].update(self.must_follow[k])

    def evaluate_solution(self, solution):
        total_cost = 0
        for i in range(len(solution) - 1):
            total_cost += self.cost_matrix[solution[i]][solution[i+1]]
        return total_cost
    
    def is_valid_solution(self, solution):
        pos = {node: idx for idx, node in enumerate(solution)}
        
        for i, j in self.precedence_constraints:
            if pos[i] >= pos[j]:
                return False
        return True


class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, elite_size=20, crossover_rate = 0.9,
                 mutation_rate=0.1, generations=500):
        self.problem = problem
        self.population_size = population_size
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        self.best_solution = None
        self.best_fitness = float('inf')
        
        self.topo_sort_info = self._build_topological_sort_info()
    
    def _build_topological_sort_info(self):
        n = self.problem.n
        
        predecessors = {node: set(self.problem.predecessors[node]) for node in range(n)}
        successors = {node: set(self.problem.successors[node]) for node in range(n)}
        
        no_predecessors = [node for node in range(n) if not predecessors[node]]
        
        return {
            'predecessors': predecessors,
            'successors': successors,
            'no_predecessors': no_predecessors
        }
    
    def generate_valid_solution(self):
        n = self.problem.n
        solution = []
        
        predecessors = {node: set(self.topo_sort_info['predecessors'][node]) for node in range(n)}
        available = set(self.topo_sort_info['no_predecessors'])
        
        while len(solution) < n:
            if not available:
                raise ValueError("No valid solution exists for the given constraints")
            
            node = random.choice(list(available))
            solution.append(node)
            available.remove(node)
            
            for successor in self.problem.successors[node]:
                predecessors[successor].remove(node)
                if not predecessors[successor]:
                    available.add(successor)
        
        return solution
    
    def initialize_population(self):
        return [self.generate_valid_solution() for _ in range(self.population_size)]
    
    def calculate_fitness(self, solution):
        return self.problem.evaluate_solution(solution)
    
    def rank_population(self, population):
        fitness_results = [(solution, self.calculate_fitness(solution)) for solution in population]
        return sorted(fitness_results, key=lambda x: x[1])
    
    def selection(self, ranked_population):
        elites = [item[0] for item in ranked_population[:self.elite_size]]
        
        fitness_value = [item[1] for item in ranked_population]

        max_fitness = max(fitness_value)
        adjusted_fitness = [max_fitness - f + 1e-6 for f in fitness_value]

        total_fitness = sum(adjusted_fitness)
        selection_probs = [fit / total_fitness for fit in adjusted_fitness]
        
        selected = []
        for _ in range(self.population_size - self.elite_size):
            pick = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if pick <= cumulative_prob:
                    selected.append(ranked_population[i][0])
                    break
        
        return elites + selected
    
    # def selection(self, ranked_population, k=3):
    #     elites = [item[0] for item in ranked_population[:self.elite_size]]
    #     selected = elites[:]
        
    #     for _ in range(self.population_size - self.elite_size):
    #         contenders = random.sample(ranked_population, k)
    #         winner = min(contenders, key=lambda x: x[1])
    #         selected.append(winner[0])
        
    #     return selected
    
    def ordered_crossover(self, parent1, parent2):
        n = len(parent1)
        
        start = random.randint(0, n-2)
        end = random.randint(start+1, n-1)
            
        child = [None] * n
        for i in range(start, end+1):
            child[i] = parent1[i]
            
        remaining = [node for node in parent2 if node not in child[start:end+1]]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = remaining[j]
                j += 1
            
        if self.problem.is_valid_solution(child):
            return child
        else:
            return self.repair_sequence(child)
    
    def repair_sequence(self, child):
        constraints = self.problem.precedence_constraints
        pos = {n: i for i, n in enumerate(child)}
        changed = True

        while changed:
            changed = False
            for a, b in constraints:
                if pos[a] > pos[b]:
                    child.remove(a)
                    index_b = child.index(b)
                    child.insert(index_b, a)
                    pos = {n: i for i, n in enumerate(child)}
                    changed = True

        return child
    
    def breed_population(self, mating_pool):
        children = []
        elites = mating_pool[:self.elite_size]
        
        for i in range(self.population_size - self.elite_size):
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            if random.random() < self.crossover_rate:
                child = self.ordered_crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])[:]
            children.append(child)
        
        return elites + children
    
    def mutate(self, solution):
        n = len(solution)
        mutated = solution.copy()
        
        for _ in range(10):
            pos1 = random.randint(0, n-1)
            pos2 = random.randint(0, n-1)
            
            if pos1 == pos2:
                continue
            
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            
            node1 = mutated[pos1]
            node2 = mutated[pos2]
            
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
            
            if self.problem.is_valid_solution(mutated):
                return mutated
            
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        
        return solution
    
    def mutate_population(self, population):
        mutated_population = []
        
        elites = population[:self.elite_size]
        mutated_population.extend(elites)
        
        for i in range(self.elite_size, self.population_size):
            individual = population[i]
            if random.random() < self.mutation_rate:
                individual = self.mutate(individual)
            mutated_population.append(individual)
        
        return mutated_population
    
    def run(self, verbose=True):
        start_time = time.time()
        
        population = self.initialize_population()
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            ranked_population = self.rank_population(population)
            
            if ranked_population[0][1] < best_fitness:
                best_fitness = ranked_population[0][1]
                best_solution = ranked_population[0][0]
                if verbose:
                    print(f"Generation {generation}: New best solution with cost {best_fitness}")
            
            mating_pool = self.selection(ranked_population)
            
            children = self.breed_population(mating_pool)
            
            population = self.mutate_population(children)
            
            if verbose and generation % 100 == 0:
                print(f"Generation {generation} completed. Best fitness: {best_fitness}")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        if verbose:
            print(f"\nGenetic Algorithm completed in {computation_time:.2f} seconds")
            print(f"Best solution cost: {best_fitness}")
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        return best_solution, best_fitness


def load_sop_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    dimension = None
    for line in lines:
        if line.strip().startswith('DIMENSION:'):
            dimension = int(line.strip().split(':')[1].strip())
            break
    
    if dimension is None:
        raise ValueError("Could not find DIMENSION in the input file")
    
    edge_weight_section_start = None
    for i, line in enumerate(lines):
        if line.strip() == "EDGE_WEIGHT_SECTION":
            edge_weight_section_start = i + 1
            break
    
    if edge_weight_section_start is None:
        raise ValueError("Could not find EDGE_WEIGHT_SECTION in the input file")
    
    if lines[edge_weight_section_start].strip().isdigit():
        edge_weight_section_start += 1
    
    cost_matrix = []
    i = edge_weight_section_start
    while i < len(lines) and not lines[i].strip().startswith("EOF"):
        row_values = lines[i].strip().split()
        if not row_values:
            i += 1
            continue
            
        row = [int(val) for val in row_values]
        cost_matrix.append(row)
        i += 1
    
    precedence_constraints = []
    for i in range(dimension):
        for j in range(dimension):
            if cost_matrix[i][j] == -1:
                precedence_constraints.append((j, i))
    
    for i in range(dimension):
        for j in range(dimension):
            if cost_matrix[i][j] == 1000000:
                cost_matrix[i][j] = float('inf')
    
    return cost_matrix, precedence_constraints


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "ESC47.sop")
    cost_matrix, precedence_constraints = load_sop_file(file_path)
    
    problem = SequentialOrderingProblem(cost_matrix, precedence_constraints)
    
    ga = GeneticAlgorithm(
        problem,
        population_size=200,
        elite_size=5,
        crossover_rate = 0.9,
        mutation_rate=0.1,
        generations=500
    )
    
    best_solution, best_cost = ga.run()
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")


if __name__ == "__main__":
    main()