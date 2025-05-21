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


class AntColonyOptimization:
    def __init__(self, problem, n_ants=10, alpha=1.0, beta=2.0, rho=0.1, 
                 q0=0.9, initial_pheromone=1.0, max_iterations=100, local_search=True):
        self.problem = problem
        self.n = problem.n
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.max_iterations = max_iterations
        self.local_search = local_search
        
        self.pheromone = np.full((self.n, self.n), self.initial_pheromone)
        
        self.eta = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.problem.cost_matrix[i][j] > 0:
                    self.eta[i][j] = 1.0 / self.problem.cost_matrix[i][j]
        
        self.best_solution = None
        self.best_cost = float('inf')
    
    def _select_next_node(self, ant, current_node, unvisited):
        feasible = []
        for node in unvisited:
            if all(pred in ant and node not in ant for pred in self.problem.predecessors[node]):
                feasible.append(node)
        
        if not feasible:
            return None
        
        if random.random() < self.q0:
            best_value = -1
            best_node = None
            for node in feasible:
                value = (self.pheromone[current_node][node] ** self.alpha) * \
                        (self.eta[current_node][node] ** self.beta)
                if value > best_value:
                    best_value = value
                    best_node = node
            return best_node
        else:
            probabilities = []
            for node in feasible:
                attr = (self.pheromone[current_node][node] ** self.alpha) * \
                       (self.eta[current_node][node] ** self.beta)
                probabilities.append((node, attr))
            
            total = sum(p[1] for p in probabilities)
            if total == 0:
                return random.choice(feasible)
            
            probabilities = [(node, attr/total) for node, attr in probabilities]
            
            r = random.random()
            cumulative = 0
            for node, prob in probabilities:
                cumulative += prob
                if r <= cumulative:
                    return node
            
            return feasible[-1]
    
    def _construct_solution(self, start_node=0):
        ant_path = [start_node]
        unvisited = set(range(self.n))
        unvisited.remove(start_node)
        current_node = start_node
        
        while unvisited:
            next_node = self._select_next_node(ant_path, current_node, unvisited)
            if next_node is None:
                break
            
            ant_path.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        
        return ant_path
    
    def _local_pheromone_update(self, i, j):
        self.pheromone[i][j] = (1 - self.rho) * self.pheromone[i][j] + self.rho * self.initial_pheromone
    
    def _global_pheromone_update(self, best_ant_path):
        best_cost = self.problem.evaluate_solution(best_ant_path)
        self.pheromone *= (1 - self.rho)
        
        for i in range(len(best_ant_path) - 1):
            node1 = best_ant_path[i]
            node2 = best_ant_path[i + 1]
            delta_pheromone = 1.0 / best_cost
            self.pheromone[node1][node2] += self.rho * delta_pheromone
    
    def _two_opt_swap(self, solution, i, j):
        new_solution = solution.copy()
        new_solution[i:j+1] = reversed(new_solution[i:j+1])
        return new_solution
    
    def _local_search(self, solution):
        improved = True
        current_solution = solution.copy()
        current_cost = self.problem.evaluate_solution(current_solution)
        
        while improved:
            improved = False
            
            for i in range(1, len(solution) - 2):
                for j in range(i + 1, len(solution) - 1):
                    new_solution = self._two_opt_swap(current_solution, i, j)
                    
                    if self.problem.is_valid_solution(new_solution):
                        new_cost = self.problem.evaluate_solution(new_solution)
                        
                        if new_cost < current_cost:
                            current_solution = new_solution
                            current_cost = new_cost
                            improved = True
                            break
                
                if improved:
                    break
        
        return current_solution
    
    def generate_valid_initial_solution(self):
        n = self.problem.n
        solution = []
        
        predecessors = {node: set(self.problem.predecessors[node]) for node in range(n)}
        
        available = [node for node in range(n) if not predecessors[node]]
        
        while len(solution) < n:
            if not available:
                raise ValueError("No valid solution exists for the given constraints")
            
            current_node = solution[-1] if solution else 0
            best_node = None
            best_cost = float('inf')
            
            for node in available:
                cost = self.problem.cost_matrix[current_node][node]
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
            
            solution.append(best_node)
            available.remove(best_node)
            
            for successor in self.problem.successors[best_node]:
                predecessors[successor].remove(best_node)
                if not predecessors[successor]:
                    available.append(successor)
        
        return solution
    
    def run(self, verbose=True):
        start_time = time.time()
        
        initial_solution = self.generate_valid_initial_solution()
        self.best_solution = initial_solution
        self.best_cost = self.problem.evaluate_solution(initial_solution)
        
        if verbose:
            print(f"Initial solution cost: {self.best_cost}")
        
        for iteration in range(self.max_iterations):
            all_solutions = []
            
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                
                for i in range(len(solution) - 1):
                    self._local_pheromone_update(solution[i], solution[i+1])
                
                if self.local_search and len(solution) == self.n:
                    solution = self._local_search(solution)
                
                if len(solution) == self.n:
                    cost = self.problem.evaluate_solution(solution)
                    all_solutions.append((solution, cost))
            
            if all_solutions:
                all_solutions.sort(key=lambda x: x[1])
                iteration_best_solution, iteration_best_cost = all_solutions[0]
                
                if iteration_best_cost < self.best_cost:
                    self.best_solution = iteration_best_solution
                    self.best_cost = iteration_best_cost
                    if verbose:
                        print(f"Iteration {iteration}: New best solution with cost {self.best_cost}")
                
                self._global_pheromone_update(iteration_best_solution)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best cost: {self.best_cost}")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        if verbose:
            print(f"\nAnt Colony Optimization completed in {computation_time:.2f} seconds")
            print(f"Best solution cost: {self.best_cost}")
        
        return self.best_solution, self.best_cost


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
    
    aco = AntColonyOptimization(
        problem,
        n_ants=20,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q0=0.9,
        initial_pheromone=0.1,
        max_iterations=100,
        local_search=True
    )
    
    best_solution, best_cost = aco.run(verbose=True)
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")


if __name__ == "__main__":
    main()