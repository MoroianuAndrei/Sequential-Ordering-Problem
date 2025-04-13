import numpy as np
import random
import time
from collections import defaultdict


class SequentialOrderingProblem:
    def __init__(self, cost_matrix, precedence_constraints):
        """
        Initialize the SOP with cost matrix and precedence constraints.
        
        Parameters:
        - cost_matrix: 2D array where cost_matrix[i][j] is the cost of going from node i to node j
        - precedence_constraints: list of tuples (i, j) meaning node i must be visited before node j
        """
        self.n = len(cost_matrix)
        self.cost_matrix = cost_matrix
        self.precedence_constraints = precedence_constraints
        
        # Build dependency graphs
        self.predecessors = defaultdict(set)
        self.successors = defaultdict(set)
        for i, j in precedence_constraints:
            self.predecessors[j].add(i)
            self.successors[i].add(j)
        
        # Compute transitive closure of precedence constraints
        self._compute_transitive_closure()
    
    def _compute_transitive_closure(self):
        """Compute the transitive closure of precedence constraints."""
        self.must_precede = defaultdict(set)
        self.must_follow = defaultdict(set)
        
        # Initialize with direct constraints
        for i, j in self.precedence_constraints:
            self.must_precede[i].add(j)
            self.must_follow[j].add(i)
        
        # Compute transitive closure
        for k in range(self.n):
            for i in range(self.n):
                if k in self.must_precede[i]:
                    self.must_precede[i].update(self.must_precede[k])
            for j in range(self.n):
                if k in self.must_follow[j]:
                    self.must_follow[j].update(self.must_follow[k])
    
    def evaluate_solution(self, solution):
        """
        Calculate the total cost of a solution (path).
        
        Parameters:
        - solution: a permutation of nodes (0 to n-1)
        
        Returns:
        - total cost of the path
        """
        total_cost = 0
        for i in range(len(solution) - 1):
            total_cost += self.cost_matrix[solution[i]][solution[i+1]]
        return total_cost
    
    def is_valid_solution(self, solution):
        """
        Check if a solution satisfies all precedence constraints.
        
        Parameters:
        - solution: a permutation of nodes (0 to n-1)
        
        Returns:
        - True if all constraints are satisfied, False otherwise
        """
        # Create a position map for O(1) lookups
        pos = {node: idx for idx, node in enumerate(solution)}
        
        # Check all precedence constraints
        for i, j in self.precedence_constraints:
            if pos[i] >= pos[j]:  # i should come before j
                return False
        return True


class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, elite_size=20, 
                 mutation_rate=0.01, generations=500):
        """
        Initialize GA for the Sequential Ordering Problem.
        
        Parameters:
        - problem: instance of SequentialOrderingProblem
        - population_size: number of individuals in the population
        - elite_size: number of best individuals to keep in each generation
        - mutation_rate: probability of mutation for each gene
        - generations: number of generations to run
        """
        self.problem = problem
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # Store the best solution found so far
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Build topological sort information for generating valid solutions
        self.topo_sort_info = self._build_topological_sort_info()
    
    def _build_topological_sort_info(self):
        """
        Precompute information needed for generating valid topological orderings.
        
        Returns:
        - Dictionary with info needed for generating valid solutions
        """
        n = self.problem.n
        
        # Deep copy the predecessors and successors to avoid modifying the original
        predecessors = {node: set(self.problem.predecessors[node]) for node in range(n)}
        successors = {node: set(self.problem.successors[node]) for node in range(n)}
        
        # Compute nodes with no predecessors (potential starting nodes)
        no_predecessors = [node for node in range(n) if not predecessors[node]]
        
        return {
            'predecessors': predecessors,
            'successors': successors,
            'no_predecessors': no_predecessors
        }
    
    def generate_valid_solution(self):
        """
        Generate a random valid solution that satisfies all precedence constraints
        using a modified topological sort.
        
        Returns:
        - A valid permutation of nodes
        """
        n = self.problem.n
        solution = []
        
        # Make deep copies to avoid modifying the original data
        predecessors = {node: set(self.topo_sort_info['predecessors'][node]) for node in range(n)}
        available = set(self.topo_sort_info['no_predecessors'])
        
        while len(solution) < n:
            # Choose randomly from available nodes
            if not available:
                # This should not happen with valid constraints
                raise ValueError("No valid solution exists for the given constraints")
            
            node = random.choice(list(available))
            solution.append(node)
            available.remove(node)
            
            # Update available nodes
            for successor in self.problem.successors[node]:
                predecessors[successor].remove(node)
                if not predecessors[successor]:  # All predecessors have been visited
                    available.add(successor)
        
        return solution
    
    def initialize_population(self):
        """
        Create an initial population of valid solutions.
        
        Returns:
        - List of valid permutations of nodes
        """
        return [self.generate_valid_solution() for _ in range(self.population_size)]
    
    def calculate_fitness(self, solution):
        """
        Calculate fitness of a solution (lower cost is better).
        
        Parameters:
        - solution: a permutation of nodes
        
        Returns:
        - fitness value (negative cost, as higher fitness is better in GA)
        """
        return -self.problem.evaluate_solution(solution)
    
    def rank_population(self, population):
        """
        Rank the population based on fitness.
        
        Parameters:
        - population: list of solutions
        
        Returns:
        - List of (solution, fitness) tuples sorted by fitness (descending)
        """
        fitness_results = [(solution, self.calculate_fitness(solution)) for solution in population]
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)
    
    def selection(self, ranked_population):
        """
        Select parents for crossover using roulette wheel selection.
        
        Parameters:
        - ranked_population: list of (solution, fitness) tuples sorted by fitness
        
        Returns:
        - Selected population of solutions
        """
        # Keep elites
        elites = [item[0] for item in ranked_population[:self.elite_size]]
        
        # Convert negative fitness values to positive for selection probability calculation
        # Add a constant to make all fitness values positive
        min_fitness = min(item[1] for item in ranked_population)
        adjusted_fitness = [item[1] - min_fitness + 1 for item in ranked_population]
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitness)
        selection_probs = [fit/total_fitness for fit in adjusted_fitness]
        
        # Perform selection
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
    
    def ordered_crossover(self, parent1, parent2):
        """
        Perform ordered crossover (OX) while preserving precedence constraints.
        
        Parameters:
        - parent1, parent2: parent solutions
        
        Returns:
        - Child solution that satisfies precedence constraints
        """
        n = len(parent1)
        
        # Try multiple times to generate a valid child
        for _ in range(10):  # Limit the number of attempts
            # Choose crossover points
            start = random.randint(0, n-2)
            end = random.randint(start+1, n-1)
            
            # Create the child with the segment from parent1
            child = [None] * n
            for i in range(start, end+1):
                child[i] = parent1[i]
            
            # Fill the remaining positions with nodes from parent2 in order
            remaining = [node for node in parent2 if node not in child[start:end+1]]
            j = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = remaining[j]
                    j += 1
            
            # Check if the child satisfies precedence constraints
            if self.problem.is_valid_solution(child):
                return child
        
        # If we couldn't create a valid child, return a valid solution
        return self.generate_valid_solution()
    
    def breed_population(self, mating_pool):
        """
        Create a new population through crossover.
        
        Parameters:
        - mating_pool: selected individuals for breeding
        
        Returns:
        - New population of children
        """
        children = []
        # Keep elites unchanged
        elites = mating_pool[:self.elite_size]
        
        # Create children from the mating pool
        for i in range(self.population_size - self.elite_size):
            # Select two parents randomly
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            # Create a child using crossover
            child = self.ordered_crossover(parent1, parent2)
            children.append(child)
        
        return elites + children
    
    def mutate(self, solution):
        """
        Apply swap mutation while maintaining precedence constraints.
        
        Parameters:
        - solution: a permutation of nodes
        
        Returns:
        - Mutated solution
        """
        n = len(solution)
        mutated = solution.copy()
        
        # Try to find a valid mutation
        for _ in range(10):  # Limit number of attempts
            # Choose two positions randomly for potential swap
            pos1 = random.randint(0, n-1)
            pos2 = random.randint(0, n-1)
            
            if pos1 == pos2:
                continue
            
            # Ensure pos1 < pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            
            # Check if swapping would violate any constraints
            node1 = mutated[pos1]
            node2 = mutated[pos2]
            
            # Check if node1 must precede node2
            if node2 in self.problem.must_precede[node1]:
                continue
            
            # Check if node2 must precede node1
            if node1 in self.problem.must_precede[node2]:
                continue
            
            # Swap the nodes
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
            
            # Further check if the solution is valid
            if self.problem.is_valid_solution(mutated):
                return mutated
            
            # Revert the swap if not valid
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        
        # If no valid mutation was found, return the original solution
        return solution
    
    def mutate_population(self, population):
        """
        Apply mutation to the population.
        
        Parameters:
        - population: list of solutions
        
        Returns:
        - Mutated population
        """
        mutated_population = []
        
        # Keep elites unchanged
        elites = population[:self.elite_size]
        mutated_population.extend(elites)
        
        # Apply mutation to the rest of the population based on mutation rate
        for i in range(self.elite_size, self.population_size):
            individual = population[i]
            for j in range(len(individual)):
                if random.random() < self.mutation_rate:
                    individual = self.mutate(individual)
                    break
            mutated_population.append(individual)
        
        return mutated_population
    
    def run(self, verbose=True):
        """
        Run the genetic algorithm.
        
        Parameters:
        - verbose: whether to print progress information
        
        Returns:
        - Best solution found and its fitness
        """
        start_time = time.time()
        
        # Initialize population with valid solutions
        population = self.initialize_population()
        
        # Track the best solution
        best_solution = None
        best_fitness = float('-inf')
        
        # Run for the specified number of generations
        for generation in range(self.generations):
            # Rank the population
            ranked_population = self.rank_population(population)
            
            # Update best solution
            if ranked_population[0][1] > best_fitness:
                best_fitness = ranked_population[0][1]
                best_solution = ranked_population[0][0]
                if verbose:
                    print(f"Generation {generation}: New best solution with cost {-best_fitness}")
            
            # Selection
            mating_pool = self.selection(ranked_population)
            
            # Crossover
            children = self.breed_population(mating_pool)
            
            # Mutation
            population = self.mutate_population(children)
            
            # Tracking generation progress
            if verbose and generation % 100 == 0:
                print(f"Generation {generation} completed. Best fitness: {-best_fitness}")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        if verbose:
            print(f"\nGenetic Algorithm completed in {computation_time:.2f} seconds")
            print(f"Best solution cost: {-best_fitness}")
        
        self.best_solution = best_solution
        self.best_fitness = -best_fitness  # Convert back to cost
        
        return best_solution, -best_fitness


def load_sop_file(file_path):
    """
    Load a SOP instance from a TSPLIB-like file.
    
    Parameters:
    - file_path: path to the SOP file
    
    Returns:
    - cost_matrix: 2D array of costs
    - precedence_constraints: list of (i,j) tuples meaning i must be visited before j
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse dimension
    dimension = None
    for line in lines:
        if line.strip().startswith('DIMENSION:'):
            dimension = int(line.strip().split(':')[1].strip())
            break
    
    if dimension is None:
        raise ValueError("Could not find DIMENSION in the input file")
    
    # Find the beginning of the edge weight section
    edge_weight_section_start = None
    for i, line in enumerate(lines):
        if line.strip() == "EDGE_WEIGHT_SECTION":
            edge_weight_section_start = i + 1
            break
    
    if edge_weight_section_start is None:
        raise ValueError("Could not find EDGE_WEIGHT_SECTION in the input file")
    
    # Skip the dimension line after EDGE_WEIGHT_SECTION if present
    if lines[edge_weight_section_start].strip().isdigit():
        edge_weight_section_start += 1
    
    # Parse the cost matrix
    cost_matrix = []
    i = edge_weight_section_start
    while i < len(lines) and not lines[i].strip().startswith("EOF"):
        row_values = lines[i].strip().split()
        if not row_values:  # Skip empty lines
            i += 1
            continue
            
        # Convert to integers, handling '-1' which indicates precedence
        row = [int(val) for val in row_values]
        cost_matrix.append(row)
        i += 1
    
    # Extract precedence constraints from the matrix
    # If matrix[i][j] = -1, it means j must be visited before i
    precedence_constraints = []
    for i in range(dimension):
        for j in range(dimension):
            if cost_matrix[i][j] == -1:
                precedence_constraints.append((j, i))  # j must be visited before i
    
    # Special case: if there's a value 1000000, consider it as infinity (no direct path)
    for i in range(dimension):
        for j in range(dimension):
            if cost_matrix[i][j] == 1000000:
                cost_matrix[i][j] = float('inf')
    
    return cost_matrix, precedence_constraints


def main():
    # Load the SOP instance from file
    file_path = "ft70.2.sop"
    cost_matrix, precedence_constraints = load_sop_file(file_path)
    
    # Create SOP instance
    problem = SequentialOrderingProblem(cost_matrix, precedence_constraints)
    
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        problem,
        population_size=100,
        elite_size=20,
        mutation_rate=0.01,
        generations=500
    )
    
    best_solution, best_cost = ga.run()
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")


if __name__ == "__main__":
    main()