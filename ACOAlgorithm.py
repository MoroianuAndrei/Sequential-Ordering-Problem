import numpy as np
import time
import random
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


class AntColonyOptimization:
    def __init__(self, problem, n_ants=10, alpha=1.0, beta=2.0, rho=0.5, 
                 q0=0.9, initial_pheromone=1.0, max_iterations=100, local_search=True):
        """
        Initialize the ACO algorithm for SOP.
        
        Parameters:
        - problem: instance of SequentialOrderingProblem
        - n_ants: number of ants
        - alpha: relative importance of pheromone
        - beta: relative importance of heuristic information
        - rho: pheromone evaporation rate (0 to 1)
        - q0: exploitation/exploration balance (0 to 1)
        - initial_pheromone: initial pheromone level
        - max_iterations: maximum number of iterations
        - local_search: whether to apply local search to improve solutions
        """
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
        
        # Initialize pheromone matrix
        self.pheromone = np.full((self.n, self.n), self.initial_pheromone)
        
        # Heuristic information (eta) - inverse of cost
        self.eta = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.problem.cost_matrix[i][j] > 0:
                    self.eta[i][j] = 1.0 / self.problem.cost_matrix[i][j]
        
        # Best solution found so far
        self.best_solution = None
        self.best_cost = float('inf')
    
    def _select_next_node(self, ant, current_node, unvisited):
        """
        Select the next node for an ant using the ACO decision rule.
        
        Parameters:
        - ant: the ant making the decision
        - current_node: the current node of the ant
        - unvisited: set of unvisited nodes
        
        Returns:
        - next node to visit
        """
        # Filter unvisited nodes based on precedence constraints
        # Only consider nodes whose predecessors have all been visited
        feasible = []
        for node in unvisited:
            if all(pred in ant and node not in ant for pred in self.problem.predecessors[node]):
                feasible.append(node)
        
        if not feasible:
            # If no feasible nodes, this is a dead end (should not happen with valid constraints)
            return None
        
        # Exploitation vs exploration decision
        if random.random() < self.q0:
            # Exploitation: choose the best node
            best_value = -1
            best_node = None
            for node in feasible:
                # Consider pheromone and heuristic information
                value = (self.pheromone[current_node][node] ** self.alpha) * \
                        (self.eta[current_node][node] ** self.beta)
                if value > best_value:
                    best_value = value
                    best_node = node
            return best_node
        else:
            # Exploration: probabilistic choice
            probabilities = []
            for node in feasible:
                # Calculate attractiveness
                attr = (self.pheromone[current_node][node] ** self.alpha) * \
                       (self.eta[current_node][node] ** self.beta)
                probabilities.append((node, attr))
            
            # Calculate total attractiveness for normalization
            total = sum(p[1] for p in probabilities)
            if total == 0:
                # If all attractiveness values are zero, choose randomly
                return random.choice(feasible)
            
            # Normalize probabilities
            probabilities = [(node, attr/total) for node, attr in probabilities]
            
            # Select a node based on probabilities
            r = random.random()
            cumulative = 0
            for node, prob in probabilities:
                cumulative += prob
                if r <= cumulative:
                    return node
            
            # If we reach here, return the last feasible node
            return feasible[-1]
    
    def _construct_solution(self, start_node=0):
        """
        Construct a solution for a single ant.
        
        Parameters:
        - start_node: the starting node
        
        Returns:
        - constructed solution
        """
        ant_path = [start_node]
        unvisited = set(range(self.n))
        unvisited.remove(start_node)
        current_node = start_node
        
        while unvisited:
            next_node = self._select_next_node(ant_path, current_node, unvisited)
            if next_node is None:
                # No feasible next node, try to repair the solution
                break
            
            ant_path.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        
        return ant_path
    
    def _local_pheromone_update(self, i, j):
        """
        Perform local pheromone update for edge (i, j).
        
        Parameters:
        - i, j: nodes forming the edge
        """
        self.pheromone[i][j] = (1 - self.rho) * self.pheromone[i][j] + self.rho * self.initial_pheromone
    
    def _global_pheromone_update(self, best_ant_path):
        """
        Perform global pheromone update based on the best ant's path.
        
        Parameters:
        - best_ant_path: the best solution found in the current iteration
        """
        # Calculate the cost of the best path
        best_cost = self.problem.evaluate_solution(best_ant_path)
        
        # Update pheromone on the edges of the best path
        for i in range(len(best_ant_path) - 1):
            node1 = best_ant_path[i]
            node2 = best_ant_path[i + 1]
            delta_pheromone = 1.0 / best_cost
            self.pheromone[node1][node2] = (1 - self.rho) * self.pheromone[node1][node2] + self.rho * delta_pheromone
    
    def _two_opt_swap(self, solution, i, j):
        """
        Perform a 2-opt swap: reverse the segment between positions i and j.
        
        Parameters:
        - solution: current solution
        - i, j: positions to swap
        
        Returns:
        - new solution with the segment reversed
        """
        new_solution = solution.copy()
        # Reverse the segment
        new_solution[i:j+1] = reversed(new_solution[i:j+1])
        return new_solution
    
    def _local_search(self, solution):
        """
        Apply 2-opt local search to improve the solution.
        
        Parameters:
        - solution: initial solution
        
        Returns:
        - improved solution
        """
        improved = True
        current_solution = solution.copy()
        current_cost = self.problem.evaluate_solution(current_solution)
        
        while improved:
            improved = False
            
            # Try all possible 2-opt swaps
            for i in range(1, len(solution) - 2):
                for j in range(i + 1, len(solution) - 1):
                    # Perform swap
                    new_solution = self._two_opt_swap(current_solution, i, j)
                    
                    # Check if the new solution is valid
                    if self.problem.is_valid_solution(new_solution):
                        new_cost = self.problem.evaluate_solution(new_solution)
                        
                        # If the new solution is better, accept it
                        if new_cost < current_cost:
                            current_solution = new_solution
                            current_cost = new_cost
                            improved = True
                            break
                
                if improved:
                    break
        
        return current_solution
    
    def generate_valid_initial_solution(self):
        """
        Generate a valid initial solution using a greedy approach.
        
        Returns:
        - A valid permutation of nodes
        """
        n = self.problem.n
        solution = []
        
        # Make deep copies to avoid modifying the original data
        predecessors = {node: set(self.problem.predecessors[node]) for node in range(n)}
        
        # Start with nodes that have no predecessors
        available = [node for node in range(n) if not predecessors[node]]
        
        while len(solution) < n:
            if not available:
                # This should not happen with valid constraints
                raise ValueError("No valid solution exists for the given constraints")
            
            # Choose the node with lowest cost from current node
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
            
            # Update available nodes
            for successor in self.problem.successors[best_node]:
                predecessors[successor].remove(best_node)
                if not predecessors[successor]:  # All predecessors have been visited
                    available.append(successor)
        
        return solution
    
    def run(self, verbose=True):
        """
        Run the ACO algorithm.
        
        Parameters:
        - verbose: whether to print progress information
        
        Returns:
        - Best solution found and its cost
        """
        start_time = time.time()
        
        # Initialize with a greedy solution
        initial_solution = self.generate_valid_initial_solution()
        self.best_solution = initial_solution
        self.best_cost = self.problem.evaluate_solution(initial_solution)
        
        if verbose:
            print(f"Initial solution cost: {self.best_cost}")
        
        # Main ACO loop
        for iteration in range(self.max_iterations):
            # Solutions found by ants in this iteration
            all_solutions = []
            
            # Each ant constructs a solution
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                
                # Apply local pheromone update
                for i in range(len(solution) - 1):
                    self._local_pheromone_update(solution[i], solution[i+1])
                
                # Apply local search if enabled
                if self.local_search and len(solution) == self.n:
                    solution = self._local_search(solution)
                
                # Check if solution is complete
                if len(solution) == self.n:
                    cost = self.problem.evaluate_solution(solution)
                    all_solutions.append((solution, cost))
            
            # Find the best solution in this iteration
            if all_solutions:
                all_solutions.sort(key=lambda x: x[1])
                iteration_best_solution, iteration_best_cost = all_solutions[0]
                
                # Update best solution overall
                if iteration_best_cost < self.best_cost:
                    self.best_solution = iteration_best_solution
                    self.best_cost = iteration_best_cost
                    if verbose:
                        print(f"Iteration {iteration}: New best solution with cost {self.best_cost}")
                
                # Apply global pheromone update using the best solution
                self._global_pheromone_update(iteration_best_solution)
            
            # Evaporate pheromones on all edges
            self.pheromone *= (1 - self.rho)
            
            # Add a minimum pheromone level to avoid stagnation
            self.pheromone = np.maximum(self.pheromone, self.initial_pheromone * 0.1)
            
            # Progress report
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best cost: {self.best_cost}")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        if verbose:
            print(f"\nAnt Colony Optimization completed in {computation_time:.2f} seconds")
            print(f"Best solution cost: {self.best_cost}")
        
        return self.best_solution, self.best_cost


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
    file_path = "br17.10.sop"
    cost_matrix, precedence_constraints = load_sop_file(file_path)
    
    # Create SOP instance
    problem = SequentialOrderingProblem(cost_matrix, precedence_constraints)
    
    # Create and run ACO algorithm
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