import itertools
from typing import Counter, Iterable, List, Set
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

DATATYPE = np.float64
N_ITER = 1000
VERBOSE = 0
tol = 1e-8

rng = np.random.default_rng(seed=1001)

def get_cycles(path: np.array) -> List[List[int]]:
    """Get all cycles that occur in a path"""
    def _inner_getter(_path: np.array) -> Iterable[List[int]]:
        visited = {k: False for k in range(_path.shape[0])}
        while not all(visited.values()):
            node = next(n for n in visited if not visited[n])
            result = []
            while not visited[node]:
                result.append(node)
                visited[node] = True
                node = np.argmax(_path[node], axis=0)
            yield result
    return list(_inner_getter(path))

def costs_of_switch_if_switch(proposed_start: int, proposed_end: int, start_connectors: Iterable, end_connectors: Iterable, distances: np.array):
    # Cost of other switches, provided that you're going to do one switch already.
    # This cost is often negative, since we're deleting two connections and replacing one.
    return np.array([
        [- distances[proposed_start, alt_start_connector]
        - distances[proposed_end, alt_end_connector]
        + distances[alt_start_connector, alt_end_connector]
        for alt_start_connector in start_connectors]
        for alt_end_connector in end_connectors
    ])

def opportunity_cost_of_switch(proposed_start: int, proposed_end: int, start_connectors: Iterable, end_connectors: Iterable, distances: np.array):
    """Compute the best possible opportunity cost of switching to a new proposed start and end"""
    if VERBOSE >= 2:
        print(proposed_start, proposed_end)
        print(start_connectors, end_connectors)
    return distances[proposed_start, proposed_end] + np.min(costs_of_switch_if_switch(proposed_start, proposed_end, start_connectors, end_connectors, distances))

def connections_of(connections, node) -> set:
    return set(np.nonzero(connections[:, node])[0]) | set(np.nonzero(connections[node, :])[0])

def invalid_solve(weights: np.array, distances: np.array) -> np.array:
    if VERBOSE >= 2:
        print(f"entering invalid solve with a {weights.shape} matrix")
    sum0 = weights.sum(axis=0)
    sum1 = weights.sum(axis=1)
    ix0 = set(np.where(sum0 != 1.)[0])
    ix1 = set(np.where(sum1 != 1.)[0])
    initial_problems = len(ix0) + len(ix1)
    if VERBOSE >= 2:
        print(ix0)
        print(sum0[list(ix0)])
        print(ix1)
        print(sum1[list(ix1)])
        print(f"There are {initial_problems} problems")
    maxdist = np.max(distances)
    _distances = distances + np.eye(distances.shape[0]) * maxdist
    for i in ix0:
        if VERBOSE >= 2:
            print(i)
        j_ = np.where(weights[:, i] > 0)[0]
        if VERBOSE >= 2:
            print(j_)
        if len(j_) == 0:
            j = np.argmin(_distances[:, i] + (maxdist) * sum1)
            weights[j, i] = 1
        elif len(j_) > 1:
            j = j_[np.argmin(_distances[j_, i])]
            if VERBOSE >= 2:
                print(j)
                print("deleting", list(set(j_) - {j}))
            weights[list(set(j_) - {j}), i] = 0
    sum0 = weights.sum(axis=0)
    ix0 = set(np.where(sum0 != 1.)[0])
    for i in ix1:
        j_ = np.where(weights[i, :] > 0)[0]
        if len(j_) == 0:
            j = np.argmin(_distances[i, :] + (maxdist) * sum0)
            weights[i, j] = 1
        elif len(j_) > 1:
            j = j_[np.argmin(_distances[i, j_])]
            weights[i, list(set(j_) - {j})] = 0
    sum1 = weights.sum(axis=1)
    ix1 = set(np.where(sum1 != 1.)[0])
    if 0 < (problems := len(ix0) + len(ix1)):
        if VERBOSE >= 2:
            print(f"There are {problems} problems")
        # To escape infinite cycles of A -> B -> A -> B -> ..., we add a few other random nodes
        # into the mix when calling recursively
        n_random_addons = rng.choice(10, 1)[0]
        problem_indexes = list(
            ix0
            | ix1
            | set(rng.choice(distances.shape[0], size=min(n_random_addons, distances.shape[0])))
        )
        slice = np.ix_(problem_indexes, problem_indexes)
        weights[slice] = invalid_solve(weights[slice], distances[slice])
        return invalid_solve(weights, distances)
    if VERBOSE >= 2:
        print("problems solved")
    return weights

def disjoint_solve(weights: np.array, distances: np.array) -> np.array:
    cycles = get_cycles(weights)
    c1, c2 = rng.choice(cycles, 2, replace=False)
    opportunity_costs = np.ones(shape=weights.shape) * np.inf
    for start_node in c1:
        start_connectors = connections_of(weights, start_node)
        for end_node in c2:
            end_connectors = connections_of(weights, end_node)
            opportunity_costs[start_node, end_node] = opportunity_costs[end_node, start_node] = \
                opportunity_cost_of_switch(start_node, end_node, start_connectors, end_connectors, distances)
    b1, a1 = np.unravel_index(opportunity_costs.argmin(), opportunity_costs.shape)
    if a1 == b1:
        raise RuntimeError()
    start_alternatives = sorted(connections_of(weights, b1))
    end_alternatives = sorted(connections_of(weights, a1))
    if alt_intersection := set(start_alternatives) & set(end_alternatives):
        relevant_matrix_part = sorted(set(start_alternatives) | set(end_alternatives) | {a1, b1})
        slicer = np.ix_(relevant_matrix_part, relevant_matrix_part)
        print(relevant_matrix_part)
        print(weights[slicer])
        raise RuntimeError(f"start {b1} {start_alternatives} and end {a1} {end_alternatives} overlap: {alt_intersection}. Cycles: {c1}, {c2}")
    switching_costs = costs_of_switch_if_switch(
        b1, a1, start_alternatives, end_alternatives, distances
    )
    a2_i, b2_i = np.unravel_index(switching_costs.argmin(), switching_costs.shape)
    b2 = start_alternatives[b2_i]
    a2 = end_alternatives[a2_i]
    if a2 == b2:
        raise RuntimeError()
    if weights[a1, a2] == 0:
        a1, a2 = a2, a1
    if weights[b2, b1] == 0:
        b1, b2 = b2, b1
    if VERBOSE >= 1:
        print(f"cycles are {cycles}")
    if all((
        weights[a1, a2] == 1,
        weights[a1, b1] == 0,
        weights[b2, b1] == 1,
        weights[b2, a2] == 0
    )):
        if VERBOSE >= 1:
            print(f"Disconnecting {a1} -> {a2}, Connecting {a1} -> {b1}")
            print(f"Disconnecting {b2} -> {b1}, Connecting {b2} -> {a2}")
        weights[a1, a2] = 0
        weights[a1, b1] = 1
        weights[b2, b1] = 0
        weights[b2, a2] = 1
    else:
        raise NotImplementedError
    if VERBOSE >= 1:
        print(f"cycles are {get_cycles(weights)}")
    if VERBOSE >= 2:
        print(weights)
    return weights

def general_distance_loss(distances: np.array, weights: np.array) -> float:
    return distances.ravel().dot(weights.ravel())

def swap_node_solve(weights: np.array, distances: np.array) -> np.array:

    if VERBOSE >= 2:
        print(f"cycles are {get_cycles(weights)}")

    alt0 = np.zeros(weights.shape)
    alt1 = np.zeros(weights.shape)

    for n1, n2 in zip(*np.where(weights > 0)):
        n0 = np.where(weights[:, n1] > 0)[0][0]
        n3 = np.where(weights[n2, :] > 0)[0][0]
        alt0[n0, n1] = 0
        alt0[n1, n2] = 0
        alt0[n2, n3] = 0
        alt0[n0, n2] = 1
        alt0[n2, n1] = 1
        alt0[n1, n3] = 1

        alt1[n0, n1] = 1
        alt1[n1, n2] = 1
        alt1[n2, n3] = 1
        alt1[n0, n2] = 0
        alt1[n2, n1] = 0
        alt1[n1, n3] = 0

        if general_distance_loss(distances, alt0) < general_distance_loss(distances, alt1):
            weights[n0, n1] = 0
            weights[n1, n2] = 0
            weights[n2, n3] = 0
            weights[n0, n2] = 1
            weights[n2, n1] = 1
            weights[n1, n3] = 1

        alt0 *= 0
        alt1 *= 0

    return weights

def random_path(n: int, distances: np.array) -> np.array:
    result = np.zeros(shape=(n, n))
    order = np.array(range(n))
    rng.shuffle(order)
    prev = order[-1]
    for node in order:
        result[prev, node] = 1.
        prev = node
    return swap_node_solve(result, distances)

def naive_solve(distances: np.array) -> np.array:
    result = np.zeros(shape=distances.shape)
    n = distances.shape[0]
    for i, j in zip(range(n), np.argmin(distances + np.eye(n) * (np.max(distances) + 100), axis=0)):
        result[i, j] = 1
    return result

def minimum_estimate(distances: np.array) -> np.array:
    return general_distance_loss(distances, naive_solve(distances))

def validate(weights, *, verbose: bool = False):
    n = weights.shape[0]
    if set(np.count_nonzero(weights, axis=0)) != {1}:
        if verbose:
            print("weights inconsistent in axis 0")
        return False
    if set(np.count_nonzero(weights, axis=1)) != {1}:
        if verbose:
            print("weights inconsistent in axis 1")
        return False
    if len(get_cycles(weights)) > 1:
        if verbose:
            print("path is disjoint")
        return False
    if n_visited := len(set.union(*map(set, get_cycles(weights)))) < n:
        if verbose:
            print(f"Not all nodes visited: {n_visited} < ")
        return False
    if cycle_length := len(get_cycles(weights)[0]) != n:
        if verbose:
            print(f"Length on path is {cycle_length}, not {n}")
        return False
    return True

def solve(distances: np.array) -> List[int]:

    naive_solution = naive_solve(distances)
    if VERBOSE >= 2:
        print("naive solution:")
        print(naive_solution)
    if validate(naive_solution):
        if VERBOSE >= 2:
            print("naive solution ok")
        return naive_solution
    
    if VERBOSE >= 2:
        print("failed to validate naive solution")

    n = distances.shape[0]

    # Initialize random weights
    weights = np.eye(n)

    _distances_ravel = distances.ravel()

    # Distance loss function
    # The cost of each route
    def distance_loss(_weights):
        return _distances_ravel.dot(_weights.ravel())

    weights = np.zeros(shape=(n, n))
    total_weight = 0.
    lowest_loss = np.inf
    for _ in range(N_ITER):
        path = random_path(n, distances)
        path_loss = distance_loss(path)
        path_weight = 1. / path_loss ** 2
        if path_loss < lowest_loss:
            lowest_loss = path_loss
            print(f"new lowest loss ({n=}): {lowest_loss}")
            # print(path)
        total_weight += path_weight
        weights += path * path_weight

    weights /= total_weight

    if VERBOSE >= 2:
        print("weights before processing")
        print(weights)

    ix0 = np.argmax(weights, axis=0)
    result = np.zeros(shape=(n, n))
    for i0, i1 in zip(ix0, range(n)):
        result[i0, i1] = 1.

    # Find and fix invalid path spec, i.e. path is not a proper path.
    # 1 -> 2 and 1 -> 3 for example.
    while set(result.sum(axis=0)) != {1.} or set(result.sum(axis=1)) != {1.}:
        if VERBOSE >= 1:
            print("Solution is invalid. Solving subset:")
        result = invalid_solve(result, distances)

    if set(result.sum(axis=0)) != {1.} or set(result.sum(axis=1)) != {1.}:
        raise RuntimeError("Invalid solve failed")

    if VERBOSE >= 2:
        print("result before connecting")
        print(result)

    # Find and fix disjoint cycles.
    # 1 -> 2 -> 1 and 3 -> 4 -> 5 -> 3 for example.
    while len(cycles := get_cycles(result)) > 1:
        if VERBOSE >= 1:
            print(f"Solution has more than 1 cycle: {cycles}")
        result = disjoint_solve(result, distances)

    if VERBOSE >= 1:
        print(f"solution loss: {distance_loss(result)}")
    # print(result)

    result = swap_node_solve(result, distances)

    return result

if __name__ == "__main__":

    N = 40

    mat = rng.random(size=(N, N)).astype(DATATYPE)
    mat = (mat + mat.T) / 2.
    mat[range(N), range(N)] = 0.
    weights = solve(mat)

    if VERBOSE >= 2:
        print("MAT")
        print(mat)
        print("WEIGHTS")
        print(weights)

    weight_validity = validate(weights, verbose=True)
    print(f"Weights are {'not ' if not weight_validity else ''}valid.")

    print(f"Solution loss: {general_distance_loss(mat, weights)}, naive loss: {general_distance_loss(mat, naive_solve(mat))}")
    print(get_cycles(weights))

    from python_tsp.heuristics import solve_tsp_simulated_annealing

    permutation, distance = solve_tsp_simulated_annealing(mat)

    ots_solver_solution = np.zeros(mat.shape)
    prev = permutation[-1]
    for node in permutation[1:]:
        ots_solver_solution[prev, node] = 1
        prev = node

    print(f"Off the shelf solver achieved loss {general_distance_loss(mat, ots_solver_solution)} {permutation}")
