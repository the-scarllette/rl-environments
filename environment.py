import networkx as nx
import numpy as np
from scipy import sparse
from typing import Any, Dict, List, Tuple, Type

from utils.progress_bar import print_progress_bar


class Environment:

    current_state: None|np.ndarray=None
    environment_name: str=""
    possible_actions: List[int]=[]
    state_dtype: Type=int
    state_shape: Tuple[int, ...]
    terminal: bool=True

    def __init__(self):
        return

    def generate_random_state(self) -> np.ndarray:
        return self.reset()

    def get_adjacency_matrix(
            self,
            directed: bool=True,
            probability_weights: bool=False,
            compressed_matrix: bool=False,
            progress_bar: bool=False
    ) -> Tuple[np.ndarray|sparse.csr_matrix, nx.DiGraph, Dict[str, Dict[str, str]]]:
        connected_states = {}
        state_indexes = {}
        all_states = []
        state_indexer = 0

        states_to_check = self.get_start_states()
        num_states_to_check = len(states_to_check)
        iteration = 0
        total_states = num_states_to_check

        while num_states_to_check > 0:
            iteration += 1

            if progress_bar:
                print_progress_bar(iteration, total_states, "Finding STG for " + self.environment_name + ":")

            state = states_to_check.pop().copy()
            num_states_to_check -= 1
            state_bytes = state.tobytes()

            try:
                # If already found successor states, skip state
                _ = connected_states[state_bytes]['states']
                continue
            except KeyError:
                # If successor states not found, find them
                successor_states, weights = self.get_successor_states(
                    state,
                    probability_weights=probability_weights
                )

            all_states.append(state)

            state_indexes[state_bytes] = state_indexer
            state_indexer += 1

            connected_states[state_bytes] = {
                'states': successor_states,
                'weights': weights
            }

        data_type = int
        if probability_weights:
            data_type = float
        if compressed_matrix:
            adj_matrix = sparse.lil_matrix((state_indexer, state_indexer), dtype=data_type)
        else:
            adj_matrix = np.zeros((state_indexer, state_indexer), dtype=data_type)
        state_num = 1
        for state in all_states:
            state_bytes = state.tobytes()
            i = state_indexes[state_bytes]

            successor_states = connected_states[state_bytes]['states']
            weights = connected_states[state_bytes]['weights']
            num_successor_states = len(successor_states)

            for j in range(num_successor_states):
                successor = successor_states[j]
                weight = 1.0
                if probability_weights:
                    weight = weights[j]
                successor_index = state_indexes[successor.tobytes()]

                adj_matrix[i, successor_index] = weight
                if not directed:
                    adj_matrix[successor_index, i] = weight
            state_num += 1

        if compressed_matrix:
            adj_matrix = adj_matrix.tocsr()
            state_transition_graph = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.MultiDiGraph)
        else:
            state_transition_graph = nx.from_numpy_array(adj_matrix, create_using=nx.MultiDiGraph)

        stg_values = {str(node) : {'state': np.array2string(all_states[node])}
                      for node in range(adj_matrix.shape[0])}
        nx.set_node_attributes(state_transition_graph, stg_values)

        return adj_matrix, state_transition_graph, stg_values

    def get_current_state(self) -> np.ndarray:
        if self.terminal or self.current_state is None:
            raise AttributeError("No current state when environment is terminal.")
        return self.current_state

    def get_possible_actions(
            self,
            state: np.ndarray
    ) -> List[int]:
        return self.possible_actions

    def get_start_states(self) -> List[np.ndarray]:
        return []

    def get_successor_states(
            self,
            state: np.ndarray,
            probability_weights: bool=False
    ) -> Tuple[List[np.ndarray], List[float]]:
        return [], []

    def get_transition_probability(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray
    ) -> float:
        return 1.0

    def is_terminal(
            self,
            state: None|np.ndarray=None
    ) -> bool:
        return self.terminal

    def print_state(
            self,
            state: None|np.ndarray=None
    ):
        if state is None:
            if self.terminal:
                raise AttributeError("Either provide a state or print state while environment is not terminal.")
            state = self.current_state

        print(np.array2string(state))
        return

    def step(
            self,
            action: int
    ) ->(np.ndarray, float, bool, Any):
        if self.terminal:
            raise AttributeError("Cannot step while environment is terminal, reset the environment.")
        return None, 0.0, False, None

    def reset(
            self,
            start_state: None|np.ndarray=None,
            seed: None|int=None
    ) -> np.ndarray:
        pass
