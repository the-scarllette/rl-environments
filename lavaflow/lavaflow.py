import networkx as nx
import numpy as np
import random as rand
from typing import List, Tuple

from environment import Environment

# Agent appears in a maze with 1 or more squares of lava
# agent can move (N, S, E, W), place a block (N, S, E, W) or terminate the environment
# each timestep all lava spreads to adjacent squares, not through walls or blocks
# terminal if:
#   take the terminate action
#   lava and agent occupy the same tile
# agent cannot take the place blocks action if there is no path such that lava could reach the agent
# reward
#   0 each timestep
#   -0.1 for an illegal action (placing a block on an occupied square, moving into a wall)
#   -1.0 for entering lava or terminating when lava can reach the agent
#   when terminating and lava can no loner reach agent, +1.0 for each empty square the agent could reach
class LavaFlow(Environment):
    north_action = 0
    south_action = 1
    east_action = 2
    west_action = 3
    north_block_action = 4
    south_block_action = 5
    east_block_action = 6
    west_block_action = 7
    terminate_action = 8
    possible_actions = [north_action, south_action, east_action, west_action,
                        north_block_action, south_block_action, east_block_action, west_block_action,
                        terminate_action]
    num_possible_actions = len(possible_actions)
    move_actions = [north_action, south_action, east_action, west_action]
    block_actions = [north_block_action, south_block_action, east_block_action, west_block_action]

    empty_tile = 0
    agent_tile = 1
    lava_tile = 2
    block_tile = 3
    is_terminal_tile = 4

    # # # # #
    # 0 0 0 #
    # 0 0 L #
    # 0 0 0 #
    # # # # #
    default_board = np.array([[block_tile, block_tile, block_tile, block_tile, block_tile, block_tile, block_tile],
                              [block_tile, empty_tile, block_tile, empty_tile, empty_tile, empty_tile, block_tile],
                              [block_tile, empty_tile, block_tile, empty_tile, block_tile, block_tile, block_tile],
                              [block_tile, empty_tile, block_tile, empty_tile, empty_tile, empty_tile, block_tile],
                              [block_tile, empty_tile, empty_tile, empty_tile, empty_tile, empty_tile, block_tile],
                              [block_tile, empty_tile, block_tile, empty_tile, empty_tile, empty_tile, block_tile],
                              [block_tile, block_tile, block_tile, lava_tile , lava_tile , block_tile, block_tile],
                              ])
    default_board_name = 'room'
    default_terminal_lookup = (0, 0)

    failure_reward = -1.0
    invalid_action_reward = -0.1
    step_reward = -0.01
    reward_per_tile = 2.0

    def __init__(
            self,
            board: None | np.ndarray=None,
            board_name: None | str =None,
            terminal_lookup_cords: None | Tuple[int, int]=None
    ):
        self.state_dtype = int

        self.board = board
        self.board_name = board_name
        if self.board is None:
            self.board = self.default_board
            self.board_name = self.default_board_name
        self.terminal_lookup_cords = terminal_lookup_cords
        if self.terminal_lookup_cords is None:
            self.terminal_lookup_cords = self.default_terminal_lookup

        self.board_graph = None
        self.state_shape = self.board.shape

        self.agent_i = None
        self.agent_j = None
        self.safe_from_lava = False
        self.lava_nodes = self.get_lava_nodes(self.board)

        self.current_state = None
        self.terminal = True
        self.environment_name = 'lavaflow_' + self.board_name
        return

    def build_state_graph(
            self,
            state: np.ndarray | None=None
    ) -> nx.Graph:
        if state is None:
            state = self.board
        state_graph = nx.Graph()
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if state[i, j] == self.block_tile:
                    continue
                node = self.cord_node_key(i, j)
                state_graph.add_node(node)

                adjacent_cords = self.get_adjacent_cords(i, j, state)
                for adjacent_cord in adjacent_cords:
                    next_i = adjacent_cord[0]
                    next_j = adjacent_cord[1]
                    connected_node = self.cord_node_key(next_i, next_j)
                    state_graph.add_node(connected_node)
                    state_graph.add_edge(connected_node, node)
        return state_graph

    def cord_node_key(
            self,
            i: int,
            j: int
    ) -> int:
        return (self.state_shape[0] * j) + i

    @staticmethod
    def generate_empty_board(
            n: int
    ) -> np.ndarray:
        board_size = n + 1
        mid = int(board_size / 2)
        board = np.zeros((board_size, board_size))
        board.fill(LavaFlow.block_tile)

        board[1:n, 1:n] = LavaFlow.empty_tile
        board[0, mid] = LavaFlow.lava_tile
        board[n, mid] = LavaFlow.lava_tile
        board[mid, n] = LavaFlow.lava_tile
        board[mid, 0] = LavaFlow.lava_tile

        return board

    # Given n rooms, generates a starting lavaflow board that has that many roms.
    @staticmethod
    def generate_n_room_board(
            n: int
    ) -> np.ndarray:
        num_squares = int(np.ceil(np.sqrt(n)))
        state_len = int((4 * num_squares) + 1)

        board = np.zeros((state_len, state_len))
        board.fill(LavaFlow.block_tile)

        i = 0
        j = 0
        while n > 0:
            start_i = 1 + (4 * i)
            start_j = 1 + (4 * j)
            board[start_i: start_i + 3, start_j: start_j + 3] = LavaFlow.empty_tile
            n -= 1

            center_i = 2 + (4 * i)
            center_j = 2 + (4 * j)
            corridor_cords = [(center_i + 2, center_j), (center_i, center_j + 2),
                              (center_i - 2, center_j), (center_i, center_j - 2)]
            for cord in corridor_cords:
                tile = LavaFlow.empty_tile
                if (cord[0] in [0, state_len - 1]) or (cord[1] in [0, state_len - 1]):
                    tile = LavaFlow.lava_tile
                board[cord[0]][cord[1]] = tile

            j += 1
            if j >= num_squares:
                j = 0
                i += 1

        return board

    @staticmethod
    def generate_scatter_board(
            n: int
    ) -> np.ndarray:
        board_size = n + 2

        board = np.zeros((board_size, board_size))
        board.fill(LavaFlow.block_tile)

        column_indicies = np.arange(1, board_size, 2)

        k = 1
        while k <= n:
            board[k, 1:board_size - 1] = LavaFlow.empty_tile
            k += 2

        k = 2
        while k <= n - 1:
            board[k, column_indicies] = LavaFlow.empty_tile
            k += 2

        corner_cords = [(1, 1), (1, n),
                        (n, 1), (n, n)]
        for corner in corner_cords:
            board[corner[0], corner[1]] = LavaFlow.lava_tile

        return board

    def get_agent_cords(
            self,
            state: np.ndarray | None = None
    ) -> Tuple[int, int] | Tuple[None, None]:
        if state is None:
            state = self.current_state

        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if state[i, j] == self.agent_tile:
                    return i, j
        return None, None

    def get_adjacent_cords(
            self,
            i: int,
            j: int,
            state: None | np.ndarray=None
    ) -> List[Tuple[int, int]]:
        adjacent_cords = []

        if (i < 0 or i >= self.state_shape[0]) or (j < 0 or j >= self.state_shape[1]):
            return adjacent_cords

        if state is None:
            if self.terminal:
                raise AttributeError("Environment must not be terminal or state must be provided")
            state = self.current_state

        for next_i in [max(i - 1, 0), min(i + 1, self.state_shape[0] - 1)]:
            if (next_i == i) or (state[next_i, j] == self.block_tile) or (
                (next_i == self.terminal_lookup_cords[0]) and (j == self.terminal_lookup_cords[1])):
                continue
            adjacent_cords.append((next_i, j))
        for next_j in [max(j - 1, 0), min(j + 1, self.state_shape[1] - 1)]:
            if (next_j == j) or (state[i, next_j] == self.block_tile) or (
                (i == self.terminal_lookup_cords[0]) and (next_j == self.terminal_lookup_cords[1])):
                continue
            adjacent_cords.append((i, next_j))

        return adjacent_cords

    def get_empty_tiles(
            self,
            state: np.ndarray | None = None
    ) -> List[Tuple[int, int]]:
        if state is None:
            state = self.current_state

        empty_tiles = []
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if state[i, j] == self.empty_tile:
                    empty_tiles.append((i, j))
        return empty_tiles

    def get_lava_nodes(
            self,
            state: None | np.ndarray=None
    ) -> List[int]:
        if state is None:
            if self.terminal:
                raise AttributeError("Environment must not be terminal or a state must be provided")
            state = self.current_state
        return [self.cord_node_key(i, j) for i in range(self.state_shape[0]) for j in range(self.state_shape[1])
                if state[i, j] == self.lava_tile]

    def get_start_states(self) -> List[np.ndarray]:
        start_states = []
        empty_tiles = self.get_empty_tiles(self.board)
        for empty_tile in empty_tiles:
            start_state = self.board.copy()
            start_state[empty_tile] = self.agent_tile
            start_states.append(start_state)
        return start_states

    def get_successor_states(
            self,
            state: np.ndarray,
            probability_weights: bool=False
    ) -> Tuple[List[np.ndarray], List[float]]:
        # if terminal:
        #   no successors
        # moving N, S, W, E
        # placing N, S, W, E
        # terminal action: move to terminal state

        stationary_actions = 0
        successor_states = []
        weights = []

        def add_successor(to_add, num_reaching_actions):
            successor_states.append(to_add)
            weights.append(num_reaching_actions)
            return


        if self.is_terminal(state):
            return successor_states, weights

        safe_from_lava = not self.has_path_to_lava(state)
        if safe_from_lava:
            stationary_actions += 4

        # Agent Taking Action
        agent_i, agent_j = self.get_agent_cords(state)
        action_locations = [(agent_i - 1, agent_j), (agent_i + 1, agent_j),
                            (agent_i, agent_j - 1), (agent_i, agent_j + 1)]
        for action_location in action_locations:
            i = action_location[0]
            j = action_location[1]
            action_possible = True
            if i < 0 or i >= self.state_shape[0] or j < 0 or j >= self.state_shape[1]:
                action_possible = False
            elif state[i, j] == self.block_tile:
                action_possible = False

            if not action_possible:
                stationary_actions += 2
                if safe_from_lava:
                    stationary_actions -= 1
                continue

            successor = state.copy()
            successor[agent_i, agent_j] = self.empty_tile
            if successor[i, j] == self.lava_tile:
                successor[self.terminal_lookup_cords] = self.is_terminal_tile
            else:
                successor[i, j] = self.agent_tile
            add_successor(successor, 1)
            if safe_from_lava:
                continue
            successor = state.copy()
            successor[i, j] = self.block_tile
            add_successor(successor, 1)
        successor = state.copy()
        successor[self.terminal_lookup_cords] = self.is_terminal_tile
        add_successor(successor, 1)

        # Adding stationary actions
        if stationary_actions > 0:
            add_successor(state.copy(), stationary_actions)

        # Spreading Lava
        num_successors = 0
        successors_after_lava = []
        weights_after_lava = []
        for k in range(len(successor_states)):
            successor = successor_states[k]
            weight = weights[k]
            successor_after_lava = self.spread_lava(successor)

            successor_found = False
            for l in range(num_successors):
                if np.array_equal(successors_after_lava[l],
                                  successor_after_lava):
                    successor_found = True
                    break

            if successor_found:
                weights_after_lava[l] += weight
                continue
            successors_after_lava.append(successor_after_lava)
            weights_after_lava.append(weight)
            num_successors += 1

        if probability_weights:
            weights = [weight / self.num_possible_actions for weight in weights_after_lava]
            return successors_after_lava, weights
        weights = [1] * num_successors
        return successors_after_lava, weights

    def has_path_to_lava(
            self,
            state: np.ndarray | None=None
    ) -> bool:
        if state is None:
            state_graph = self.board_graph
            i, j = self.agent_i, self.agent_j
            lava_nodes = self.lava_nodes
        else:
            i, j = self.get_agent_cords(state)
            if i is None or j is None:
                return True
            state_graph = self.build_state_graph(state)
            lava_nodes = self.get_lava_nodes(state)

        agent_node = self.cord_node_key(i, j)

        for lava_node in lava_nodes:
            if nx.has_path(state_graph, lava_node, agent_node):
                return True
        return False

    def is_terminal(
            self,
            state: None | np.ndarray=None
    ) -> bool:
        if state is None:
            state = self.current_state

        return state[self.terminal_lookup_cords] == self.is_terminal_tile

    def num_reachable_tiles(
            self,
            state: None | np.ndarray=None
    ) -> int:
        state_graph = self.board_graph
        i, j = self.agent_i, self.agent_j
        if state is not None:
            state_graph = self.build_state_graph(state)
            i, j = self.get_agent_cords(state)
        reachable_nodes = nx.descendants(state_graph, self.cord_node_key(i, j))
        return len(reachable_nodes) + 1

    def reset(
            self,
            state: np.ndarray | None=None,
            seed: None | int=None
    ) -> np.ndarray:
        if seed is not None:
            rand.seed(seed)

        if state is None:
            self.current_state = self.board.copy()
            self.board_graph = self.build_state_graph()
            empty_tiles = self.get_empty_tiles(self.board)
            self.agent_i, self.agent_j = rand.choice(empty_tiles)
            self.current_state[self.agent_i, self.agent_j] = self.agent_tile
            self.terminal = False
        else:
            self.current_state = state.copy()
            self.board_graph = self.build_state_graph()
            self.agent_i, self.agent_j = self.get_agent_cords(self.current_state)
            self.terminal = self.is_terminal(self.current_state)

        self.safe_from_lava = not self.has_path_to_lava(self.current_state)
        self.lava_nodes = self.get_lava_nodes(self.current_state)
        return self.current_state.copy()

    def spread_lava(
            self,
            state: np.ndarray | None = None
    ) -> np.ndarray:
        environment_running = False
        if state is None:
            if self.current_state is None:
                raise AttributeError("Must provide a state or environment must not be terminal")
            environment_running = True
            state = self.current_state

        state_after_lava = state.copy()
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if state[i, j] == self.lava_tile:
                    adjacent_cords = self.get_adjacent_cords(i, j, state)
                    for adjacent_cord in adjacent_cords:
                        next_i = adjacent_cord[0]
                        next_j = adjacent_cord[1]

                        if state[next_i, next_j] == self.agent_tile:
                            state_after_lava[self.terminal_lookup_cords] = self.is_terminal_tile
                            if environment_running:
                                self.terminal = True
                        state_after_lava[next_i, next_j] = self.lava_tile
                        if environment_running:
                            lava_node = self.cord_node_key(next_i, next_j)
                            if lava_node not in self.lava_nodes:
                                self.lava_nodes.append(lava_node)
        return state_after_lava

    def step(
            self,
            action: int
    ) -> (np.ndarray, float, bool, None):
        reward = self.step_reward
        i, j = self.agent_i, self.agent_j

        # Finding position of action
        if action in [self.north_action, self.north_block_action]:
            i -= 1
        elif action in [self.south_action, self.south_block_action]:
            i += 1
        elif action in [self.east_action, self.east_block_action]:
            j += 1
        elif action in [self.west_action, self.west_block_action]:
            j -= 1
        # Finding out if action possible
        action_possible = True
        if i < 0 or i >= self.state_shape[0]:
            reward += self.invalid_action_reward
            action_possible = False
        elif j < 0 or j >= self.state_shape[1]:
            reward += self.invalid_action_reward
            action_possible = False
        else:
            next_tile = self.current_state[i, j]
            if next_tile == self.block_tile:
                reward += self.invalid_action_reward
                action_possible = False
        # Moving Agent
        if (action in self.move_actions) and action_possible:
            self.current_state[self.agent_i, self.agent_j] = self.empty_tile
            self.agent_i, self.agent_j = i, j
            if next_tile == self.lava_tile:
                self.current_state[self.terminal_lookup_cords] = self.is_terminal_tile
                self.terminal = True
            else:
                self.current_state[self.agent_i, self.agent_j] = self.agent_tile
        #Placing Block
        elif (action in self.block_actions) and action_possible:
            if self.safe_from_lava:
                reward += self.invalid_action_reward
                action_possible = False
            else:
                node = self.cord_node_key(i, j)
                if self.current_state[i, j] == self.lava_tile:
                    self.lava_nodes.remove(node)
                self.current_state[i, j] = self.block_tile
                self.board_graph.remove_node(node) # updating graph
        elif action_possible and (action == self.terminate_action):
            self.current_state[self.terminal_lookup_cords] = self.is_terminal_tile
            self.terminal = True

        # Spread Lava;
        self.current_state = self.spread_lava()

        # Check if path from agent to lava exists
        if not self.safe_from_lava or self.terminal:
            self.safe_from_lava = not self.has_path_to_lava()

        # Finding reward if terminal
        if self.terminal:
            if self.safe_from_lava:
                reward += (self.num_reachable_tiles() * self.reward_per_tile)
            else:
                reward += self.failure_reward

        # Check if terminal
        return self.current_state.copy(), reward, self.terminal, None
