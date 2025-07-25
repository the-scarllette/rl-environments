import numpy as np
import random as rand
from typing import Any, List, Tuple

from environment import Environment

class Building:

    def __init__(
            self,
            index: int,
            building_pattern: np.ndarray,
            feeds: int,
            needs_feeding: bool,
            num_resources: int,
            fed_score: float
    ):
        self.index = index
        self.building_pattern = building_pattern
        self.feeds = feeds
        self.fed_score = fed_score
        self.needs_feeding = needs_feeding
        self.num_resources = num_resources
        return

    def rotate_plan(
            self,
            rotation_index: int
    ) -> np.ndarray:
        building_plan = self.building_pattern
        if rotation_index >= 4:
            building_plan = np.fliplr(building_plan)
            rotation_index -= 4
        for _ in range(rotation_index):
            building_plan = np.rot90(building_plan)
        return building_plan


class TinyTown(Environment):
    any_tile = -1
    empty_tile = 0
    brick = 1
    glass = 2

    cottage_index = 4
    cottage_plan = np.array([[any_tile, any_tile, any_tile],
                             [any_tile, glass, brick],
                             [any_tile, brick, any_tile]])
    cottage = Building(cottage_index, cottage_plan, 0, True, 3, 3)

    greenhouse_index = 5
    greenhouse_plan = np.array([[any_tile, any_tile, any_tile],
                                [any_tile, brick, brick],
                                [any_tile, glass, glass]])
    greenhouse = Building(greenhouse_index, greenhouse_plan, 4, False, 4, 0)

    default_resources = [brick, glass]
    default_buildings = [cottage, greenhouse]

    step_reward = 0.0

    def __init__(
            self,
            width: int,
            height: int,
            resources: List[int]=default_resources,
            buildings: List[Building]=default_buildings,
            pick_every: int=1
    ):
        self.width = width
        self.height = height
        self.resources = resources
        self.num_resources = len(self.resources)
        self.buildings = buildings
        self.num_buildings = len(self.buildings)
        self.building_indexes = {b.index: b for b in self.buildings}
        self.pick_every = pick_every  # <= 0: never pick, 1: choose every time, 2: choose every other etc.

        self.board = None
        self.building_phase = False
        self.next_resource = None
        self.total_num_actions = 0
        self.action_lookup = {}
        for resource in range(self.num_resources):
            for x in range(self.width):
                for y in range(self.height):
                    self.action_lookup[self.total_num_actions] = {'resource': self.resources[resource], 'x': x, 'y': y}
                    self.total_num_actions += 1
        for building in range(self.num_buildings):
            for rot_index in range(1):
                for x_1 in range(self.width):
                    for x_2 in range(self.width):
                        for y_1 in range(self.height):
                            for y_2 in range(self.height):
                                self.action_lookup[self.total_num_actions] = {'building': self.buildings[building],
                                                                    'rotation': rot_index,
                                                                    'x_plan': x_1, 'y_plan': y_1, 'x': x_2, 'y': y_2}
                                self.total_num_actions += 1
        self.action_lookup[self.total_num_actions] = 'stop_building'
        self.total_num_actions += 1
        self.possible_actions = list(range(self.total_num_actions))
        self.terminal = True
        self.turn_count = 0

        name_key = ['random', 'choice']
        self.environment_name = "tiny_town_" + name_key[self.pick_every] + "_" +\
                                str(len(self.resources)) + "x" + str(self.width) + "x" + str(self.height)

        self.state_dtype = int
        self.state_shape = (self.width + 1, self.height + 1)
        self.available_actions = {}
        return

    def board_full(self) -> bool:
        if self.terminal:
            raise AttributeError("Environment is terminal")

        for j in range(self.height):
            for i in range(self.width):
                if self.board[j, i] == self.empty_tile:
                    return False
        return True

    def can_build(
            self,
            state: np.ndarray,
            plan: np.ndarray,
            x: int,
            y: int
    ) -> bool:
        surrounding_tiles = np.full((3, 3), self.any_tile)

        i = 0
        for x_new in range(x - 1, x + 2):
            j = 0
            for y_new in range(y - 1, y + 2):
                if (0 <= y_new < self.height) and (0 <= x_new < self.width):
                    surrounding_tiles.itemset((j, i), state[y_new, x_new])
                j += 1
            i += 1

        for i in range(3):
            for j in range(3):
                if plan[j, i] == self.any_tile:
                    continue
                if not plan[j, i] == surrounding_tiles[j, i]:
                    return False

        return True

    def clear_resources(
            self,
            state: None|np.ndarray=None
    ) -> np.ndarray:
        if self.terminal and state is None:
            raise AttributeError("Environment must not be terminal in order to clear board of resources")

        if state is None:
            state = self.board
        for i in range(self.width):
            for j in range(self.height):
                if state[j, i] in self.resources:
                    state.itemset((j, i), self.empty_tile)
        return state

    def get_possible_actions(
            self,
            state: np.ndarray
    ) -> List[int]:
        if state is None:
            state_str = np.array2string(self.board)
        else:
            state_str = np.array2string(state)

        try:
            possible_actions = self.available_actions[state_str]
            return possible_actions
        except KeyError:
            ()

        if self.terminal:
            return []

        possible_actions = []
        building_phase_actions = self.width * self.height * self.num_resources

        # 0 <-> width*height*num_resources - 1 : place a resource at that location
        if not self.building_phase:
            if self.next_resource == self.any_tile:
                can_place = self.resources
            else:
                can_place = [self.next_resource]

            possible_actions = [i for i in range(building_phase_actions)
                                if self.board[(action_dict := self.action_lookup[i])['y'],
                                              action_dict['x']] == self.empty_tile and
                                (action_dict['resource'] in can_place)]

            return possible_actions

        # width*height*num_resources <-> end : at location, build building, with rotation, place building at location
        end_building_action = self.total_num_actions - 1
        for i in range(building_phase_actions, end_building_action):
            action_dict = self.action_lookup[i]
            building = action_dict['building']
            rotation = action_dict['rotation']
            x_plan_center = action_dict['x_plan']
            y_plan_center = action_dict['y_plan']
            x_to_build = action_dict['x']
            y_to_build = action_dict['y']

            building_plan = building.rotate_plan(rotation)

            if not self.can_build(self.board, building_plan, x_plan_center, y_plan_center):
                continue

            y_in_plan = -y_plan_center + y_to_build + 1
            x_in_plan = -x_plan_center + x_to_build + 1
            if (not 0 <= y_in_plan < 3) or (not 0 <= x_in_plan < 3) or (
                    building_plan[y_in_plan, x_in_plan] == self.any_tile):
                continue
            possible_actions.append(i)

        possible_actions.append(end_building_action)

        self.available_actions[state_str] = possible_actions
        return possible_actions

    def get_start_states(self) -> List[np.ndarray]:
        possible_resources = [self.any_tile]
        if not self.pick_every == 1:
            possible_resources = self.resources

        start_states = []
        start = np.full((self.width + 1, self.height + 1), self.empty_tile)
        for resource in possible_resources:
            start_state = start.copy()
            start_state.itemset((self.width, self.height), resource)
            start_states.append(start_state)
        return start_states

    def get_successor_states(
            self,
            state: np.ndarray,
            probability_weights: bool=False
    ) -> Tuple[List[np.ndarray], List[float]]:
        if self.is_state_terminal(state):
            return [], []

        successor_states = []
        possible_actions = 0
        resource_to_place = state[self.width, self.height]

        # Building Phase
        if resource_to_place == self.empty_tile:

            for y in range(self.height):
                for x in range(self.width):
                    for building in self.buildings:
                        if not self.can_build(state, building.building_pattern, x, y):
                            continue

                        # Removing Building Materials
                        successor_template = state.copy()
                        build_locations = []
                        for i in range(3):
                            for j in range(3):
                                if building.building_pattern[j, i] == self.any_tile:
                                    continue

                                y_build = y - 1 + j
                                x_build = x - 1 + i
                                possible_actions += 1
                                build_locations.append((y_build, x_build))
                                successor_template.itemset((y_build, x_build),
                                                           self.empty_tile)

                        # Placing Building
                        for build_location in build_locations:
                            successor = successor_template.copy()
                            successor.itemset(build_location, building.index)
                            successor_states.append(successor)

            # End Building Phase Action
            possible_actions += 1
            next_resources = self.resources
            num_next_resources = self.num_resources
            if self.pick_every == 1:
                next_resources = [self.any_tile]
                num_next_resources = 1
            for resource in next_resources:
                successor = state.copy()
                successor.itemset((self.width, self.height), resource)
                successor_states.append(successor)

            # Finding Probability Weights
            if not probability_weights:
                probabilities = [1.0] * len(successor_states)
                return successor_states, probabilities

            # TODO: Fix probabilities for random tile choice
            probabilities = [1.0 / possible_actions] * (possible_actions - 1)
            probabilities += [1 / (num_next_resources * possible_actions)] * num_next_resources
            return successor_states, probabilities

        # In Resource Phase
        can_place = [resource_to_place]
        if resource_to_place == self.any_tile:
            can_place = self.resources

        for resource in can_place:
            for y in range(self.height):
                for x in range(self.width):
                    if not state[y, x] == self.empty_tile:
                        continue
                    possible_actions += 1
                    successor = state.copy()
                    successor.itemset((y, x), resource)
                    successor.itemset((self.width, self.height), self.empty_tile)
                    successor_states.append(successor)

        probability = 1.0
        if probability_weights:
            probability = 1.0 / possible_actions

        probabilities = [probability] * possible_actions
        return successor_states, probabilities

    def is_terminal(
            self,
            state: None|np.ndarray=None
    ) -> bool:
        if state is None:
            if self.terminal:
                return True
            state = self.current_state

        # All squares are full
        for i in range(self.width):
            for j in range(self.height):
                if state[j, i] == self.empty_tile:
                    return False

        # State is full and environment is in resource phase
        if not (state[(self.width, self.height)] == self.empty_tile):
            return True

        # Can build
        for i in range(self.width):
            for j in range(self.height):
                for building in self.buildings:
                    for k in range(1):
                        building_plan = building.rotate_plan(k)
                        if self.can_build(state, building_plan, i, j):
                            return False
        return True

    def score_state(
            self,
            state
    ) -> float:
        has_greenhouse = False
        need_feeding = 0
        empty_tiles = 0
        for i in range(self.width):
            for j in range(self.height):
                if state[j, i] == self.greenhouse.index:
                    has_greenhouse = True
                elif state[j, i] == self.cottage.index:
                    need_feeding += 1
                else:
                    empty_tiles += 1

        fed_cottages = 0
        if has_greenhouse:
            fed_cottages = need_feeding

        return (self.cottage.fed_score * fed_cottages) - empty_tiles

    def step(
            self,
            action: int
    ) ->(np.ndarray, float, bool, Any):
        # Place resource
        # If no more buildings can be built:
        #   End game and score
        # Else:
        #   Continue to place resources or Build a building,
        #   If building jump back to no more buildings check

        if self.terminal:
            raise AttributeError("Environment must not be reset before step action")

        possible_actions = self.get_possible_actions()
        if action not in possible_actions:
            raise AttributeError("Invalid action for current state")

        if not self.building_phase:
            action_dict = self.action_lookup[action]
            resource = action_dict['resource']
            x = action_dict['x']
            y = action_dict['y']
            self.board.itemset((y, x), resource)

            self.building_phase = True
            self.next_resource = self.empty_tile
            self.board.itemset((self.width, self.height), self.next_resource)

        elif self.building_phase and not action == self.total_num_actions - 1:
            action_dict = self.action_lookup[action]
            building = action_dict['building']
            rotation = action_dict['rotation']
            x_plan_center = action_dict['x_plan']
            y_plan_center = action_dict['y_plan']
            x_to_build = action_dict['x']
            y_to_build = action_dict['y']

            plan = building.rotate_plan(rotation)

            for i in range(x_plan_center - 1, x_plan_center + 2):
                plan_x = i - x_plan_center + 1
                for j in range(y_plan_center - 1, y_plan_center + 2):
                    plan_y = j - y_plan_center + 1
                    if plan[plan_y, plan_x] == self.any_tile:
                        continue
                    self.board.itemset((j, i), self.empty_tile)

            self.board.itemset((y_to_build, x_to_build), building.index)

        elif action == self.total_num_actions - 1:
            self.building_phase = False

            self.turn_count += 1
            self.next_resource = rand.choice(self.resources)
            if self.pick_every == 1:
                self.next_resource = self.any_tile
            self.board.itemset((self.width, self.height), self.next_resource)

        if self.is_state_terminal(self.board) or (action == self.total_num_actions - 1 and self.board_full()):
            reward = self.score_state(self.board)
            self.building_phase = False
            self.terminal = True
            self.next_resource = self.empty_tile
            self.board.itemset((self.width, self.height), self.next_resource)

            return self.board.copy(), reward, self.terminal, None

        return self.board.copy(), self.step_reward, False, None

    def reset(
            self,
            start_state: None|np.ndarray=None,
            seed: None|int=None
    ) -> np.ndarray:
        if seed is not None:
            rand.seed(seed)

        if start_state is not None:
            self.board = start_state.copy()
            self.next_resource = self.board[self.width, self.height]
            self.building_phase = self.next_resource == self.empty_tile
            self.terminal = False
            return self.board.copy()

        self.board = np.full((self.width + 1, self.height + 1), self.empty_tile)

        if self.pick_every == 1:
            self.next_resource = self.any_tile
        else:
            self.next_resource = rand.choice(self.resources)
        self.board.itemset((self.width, self.height), self.next_resource)

        self.building_phase = False
        self.terminal = False
        self.turn_count = 1

        return self.board.copy()
