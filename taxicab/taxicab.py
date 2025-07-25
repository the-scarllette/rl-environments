import numpy as np
import random as rand
from typing import List, Tuple

from environment import Environment

'''
actions:
0: North
1: South
2: East
3: West
4: pickup
5: putdown
'''

class TaxiCab(Environment):

    possible_actions = [0, 1, 2, 3, 4, 5]

    no_passenger_index = 6

    fuel_station = (2, 1)
    stops = {0: (0, 0),
             1: (0, 4),
             2: (4, 4),
             3: (3, 0),
             no_passenger_index: (None, None)}
    num_stops = 4

    no_right = [(0, 0), (0, 1), (1, 3), (1, 4), (2, 0), (2, 1)]
    no_left = [(1, 0), (1, 1), (2, 3), (2, 4), (3, 0), (3, 1)]

    max_time = 500

    step_reward = 0.0
    success_reward = 2
    failure_reward = -2
    illegal_action_reward = -1

    def __init__(
            self,
            arrival_probabilities: None|List[float]=None,
            continuous: bool=False

    ):

        self.terminal = True

        self.taxi_x = None
        self.taxi_y = None
        self.fuel_level = None

        self.passenger_loc = None  # 0 - 3 for each stop, 4 in taxi, 5 at destination, 6 no passenger
        self.passenger_destination = None # 0 - 3 for each stop, 6 no passenger
        self.possible_passenger_locations = [0, 1, 2, 3]
        self.possible_passenger_destinations = [0, 1, 2, 3]

        self.arrival_probabilities = arrival_probabilities
        self.using_arrival_probabilities = self.arrival_probabilities is not None
        if self.using_arrival_probabilities:
            if abs(sum(self.arrival_probabilities) - 1.0) > 0.001:
                raise ValueError("Arrival Probabilities must sum to 1")
            if len(self.arrival_probabilities) != self.num_stops + 1:
                raise ValueError("Must be an arrival probability for each stop and 1 for no passenger")
            self.arrival_probabilities = {location: arrival_probabilities[location]
                                          for location in self.possible_passenger_locations}
            self.arrival_probabilities[self.no_passenger_index] = arrival_probabilities[4]

            self.possible_passenger_locations = []
            self.arrival_probabilities_list = []
            for possible_location in self.arrival_probabilities:
                if self.arrival_probabilities[possible_location] > 0:
                    self.possible_passenger_locations.append(possible_location)
                    self.arrival_probabilities_list.append(self.arrival_probabilities[possible_location])

        self.no_right = TaxiCab.no_right
        self.no_left = TaxiCab.no_left

        self.current_state = None

        self.output_true_state = False

        self.environment_name = 'taxicab'
        if self.using_arrival_probabilities:
            self.environment_name += '_arrival_probabilities'
            for prob in self.arrival_probabilities_list:
                self.environment_name += '_' + str(round(prob, 3))

        self.state_shape = (4,)
        self.state_dtype = int

        self.continuous = continuous
        return

    def generate_random_state(self) -> np.ndarray:
        x = rand.randint(0, 4)
        y = rand.randint(0, 4)
        passenger_location = rand.choice([0, 1, 2, 3, 4, 6])
        passenger_destination = 6

        if passenger_location != 6:
            passenger_destination = rand.randint(0, 3)

        state = np.array([x, y, passenger_location, passenger_destination])
        return state

    def get_start_states(self) -> List[np.ndarray]:
        # location x: 0-4
        # location y: 0-4
        # passenger location: 0-3, 6
        # passenger destination: 0-3, 6

        start_states = []
        start_state_template = np.full(self.state_shape, 0)

        for x in range(5):
            start_state_template[0] = x
            for y in range(5):
                start_state_template[1] = y
                for passenger_location in self.possible_passenger_locations:
                    start_state_template[2] = passenger_location

                    if passenger_location == self.no_passenger_index:
                        start_state_template[3] = self.no_passenger_index
                        start_states.append(start_state_template.copy())
                        continue

                    for passenger_destination in self.possible_passenger_destinations:
                        start_state_template[3] = passenger_destination
                        start_states.append(start_state_template.copy())

        return start_states

    def get_successor_states(self, state: np.ndarray, probability_weights: bool=False) ->(
            Tuple)[List[np.ndarray], List[float]]:
        successor_states = []
        weights = []

        stationary_actions = 0
        total_actions = 6

        taxi_x = state[0]
        taxi_y = state[1]
        passenger_location = state[2]
        passenger_destination = state[3]
        fuel_level = None
        state = state.copy()

        no_passenger_index = self.no_passenger_index
        possible_passenger_destinations = self.possible_passenger_destinations

        if passenger_location == 5:
            return successor_states, weights
        if (not self.continuous) and self.is_terminal(state):
            return successor_states, weights

        def add_successor_state(index: int, new_value: int, weight: float) -> None:
            successor_state = state.copy()

            if (index is not None) and (new_value is not None):
                successor_state[index] = new_value

            successor_states.append(successor_state)

            if not probability_weights:
                weight = 1.0
            weights.append(weight)
            return
        if self.using_arrival_probabilities and (passenger_location == self.no_passenger_index):
            def add_successor_state(index, new_value, weight):
                successor_state_template = state.copy()
                if (index is not None) and (new_value is not None):
                    successor_state_template[index] = new_value
                for loc in self.possible_passenger_locations:
                    successor_state = successor_state_template.copy()
                    successor_state[2] = loc

                    if loc == no_passenger_index:
                        successor_state[3] = no_passenger_index
                        successor_weight = 1.0
                        if probability_weights:
                            successor_weight = weight * self.arrival_probabilities[no_passenger_index]
                        successor_states.append(successor_state)
                        weights.append(successor_weight)
                        continue

                    for des in possible_passenger_destinations:
                        successor_state = successor_state.copy()
                        successor_state[3] = des
                        successor_weight = 1.0
                        if probability_weights:
                            successor_weight = weight * self.arrival_probabilities[loc] * (1 / self.num_stops)
                        successor_states.append(successor_state)
                        weights.append(successor_weight)
                return

        # Move successor states
        base_cord = (state[0], state[1])
        if base_cord[1] < 4:
            add_successor_state(1, base_cord[1] + 1, 1/total_actions)
        else:
            stationary_actions += 1
        if 0 < base_cord[1]:
            add_successor_state(1, base_cord[1] - 1, 1/total_actions)
        else:
            stationary_actions += 1
        if (base_cord not in self.no_right) and (base_cord[0] < 4):
            add_successor_state(0, base_cord[0] + 1, 1/total_actions)
        else:
            stationary_actions += 1
        if (base_cord not in self.no_left) and (0 < base_cord[0]):
            add_successor_state(0, base_cord[0] - 1, 1/total_actions)
        else:
            stationary_actions += 1

        # Pickup successor state
        if passenger_location <= 3 and (taxi_x, taxi_y) == self.stops[passenger_location]:
            add_successor_state(2, 4, 1/total_actions)
        else:
            stationary_actions += 1

        # Putdown successor state
        if passenger_location == 4 and (taxi_x, taxi_y) == self.stops[passenger_destination]:
            if not self.continuous:
                successor_state = state.copy()
                successor_state[2] = 5
                successor_state[3] = 5
                successor_states.append(successor_state)
                weight = 1.0
                if probability_weights:
                    weight = 1 / total_actions
                weights.append(weight)
            elif self.arrival_probabilities:
                successor_state = state.copy()
                successor_state[2] = self.no_passenger_index
                successor_state[3] = self.no_passenger_index
                successor_states.append(successor_state)
                weight = 1.0
                if probability_weights:
                    weight = 1/total_actions
                weights.append(weight)
            else:
                for loc in self.possible_passenger_locations:
                    for des in possible_passenger_destinations:
                        successor_state = successor_state.copy()
                        successor_state[2] = loc
                        successor_state[3] = des
                        successor_states.append(successor_state)
                        weight = 1.0
                        if probability_weights:
                            weight = 1 / (total_actions * self.num_stops * self.num_stops)
                        weights.append(weight)
        else:
            stationary_actions += 1

        # Finding probability weights
        add_successor_state(None, None, stationary_actions / total_actions)

        return successor_states, weights

    def get_transition_probability(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray
    ) -> float:
        if self.is_terminal(state):
            return 0.0

        after_state = state.copy()
        if action <= 3:
            taxi_cord = (state[0], state[1])
            next_x = state[0]
            next_y = state[1]
            if action == 0:
                next_y += 1
            elif action == 1:
                next_y -= 1
            elif action == 2 and taxi_cord not in self.no_right:
                next_x += 1
            elif action == 3 and taxi_cord not in self.no_left:
                next_x -= 1

            if not (next_x < 0 or next_y < 0 or next_x > 4 or next_y > 4):
                after_state[0] = next_x
                after_state[1] = next_y
        elif action == 4:
            passenger_loc = state[2]
            if passenger_loc <= 3:
                passenger_cords = self.stops[passenger_loc]
                if state[0] == passenger_cords[0] and state[1] == passenger_cords[1]:
                    after_state[2] = 4
        elif action == 5:
            if state[2] == 4:
                passenger_destination_cords = self.stops[state[3]]
                if state[0] == passenger_destination_cords[0] and state[1] == passenger_destination_cords[1]:
                    after_state[2] = 5

        if (after_state[2] != self.no_passenger_index) or not self.using_arrival_probabilities:
            if np.array_equal(next_state, after_state):
                return 1.0
            return 0

        # after_state has no passenger appeared, so now checking arrival probabilities
        if next_state[2] in [4, 5] or next_state[0] != after_state[0] or next_state[1] != after_state[1]:
            return 0.0

        return self.arrival_probabilities[next_state[2]]

    def is_terminal(
            self,
            state: None|np.ndarray=None
    ) -> bool:
        if state is None:
            state = self.current_state
            if state is None:
                return True

        return state[2] == 5

    def reset(
            self,
            start_state=None,
            seed: None|int=None) -> np.ndarray:
        if seed is not None:
            rand.seed(seed)

        state_len = 4
        if start_state is None:
            start_state = [None] * state_len

        def draw_from_start_state(currrent_value, index):
            if start_state[index] is None:
                return currrent_value
            return start_state[index]

        self.terminal = False

        self.taxi_x = rand.randint(0, 4)
        self.taxi_x = draw_from_start_state(self.taxi_x, 0)
        self.taxi_y = rand.randint(0, 4)
        self.taxi_y = draw_from_start_state(self.taxi_y, 1)

        if self.using_arrival_probabilities:
            self.passenger_loc = rand.choices(self.possible_passenger_locations,
                                              self.arrival_probabilities_list)[0]
            self.passenger_loc = draw_from_start_state(self.passenger_loc, 2)

            if self.passenger_loc == self.no_passenger_index:
                self.passenger_destination = self.no_passenger_index
            else:
                self.passenger_destination = rand.choice(self.possible_passenger_destinations)
                self.passenger_destination = draw_from_start_state(self.passenger_destination, 3)
                if self.passenger_destination == self.no_passenger_index:
                    raise ValueError("Cannot assign passenger destination to no location when they are"
                                     "at a location")
        else:
            self.passenger_loc = rand.randint(0, 3)
            self.passenger_loc = draw_from_start_state(self.passenger_loc, 2)
            self.passenger_destination = rand.randint(0, 3)
            self.passenger_destination = draw_from_start_state(self.passenger_destination, 3)

        self.update_state()
        return self.get_current_state()

    def step(
            self,
            action: int
    ) -> np.ndarray:
        if self.terminal:
            raise AttributeError("Environment must be reset before calling step")

        if action not in self.possible_actions:
            raise ValueError("No valid action " + str(action))

        reward = self.step_reward

        next_passenger_location = None
        next_passenger_destination = None
        if self.using_arrival_probabilities and (self.passenger_loc == self.no_passenger_index):
            next_passenger_location = rand.choices(self.possible_passenger_locations,
                                                   self.arrival_probabilities_list)[0]
            if next_passenger_location != self.no_passenger_index:
                next_passenger_destination = rand.choice(self.possible_passenger_destinations)

        if action <= 3:  # Move Action
            taxi_cord = (self.taxi_x, self.taxi_y)
            if (action == 2 and taxi_cord in self.no_right) or (
               (action == 3 and taxi_cord in self.no_left)):
                if next_passenger_destination is not None:
                    self.passenger_loc = next_passenger_location
                    self.passenger_destination = next_passenger_destination
                self.update_state()
                reward += self.illegal_action_reward
                return self.current_state, reward, False, None

            next_x = self.taxi_x
            next_y = self.taxi_y
            if action == 0:
                next_y += 1
            elif action == 1:
                next_y -= 1
            elif action == 2:
                next_x += 1
            else:
                next_x -= 1

            if not (next_x < 0 or next_y < 0 or next_x > 4 or next_y > 4):
                self.taxi_x = next_x
                self.taxi_y = next_y

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, False, None

        if action == 4:  # Pickup Action
            can_pickup = False
            if self.passenger_loc < 4:
                passenger_cords = self.stops[int(self.passenger_loc)]
                can_pickup = passenger_cords[0] == self.taxi_x and passenger_cords[1] == self.taxi_y

            if can_pickup:
                self.passenger_loc = 4
            else:
                reward += self.illegal_action_reward

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, False, None

        if action == 5:  # Putdown
            can_putdown = False
            if self.passenger_loc == 4:
                destination_cords = self.stops[int(self.passenger_destination)]
                can_putdown = destination_cords[0] == self.taxi_x and destination_cords[1] == self.taxi_y

            info = None
            if can_putdown:
                putdown_place = self.no_passenger_index
                if not self.continuous:
                    self.terminal = True
                    putdown_place = 5
                self.passenger_loc = putdown_place
                self.passenger_destination = putdown_place
                reward += self.success_reward
                info = {'success': True}
            else:
                reward += self.illegal_action_reward

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, self.terminal, info

        raise IndexError("Invalid action provided")

    def taxi_at_cords(
            self,
            cord: Tuple[int, int]
    ) -> bool:
        return self.taxi_x == cord[0] and self.taxi_y == cord[1]

    def update_state(self):
        self.current_state = np.array(
            [self.taxi_x, self.taxi_y,self.passenger_loc, self.passenger_destination],
            dtype=float
        )
        return
