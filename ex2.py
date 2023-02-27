from utils import FIFOQueue
from copy import deepcopy
import itertools
import json
from datetime import datetime
import time

ids = ["206418642", "208142612"]

def adjacency_list_creator(map):
    adj_dict = {}
    for r_idx in range(len(map)):
        for c_idx in range(len(map[0])):
            if map[r_idx][c_idx] != "I":
                adj_list = []
                if c_idx + 1 <= len(map[0]) - 1 and map[r_idx][c_idx + 1] != "I":
                    adj_list.append((r_idx, c_idx + 1))

                if r_idx + 1 <= len(map) - 1 and map[r_idx + 1][c_idx] != "I":
                    adj_list.append((r_idx + 1, c_idx))

                if r_idx - 1 >= 0 and map[r_idx + - 1][c_idx] != "I":
                    adj_list.append((r_idx - 1, c_idx))

                if c_idx - 1 >= 0 and map[r_idx][c_idx - 1] != "I":
                    adj_list.append((r_idx, c_idx - 1))

                adj_dict.update({(r_idx, c_idx): adj_list})

    return adj_dict


def BFS(curr, N, visited, distance, v, adj_dict, source):
    while (curr <= N):
        node = v[curr - 1]
        for i in range(len(adj_dict[node])):
            next = adj_dict[node][i]
            if (not visited[next]) and (distance[next] < distance[node] + 1):
                v.append(next)
                distance[next] = distance[node] + 1
                visited[next] = True

        curr += 1
        if curr > len(v):
            for key in distance.keys():
                if distance[key] == 0 and key != source:
                    distance[key] = float("inf")
            break
    return distance


def bfsTraversal(adj_dict, N, source):
    visited = {}
    distance = {}
    for key in adj_dict.keys():
        visited[key] = False
        distance[key] = 0
    v = []
    v.append(source)
    visited[v[0]] = True
    return BFS(1, N, visited, distance, v, adj_dict, source)


def actions(state):
    """Returns all the actions that can be executed in the given
    state. The result should be a tuple (or other iterable) of actions
    as defined in the problem description file"""
    actions = []
    # actions.append("terminate")
    actions.append("reset")

    actions_by_taxi = []
    for taxi in state["taxis"]:
        taxi_actions = []
        current_location = state["taxis"][taxi]["location"]

        # checks for drop offs
        for person in state["passengers"]:
            if state["passengers"][person]["location"] == taxi and \
                    current_location == state["passengers"][person]["destination"]:
                taxi_actions.append(("drop off", taxi, person))

        # checks for gas stations
        if state["map"][current_location[0]][current_location[1]] == "G":
            taxi_actions.append(("refuel", taxi))

        # checks for valid movements
        rows_num = len(state["map"]) - 1
        columns_num = len(state["map"][0]) - 1

        if state["taxis"][taxi]["fuel"] > 0:
            if current_location[0] + 1 <= rows_num and \
                    state["map"][current_location[0] + 1][current_location[1]] != "I":
                taxi_actions.append(("move", taxi, (current_location[0] + 1, current_location[1])))

            if current_location[1] + 1 <= columns_num and state["map"][current_location[0]][
                current_location[1] + 1] != "I":
                taxi_actions.append(("move", taxi, (current_location[0], current_location[1] + 1)))

            if current_location[0] - 1 >= 0 and state["map"][current_location[0] - 1][current_location[1]] != "I":
                taxi_actions.append(("move", taxi, (current_location[0] - 1, current_location[1])))

            if current_location[1] - 1 >= 0 and state["map"][current_location[0]][current_location[1] - 1] != "I":
                taxi_actions.append(("move", taxi, (current_location[0], current_location[1] - 1)))

        # checks for pickups
        for person in state["passengers"]:
            if current_location == state["passengers"][person]["location"] and \
                    state["passengers"][person]["location"] != state["passengers"][person]["destination"] and \
                    state["taxis"][taxi]["capacity"] > 0:
                taxi_actions.append(("pick up", taxi, person))

        taxi_actions.append(("wait", taxi))

        actions_by_taxi.append(taxi_actions)

    for element in itertools.product(*actions_by_taxi):
        locations = []
        flag = True
        for i in element:
            if i[0] == "move":
                locations.append(i[2])
            else:
                locations.append(state["taxis"][i[1]]["location"])
            if len(locations) != len(set(locations)):
                flag = False
                break
        if flag:
            actions.append(element)
    return actions


# def adjacency_list_creator(map1):
#     adj_dict = {}
#     for r_idx in range(len(map1)):
#         for c_idx in range(len(map1[0])):
#             if map1[r_idx][c_idx] != "I":
#                 adj_list = []
#                 if c_idx + 1 <= len(map1[0]) - 1 and map1[r_idx][c_idx + 1] != "I":
#                     adj_list.append((r_idx, c_idx + 1))
#
#                 if r_idx + 1 <= len(map1) - 1 and map1[r_idx + 1][c_idx] != "I":
#                     adj_list.append((r_idx + 1, c_idx))
#
#                 if r_idx - 1 >= 0 and map1[r_idx + - 1][c_idx] != "I":
#                     adj_list.append((r_idx - 1, c_idx))
#
#                 if c_idx - 1 >= 0 and map1[r_idx][c_idx - 1] != "I":
#                     adj_list.append((r_idx, c_idx - 1))
#
#                 adj_dict.update({(r_idx, c_idx): adj_list})
#
#     return adj_dict


class OptimalTaxiAgent:
    def __init__(self, initial):
        start = time.time()
        self.initial = initial
        self.initial_turns = self.initial["turns to go"]
        self.dest_by_person = []

        for person in initial["passengers"]:
            temp_set = set(initial["passengers"][person]["possible_goals"])
            temp_set.add(initial["passengers"][person]["destination"])
            temp_set = tuple(temp_set)
            self.dest_by_person.append(temp_set)

        self.passengers = list(initial["passengers"].keys())
        self.possible_destinations = []
        for destinations in itertools.product(*self.dest_by_person):
            self.possible_destinations.append(destinations)

        self.states = {}
        self.neighbors = {}
        self.create_states()

    # ------------------------------- just a check
    #     missing_states = set()
    #     for key in self.neighbors:
    #         for state, prob in self.neighbors[key]:
    #             if state not in self.states:
    #                 missing_states.add(state)
    #     for state in missing_states:
    #         print(state)
    # ------------------------------ just a check

        self.policy, self.value = self.VI()

        # print(f"Initial Runtime: {time.time() - start}")
        # print(f"Expected value: {self.value[(json.dumps(self.initial, sort_keys=True), self.initial['turns to go'])]}")

    def create_states(self):
        queue = FIFOQueue()
        initial = deepcopy(self.initial)
        str_initial = json.dumps(initial, sort_keys=True)
        queue.append(initial)
        step = 0
        num_of_states = 1
        while step <= self.initial["turns to go"]:
            for j in range(num_of_states):
                current_state = queue.pop()
                str_current_state = json.dumps(current_state, sort_keys=True)
                possible_actions = actions(current_state)
                for action in possible_actions:
                    new_state, im_reward = self.result(current_state, action)
                    if not self.states.get(str_current_state):
                        self.states[str_current_state] = []
                    if action == "reset":
                        if not self.neighbors.get((str_current_state, action)):
                            self.neighbors[(str_current_state, action)] = []
                        self.neighbors[(str_current_state, action)].append((str_initial, 1))
                        if step <= self.initial_turns - 1:
                            self.states[str_current_state].append((action, im_reward))

                    else:
                        if step <= self.initial_turns - 1:
                            self.states[str_current_state].append((action, im_reward))

                        for dest in self.possible_destinations:
                            new_state = deepcopy(new_state)
                            flag = True
                            for i, sub_dest in enumerate(dest):
                                initial_dest = self.initial["passengers"][self.passengers[i]]["destination"]
                                initial_goals = self.initial["passengers"][self.passengers[i]]["possible_goals"]
                                #
                                if initial_dest not in initial_goals:
                                    if current_state["passengers"][self.passengers[i]]["destination"] != initial_dest\
                                            and sub_dest == initial_dest:
                                        flag = False
                                        break
                                new_state["passengers"][self.passengers[i]]["destination"] = dest[i]
                            #
                            if not flag:
                                continue

                            str_new_state = json.dumps(new_state, sort_keys=True)

                            # if step <= self.initial_turns:
                            if step <= self.initial_turns - 1:
                                if not self.neighbors.get((str_current_state, action)):
                                    self.neighbors[(str_current_state, action)] = []
                                p = self.P(current_state, new_state)
                                self.neighbors[(str_current_state, action)].append((str_new_state, p))
                            else:
                                if not self.neighbors.get((str_current_state, action)):
                                    self.neighbors[(str_current_state, action)] = []

                            if not self.states.get(str_new_state) and new_state not in queue:
                                queue.append(new_state)

            step += 1
            num_of_states = len(queue)

        # print(f"len queue end: {len(queue)}")
        # while len(queue) > 0:
        #     temp_state = queue.pop()
        #     if not self.states.get(json.dumps(temp_state, sort_keys=True)):
        #         self.states[json.dumps(temp_state, sort_keys=True)] = []

    def VI(self):
        a_value = {}
        for key in self.states.keys():
            a_value[(key, 0)] = 0
        a_policy = {}
        i = 1
        while i < self.initial["turns to go"] + 1:
            for state in self.states.keys():
                Q = {}
                if len(self.states[state]) > 0:
                    for action, reward in self.states[state]:
                        neighbors = self.neighbors[(state, action)]
                        temp1 = 0
                        for neighbor in neighbors:
                            if a_value.get((neighbor[0], i - 1)):
                                tempo = a_value[(neighbor[0], i - 1)]
                            else:
                                tempo = 0
                            temp1 += neighbor[1] * tempo
                        Q[(action, i)] = reward + temp1

                    temp = max(Q, key=Q.get)
                    a_value[(state, i)] = max(Q.values())
                    a_policy[(state, temp[1])] = temp[0]

            i += 1

        return a_policy, a_value

    def P(self, state1, state2):
        probability = 1
        for passenger in self.initial["passengers"]:
            passenger_prob_change_goal = self.initial["passengers"][passenger]["prob_change_goal"]
            passenger_possible_goals = self.initial["passengers"][passenger]["possible_goals"]
            if state1["passengers"][passenger]["destination"] in passenger_possible_goals:
                if state1["passengers"][passenger]["destination"] == state2["passengers"][passenger]["destination"]:
                    probability = probability * ((1 - passenger_prob_change_goal) + (
                            passenger_prob_change_goal * (1 / len(passenger_possible_goals))))
                else:
                    probability = probability * (passenger_prob_change_goal *
                                                 (1/(len(passenger_possible_goals))))
                # לשאול את אריק על ההסתברות הזו ^

            else:
                if state1["passengers"][passenger]["destination"] == state2["passengers"][passenger]["destination"]:
                    probability = probability * (1 - passenger_prob_change_goal)
                else:
                    probability = probability * (passenger_prob_change_goal / len(passenger_possible_goals))
        return probability

    def act(self, state):
        turn = state["turns to go"]
        state1 = deepcopy(state)
        state1["turns to go"] = self.initial_turns
        state1 = json.dumps(state1, sort_keys=True)
        temp = self.policy[(state1, turn)]
        # print(temp)
        return temp

        # raise NotImplemented

    def refuel(self, state, action):
        state["taxis"][action[1]]["fuel"] = self.initial["taxis"][action[1]]["fuel"]
        return state

    def move(self, state, action):
        taxi_name = action[1]
        new_location = action[2]
        state["taxis"][taxi_name]["fuel"] -= 1
        state["taxis"][taxi_name]["location"] = new_location
        return state

    def pick_up(self, state, action):
        taxi_name = action[1]
        passenger_name = action[2]
        state["passengers"][passenger_name]["location"] = taxi_name
        state["taxis"][taxi_name]["capacity"] -= 1
        return state

    def drop_off(self, state, action):
        taxi_name = action[1]
        passenger_name = action[2]
        state["passengers"][passenger_name]["location"] = state["taxis"][taxi_name]["location"]
        state["taxis"][taxi_name]["capacity"] += 1
        return state

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        im_reward = 0
        if action == "reset":
            im_reward = -50
            state = deepcopy(self.initial)
        else:
            state = deepcopy(state)
            for taxi_action in action:
                if taxi_action[0] == "wait":
                    continue
                elif taxi_action[0] == "move":
                    state = self.move(state, taxi_action)
                elif taxi_action[0] == "pick up":
                    state = self.pick_up(state, taxi_action)
                elif taxi_action[0] == "drop off":
                    im_reward += 100
                    state = self.drop_off(state, taxi_action)
                elif taxi_action[0] == "refuel":
                    im_reward -= 10
                    state = self.refuel(state, taxi_action)
        return state, im_reward


MAX_DEST = 4
NEW_CYCLE_TRASHOLD = 75
class TaxiAgent:
    def __init__(self, initial):
        self.total_num_dest = 0
        self.original_taxis = initial["taxis"].keys()
        self.reduced_map = []
        self.distances_dict = {}
        self.desired_pass = []
        self.desired_taxi = ""
        self.initial = self.problem_reducer(initial)
        self.TURNS = self.turns_decider()
        self.initial_turns = self.initial["turns to go"]

        self.dest_by_person = []
        for person in self.initial["passengers"]:
            temp_set = set(self.initial["passengers"][person]["possible_goals"])
            temp_set.add(self.initial["passengers"][person]["destination"])
            temp_set = tuple(temp_set)
            self.dest_by_person.append(temp_set)

        self.neighbors = {}
        self.states = {}
        self.passengers = list(self.initial["passengers"].keys())
        self.possible_destinations = []
        for destinations in itertools.product(*self.dest_by_person):
            self.possible_destinations.append(destinations)

        self.create_states()
        self.policy, self.value = self.VI()
        self.reset_counter = 0
        self.turn = self.TURNS



    def turns_decider(self):
        state = self.initial
        if len(state["map"]) * len(state["map"][0]) >= 100:
            return min(state["turns to go"], 100)

        if len(state["passengers"].keys()) == 1 and len(state["taxis"].keys()) == 1:
            return state["turns to go"]

        if len(state["map"]) * len(state["map"][0]) <= 49:
            return min(state["turns to go"], 200)
        else:
            return min(state["turns to go"], 125)


    def problem_reducer(self, state):
        num_of_taxis = len(state["taxis"].keys())
        num_of_passengers = len(state["passengers"].keys())
        if num_of_taxis > 1 or num_of_passengers > 1:
            pass_dest_dict = {}
            dest_set_dict = {}
            for passenger in state["passengers"]:
                temp = state["passengers"][passenger]["destination"]
                dest_set = set(list(state["passengers"][passenger]["possible_goals"]))
                dest_set.add(temp)
                dest_set_dict[passenger] = dest_set
                pass_dest_dict[passenger] = len(dest_set)

            one_taxi_dest = {}
            maps = {}
            for taxi1 in state["taxis"].keys():
                distances_dict = {}
                map = deepcopy(state["map"])
                for taxi2 in state["taxis"].keys():
                    if taxi1 != taxi2:
                        map = self.map_i_changer(state["taxis"][taxi2]["location"], map)
                maps[taxi1] = map
                adj_dict = adjacency_list_creator(map)
                N = len(adj_dict.keys())
                for key in adj_dict.keys():
                    distances_dict[key] = bfsTraversal(adj_dict, N, key)
                one_taxi_dest[taxi1] = distances_dict

            taxi_pass_dest = {}
            passenger_scores_by_taxi = {}
            for taxi in state["taxis"].keys():
                passenger_scores = []
                for passenger in state["passengers"].keys():
                    a = state["taxis"][taxi]["location"] # taxi initial location
                    b = state["passengers"][passenger]["location"] # passenger initial location
                    if not taxi_pass_dest.get(taxi):
                        taxi_pass_dest[taxi] = []
                    if maps[taxi][b[0]][b[1]] == "I":
                        temp = float("inf")
                    else:
                        temp = one_taxi_dest[taxi][a][b]

                    if temp != float("inf"):
                        available_dest = 0
                        unavailable_initial_dest = 0
                        prob_change = state["passengers"][passenger]["prob_change_goal"]
                        for dest in dest_set_dict[passenger]:
                            if maps[taxi][dest[0]][dest[1]] == "I" or one_taxi_dest[taxi][b][dest] == float("inf"):
                                if dest == state["passengers"][passenger]["destination"]:
                                    unavailable_initial_dest = 1
                            else:
                                available_dest += 1

                        if available_dest != 0:
                            available_dest = available_dest / pass_dest_dict[passenger]
                            score = 0
                            if not unavailable_initial_dest:
                                score += 1
                                dist_to_dest = one_taxi_dest[taxi][b][state["passengers"][passenger]["destination"]]
                            else:
                                dist_to_dest = float("inf")
                                if prob_change <= 0.2:
                                    score -= 2
                                if prob_change <= 0.1:
                                    score -= 1.5

                            if available_dest == 1:
                                score += 0.5
                            if pass_dest_dict[passenger] == 1:
                                score += 1.25
                            else:
                                if prob_change >= 0.5:
                                    score -= 1.1
                                if prob_change <= 0.25:
                                    score += 0.5
                                if prob_change <= 0.05:
                                    score += 0.5

                            passenger_scores.append([passenger, score, pass_dest_dict[passenger], temp, dist_to_dest])

                min_dist_to_dest = float("inf")
                min_dist_to_pass = float("inf")
                for passenger in passenger_scores:
                    if passenger[4] < min_dist_to_dest:
                        min_dist_to_dest = passenger[4]
                    if passenger[3] < min_dist_to_pass:
                        min_dist_to_pass = passenger[3]
                for passenger in passenger_scores:
                    if passenger[4] == min_dist_to_dest:
                        passenger[1] += 1
                    if passenger[3] == min_dist_to_pass:
                        passenger[1] += 1
                        if min_dist_to_pass == 0:
                            passenger[1] += 1

                passenger_scores_by_taxi[taxi] = passenger_scores

            max_pass = 0
            max_pass_taxis = []
            for taxi in taxi_pass_dest.keys():
                passenger_scores_by_taxi[taxi].sort(key=lambda i: (i[1]))
                desired_pass = []
                if len(passenger_scores_by_taxi[taxi]) == 0:
                    continue
                min_pass = passenger_scores_by_taxi[taxi].pop()
                desired_pass.append(min_pass)
                total_dest = min_pass[2]
                if total_dest >= MAX_DEST:
                    passenger_scores_by_taxi[taxi] = desired_pass
                while total_dest < MAX_DEST:
                    if len(passenger_scores_by_taxi[taxi]) == 0:
                        passenger_scores_by_taxi[taxi] = desired_pass
                        break
                    temp = passenger_scores_by_taxi[taxi].pop()
                    if total_dest + temp[2] <= MAX_DEST:
                        desired_pass.append(temp)
                        total_dest += temp[2]
                    else:
                        continue
                passenger_scores_by_taxi[taxi] = desired_pass

                if len(passenger_scores_by_taxi[taxi]) > max_pass:
                    max_pass = len(passenger_scores_by_taxi[taxi])
                    max_pass_taxis = []
                    max_pass_taxis.append(taxi)
                elif len(passenger_scores_by_taxi[taxi]) == max_pass:
                    max_pass_taxis.append(taxi)

            desired_taxi = []
            if len(max_pass_taxis) == 1:
                desired_taxi = max_pass_taxis[0]
            else:
                max_mean_score = float('-inf')
                for taxi in max_pass_taxis:
                    mean_score = 0
                    for passenger, score, dest, dist_to_pass, dist_to_dest in passenger_scores_by_taxi[taxi]:
                        mean_score += score / len(passenger_scores_by_taxi[taxi])
                    if mean_score > max_mean_score:
                        desired_taxi = [taxi]
                        max_mean_score = mean_score
                    elif mean_score == max_mean_score:
                        desired_taxi.append(taxi)
                if len(desired_taxi) == 1:
                    desired_taxi = desired_taxi[0]
                else:
                    min_mean_distances = float("inf")
                    for taxi in desired_taxi:
                        mean_distances = 0
                        for passenger in passenger_scores_by_taxi[taxi]:
                            mean_distances += (passenger[3] + passenger[4]) / len(passenger_scores_by_taxi[taxi])
                        if mean_distances < min_mean_distances:
                            min_mean_distances = mean_distances
                            desired_taxi = [taxi]
                        elif mean_distances == min_mean_distances:
                            desired_taxi.append(taxi)
                    if len(desired_taxi) == 1:
                        desired_taxi = desired_taxi[0]
                    else:
                        minimum_dist = float("inf")
                        for taxi in desired_taxi:
                            min_pass_dist = float("inf")
                            for passenger in passenger_scores_by_taxi[taxi]:
                                pass_dist = passenger[3]
                                if pass_dist < min_pass_dist:
                                    min_pass_dist = pass_dist
                            if min_pass_dist < minimum_dist:
                                minimum_dist = min_pass_dist
                                desired_taxi = taxi


            for i, passenger in enumerate(passenger_scores_by_taxi[desired_taxi]):
                if i <= 1:
                    self.desired_pass.append(passenger[0])
                else:
                    break
            self.desired_taxi = desired_taxi

            reduced_state = deepcopy(state)
            reduced_state["passengers"] = {}
            for passenger in self.desired_pass:
                self.total_num_dest += pass_dest_dict[passenger]
                reduced_state["passengers"][passenger] = state["passengers"][passenger]

            reduced_state["taxis"] = {}
            reduced_state["taxis"][desired_taxi] = state["taxis"][desired_taxi]

            reduced_state["map"] = maps[desired_taxi]
            self.reduced_map = maps[desired_taxi]

            self.distances_dict = one_taxi_dest[desired_taxi]
            return reduced_state

        else:

            adj_dict = adjacency_list_creator(state["map"])
            N = len(adj_dict.keys())
            for key in adj_dict.keys():
                self.distances_dict[key] = bfsTraversal(adj_dict, N, key)
            self.reduced_map = state["map"]
            self.desired_pass = state["passengers"].keys()
            self.desired_taxi = list(self.original_taxis)[0]

            return state

    def distance_calculator(self, a, b):
        if self.distances_dict.get(a) is not None:
            if self.distances_dict[a].get(b) is not None:
                return self.distances_dict[a][b]
        return float("inf")

    def map_i_changer(self, location, map):
        map[location[0]][location[1]] = "I"
        return map

    def create_states(self):
        queue = FIFOQueue()
        initial = deepcopy(self.initial)
        str_initial = json.dumps(initial, sort_keys=True)
        queue.append(initial)
        step = 0
        num_of_states = 1

        turns = self.TURNS
        if self.initial["turns to go"] < self.TURNS:
            turns = self.initial["turns to go"]

        while step <= turns:
            for j in range(num_of_states):
                current_state = queue.pop()
                str_current_state = json.dumps(current_state, sort_keys=True)
                possible_actions = actions(current_state)
                for action in possible_actions:
                    new_state, im_reward = self.result(current_state, action)
                    if not self.states.get(str_current_state):
                        self.states[str_current_state] = []
                    if action == "reset":
                        if not self.neighbors.get((str_current_state, action)):
                            self.neighbors[(str_current_state, action)] = []
                        self.neighbors[(str_current_state, action)].append((str_initial, 1))
                        if step <= turns - 1:
                            self.states[str_current_state].append((action, im_reward))

                    else:
                        if step <= turns - 1:
                            self.states[str_current_state].append((action, im_reward))

                        for dest in self.possible_destinations:
                            new_state = deepcopy(new_state)
                            flag = True
                            for i, sub_dest in enumerate(dest):
                                initial_dest = self.initial["passengers"][self.passengers[i]]["destination"]
                                initial_goals = self.initial["passengers"][self.passengers[i]]["possible_goals"]
                                #
                                if initial_dest not in initial_goals:
                                    if current_state["passengers"][self.passengers[i]]["destination"] != initial_dest\
                                            and sub_dest == initial_dest:
                                        flag = False
                                        break
                                new_state["passengers"][self.passengers[i]]["destination"] = dest[i]

                            if not flag:
                                continue

                            str_new_state = json.dumps(new_state, sort_keys=True)

                            if step <= turns - 1:
                                if not self.neighbors.get((str_current_state, action)):
                                    self.neighbors[(str_current_state, action)] = []
                                p = self.P(current_state, new_state)
                                self.neighbors[(str_current_state, action)].append((str_new_state, p))
                            else:
                                if not self.neighbors.get((str_current_state, action)):
                                    self.neighbors[(str_current_state, action)] = []

                            if not self.states.get(str_new_state) and new_state not in queue:
                                queue.append(new_state)

            step += 1
            num_of_states = len(queue)

    def VI(self):
        a_value = {}
        for key in self.states.keys():
            a_value[(key, 0)] = 0
        a_policy = {}
        i = 1
        while i < self.TURNS + 1:
            for state in self.states.keys():
                Q = {}
                if len(self.states[state]) > 0:
                    for action, reward in self.states[state]:
                        neighbors = self.neighbors[(state, action)]
                        temp1 = 0
                        for neighbor in neighbors:
                            if a_value.get((neighbor[0], i - 1)):
                                tempo = a_value[(neighbor[0], i - 1)]
                            else:
                                tempo = 0
                            temp1 += neighbor[1] * tempo
                        Q[(action, i)] = reward + temp1

                    temp = max(Q, key=Q.get)
                    a_value[(state, i)] = max(Q.values())
                    a_policy[(state, temp[1])] = temp[0]

            i += 1

        return a_policy, a_value


    def enlarge_action(self, action):
        if action == "reset":
            return "reset"
        elif action == "terminate":
            return "terminate"

        full_action = []
        for taxi in self.original_taxis:
            if taxi != self.desired_taxi:
                full_action.append(("wait", taxi))
            else:
                full_action.append(action[0])
        return tuple(full_action)


    def act(self, state):
        state = deepcopy(state)
        passengers = {}
        for passenger in self.desired_pass:
            passengers[passenger] = state["passengers"][passenger]
        state["passengers"] = passengers

        taxis = {}
        taxis[self.desired_taxi] = state["taxis"][self.desired_taxi]
        state["taxis"] = taxis

        state["map"] = self.reduced_map

        if self.turn == 0:
            state["turns to go"] = self.initial_turns
            state_str = json.dumps(state, sort_keys=True)
            if state["turns to go"] >= self.TURNS + 1:
                self.turn = self.TURNS
                if self.value[(state_str, self.turn)] > NEW_CYCLE_TRASHOLD:
                    return "reset"
                else:
                    return "terminate"
            else:
                self.turn = state["turns to go"]
                if self.value[(state_str, self.turn)] > NEW_CYCLE_TRASHOLD:
                    return "reset"
                else:
                    return "terminate"

        else:
            state["turns to go"] = self.initial_turns
            state = json.dumps(state, sort_keys=True)
            temp = self.policy[(state, self.turn)]
            full_action = self.enlarge_action(temp)
            self.turn -= 1
            return full_action

        # raise NotImplemented



    def P(self, state1, state2):
        probability = 1
        for passenger in self.initial["passengers"]:
            passenger_prob_change_goal = self.initial["passengers"][passenger]["prob_change_goal"]
            passenger_possible_goals = self.initial["passengers"][passenger]["possible_goals"]
            if state1["passengers"][passenger]["destination"] in passenger_possible_goals:
                if state1["passengers"][passenger]["destination"] == state2["passengers"][passenger]["destination"]:
                    probability = probability * ((1 - passenger_prob_change_goal) + (
                            passenger_prob_change_goal * (1 / len(passenger_possible_goals))))
                else:
                    probability = probability * (passenger_prob_change_goal * (1 / len(passenger_possible_goals)))
            else:
                if state1["passengers"][passenger]["destination"] == state2["passengers"][passenger]["destination"]:
                    probability = probability * (1 - passenger_prob_change_goal)
                else:
                    probability = probability * (passenger_prob_change_goal / len(passenger_possible_goals))
        return probability

    def refuel(self, state, action):
        state["taxis"][action[1]]["fuel"] = self.initial["taxis"][action[1]]["fuel"]
        return state

    def move(self, state, action):
        taxi_name = action[1]
        new_location = action[2]
        state["taxis"][taxi_name]["fuel"] -= 1
        state["taxis"][taxi_name]["location"] = new_location
        return state

    def pick_up(self, state, action):
        taxi_name = action[1]
        passenger_name = action[2]
        state["passengers"][passenger_name]["location"] = taxi_name
        state["taxis"][taxi_name]["capacity"] -= 1
        return state

    def drop_off(self, state, action):
        taxi_name = action[1]
        passenger_name = action[2]
        state["passengers"][passenger_name]["location"] = state["taxis"][taxi_name]["location"]
        state["taxis"][taxi_name]["capacity"] += 1
        return state

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        im_reward = 0
        if action == "reset":
            im_reward = -50
            state = deepcopy(self.initial)
        else:
            state = deepcopy(state)
            for taxi_action in action:
                if taxi_action[0] == "wait":
                    continue
                elif taxi_action[0] == "move":
                    state = self.move(state, taxi_action)
                elif taxi_action[0] == "pick up":
                    state = self.pick_up(state, taxi_action)
                elif taxi_action[0] == "drop off":
                    im_reward += 100
                    state = self.drop_off(state, taxi_action)
                elif taxi_action[0] == "refuel":
                    im_reward -= 10
                    state = self.refuel(state, taxi_action)
        return state, im_reward
