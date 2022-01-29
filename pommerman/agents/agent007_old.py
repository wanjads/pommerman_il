
"""The base simple agent use to train agents.
This agent is also the benchmark for other agents.
"""

from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility
from ..constants import Action

from . import action_prune


class Agent007(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """

    def __init__(self, *args, **kwargs):
        super(Agent007, self).__init__(*args, **kwargs)

        # Keep track of recently visited uninteresting positions so that we
        # don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        self.attack_distance = 6
        self.powerup_distance = 2
        # Keep track of the previous direction to help with the enemy standoffs.
        self._prev_direction = None
        self.role = 0  # 0: unknown, 1: leader, 2: assistant
        self.old_message = [0, 0]
        self.ongoing_attack = False

    def act(self, obs, action_space):
        def convert_bombs(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                })
            return ret

        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        enemies = [constants.Item(e) for e in obs['enemies']]
        ammo = int(obs['ammo'])
        blast_strength = int(obs['blast_strength'])
        items, dist, prev = self._djikstra(
            board, my_position, bombs, enemies, depth=10)

        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]

        # action pruning
        valid_directions = action_prune.get_filtered_actions(obs)
        if len(valid_directions) == 0:
            valid_directions.append(Action.Stop.value)

        directions = [k for k in directions if k.value in valid_directions]

        message = self.communicator(obs)

        if not self.ongoing_attack:
            # Lay pomme if we are adjacent to an enemy.
            if self._is_adjacent_enemy(items, dist, enemies) and self._maybe_bomb(
                    ammo, blast_strength, items, dist, my_position, directions):
                return [constants.Action.Bomb.value] + message

            # Move towards an enemy if there is one in exactly self.attack_distance reachable spaces.
            direction = self._near_enemy(my_position, items, dist, prev, enemies, self.attack_distance)
            if direction is not None and (self._prev_direction != direction or
                                          random.random() < .5) and direction in directions:
                self._prev_direction = direction
                return [direction.value] + message

            # Move towards a good item if there is one within two reachable spaces.
            direction = self._near_good_powerup(my_position, items, dist, prev, self.powerup_distance)
            if direction is not None and direction in directions:
                return [direction.value] + message

            # Maybe lay a bomb if we are within a space of a wooden wall.
            if self._near_wood(my_position, items, dist, prev, 1):
                if self._maybe_bomb(ammo, blast_strength, items, dist, my_position, directions):
                    return [constants.Action.Bomb.value] + message

            # Move towards a wooden wall if there is one within two reachable spaces and you have a bomb.
            direction = self._near_wood(my_position, items, dist, prev, 2)
            if direction is not None and direction in directions:
                return [direction.value] + message

        # TODO Bewertungsnetzwerk mit Input obs und output value:
        # TODO das netzwerk könnte versuchen zu predicten, wie viele züge es nch dauert, bis der agent gewinnt
        # TODO für verlieren muss es sehr hohe werte ausgeben (z.b. 1000)

        if len(directions) >= 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        received_message = obs['message']

        # Choose a random but valid direction.
        probabilities = self.get_probabilities(
            directions, my_position, self._recently_visited_positions, received_message)

        # Add this position to the recently visited uninteresting positions so we don't return immediately.
        self._recently_visited_positions.append(my_position)
        self._recently_visited_positions = self._recently_visited_positions[
            -self._recently_visited_length:]

        return [np.random.choice(directions, p=probabilities).value] + message

    @staticmethod
    def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1, p_2):
            '''Determines if two points are out of rang of each other'''
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.Queue()

        my_x, my_y = my_position
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
                position = (r, c)
                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude),
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)
                
                if position == my_position:
                    Q.put(position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf


        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            position = Q.get()

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                val = dist[(x, y)] + 1
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + x, col + y)
                    if new_position not in dist:
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position
                        Q.put(new_position)
                    elif (val == dist[new_position] and random.random() < .5):
                        dist[new_position] = val
                        prev[new_position] = position   


        return items, dist, prev

    def _directions_in_range_of_bomb(self, board, my_position, bombs, dist):
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                        constants.Action.Right,
                        constants.Action.Left,
                        constants.Action.Up,
                        constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])
            elif x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _find_safe_directions(self, board, my_position, unsafe_directions,
                              bombs, enemies):

        def is_stuck_direction(next_position, bomb_range, next_board, enemies):
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board,
                                                        new_position, enemies):
                        continue

                    dist = abs(row + position_x - next_x) + abs(col + position_y - next_y)
                    Q.put((dist, new_position))
            return is_stuck

        # All directions are unsafe. Return a position that won't leave us locked.
        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(
                    my_position, direction)
                next_x, next_y = next_position
                if not utility.position_on_board(next_board, next_position) or \
                   not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board,
                                          enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []  # The directions that will go off the board.

        for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row, y + col)
            direction = utility.get_direction(my_position, position)

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position,
                                            enemies) or utility.position_is_fog(
                                                board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    @staticmethod
    def _maybe_bomb(ammo, blast_strength, items, dist, my_position, directions):
        """Returns whether we can safely bomb right now.

        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        if 0 not in [k.value for k in directions]:
            return False

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] == np.inf:
                continue

            # We can reach a passage that's outside of the bomb strength.
            if dist[position] > blast_strength:
                return True

            # We can reach a passage that's outside of the bomb scope.
            position_x, position_y = position
            if position_x != x and position_y != y:
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        dist_to = max(dist.values())

        for obj in objs:
            for position in items.get(obj, []):
                d = dist[position]
                if d <= radius and d <= dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    @staticmethod
    def _get_direction_towards_position(my_position, position, prev):
        if not position:
            return None

        next_position = position
        while prev[next_position] != my_position:
            next_position = prev[next_position]

        return utility.get_direction(my_position, next_position)

    @classmethod
    def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
        nearest_enemy_position = cls._nearest_position(dist, enemies, items,
                                                       radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_enemy_position, prev)

    @classmethod
    def _near_good_powerup(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_wood(cls, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                        board, position, enemies):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_unsafe_directions(board, my_position, directions, bombs):
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bomb_x, bomb_y = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bomb_x and abs(bomb_y - y) <= blast_strength) or \
                   (y == bomb_y and abs(bomb_x - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret

    def get_probabilities(self, directions, my_position, recently_visited_positions, received_message):

        team_direction, second_team_direction = self.get_team_direction(my_position, received_message)

        probs = []
        for direction in directions:
            if direction == team_direction and team_direction != constants.Action.Stop:
                team_boost = 5
            elif direction == second_team_direction and team_direction != constants.Action.Stop:
                team_boost = 3
            else:
                team_boost = 1
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                probs.append(5 * team_boost)
            else:
                probs.append(1 * team_boost)

        return np.array(probs)/sum(probs)

    def communicator(self, obs):

        received = obs['message']

        # if (8, 8) is received, the attack begins or ends
        if received == (8, 8) and not self.ongoing_attack:
            self.ongoing_attack = True
        elif received == (8, 8):
            self.ongoing_attack = False

        # decide if this agent is leader or assistant
        if not self.role and received != (0, 0):
            if 8 * received[0] + received[1] < 8 * self.old_message[0] + self.old_message[1]:
                self.role = 1
            elif not self.old_message == received:
                self.role = 2

        # if the agent has no role yet, send a random message, else communicate position with precision 1/2
        if not self.role:
            message = [random.randint(1, 8), random.randint(1, 8)]
        else:
            # assistant normal mode
            if not self.ongoing_attack and self.role == 2:
                message = [obs['position'][0]//2 + 2, obs['position'][1]//2 + 2]
            # leader normal mode may trigger an attack
            elif not self.ongoing_attack:
                if self.should_we_attack(obs):
                    message = [8, 8]
                else:
                    message = [obs['position'][0]//2 + 2, obs['position'][1]//2 + 2]
            # attack mode
            else:
                message = self.get_attack_message(obs)

        self.old_message = message
        return message

    def should_we_attack(self, obs):

        board = obs['board']
        teammate = obs['teammate'].value
        enemies = [e.value for e in obs['enemies']]
        my_pos = obs['position']

        # cut the right corner from the board:
        in_corner = False
        if my_pos[0] <= 4:
            if my_pos[1] <= 4:
                board = board[:5, :5]
                corner = [0, 0]
                in_corner = True
            elif my_pos[1] >= 6:
                board = board[:5, 6:]
                corner = [0, 1]
                in_corner = True
        elif my_pos[0] >= 6:
            if my_pos[1] <= 4:
                board = board[6:, :5]
                corner = [1, 0]
                in_corner = True
            elif my_pos[1] >= 6:
                board = board[6:, 6:]
                corner = [1, 1]
                in_corner = True

        target = -1
        # cut the board to see, if the enemy can be cornered
        if in_corner and teammate in board:
            teammate_pos = (np.where(board == teammate)[0][0], np.where(board == teammate)[1][0])
            if corner[0] == 0:
                if corner[1] == 0:
                    board = board[:max(teammate_pos[0], my_pos[0]), :max(teammate_pos[1], my_pos[1])]
                elif corner[1] == 1:
                    board = board[:max(teammate_pos[0], my_pos[0]), :min(teammate_pos[1], my_pos[1])]
            elif corner[0] == 1:
                if corner[1] == 0:
                    board = board[:min(teammate_pos[0], my_pos[0]), :max(teammate_pos[1], my_pos[1])]
                elif corner[1] == 1:
                    board = board[:min(teammate_pos[0], my_pos[0]), :min(teammate_pos[1], my_pos[1])]

            # Is teammate still in board?
            if teammate in board:
                if enemies[0] in board:
                    target = enemies[0]
                elif enemies[1] in board:
                    target = enemies[1]

        if target != -1:
            if corner[0] == corner[1]:
                if my_pos[0] > teammate_pos[0] and my_pos[1] < teammate_pos[1] or \
                        my_pos[0] < teammate_pos[0] and my_pos[1] > teammate_pos[1]:
                    return True
            else:
                if my_pos[0] > teammate_pos[0] and my_pos[1] > teammate_pos[1] or \
                        my_pos[0] < teammate_pos[0] and my_pos[1] < teammate_pos[1]:
                    return True

        return False

    def get_attack_message(self, obs):

        if self.return_to_normal(obs):
            return [8, 8]
        else:
            return [7, 7]  # TODO

    def return_to_normal(self, obs):
        return False  # TODO

    def get_team_direction(self, my_position, received_message):

        if received_message[0] > 1 and received_message[1] > 1:

            estimated_position = [(received_message[0] - 2) * 2 + random.randint(0, 1),
                                  (received_message[1] - 2) * 2 + random.randint(0, 1)]

            diff = [estimated_position[0] - my_position[0], estimated_position[1] - my_position[1]]

            if diff[0] >= diff[1]:
                if diff[0] >= 0:
                    team_direction = constants.Action.Down
                else:
                    team_direction = constants.Action.Up
                if diff[1] >= 0:
                    return team_direction, constants.Action.Right
                else:
                    return team_direction, constants.Action.Left
            else:
                if diff[1] >= 0:
                    team_direction = constants.Action.Right
                else:
                    team_direction = constants.Action.Left
                if diff[0] >= 0:
                    return team_direction, constants.Action.Down
                else:
                    return team_direction, constants.Action.Right

        else:
            return constants.Action.Stop, constants.Action.Stop
