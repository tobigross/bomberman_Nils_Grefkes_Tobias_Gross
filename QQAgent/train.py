from typing import List, Tuple
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS_to_index
from collections import deque

# Hyper parameters
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
CPOT_NORM = 0.5

GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"
INSIDE_EXPLOSION_RADIUS = "INSIDE_EXPLOSION_RADIUS"

# Events

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug(f'Starting training')
    self.num_inv_act = 0
    self.integrated_reward = 0
    with open("progression.txt", "w") as f:
        f.write("#round;reward;#coins,#invalid_actions\n")
    return


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is None:
        self.logger.debug(f'weired first state: pos_next={new_game_state["self"][3]}, action={self_action}')
        return

    self.logger.debug(f'step={new_game_state["step"]:03d}: Encountered game event(s) {", ".join(map(repr, events))}')

    # reduce states and get action index
    state = state_to_features(self, old_game_state, normalize=True)
    action = ACTIONS_to_index[self_action]
    next_state = state_to_features(self, new_game_state, normalize=True)

    self.action_buffer.append(action)  # append action to last actions

    # compute rewards
    re = reward_from_events(self, events)
    rc = reward_from_coin_potential(self, old_game_state, new_game_state)
    rbd = 0
    if e.BOMB_DROPPED in events:
        rbd += bomb_placement_reward(new_game_state)
    rie = 0
    if old_game_state["bombs"]:
        rie += step_inside_explosion_rad(new_game_state["self"][3], new_game_state["bombs"])
    reward = re + rc + rbd + rie
    self.integrated_reward += reward

    self.logger.debug(f'step={new_game_state["step"]:03d}: reward constructed: re={re}, rc={rc}, rbd={rbd}, rie={rie}')
    self.logger.debug(f'step={new_game_state["step"]:03d}: reward r={reward} given for action a={action}')
    self.model.cache(state, next_state, action, reward, 0)
    self.logger.debug(f'step={new_game_state["step"]:03d}: cached o-a-n-r')
    q, loss = self.model.learn()
    self.logger.debug(f'step={new_game_state["step"]:03d}: learned q={q} , loss={loss}')
    return


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    state = state_to_features(self, last_game_state, normalize=True)
    action = ACTIONS_to_index[last_action]

    # compute custom events and give rewards
    reward = reward_from_events(self, events)
    self.model.cache(state, state, action, reward, 1)
    self.logger.debug(f'step={last_game_state["step"]:03d}: cached action={action}, reward={reward}')
    q, loss = self.model.learn()
    self.logger.debug(f'step={last_game_state["step"]:03d} learned q={q} , loss={loss}')

    # write integrated reward score and number of invalid actions to file
    with open("progression.txt", "a") as f:
        f.write(f'{last_game_state["round"]};{self.integrated_reward};{last_game_state["self"][1]};{self.num_inv_act}\n')

    # reset variables for next game
    self.num_inv_act = 0
    self.integrated_reward = 0
    self.action_buffer = deque([4] * 6, maxlen=6)
    return


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -0.5,
        e.KILLED_SELF: -10,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 1,
        e.GOT_KILLED: -5,
        e.WAITED: -0.1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event == e.INVALID_ACTION:
            self.num_inv_act += 1
    return reward_sum


def bomberman_dist(x1: int, y1: int, x2: int, y2: int) -> int:
    # one norm but with correction_term, which accounts for columns in even rows/columns
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if (dx == 0) and (dy == 0):
        return 0
    correction_term = [0, 2][((dx == 0) and (x1 % 2 == 0)) or ((dy == 0) and (y1 % 2 == 0))]
    return dx + dy + correction_term


def reward_from_coin_potential(self, old_game_state: dict, new_game_state: dict) -> float:
    """

    Args:
        self:
        old_game_state:
        new_game_state:

    Returns:
        reward: proportional to the inverse of the distance
    """
    px_old, py_old = old_game_state["self"][3]
    px_new, py_new = new_game_state["self"][3]
    coins = old_game_state["coins"]
    any_coins = bool(coins)
    ncx = 0
    ncy = 0
    if any_coins:
        coins.sort(key=lambda x: bomberman_dist(x[0], x[1], px_old, py_old))
        ncx, ncy = coins[0]
    dist_old = bomberman_dist(ncx, ncy, px_old, py_old)
    dist_new = bomberman_dist(ncx, ncy, px_new, py_new)

    if any_coins and dist_new != 0:
        return CPOT_NORM * (1 / dist_new - 1 / dist_old)
    return 0


def step_inside_explosion_rad(pos_new: Tuple[int, int], bombs: List) -> float:
    """
    
    Args:
        pos_new: 
        bombs: 

    Returns:
        reward: negative if inside the explosion radius
    """
    px, py = pos_new
    for bomb in bombs:
        (bx, by), cd = bomb
        if px == bx and py == by:
            dist = 0
        elif px == bx and px % 2 == 1:
            dist = abs(py - by)
        elif px == bx and py % 2 == 1:
            dist = abs(px - bx)
        else:
            dist = 5
        save_dist = 3 - cd
        if dist < save_dist:
            return -5
    return 0


def bomb_placement_reward(new_game_state: dict) -> float:
    """

    Args:
        new_game_state:

    Returns:
        reward: becomes positive if placed directly adjacent to a box or player
    """
    board = new_game_state["field"]
    player_or_crate = board == 1
    for p in new_game_state["others"]:
        player_or_crate[p[3]] = True
    pos = new_game_state["self"][3]
    # padding
    pad_pos = (pos[0] + 3, pos[1] + 3)
    pad_poc = np.pad(player_or_crate, 3)
    reward_map = np.zeros((23, 23))
    rtmp = np.array([0.25, 0.5, 1, 0, 1, 0.5, 0.25]) # !säulen
    reward_map[pad_pos[0], pad_pos[1] - 3:pad_pos[1] + 4] = rtmp
    reward_map[pad_pos[0] - 3:pad_pos[0] + 4, pad_pos[1]] = rtmp
    reward = - 0.9
    reward += np.sum(reward_map[pad_poc])
    return reward
