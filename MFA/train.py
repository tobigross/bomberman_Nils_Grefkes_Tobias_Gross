import numpy as np
from typing import List, Tuple
import events as e
from .callbacks import state_to_features, bomberman_dist, ACTIONS


# Hyper parameters
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.3  # LEARNING_RATE
GAMMA = 0.3  # DAMPENING
GAMMA_EXP_N_ARR = np.ones(TRANSITION_HISTORY_SIZE)
for i_ in range(TRANSITION_HISTORY_SIZE):
    GAMMA_EXP_N_ARR[i_] = GAMMA ** i_
CPOT_NORM = 0.5

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = [0] * TRANSITION_HISTORY_SIZE
    self.num_inv_act = 0
    self.int_reward = 0
    with open("progression.txt", "w") as f:
        f.write("#round;#coins,#invalid_actions;integrated_reward\n")
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
    if not old_game_state:  # skip first step due to None type old board
        return

    #determine the distances to the closest coin for the old and new state
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
    if e.COIN_COLLECTED in events:  # sets the new distance to 0 if the coin is collected (prevents dividing by 0)
        dist_new = 0
    #f_args for the reward function
    potential_param = (any_coins, dist_old, dist_new)
    self.logger.debug(f'step {old_game_state["step"]:03d}: player ({px_old},{py_old}) -> ({px_new},{py_new})')
    self.logger.debug(f'step {old_game_state["step"]:03d}: coin @ ({ncx},{ncy}) rel: ({dist_old},{dist_new})')

    # compute total reward and store all information in transition history
    rtot = reward_total(self, events, potential_param)
    self.int_reward += rtot
    self.transitions.append([state_to_features(old_game_state),
                             self_action,
                             state_to_features(new_game_state),
                             rtot])
    self.transitions.pop(0)
    self.logger.debug(f'step {old_game_state["step"]:03d}: state={state_to_features(old_game_state)}')
    self.logger.debug(f'step {old_game_state["step"]:03d}: reward={rtot}')

    t = old_game_state["step"]
    if t >= TRANSITION_HISTORY_SIZE:  # wait until enough experience has been gathered

        tm3_state = self.transitions[0][0]  # update the latest state in history (here t minus 3)
        qtm3 = self.model[tuple(tm3_state)]  # get its Q-Values
        action_index = ACTIONS.index(self.transitions[0][1])
        self.logger.debug(f'step {old_game_state["step"]:03d}: update Q{tm3_state}={qtm3}')

        now_state = self.transitions[-1][2]
        vtn = np.max(self.model[tuple(now_state)])  # q-learning
        # Vtn = self.model[tuple(now_state)])[tn_action] # SARSA: action of t+n needed

        rewards = np.zeros(TRANSITION_HISTORY_SIZE)
        for i in range(TRANSITION_HISTORY_SIZE):
            rewards[i] = self.transitions[i][3]

        # Here we update Q = Q + alpha*(sum(r_i *gamma**i) + gamma**n * Vtn - Q)
        qtm3[action_index] += ALPHA * (np.sum(GAMMA_EXP_N_ARR * rewards)
                                       + GAMMA ** TRANSITION_HISTORY_SIZE * vtn
                                       - qtm3[action_index])
        self.logger.debug(f'step {old_game_state["step"]:03d}: to     Q{tm3_state}={qtm3}')
    return


def end_of_round(self, last_game_state: dict, last_action: str, events: Tuple[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The last game state.
    :param last_action: The action that you took.
    :param events: The last events that occurred.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    np.save("my-saved-model.npy", self.model)

    # write in progression round, score, num_inv_act
    with open("progression.txt", "a") as f:
        f.write(f'{last_game_state["round"]};{last_game_state["self"][1]};{self.num_inv_act};{self.int_reward}\n')
    # clear transitions
    self.transitions = [0] * TRANSITION_HISTORY_SIZE
    self.num_inv_act = 0
    self.int_reward = 0
    return


def reward_total(self, events: List[str], potential_param: Tuple[bool, int, int]) -> float:
    """

    Args:
        events: list of events occured in one game step
        potential_param: f_args for reward_from_coin_potential

    Returns:
        total reward
    """
    re = reward_from_events(self, events)
    rp = reward_from_coin_potential(self, potential_param)
    self.logger.debug(f'Reward R={re + rp} given, with R_e={re} and R_p={rp}')
    return re + rp


def reward_from_coin_potential(self, potential_param: Tuple[bool, int, int]) -> float:
    """
    Args:
        potential_param: Tuple(are there any coins?, old distance, new distance)

    Returns:
        reward from potential (proportional to the inverse of the distance)
    """
    any_coins, dist_last, dist_next = potential_param
    if any_coins and dist_next != 0:
        return CPOT_NORM * (1 / dist_next - 1 / dist_last)
    return 0


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event == e.INVALID_ACTION:
            self.num_inv_act += 1
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
