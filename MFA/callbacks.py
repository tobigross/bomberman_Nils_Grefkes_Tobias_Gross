import os
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.npy"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.random.random((2, 2, 2, 2, 2, 3, 3, 3, 4))
    else:
        self.logger.info("Loading model from saved state.")
        self.model = np.load("my-saved-model.npy", allow_pickle=True)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    random_prob = 1/game_state["round"]
    if self.train and np.random.random() < random_prob:
        # 100%: walk in any direction. 0% wait. 0% bomb.
        action_ret = np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
        self.logger.debug(f'step {game_state["step"]:03d} : Random {action_ret}')
        return action_ret

    # self.logger.debug("Querying model for action.")
    reduced_state = state_to_features(game_state)
    q_values = self.model[tuple(reduced_state)]
    # eps greedy
    model_choice = np.argmax(q_values)
    self.logger.debug(f'step {game_state["step"]:03d} : Greedy {ACTIONS[model_choice]}')

    return ACTIONS[model_choice]


def state_to_features(game_state: dict) -> np.array:
    """
    list of the features returned:
    - wall left?
    -wall right?
    -wall up?
    -wall down?
    -any coins?
    -sign(dist_x)
    -sign(dist_y)
    -sign(dist_x-dist_y)

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    player_xpos = game_state["self"][3][0]
    player_ypos = game_state["self"][3][1]
    board = game_state["field"]
    wall_left = board[player_xpos - 1, player_ypos] == -1
    wall_right = board[player_xpos + 1, player_ypos] == -1
    wall_up = board[player_xpos, player_ypos + 1] == -1
    wall_down = board[player_xpos, player_ypos - 1] == -1

    # find closest coin by sorting all coins by distance
    coins = game_state["coins"]
    any_coins = len(coins) > 0
    cc_x_dist = 0
    cc_y_dist = 0
    # closest_coin_steps = 0
    if any_coins:
        coins.sort(key=lambda x: bomberman_dist(x[0], x[1], player_xpos, player_ypos))
        cc_xpos, cc_ypos = coins[0]
        cc_x_dist = cc_xpos - player_xpos
        cc_y_dist = cc_ypos - player_ypos
    dx_dy_sign = sign(abs(cc_x_dist) - abs(cc_y_dist))
    cc_x_sign = sign(cc_x_dist)
    cc_y_sign = sign(cc_y_dist)

    features = np.array([wall_left, wall_right, wall_up, wall_down, any_coins,
                         cc_x_sign + 1, cc_y_sign + 1, dx_dy_sign + 1
                         ])
    return features.astype(int)


def bomberman_dist(x1: int, y1: int, x2: int, y2: int) -> int:
    # one norm but with correction_term, which accounts for columns in even rows/columns
    dx = abs(x1-x2)
    dy = abs(y1-y2)
    if (dx == 0) and (dy == 0):
        return 0
    correction_term = [0, 2][((dx == 0) and (x1 % 2 == 0)) or ((dy == 0) and (y1 % 2 == 0))]
    return dx + dy + correction_term


def sign(x, eps=1e-8):
    if x < eps:
        return 0
    return x/abs(x)

