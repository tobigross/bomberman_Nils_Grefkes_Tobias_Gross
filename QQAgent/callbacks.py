from collections import deque
from .AgentCls import Agent
import numpy as np
import datetime
from pathlib import Path

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_to_index = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}
index_to_ACTIONS = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT', 4: 'WAIT', 5: 'BOMB'}

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
    load = False
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True)
    load_dir = Path('checkpoints') / "20220323_004723" / "agent_net_108.chkpt"  # Path('from_IL.chkpt')
    if load:
        self.logger.info("Loading Model from checkpoint")
        self.model = Agent(state_dim=308, action_dim=6, save_dir=save_dir, checkpoint=load_dir)
        self.logger.debug(f'loaded model with exploration rate={self.model.exploration_rate}')
    else:
        self.logger.info("Setting up model from scratch.")
        self.model = Agent(state_dim=308, action_dim=6, save_dir=save_dir, checkpoint=None)

    self.action_buffer = deque([4] * 6, maxlen=6)
    self.append_action_next_round = 4

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    reduced_game_state = state_to_features(self, game_state, normalize=True)
    use_softmax = False # toggle between softmax and argmax
    if use_softmax:
        action_prob = self.model.act2(reduced_game_state)
        action_index = np.random.choice(np.arange(6), p=np.squeeze(action_prob))
    else:
        action_index = self.model.act(reduced_game_state)
    return ACTIONS[action_index]


def state_to_features(self, game_state: dict, normalize=False) -> np.array:
    """
    visual_board:
        stone_walls : 0
        ###############
        nothing : 2
        ###############
        crates: 4
        coins: 5
        ##############
        explosion1: 10
        explosion2: 11
        ##############
        bomb0 : 13
        bomb1 : 14
        bomb2 : 15
        bomb3: 16
        ##############
        opp1: 20
        opp2: 21
        opp3: 22
        ##############
        self: 32
    rest:
        0: step
        1-6: last actions
        7-10: self ID-Bomb?-score
        ...: other ID-Bomb?-score ...
    """
    if game_state is None:
        return None

    board = (game_state["field"] + 1)*2
    coins = game_state["coins"]
    for coin in coins:
        board[coin[0], coin[1]] = 5
    explosion = game_state["explosion_map"]
    board[explosion > 0] = explosion[explosion > 0] + 9
    bombs = game_state["bombs"]
    for bomb in bombs:
        board[bomb[0][0], bomb[0][1]] = bomb[1] + 13
    rest = [0] * 13
    rest[0] = game_state["step"]
    pself = game_state["self"]
    if pself:
        board[pself[3][0], pself[3][1]] = 32
        rest[1] = 32
        rest[2] = pself[1]
        rest[3] = pself[2]
    others = game_state["others"]
    for i, p in enumerate(others):
        board[p[3][0], p[3][1]] = 20 + i
        rest[4 + 3 * i] = 20 + i
        rest[4 + 3 * i + 1] = p[1]
        rest[4 + 3 * i + 2] = p[2]

    f_ret = np.append(board.flatten(), rest).astype(float)

    # normalize features
    if normalize:
        f_ret[:17 * 17] /= 32  # board norm
        f_ret[17 * 17] /= 400  # rounds norm
        f_ret[[-3, -6, -9, -12]] /= 32  # player norm
        f_ret[[-2, -5, -8, -11]] /= 50  # score norm , might need adjustments

    # also include last actions
    action_append = np.array(self.action_buffer).astype(float)
    if normalize:
        action_append /= 5

    f_ret = np.append(f_ret, action_append)

    return f_ret
