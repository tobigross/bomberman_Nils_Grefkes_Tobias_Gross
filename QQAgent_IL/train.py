from typing import List
from .callbacks import state_to_features
from .AgentCls import Agent
from pathlib import Path
import datetime

ACTIONS_to_index = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    load = False
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True)
    load_dir = Path('checkpoints') / "20220317_173151" / "agent_net_72.chkpt"
    # print(f"debug: {load_dir.exists()}")
    if load:
        self.logger.info("Loading Model from checkpoint")
        self.model = Agent(state_dim=308, action_dim=6, save_dir=save_dir, checkpoint=load_dir)
    else:
        self.logger.info("Setting up model from scratch.")
        self.model = Agent(state_dim=308, action_dim=6, save_dir=save_dir, checkpoint=None)
    self.test_model_gui = False  # set to false
    self.logger.debug(f'Starting training')
    self.train_round = True
    self.true_pred = 0
    self.false_pred = 0
    with open("prog_cl.txt", "w") as f:
        f.write("#round;err\n")
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
        return
    reduced_old_state = state_to_features(self, old_game_state, normalize=True)
    try:
        action = ACTIONS_to_index[self_action]
    except KeyError:  # sometimes the RBA does not select an action -> agent waits
        action = 4
    if self.train_round:
        self.model.cache(reduced_old_state, action)
        loss = self.model.learn()
    else:
        action_pred = self.model.act(reduced_old_state)
        if action_pred == action:
            self.true_pred += 1
        else:
            self.false_pred += 1
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
    rounds = last_game_state["round"]
    if not self.train_round:
        total_steps = self.true_pred + self.false_pred
        error = self.false_pred / total_steps
        self.logger.debug(f' EPOCH {rounds:03d}: Error={error}')
        self.true_pred = 0
        self.false_pred = 0
        with open("prog_cl.txt", "a") as f:
            f.write(f"{last_game_state['round']:05d};{error}\n")
    self.train_round = rounds % 50 != 49
    return
