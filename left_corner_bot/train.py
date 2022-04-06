from collections import deque, namedtuple
from copy import copy
import numpy as np
from typing import List, Tuple
import events as e
import pickle
from .callbacks import state_to_features, bomberman_dist, ACTIONS,action_spec_f, create_all_Q
import pyastar2d
"""
This is part of the final project of the fundamental of machine learning course in winter semester 2021/22 Heidelberg.
It is a joint projekt of Nils Grefkes and Tobias Groß. All parts of this work were created in colaboration.
For grading purposes we marked who did most oft the work in which part.(N representing Nils Grefkes and T representing Tobias Groß)

"""

# Hyper parameters
n=1
ALPHA = 0.001  # LEARNING_RATE
GAMMA = 0.99  # DAMPENING
INT_DIST_INFINITY=18 #default dist.
CPOT_NORM = 0.5 #potential parameter


def setup_training(self):#T
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.num_inv_act = 0
    self.n=1
    self.q_values=np.zeros(6)
    self.i=0
    self.rew=0
    self.reward_buffer=[]
    self.old_game_state=[]

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):#T
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
    reduced_state_1=state_to_features(self,old_game_state,self.past_action)
    self.past_action.append(ACTIONS.index(self_action))
    #get values for potential rewards
    create_pot_param(self,old_game_state,new_game_state,events)
    #get rewards
    rewards=reward_total(self, events, self.potential_param,self.pot_param_2,self.pot_bomb,self.player_pot,self_action,old_game_state,new_game_state)
    #predict Y
    if old_game_state is None:
        Y=rewards
       
    else:
        reduced_state_2=state_to_features(self,new_game_state,self.past_action)
        create_all_Q(self,reduced_state_2)
        Y=rewards+GAMMA*np.max(self.q_values)
        
    #sum of rewards for matric
    self.rew=self.rew+rewards   
    #crate q
    X=action_spec_f(reduced_state_1,self_action)
    self.q=X@self.beta
    #update beta
    self.beta=self.beta+ALPHA*(Y-self.q)*X
    #reset varibles
    self.q=np.zeros(6)
    self.i=0
    self.red_state_end=reduced_state_2
    self.old_game_state=new_game_state
    
    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):#T
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.

    """
    rewards=reward_total(self, events, self.potential_param,self.pot_param_2,self.pot_bomb,self.player_pot,last_action,self.old_game_state,last_game_state)
    reduced_state_2=state_to_features(self,last_game_state,self.past_action)
    create_all_Q(self,reduced_state_2)
    Y=rewards+GAMMA*np.max(self.q_values)
    self.logger.debug(f'rewards{rewards}')
    reduced_state_1=self.red_state_end
    X=action_spec_f(reduced_state_1,last_action)
    self.q=X@self.beta
    #update beta
    self.beta=self.beta+ALPHA*(Y-self.q)*X
    loss=1/2*(Y-self.q)**2


    try:
        with open("progression.txt", "a") as f:
            f.write(f'{last_game_state["round"]};{last_game_state["self"][1]};{loss};{self.rew+rewards}\n')
    except PermissionError:
        print("could not save")
    try:
        
    # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.beta, file)
    except PermissionError:
        print("could not save")
    self.num_inv_act = 0
    self.rew=0
    self.reward_buffer.clear()
    



def reward_total(self, events: List[str], potential_param: Tuple[bool, int, int],pot_param_2,pot_bomb,pot_param_3,action,old_games_state,new_game_state) -> float:#T
    #this function give the reward sum
    re = reward_from_events(self, events)
    rp = reward_from_coin_potential(self, potential_param)
   # rc = reward_from_coin_potential(self,pot_param_2)
    #rplayer=reward_from_coin_potential(self,pot_param_3)
    buffer=reward_repeat_move(self,old_games_state,new_game_state)
    #rb=reward_bomb(pot_bomb)
    
    return re+buffer+rp#+rplayer+rp+rc+rb
def reward_repeat_move(self,old_game_state,new_game_state):#T
    #this function punishes repeated moves
    self.reward_buffer.append(old_game_state["self"][3])
    if len(self.reward_buffer)>10:
        if old_game_state["self"][3][0]==new_game_state["self"][3][0] and old_game_state["self"][3][1]==new_game_state["self"][3][1]:
            self.reward_buffer.pop(0)
            return -1
        if self.reward_buffer[8]==self.reward_buffer[10]:
            self.reward_buffer.pop(0)
            return -1
        self.reward_buffer.pop(0)
    
    return 0
        
def reward_from_coin_potential(self, potential_param: Tuple[bool, int, int]) -> float:#N
    #func. for potential rewards
    any_coins, dist_last, dist_next = potential_param
    if any_coins and dist_next != 0 and dist_last!=0:
        return CPOT_NORM * (1 / dist_next - 1 / dist_last)
    return 0
def reward_bomb(pot_bomb):#N
    #defensive rewards for bombs
    x_pos,y_pos,x_bomb,y_bomb,dist_bomb_x,dist_bomb_y,t_bomb=pot_bomb
    if x_pos==x_bomb and abs(dist_bomb_y)<4:
        if dist_bomb_x==0 and dist_bomb_y==0:
            return -0.5
        if x_pos%2==0:
           
            return -1/dist_bomb_y
        else:
            return 0
    if y_pos==y_bomb and abs(dist_bomb_x)<4:
        if y_pos%2==0:
            
            return -1/dist_bomb_x
        else:
            return 0
    else:
        return 0        
def reward_from_events(self, events: List[str]) -> float:#T
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.INVALID_ACTION: -0.5,
        e.CRATE_DESTROYED: 1.5,
        e.COIN_FOUND: 0.5,
        e.KILLED_SELF:-5,
        e.KILLED_OPPONENT:5,
        e.GOT_KILLED:-1,
        #e.BOMB_DROPPED:0.5,

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            
            reward_sum += game_rewards[event]
    return reward_sum
def create_pot_param(self,old_game_state,new_game_state,events):#T
    #a function that creates all parameters for all potentials
    px_old, py_old = old_game_state["self"][3]
    px_new, py_new = new_game_state["self"][3]
    coins = old_game_state["coins"]
    crates= list(np.where(old_game_state["field"]==1))
    crates=[crates[0],crates[1]]
    any_coins = bool(coins)
    any_crates= bool(crates)
    ncx = 0
    ncy = 0
    cx=0
    cy=0
    if any_coins:
        coins.sort(key=lambda x: bomberman_dist(x[0], x[1], px_old, py_old))
        ncx, ncy = coins[0]
    if crates[0]!=[]:
        crates.sort(key=lambda x: bomberman_dist(x[0], x[1], px_old, py_old))
        cx,cy=crates[0][0],crates[1][0]
    d_old=bomberman_dist(cx, cy, px_old, py_old)
    d_new=bomberman_dist(cx, cy, px_new, py_new)
    dist_old = bomberman_dist(ncx, ncy, px_old, py_old)
    dist_new = bomberman_dist(ncx, ncy, px_new, py_new)
    self.pot_param_2=(any_crates,d_old,d_new)
    if e.COIN_COLLECTED in events:
        dist_new = 0
    self.potential_param = (any_coins, dist_old, dist_new)
    if old_game_state["bombs"]!=[]:
        a=list(old_game_state["bombs"][0])

        t_bomb=list(a)[1]
        x_bomb,y_bomb=list(a)[0]
        dist_bomb_x=x_bomb-px_old
        dist_bomb_y=y_bomb-py_old
    else:
        dist_bomb_x=18
        dist_bomb_y=18
        x_bomb=21
        y_bomb=21
        t_bomb=10
    
    self.pot_bomb=(px_old,py_old,x_bomb,y_bomb,dist_bomb_x,dist_bomb_y,t_bomb)
    players=[num[3] for num in old_game_state["others"]]
    any_players = bool(players)
    if any_players:
        players.sort(key=lambda x: bomberman_dist(x[0], x[1], px_old, py_old))
        ncx,ncy=players[0]
    dist_players_old=bomberman_dist(ncx, ncy, px_old, py_old)
    dist_players_new=bomberman_dist(ncx, ncy, px_new, py_new)
    self.player_pot=(any_players, dist_players_old, dist_players_new)
    return 