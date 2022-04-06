#from cmath import nan
#from http.client import _DataType
import os
from tkinter import TRUE
import numpy as np
from collections import deque, namedtuple
import pickle
#from pathlib import Path
#from scipy.special import softmax
#from pyinstrument import Profiler
import pyastar2d
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT','WAIT','BOMB']
N_f=49
INT_DIST_INFINITY=18
"""
This is part of the final project of the fundamental of machine learning course in winter semester 2021/22 Heidelberg.
It is a joint projekt of Nils Grefkes and Tobias Groß. All parts of this work were created in colaboration.
For grading purposes we marked who did most oft the work in which part.(N representing Nils Grefkes and T representing Tobias Groß)

"""
def setup(self):#T
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
    #if self.train or not os.path.isfile("my-saved-model.pt"):
     #   self.logger.info("Setting up model from scratch.")
      #  a=len(ACTIONS)*N_f+1
       # self.beta = np.random.random(a)
    #else:
     #   self.logger.info("Loading model from saved state.")
      #  with open("my-saved-model.pt", "rb") as file:
       #     self.beta = pickle.load(file)
    a=len(ACTIONS)*N_f
    self.beta = np.random.random(a)
    with open("my-saved-model.pt", "rb") as file:
        self.beta = pickle.load(file)
    self.getID = None
    self.past_action=deque([4]*3,maxlen=3)
    self.past_action_a=deque([4]*3,maxlen=3)
    
    

def act(self, game_state: dict) -> str:#T
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.getID is None:
        self.getID = {}
        self.getScore = {}
        for i,o in enumerate(game_state["others"]):
            self.getID[i+1] = o[0]
            self.getScore[o[0]] = 0
    # todo Exploration vs exploitation
    self.q_values=np.zeros(6)
    self.i=0
    random_prob = 1/(game_state["round"])
    self.round=game_state["round"]
    if game_state["step"]<2 and game_state["round"]==1:
        action_ret = np.random.choice(ACTIONS, p=[1/6,1/6,1/6,1/6,1/6,1/6])
        
        return action_ret
    if self.train and np.random.random() < random_prob:
        # 100%: walk in any direction. 0% wait. 0% bomb.
        action_ret = np.random.choice(ACTIONS, p=[1/6,1/6,1/6,1/6,1/6,1/6])
        self.logger.debug(f'step {game_state["step"]:03d} : Random {action_ret}')
        return action_ret
    
    reduced_state = state_to_features(self,game_state,self.past_action)
    create_all_Q(self,reduced_state)
    soft=1
    if soft == 0:
        model_choice = softmax(self)
        model_choice= np.random.choice(ACTIONS,p=model_choice)
        self.past_action_a.append(ACTIONS.index(model_choice))
        self.q_values=np.zeros(6)
        self.i=0
        return (model_choice)
    else:
        model_choice=np.argmax(self.q_values)
        self.past_action_a.append(model_choice)
        self.q_values=np.zeros(6)
        self.i=0
        return ACTIONS[model_choice]


def state_to_features(self, game_state,past_actions_list,normalize=TRUE):#N
    """
    returns features based on current game_state
    side effects: updates player scores stored in self.getScore, writes to log file
    awareness:
        -direction is invalid
        - can move in dead end TODO
        - good place to bomb:
            -adjacend to enemy
            -adjacend to crate
    pathfinding:
        -closest coin:
            -am i there first?
            -distance
            -direction
        -near crate:
            -distance
            -direction
        -nearest enemy:
            - number of connected enemies
            - distance to closest enemy
            - direction to closest enemy
    live_saving:
        -waiting/bombing  is unsave
        -direction of choice is unsave
    """
    field = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    player_ID, player_score, player_can_bomb, player_pos = game_state["self"]
    px, py = player_pos
    others = game_state["others"]
    if others==[]:
        others_ID, others_score, others_can_bomb, others_pos=[],[],[],[]
    else:
        others_ID, others_score, others_can_bomb, others_pos = map(list, zip(*others))
    for ID,score in zip(others_ID,others_score):
        self.getScore[ID]=score
    # valid
    up_is_valid_and_safe = (field[px, py + 1] == 0) and is_new_pos_save((px, py + 1), bombs, explosion_map)
    right_is_valid_and_safe = (field[px + 1, py] == 0) and is_new_pos_save((px + 1, py), bombs, explosion_map)
    down_is_valid_and_safe = (field[px, py - 1] == 0) and is_new_pos_save((px, py - 1), bombs, explosion_map)
    left_is_valid_and_safe = (field[px - 1, py] == 0) and is_new_pos_save((px - 1, py), bombs, explosion_map)
    
    #
    current_pos_is_save,_dir = am_i_in_danger(field,(px, py),bombs)
    can_bomb = player_can_bomb
    should_bomb = False
    # coins
    coin_exists = len(coins) > 0
    i_am_closest_to_closest_coin = False
    closest_coin_dist = INT_DIST_INFINITY
    up_is_towards_closest_coin = False
    right_is_towards_closest_coin = False
    down_is_towards_closest_coin = False
    left_is_towards_closest_coin = False
    if coin_exists:
        # get distance map of closest coin
        my_dist = INT_DIST_INFINITY
        closes_coin_pos = coins[0]
        for coin in coins:
            coin_distance = astar_dist(field,player_pos,coin)
            if coin_distance < my_dist:
                closes_coin_pos = coin
        i_am_closest_to_closest_coin = True
        for o_pos in others_pos:
            if astar_dist(field,o_pos,closes_coin_pos) < my_dist:
                i_am_closest_to_closest_coin = False
        closest_coin_dist, ccdir = astar_dist_and_dir(field,player_pos,closes_coin_pos)
        up_is_towards_closest_coin = ccdir[0]
        right_is_towards_closest_coin = ccdir[1]
        down_is_towards_closest_coin = ccdir[2]
        left_is_towards_closest_coin = ccdir[3]
    #crates
    crate_exists = np.any(field > 0)
    
    closest_crate_dist = INT_DIST_INFINITY
    up_is_towards_crate = False
    right_is_towards_crate = False
    down_is_towards_crate = False
    left_is_towards_crate = False

    if crate_exists:
        closest_crate_dist, ccrdir = best_crate_dist_and_dir(field, player_pos)
        up_is_towards_crate = ccrdir[0]
        right_is_towards_crate = ccrdir[1]
        down_is_towards_crate = ccrdir[2]
        left_is_towards_crate = ccrdir[3]
        if closest_crate_dist == 0:
            should_bomb = True
    # enemies
    enemy_1_ID = self.getID.get(1, "default")
    enemy_1_alive = enemy_1_ID in others_ID
    enemy_1_dist = INT_DIST_INFINITY
    up_is_towards_enemy_1 = False
    right_is_towards_enemy_1 = False
    down_is_towards_enemy_1 = False
    left_is_towards_enemy_1 = False
    enemy_1_has_higher_score_than_me = self.getScore.get(enemy_1_ID, 0) > player_score
    if enemy_1_alive:
        enemy_1_index = others_ID.index(enemy_1_ID)
        enemy_1_dist, enemy_1_dir = astar_dist_and_dir(field,player_pos,others_pos[enemy_1_index])
        if enemy_1_dist==1:
            should_bomb = True
        up_is_towards_enemy_1 = enemy_1_dir[0]
        right_is_towards_enemy_1 = enemy_1_dir[1]
        down_is_towards_enemy_1 = enemy_1_dir[2]
        left_is_towards_enemy_1 = enemy_1_dir[3]

    enemy_2_ID = self.getID.get(2, "default")
    enemy_2_alive = enemy_2_ID in others_ID
    enemy_2_dist = INT_DIST_INFINITY
    up_is_towards_enemy_2 = False
    right_is_towards_enemy_2 = False
    down_is_towards_enemy_2 = False
    left_is_towards_enemy_2 = False
    enemy_2_has_higher_score_than_me = self.getScore.get(enemy_2_ID, 0) > player_score
    if enemy_2_alive:
        enemy_2_index = others_ID.index(enemy_2_ID)
        enemy_2_dist, enemy_2_dir = astar_dist_and_dir(field,player_pos,others_pos[enemy_2_index])
        if enemy_2_dist==1:
            should_bomb = True
        up_is_towards_enemy_2 = enemy_2_dir[0]
        right_is_towards_enemy_2 = enemy_2_dir[1]
        down_is_towards_enemy_2 = enemy_2_dir[2]
        left_is_towards_enemy_2 = enemy_2_dir[3]

    enemy_3_ID = self.getID.get(3, "default")
    enemy_3_alive = enemy_3_ID in others_ID
    enemy_3_dist = INT_DIST_INFINITY
    up_is_towards_enemy_3 = False
    right_is_towards_enemy_3 = False
    down_is_towards_enemy_3 = False
    left_is_towards_enemy_3 = False
    enemy_3_has_higher_score_than_me = self.getScore.get(enemy_3_ID, 0) > player_score
    if enemy_3_alive:
        enemy_3_index = others_ID.index(enemy_3_ID)
        enemy_3_dist, enemy_3_dir = astar_dist_and_dir(field,player_pos,others_pos[enemy_3_index])
        if enemy_3_dist==1:
            should_bomb = True
        up_is_towards_enemy_3 = enemy_3_dir[0]
        right_is_towards_enemy_3 = enemy_3_dir[1]
        down_is_towards_enemy_3 = enemy_3_dir[2]
        left_is_towards_enemy_3 = enemy_3_dir[3]
    
    enemy_is_connected_with_me = min(enemy_1_dist, enemy_2_dist, enemy_3_dist) < INT_DIST_INFINITY
    # construct_feature_vector
    feature_vector = np.array([
        up_is_valid_and_safe*1,
        right_is_valid_and_safe*1,
        down_is_valid_and_safe*1,
        left_is_valid_and_safe*1,
        current_pos_is_save*1,
        can_bomb*1,
        should_bomb*1,
        coin_exists*1,
        i_am_closest_to_closest_coin*1,
        closest_coin_dist,
        up_is_towards_closest_coin*1,
        right_is_towards_closest_coin*1,
        down_is_towards_closest_coin*1,
        left_is_towards_closest_coin*1,
        crate_exists*1,
        closest_crate_dist,
        up_is_towards_crate*1,
        right_is_towards_crate*1,
        down_is_towards_crate*1,
        left_is_towards_crate*1,
        enemy_is_connected_with_me*1,
        enemy_1_alive*1,
        enemy_1_dist,
        up_is_towards_enemy_1*1,
        right_is_towards_enemy_1*1,
        down_is_towards_enemy_1*1,
        left_is_towards_enemy_1*1,
        enemy_1_has_higher_score_than_me*1,
        enemy_2_alive*1,
        enemy_2_dist,
        up_is_towards_enemy_2*1,
        right_is_towards_enemy_2*1,
        down_is_towards_enemy_2*1,
        left_is_towards_enemy_2*1,
        enemy_2_has_higher_score_than_me*1,
        enemy_3_alive*1,
        enemy_3_dist,
        up_is_towards_enemy_3*1,
        right_is_towards_enemy_3*1,
        down_is_towards_enemy_3*1,
        left_is_towards_enemy_3*1,
        enemy_3_has_higher_score_than_me*1,
        past_actions_list[0],
        past_actions_list[1],
        past_actions_list[2],
        _dir[0]*1,
        _dir[1]*1,
        _dir[2]*1,
        _dir[3]*1

    ])
    
    if normalize:
        feature_vector = feature_vector.astype(np.float32)
        dist_upper_limit = 16 # dont go below 5
        feature_vector[feature_vector>dist_upper_limit] = dist_upper_limit

        feature_vector[[9,15,22,29,36]] /= dist_upper_limit
        feature_vector[[-1,-2,-3,-4]] /= 5 #actions betweem 0 and 5
    
    return feature_vector

def create_all_Q(self,reduced_state):#T
    #function to be called to create all q values
    create_q(self,reduced_state,"UP")
    create_q(self,reduced_state,"RIGHT")
    create_q(self,reduced_state,"DOWN")
    create_q(self,reduced_state,"LEFT")
    create_q(self,reduced_state,"WAIT")
    create_q(self,reduced_state,"BOMB")
    return



def sign(x, eps=1e-8):#N
    if x < eps:
        return 0
    return x/abs(x)

def action_spec_f(features,action):#T
    #function for reating the longer version of the features
    f=np.zeros(len(features)*6)
    if action=="UP":
        f[:len(features)]=features
    elif action=="RIGHT":
        f[len(features):len(features)*2]=features
    elif action=="DOWN":
        f[len(features)*2:len(features)*3]=features
    elif action=="LEFT":
        f[len(features)*3:len(features)*4]=features
    elif action=="WAIT":
        f[len(features)*4:len(features)*5]=features
    else :
        f[len(features)*5:len(features)*6]=features
    f=np.append(f,1)
    return f


def softmax(self):#T
    #softmax implimentation
    #subtract max to get stable comp.
    self.q_values=self.q_values-np.max(self.q_values)
    temp=1/3#(1/self.round)**2
    a=np.exp(self.q_values/temp)
    b=np.sum(a)
    return a/b


def create_q(self,reduced_state,action):#T
    #function to handle q creation
    X=action_spec_f(reduced_state,action)
    q=X@self.beta
    #self.logger.debug(f'one q values to be appendet: {X}')
    
    self.q_values[self.i]=q
    self.i=self.i+1
    return   

def bomberman_dist(x1: int, y1: int, x2: int, y2: int) -> int:#N
    # one norm but with correction_term, which accounts for columns in even rows/columns
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if (dx == 0) and (dy == 0):
        return 0
    correction_term = [0, 2][((dx == 0) and (x1 % 2 == 0)) or ((dy == 0) and (y1 % 2 == 0))]
    return dx + dy + correction_term


def astar_dist(board,pos,target,crate_weight=np.inf):#N
    weights = np.ones(board.shape,dtype=np.float32)
    weights[board==-1]=np.inf
    weights[board==1]=crate_weight
    path = pyastar2d.astar_path(weights, pos, target, allow_diagonal=False)
    if path is not None:
        return len(path)-1
    return INT_DIST_INFINITY

def astar_dist_and_dir(board,pos,target,crate_weight=np.inf):#N
    weights = np.ones(board.shape,dtype=np.float32)
    weights[board==-1]=np.inf
    weights[board==1]=crate_weight
    path = pyastar2d.astar_path(weights, pos, target, allow_diagonal=False)
    _dir = np.zeros(4,dtype="?")
    if path is not None:
        distance = len(path) -1
        pos =  np.array(pos)
        for i,direction in enumerate(np.array([[0,1],[1,0],[0,-1],[-1,0]])):
            adj_pos = pos + direction
            adj_path = pyastar2d.astar_path(weights, tuple(adj_pos), target, allow_diagonal=False)
            if (adj_path is not None) and len(adj_path)<len(path):
                _dir[i]=True
        return distance, _dir
    return INT_DIST_INFINITY, _dir

def am_i_first_at_target(board,my_pos,enemies_pos,target):#N
    my_dist = astar_dist(board,my_pos,target)
    for e_pos in enemies_pos:
        if astar_dist(board,e_pos,target)<my_dist:
            return False
    return True

def is_inside_explosion_radius(px,py,bx,by):#N
    if px==bx and py==by:
        return True
    dx=abs(px-bx)
    dy=abs(py-by)
    if px%2==1 and dx==0 and dy<4:
        return True
    if py%2==1 and dy==0 and dx<4:
        return True
    return False

def is_my_pos_save(pos,bombs):#N
    px, py  = pos
    for (bx, by), bt in bombs:
        if bt<1 and is_inside_explosion_radius(px,py,bx,by):
            return False
        if bt==1 and is_inside_explosion_radius(px,py,bx,by) and (px%2==0 or py%2==0):
            return False
    return True

def reduce_bomb_timer(bombs):#N
    return [[(bx,by),bt-1] for (bx,by),bt in bombs]

def is_new_pos_save(new_pos,bombs,explosion_map):#N
    
    if not is_my_pos_save(new_pos,reduce_bomb_timer(bombs)):
        return False
    if explosion_map[new_pos]>0:
        return False
    return True

def get_closest_crate(board,pos):#N
    crates=np.column_stack(np.where(board>0))
    dcmin = INT_DIST_INFINITY
    crate_min_pos=crates[0]
    for crate in crates:
        dc = bomberman_dist(pos[0], pos[1] ,crate[0], crate[1])
        if dc<dcmin:
            dcmin=dc
            crate_min_pos=tuple(crate)
    return dcmin, tuple(crate_min_pos)

def generate_explosion_mask(board,bomb_pos):#N
    ma_ret = np.zeros(board.shape,dtype=bool)
    ma_ret[bomb_pos]=True
    bomb_pos = np.array(bomb_pos)
    directions =  np.array([[0,1],[0,-1],[1,0],[-1,0]])
    for direction in directions:
        for factor in range(1,4):
            new_pos = bomb_pos + factor*direction
            if board[tuple(new_pos)]>-1:
                ma_ret[tuple(new_pos)]=True
            else:
                break
    return ma_ret

def number_of_destroyed_crates(board,pos):#N
    return np.sum(board[generate_explosion_mask(board,pos)]==1)

def find_best_local_bomb_spot(board,pos):#N
    pos = np.array(pos)
    best_ndcm = number_of_destroyed_crates(board,tuple(pos))
    best_pos = pos
    directions = np.array([[0,1],[1,0],[0,-1],[-1,0]])
    for step1 in directions:
        p1 = pos + step1
        if board[tuple(p1)]==0:
            ndcm1 = number_of_destroyed_crates(board,tuple(p1))
            if ndcm1>best_ndcm:
                best_ndcm=ndcm1
                best_pos = p1
            for step2 in directions:
                p2 = p1 + step2
                if board[tuple(p2)]==0:
                    ndcm2 = number_of_destroyed_crates(board,tuple(p1))
                    if ndcm2>best_ndcm:
                        best_ndcm = ndcm2
                        best_pos = p2
    return tuple(best_pos)

def best_crate_dist_and_dir(board,pos):#N
    ccdist, ccpos = get_closest_crate(board,pos)
    if ccdist>2:
        return astar_dist_and_dir(board,pos,ccpos)
    else:
        best_local_pos = find_best_local_bomb_spot(board,pos)
        return astar_dist_and_dir(board,pos,ccpos)
def can_reach_after_n_steps(board,pos,n=1): #dont use for large n #N
    crm = np.zeros(board.shape,dtype="?")
    crm[pos]=True
    if n:
        pos = np.array(pos)
        directions= np.array([[0,1],[1,0],[0,-1],[-1,0]])
        stack = [(pos+direction,n-1) for direction in directions]
        while stack:
            new_pos, n = stack.pop(0)
            new_pos_tup=tuple(new_pos)
            if board[new_pos_tup]==0:
                crm[new_pos_tup]=True
                if n:
                    for direciton in directions:
                        stack.append((new_pos+direciton,n-1))
    return crm

def safe_tile_dir(board,pos,bomb_pos,bomb_t):#N
    em = ~ generate_explosion_mask(board,bomb_pos)
    _dir = np.zeros(4,dtype="?")
    directions= np.array([[0,1],[1,0],[0,-1],[-1,0]])
    for i,direction in enumerate(directions):
        new_pos = tuple(np.array(pos) + direction)
        if board[new_pos]==0:
            crm  = can_reach_after_n_steps(board,new_pos,bomb_t)
            
            if np.any(em[crm]):
                _dir[i]=True
    return _dir

def am_i_in_danger(board,pos,bombs):#N
    danger_flag = False
    _dir = np.ones(4,dtype="?")
    for bomb_pos,bomb_t in bombs:
        em = generate_explosion_mask(board,bomb_pos)
        if em[pos]:
            danger_flag = True
            _dir = _dir & (safe_tile_dir(board,pos,bomb_pos,bomb_t))
    return danger_flag, _dir