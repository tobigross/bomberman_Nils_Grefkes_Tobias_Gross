import torch
import random
from .AgentNetCls import AgentNet
from collections import deque


class Agent:
    """"Imitation learning"""
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.curr_step = 0
        self.learn_every = 3
        self.warmup = 1e3
        self.save_every = 5e4
        self.sync_every = 1e3
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()
        self.net = AgentNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=2.5e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def cache(self, state, action):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        self.memory.append((state, action))

    def act(self, state):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        state = state.unsqueeze(0)
        return torch.argmax(self.net(state, model="target"))

    def act2(self, state):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        state = state.unsqueeze(0)
        action_values = self.net(state, model="target")
        return action_values.cpu().numpy()

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, action = map(torch.stack, zip(*batch))
        return state, action.squeeze()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        self.curr_step += 1
        if self.curr_step < self.warmup:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        state, action = self.recall()
        pred = self.net(state, model="online")  # [np.arange(0, self.batch_size), action]
        loss = self.loss_fn(pred, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        save_path = self.save_dir / f"agent_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=0.1
            ),
            save_path
        )

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
