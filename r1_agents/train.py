# rl_agent/train.py
import torch
import torch.optim as optim
import numpy as np
from networks import QNetwork
from replay_buffer import ReplayBuffer

# Hyperparams
STATE_DIM = 4        # example
ACTION_DIM = 2       # e.g. switch or hold
BUFFER_CAP = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE = 1000
MAX_STEPS = 50000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
target_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAP)

def select_action(state, eps):
    if np.random.rand() < eps:
        return np.random.randint(ACTION_DIM)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = policy_net(state_t)
    return q.argmax().item()

step_count = 0
for step in range(MAX_STEPS):
    # TODO: get initial state from env or dataset
    state = np.zeros(STATE_DIM)
    action = select_action(state, eps=max(0.01, 1 - step/MAX_STEPS))
    # TODO: apply action to env, receive reward, next_state, done
    reward, next_state, done = 0, state, False
    replay_buffer.push(state, action, reward, next_state, done)

    if len(replay_buffer) > BATCH_SIZE:
        s, a, r, ns, d = replay_buffer.sample(BATCH_SIZE)
        s_t = torch.FloatTensor(s).to(device)
        a_t = torch.LongTensor(a).unsqueeze(1).to(device)
        r_t = torch.FloatTensor(r).unsqueeze(1).to(device)
        ns_t = torch.FloatTensor(ns).to(device)
        d_t = torch.FloatTensor(d).unsqueeze(1).to(device)

        # Compute Q targets
        q_values = policy_net(s_t).gather(1, a_t)
        next_q = target_net(ns_t).max(1)[0].unsqueeze(1)
        target_q = r_t + GAMMA * next_q * (1 - d_t)

        loss = nn.MSELoss()(q_values, target_q) # type: ignore
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step_count % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    step_count += 1

    if done:
        # log metrics
        pass

# Save model
torch.save(policy_net.state_dict(), 'rl_agent/dqn_model.pth')