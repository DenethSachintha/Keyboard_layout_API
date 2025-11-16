import torch
import numpy as np
from collections import OrderedDict
from keyboard_env import LETTERS, LETTER_TO_IDX, IDX_TO_LETTER

class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def load_model(model_path):
    model = PolicyNet(input_dim=702, n_actions=325)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different save formats
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Fix key names if needed
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('net.'):
            new_state_dict[k] = v
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict['net.' + k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def generate_layout(policy, letter_freqs, bigram_freqs, top9_list=None, steps=300, start_layout_qwerty=True):
    from keyboard_env import KeyboardEnv
    
    env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=steps)
    if start_layout_qwerty:
        obs = env.reset(start_layout=env._qwerty_layout_indices())
    else:
        obs = env.reset()
    
    best_score = env.prev_score
    best_layout = env.layout.copy()

    for _ in range(steps):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_t)

            # 🔥 FIX: Use sampling instead of deterministic argmax
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        obs, _, _, info = env.step(action)
        
        if info["score"] > best_score:
            best_score = info["score"]
            best_layout = env.layout.copy()

    mapping = [IDX_TO_LETTER[i] for i in best_layout]
    return mapping, best_score, env.render_layout()
