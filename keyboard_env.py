import random
import numpy as np

# Constants
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
LETTER_TO_IDX = {c: i for i, c in enumerate(LETTERS)}
IDX_TO_LETTER = {i: c for c, i in LETTER_TO_IDX.items()}

TOP_RANGE = range(0, 10)
HOME_RANGE = range(10, 19)
BOTTOM_RANGE = range(19, 26)

LEFT_POSITIONS = set([0,1,2,3,4,10,11,12,13,14,19,20,21,22,23])
RIGHT_POSITIONS = set(range(26)) - LEFT_POSITIONS

ACTION_PAIRS = [(i,j) for i in range(26) for j in range(i+1,26)]
N_ACTIONS = len(ACTION_PAIRS)

QWERTY_ORDER = list("qwertyuiopasdfghjklzxcvbnm")

def get_hand_and_finger(pos):
    """Return (hand, finger_index) for given slot position."""
    if pos in range(0,10):  # top row
        if pos <= 4: return "L", pos
        else: return "R", pos-5
    if pos in range(10,19):  # home row
        if pos <= 14: return "L", pos-10
        else: return "R", pos-15
    if pos in range(19,26):  # bottom row
        if pos <= 23: return "L", pos-19
        else: return "R", pos-24
    raise ValueError("Invalid position")

class KeyboardEnv:
    def __init__(self, letter_freqs, bigram_freqs, top9_list=None, max_steps=200):
        freqs = np.zeros(26, dtype=np.float32)
        for c, v in letter_freqs.items():
            c = c.lower()
            if c in LETTER_TO_IDX:
                freqs[LETTER_TO_IDX[c]] = float(v)
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()

        self.letter_freqs = freqs
        self.bigram_freqs = {k.lower(): float(v) for k,v in bigram_freqs.items()}
        self.top9 = [c.lower() for c in top9_list] if top9_list else self._get_default_top9()
        self.max_steps = max_steps
        self.qwerty_indices = self._qwerty_layout_indices()
        self.reset()

    def _get_default_top9(self):
        top_idxs = np.argsort(-self.letter_freqs)[:9]
        return [IDX_TO_LETTER[i] for i in top_idxs]

    def _qwerty_layout_indices(self):
        return [LETTER_TO_IDX[c] for c in QWERTY_ORDER]

    def reset(self, start_layout=None, randomize=False):
        if start_layout is None:
            self.layout = list(range(26))
            random.shuffle(self.layout)
        else:
            self.layout = start_layout.copy()
            if randomize:
                for _ in range(3):
                    i,j = random.randrange(26), random.randrange(26)
                    self.layout[i], self.layout[j] = self.layout[j], self.layout[i]

        self.step_count = 0
        self.prev_score = self._compute_score()
        return self._get_obs()

    def _get_obs(self):
        one_hot = np.zeros((26,26), dtype=np.float32)
        for slot, letter_idx in enumerate(self.layout):
            one_hot[slot, letter_idx] = 1.0
        slot_freqs = np.array([self.letter_freqs[self.layout[s]] for s in range(26)], dtype=np.float32)
        return np.concatenate([one_hot.flatten(), slot_freqs])

    def step(self, action_idx):
        i,j = ACTION_PAIRS[action_idx]
        self.layout[i], self.layout[j] = self.layout[j], self.layout[i]
        new_score = self._compute_score()
        reward = float(new_score - self.prev_score)
        self.prev_score = new_score
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, {"score": new_score}

    def _compute_score(self):
        pos_of_letter = {letter_idx: pos for pos, letter_idx in enumerate(self.layout)}
        top9_idxs = [LETTER_TO_IDX[c] for c in self.top9 if c in LETTER_TO_IDX]
        top9_in_home = sum(1 for pos in HOME_RANGE if self.layout[pos] in top9_idxs)
        top9_reward = 5.0 * top9_in_home

        bigram_reward = 0.0
        for bigram, freq in self.bigram_freqs.items():
            if len(bigram) != 2: continue
            a,b = bigram[0], bigram[1]
            if a not in LETTER_TO_IDX or b not in LETTER_TO_IDX: continue
            ia, ib = LETTER_TO_IDX[a], LETTER_TO_IDX[b]
            pa, pb = pos_of_letter[ia], pos_of_letter[ib]
            hand_a, finger_a = get_hand_and_finger(pa)
            hand_b, finger_b = get_hand_and_finger(pb)
            if hand_a != hand_b:
                bigram_reward += freq * 2.0
            elif finger_a != finger_b:
                bigram_reward += freq * 1.0

        left_home = sum(self.letter_freqs[self.layout[p]] for p in range(10,15))
        right_home = sum(self.letter_freqs[self.layout[p]] for p in range(15,19))
        balance_score = 1.0 - abs(left_home - right_home)/(left_home + right_home + 1e-8)
        balance_reward = balance_score * (30.0 if top9_in_home == len(self.top9) else 10.0)

        return float(top9_reward + bigram_reward + balance_reward)

    def render_layout(self):
        top = "".join(IDX_TO_LETTER[self.layout[i]] for i in TOP_RANGE)
        home = "".join(IDX_TO_LETTER[self.layout[i]] for i in HOME_RANGE)
        bottom = "".join(IDX_TO_LETTER[self.layout[i]] for i in BOTTOM_RANGE)
        return f"TOP:    {top}\nHOME:   {home}\nBOTTOM: {bottom}"