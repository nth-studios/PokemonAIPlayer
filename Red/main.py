# multi_threaded_pokemon_train.py
from PIL import Image
import concurrent.futures
import random
import numpy as np
import tensorflow as tf
from pyboy import PyBoy
import cv2

# --- CONFIGURATION ---
NUM_EPISODES = 40
NUM_WORKERS = 4
MAX_STEPS = 3600
RAND_CHANCE = 0.05
SKIP_FRAMES = 24  # emulate every SKIP_FRAMES ticks

# --- UTILITY FUNCTIONS ---

def preprocess_image(pyboy):
    """Capture screen from PyBoy and return (84, 84, 1) normalized grayscale numpy array."""
    try:
        # Get the screen as a PIL image
        img = pyboy.screen.image

        # Force convert to RGB if not already
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(img)}")
        img = img.convert("RGB")

        # Convert to numpy array
        np_img = np.asarray(img)
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {np_img.shape}")

        # Convert RGB -> Grayscale
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # Resize to (84x84)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0

        # Add channel dimension: (84, 84, 1)
        return np.expand_dims(normalized, axis=-1)

    except Exception as e:
        print(f"[preprocess_image ERROR] {e}")
        raise

def get_memory_state(pyboy):
    """Extract memory-based features for Pokémon + location + menu."""
    def m(addr): return pyboy.memory[addr]
    def get_mon(addr):
        return [
            m(addr + offset) for offset in (
                0, 33, 1, 4, 5, 6, 8, 9, 10, 11, 29, 30, 31, 32
            )
        ]
    mons = []
    for base in [0xD16B, 0xD197, 0xD1C3, 0xD1EF]:
        mons += get_mon(base)
    mons += [m(0xD35E), m(0xD362), m(0xD361), m(0xD057), m(0xCC26)]
    return np.array(mons, dtype=np.float32) / 255.0  # shape: (89,)

def compute_reward(pyboy, visited):
    """Combine level-sum and exploration reward."""
    levels = sum(get_memory_state(pyboy)[i] for i in range(4 * 2))  # first 8 entries approximate levels
    pos = tuple(get_memory_state(pyboy)[-5:-2])
    visited.add(pos)
    return levels + 0.1 * len(visited)

def discount_rewards(xs, gamma=0.99):
    """Standard discounted reward (REINFORCE)."""
    ds = np.zeros_like(xs, dtype=np.float32)
    g = 0.0
    for i in reversed(range(len(xs))):
        g = xs[i] + gamma * g
        ds[i] = g
    return ds

def log_prob_from_logits(logits, actions):
    """Compute log-probs for selected actions."""
    neglog = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    return -neglog

# --- MODEL DEFINITION ---

class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CNN branch
        self.conv1 = tf.keras.layers.Conv2D(16, 8, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu')
        self.flat = tf.keras.layers.Flatten()
        # Memory branch
        self.mem_dense = tf.keras.layers.Dense(128, activation='relu')
        # Combined branch
        self.comb_dense = tf.keras.layers.Dense(256, activation='relu')
        self.logits = tf.keras.layers.Dense(8)

    def call(self, img, mem):
        x1 = self.conv1(tf.expand_dims(img, -1))
        x1 = self.conv2(x1)
        x1 = self.flat(x1)
        x2 = self.mem_dense(mem)
        x = self.comb_dense(tf.concat([x1, x2], axis=1))
        return self.logits(x)

# --- EPISODE RUNNER ---

def run_episode(agent, ep_id):
    pyboy = PyBoy('red.gb')
    pyboy.set_emulation_speed(0)
    with open("has_pokedex_nballs.state", "rb") as f:
        pyboy.load_state(f)
    pyboy.game_wrapper.start_game()

    visited = set()
    log = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        img = preprocess_image(pyboy)
        mem = get_memory_state(pyboy).reshape(1, -1)
        logits = agent(img.reshape(1,84,84), mem)
        probs = tf.nn.softmax(logits).numpy()[0]
        action = np.random.choice(8, p=probs)
        if random.random() < RAND_CHANCE:
            action = random.randint(0,7)
        log_prob = log_prob_from_logits(logits, np.array([action]))[0].numpy()

        pyboy.tick(SKIP_FRAMES)
        pyboy.button(["a","b","start","select","left","right","up","down"][action], 12)

        reward = compute_reward(pyboy, visited)
        total_reward += reward
        log.append((img, mem.squeeze(), action, log_prob, reward))

    pyboy.stop()
    imgs, mems, acts, lps, rews = zip(*log)
    return dict(
        ep_id=ep_id, total_reward=total_reward,
        states=(np.stack(imgs), np.stack(mems)),
        actions=np.array(acts, dtype=np.int32),
        old_log_probs=np.array(lps, dtype=np.float32),
        rewards=discount_rewards(rews)
    )

# --- TRAINING LOOP ---

def train():
    agent = PolicyNetwork()
    optimizer = tf.keras.optimizers.Adam(3e-4)

    for ep in range(NUM_EPISODES):
        episodes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(run_episode, agent, f"{ep}_{i}") for i in range(NUM_WORKERS)]
            for future in concurrent.futures.as_completed(futures):
                episodes.append(future.result())

        best = max(episodes, key=lambda x: x["total_reward"])
        print(f"Ep {ep}: Best Reward={best['total_reward']:.1f} ({best['ep_id']})")

        imgs, mems = best["states"]
        acts, old_lps, rews = best["actions"], best["old_log_probs"], best["rewards"]
        # train batch
        with tf.GradientTape() as tape:
            logits = agent(imgs, mems)
            lp = log_prob_from_logits(logits, acts)
            loss = -tf.reduce_mean(lp * rews)
        grads = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.trainable_variables))

    agent.save_weights("best_policy.h5")
    print("Training complete.")

# Run if executed
if __name__ == "__main__":
    train()
