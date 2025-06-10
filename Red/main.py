import random
import time
from collections import Counter

from pyboy import PyBoy
import tensorflow as tf
import numpy as np
import cv2

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99
clip_ratio = 0.2
batch_size = 64
rand_chance = 0.05
time_train = 3600  # 1 minute of frames at 60fps (~60 steps per second * 60 sec)

outputs = ["a", "b", "start", "select", "left", "right", "up", "down"]


# Define the policy network with image and memory input
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Image processing branch
        self.conv1 = tf.keras.layers.Conv2D(16, 5, strides=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, strides=2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        # Memory input branch
        self.mem_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.mem_dense2 = tf.keras.layers.Dense(64, activation='relu')

        # Combined layers
        self.concat_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.concat_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(8)  # 8 actions

    def call(self, inputs):
        img_input, mem_input = inputs
        x1 = self.conv1(img_input)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.flatten(x1)

        x2 = self.mem_dense1(mem_input)
        x2 = self.mem_dense2(x2)

        x = tf.concat([x1, x2], axis=1)
        x = self.concat_dense1(x)
        x = self.concat_dense2(x)
        return self.logits(x)


def preprocess_image(pyboy):
    # Get a PIL image from PyBoy
    image = pyboy.screen.image  # PIL.Image
    image = image.resize((84, 84)).convert('L')  # Resize to 84x84 grayscale
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to 0-1
    image = image.reshape(1, 84, 84, 1)  # Add batch dimension
    return image


def get_mem(pyboy, addr):
    return pyboy.memory[addr]


def get_pokemon(env, addr):
    dex_no = get_mem(env, addr)
    hp1 = get_mem(env, addr + 1)
    hp2 = get_mem(env, addr + 2)
    status = get_mem(env, addr + 4)
    type1 = get_mem(env, addr + 5)
    type2 = get_mem(env, addr + 6)
    move1 = get_mem(env, addr + 8)
    move2 = get_mem(env, addr + 9)
    move3 = get_mem(env, addr + 10)
    move4 = get_mem(env, addr + 11)
    pp_move1 = get_mem(env, addr + 29)
    pp_move2 = get_mem(env, addr + 30)
    pp_move3 = get_mem(env, addr + 31)
    pp_move4 = get_mem(env, addr + 32)
    level = get_mem(env, addr + 33)

    return [level, dex_no, (hp1 << 8) + hp2, status, type1, type2,
            move1, move2, move3, move4, pp_move1, pp_move2, pp_move3, pp_move4]


def log_prob_from_logits(logits, actions):
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    return -neg_log_prob


def discount_rewards(rewards, gamma):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0
    for t in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[t]
        discounted[t] = cumulative
    discounted -= np.mean(discounted)
    discounted /= (np.std(discounted) + 1e-10)
    return discounted


def train(agent, optimizer, episodes=100):
    for episode in range(episodes):
        pyboy = PyBoy('red.gb')
        pyboy.set_emulation_speed(0)

        with open("has_pokedex_nballs.state", "rb") as f:
            pyboy.load_state(f)

        red = pyboy.game_wrapper
        red.start_game()

        states_img, states_mem, actions, rewards, old_log_probs = [], [], [], [], []

        total_reward = 0
        count = 0
        squares = set()
        random.seed(time.process_time())

        while count < time_train:
            # Hybrid state
            mon1 = get_pokemon(pyboy, 0xD16B)
            mon2 = get_pokemon(pyboy, 0xD197)
            mon3 = get_pokemon(pyboy, 0xD1C3)
            mon4 = get_pokemon(pyboy, 0xD1EF)
            mon5 = get_pokemon(pyboy, 0xD21B)
            mon6 = get_pokemon(pyboy, 0xD247)

            sq_x = pyboy.memory[0xD362]
            sq_y = pyboy.memory[0xD361]
            sq_m = pyboy.memory[0xD35E]
            in_battle = get_mem(pyboy, 0xD057)
            menu_item = get_mem(pyboy, 0xCC26)

            squares.add((sq_m, sq_x, sq_y))

            mem_state = np.array(mon1 + mon2 + mon3 + mon4 + mon5 + mon6 + [sq_m, sq_x, sq_y, in_battle, menu_item],
                                 dtype=np.float32) / 255.0
            mem_state = mem_state.reshape(1, -1)

            img_state = preprocess_image(pyboy)

            logits = agent([img_state, mem_state])
            probs = tf.nn.softmax(logits)[0].numpy()

            action = np.random.choice(8, p=probs)
            # Exploration noise
            if random.random() < rand_chance:
                action = random.randint(0, 7)

            action_tensor = tf.convert_to_tensor([action], dtype=tf.int32)
            log_prob = log_prob_from_logits(logits, action_tensor)[0].numpy()

            pyboy.button(outputs[action], 12)
            pyboy.tick(24)  # Advance 24 ticks/frame

            level_reward = sum([m[0] for m in [mon1, mon2, mon3, mon4, mon5, mon6]]) - 5
            exploration_reward = len(squares) * 0.1
            reward = level_reward + exploration_reward

            total_reward += reward

            states_img.append(img_state)
            states_mem.append(mem_state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)

            count += 1

        # Convert to tensors and numpy arrays
        states_img = np.vstack(states_img)
        states_mem = np.vstack(states_mem)
        actions = np.array(actions, dtype=np.int32)
        rewards = discount_rewards(rewards, gamma)
        old_log_probs = np.array(old_log_probs, dtype=np.float32)

        # Mini-batch PPO updates
        dataset_size = len(rewards)
        indices = np.arange(dataset_size)

        for _ in range(dataset_size // batch_size):
            batch_indices = np.random.choice(indices, batch_size, replace=False)
            batch_img = states_img[batch_indices]
            batch_mem = states_mem[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]

            with tf.GradientTape() as tape:
                logits = agent([batch_img, batch_mem])
                log_probs = log_prob_from_logits(logits, batch_actions)

                ratios = tf.exp(log_probs - batch_old_log_probs)
                clipped_ratios = tf.clip_by_value(ratios, 1 - clip_ratio, 1 + clip_ratio)
                policy_loss = -tf.reduce_mean(tf.minimum(ratios * batch_rewards, clipped_ratios * batch_rewards))

                entropy = -tf.reduce_mean(
                    tf.reduce_sum(tf.nn.softmax(logits) * tf.math.log(tf.nn.softmax(logits) + 1e-10), axis=1))
                loss = policy_loss - 0.01 * entropy

            grads = tape.gradient(loss, agent.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, agent.trainable_variables))

        pyboy.stop()
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {count}")


agent = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train(agent, optimizer, episodes=100)
