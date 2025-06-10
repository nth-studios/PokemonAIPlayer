import random
import time
from collections import Counter

from pytesseract import pytesseract
from pyboy import PyBoy
import tensorflow as tf
import numpy as np
import math


# Hyperparameters
learning_rate = 0.005
gamma = 0.99  # Discount factor for future rewards
rand_chance = 0.05
time_train = 30 * 60 * 2


class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.dense4 = tf.keras.layers.Dense(8)  # logits output

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)


# Function to normalize rewards
def normalize_rewards(rewards):
    rewards = np.array(rewards)
    rewards -= np.mean(rewards)
    rewards /= (np.std(rewards) + 1e-10)  # Avoid division by zero
    return rewards


# Discount rewards function (unchanged)
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    cumulative = 0
    for t in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[t]
        discounted_rewards[t] = cumulative
    return normalize_rewards(discounted_rewards)


class Mon:
    def __init__(self, dex_no, level, hp, status, type1, type2, move1, move2, move3, move4, pp_move1, pp_move2,
                 pp_move3, pp_move4):
        self.dex_no = dex_no
        self.level = level
        self.hp = hp
        self.status = status
        self.type1 = type1
        self.type2 = type2
        self.move1 = move1
        self.move2 = move2
        self.move3 = move3
        self.move4 = move4
        self.pp_move1 = pp_move1
        self.pp_move2 = pp_move2
        self.pp_move3 = pp_move3
        self.pp_move4 = pp_move4

    def __str__(self):
        return "ID: " + str(self.dex_no) + ", Lvl: " + str(self.level)


def get_mem(env, addr):
    return env.memory[addr]


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

    return Mon(dex_no, level, hp1 << 8 + hp2, status, type1, type2, move1, move2, move3, move4, pp_move1, pp_move2,
               pp_move3, pp_move4)




outputs = ["a", "b", "start", "select", "left", "right", "up", "down"]

times = []
# Train Policy Gradient
def train(policy, optimizer, episodes=1000):
    for episode in range(episodes):

        start_time = time.process_time()

        pyboy = PyBoy('red.gb')
        pyboy.set_emulation_speed(0)

        with open("has_pokedex_nballs.state", "rb") as f:
            pyboy.load_state(f)

        red = pyboy.game_wrapper
        red.start_game()

        count = 0
        random.seed(time.process_time())

        squares = set()

        states, actions, rewards = [], [], []
        total_reward = 0

        while count < time_train:
            # Setup state
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

            state = np.array([mon1.level, mon1.dex_no, mon1.hp, mon1.type1, mon1.type2, mon1.status,
                     mon1.move1, mon1.move2, mon1.move3, mon1.move4,
                     mon1.pp_move1, mon1.pp_move2, mon1.pp_move3, mon1.pp_move4,
                     mon2.level, mon2.dex_no, mon2.hp, mon2.type1, mon2.type2, mon2.status,
                     mon2.move1, mon2.move2, mon2.move3, mon2.move4,
                     mon2.pp_move1, mon2.pp_move2, mon2.pp_move3, mon2.pp_move4,
                     mon3.level, mon3.dex_no, mon3.hp, mon3.type1, mon3.type2, mon3.status,
                     mon3.move1, mon3.move2, mon3.move3, mon3.move4,
                     mon3.pp_move1, mon3.pp_move2, mon3.pp_move3, mon3.pp_move4,
                     mon4.level, mon4.dex_no, mon4.hp, mon4.type1, mon4.type2, mon4.status,
                     mon4.move1, mon4.move2, mon4.move3, mon4.move4,
                     mon4.pp_move1, mon4.pp_move2, mon4.pp_move3, mon4.pp_move4,
                     sq_m, sq_x, sq_y, in_battle, menu_item])

            state = state / 255.0
            state = state.reshape(1, len(state))
            states.append(state)

            # Get next action

            action_logits = policy(state)
            action_prob = tf.nn.softmax(action_logits).numpy()[0]
            action = np.random.choice(8, p=action_prob)
            action = np.random.choice([action, np.random.choice(8)], p=[1-rand_chance, rand_chance])
            actions.append(action)

            pyboy.tick(24)

            level_reward = mon1.level + mon2.level + mon3.level + mon4.level + mon5.level + mon6.level - 5
            exploration_reward = len(squares) * 0.1

            reward = level_reward + exploration_reward
            # if len(actions) > 1 and actions[-1] == actions[-2]:
            #     reward -= 0.2  # or some small penalty

            rewards.append(reward)
            total_reward = reward

            pyboy.button(outputs[action], 12)
            count = count + 1

            if count == time_train:
                # print("Logits:", np.round(action_logits.numpy()[0], 2))
                print("Probs :", np.round(action_prob, 2))

        states = np.vstack(states)
        actions = np.array(actions)
        rewards = discount_rewards(rewards, gamma)

        with tf.GradientTape() as tape:
            logits = policy(states)
            action_probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
            loss = tf.reduce_mean(log_probs * rewards - 0.01 * entropy)  # Encourage randomness

        # Apply gradients to update the network
        grads = tape.gradient(loss, policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        # Print progress
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        # for i in range(8):
        #     print(outputs[i], Counter(actions)[i])

        pyboy.stop()
        time_taken = time.process_time()-start_time
        print(f"Time for episode {episode + 1} - {time_taken}")
        times.append(time_taken)

        episode_entropy = tf.reduce_mean(entropy).numpy()
        print(f"Episode {episode + 1}, Avg Entropy: {episode_entropy:.4f}")
        print()


policy = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Train the policy agent
train(policy, optimizer, episodes=500)

