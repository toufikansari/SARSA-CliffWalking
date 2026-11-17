!pip install gymnasium
!pip install "gymnasium[toy-text]"
import gymnasium as gym
import numpy as np
import random
env = gym.make("CliffWalking-v1")
gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500
# Q-Table = store q values

Q = np.zeros((48,4))
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])



for episode in range(episodes):

    env = gym.make("CliffWalking-v1")

    done = False # When the episode has ended
    state, _ = env.reset()
    action = epsilon_greedy(state)

    total_reward = 0
    episode_len = 0

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_action = epsilon_greedy(next_state)

        # SARSA update
        if done:
            Q[state, action] += alpha * (reward - Q[state, action])
        else:
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_len += 1

    print(f"episode number = {episode+1}/{episodes} total reward = {total_reward} & ep length = {episode_len}")

    env.close()
