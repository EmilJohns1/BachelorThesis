from env_manager import EnvironmentManager
from model import Model
from agent import Agent
import pygame
import matplotlib.pyplot as plt
import numpy as np
import time

episode_rewards = []

render_mode = None  # Set to None to run without graphics

env_manager = EnvironmentManager(render_mode=render_mode, seed=0)
model = Model(action_space_n=env_manager.env.action_space.n, _discount_factor=0.9999)
agent = Agent(model)

rewards = 0.
actions = []
states = []
state, info = env_manager.reset()
states.append(state)

episodes = 0
training_time = 100
while True:
    if render_mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                env_manager.close()
                exit()

    states_mean, states_std = agent.normalize_states()
    action_rewards, weight_sums = agent.compute_action_rewards(state, states_mean, states_std)
    action = agent.get_action(action_rewards, weight_sums)

    actions.append(action)
    state, reward, terminated, truncated, info = env_manager.step(action)
    states.append(state)
    rewards += float(reward)

    if terminated or truncated:
        print(f"rewards: {rewards}")

        episode_rewards.append(rewards)

        if episodes == training_time:
            model.run_k_means()

             # Calculate running mean and std
            running_means = np.cumsum(episode_rewards) / np.arange(1, len(episode_rewards) + 1)
            running_stds = [np.std(episode_rewards[:i + 1]) for i in range(len(episode_rewards))]
            
            # Plot rewards, mean, and standard deviation
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label="Rewards", alpha=0.5)
            plt.plot(range(1, len(running_means) + 1), running_means, label="Running Mean", color="orange")
            plt.fill_between(range(1, len(running_means) + 1),
                             np.array(running_means) - np.array(running_stds),
                             np.array(running_means) + np.array(running_stds),
                             color="orange", alpha=0.3, label="Mean ± Std")
            plt.xlabel("Episode")
            plt.ylabel("Rewards")
            plt.title("Episode Rewards with Running Mean and Std")
            plt.legend()
            plt.show()


        elif episodes < training_time:
            model.update_model(states, actions, rewards)
        rewards = 0.
        actions.clear()
        states.clear()
        state, info = env_manager.reset()
        states.append(state)
        episodes += 1
