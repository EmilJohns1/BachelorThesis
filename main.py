from env_manager import EnvironmentManager
from model import Model
from agent import Agent
import pygame
import numpy as np
import time
from util import plot_rewards

episode_rewards = []

render_mode = "human"  # Set to None to run without graphics

env_manager = EnvironmentManager(render_mode=render_mode)
model = Model(action_space_n=env_manager.env.action_space.n, _discount_factor=1, _observation_space=env_manager.env.observation_space)
agent = Agent(model)

rewards = 0.
actions = []
states = []
state, info = env_manager.reset()
states.append(state)

episodes = 0
training_time = 150
testing_time = 50
finished_training = False
start = time.time()
while True:
    if render_mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                env_manager.close()
                exit()

    states_mean, states_std = agent.normalize_states()
    action_rewards, action_weights = agent.compute_action_rewards(state, states_mean, states_std)
    action = agent.get_action(action_rewards, action_weights)

    actions.append(action)
    state, reward, terminated, truncated, info = env_manager.step(action)
    states.append(state)
    rewards += float(reward)

    if terminated or truncated:
        print(f"episode {episodes + 1} rewards: {rewards}")

        episode_rewards.append(rewards)

        if episodes == training_time and not finished_training:
            episodes = 0
            end = time.time()
            print("Time :{}".format(end-start))
            env_manager = EnvironmentManager(render_mode="human")
            
            model.run_k_means(k=3000)
            model.update_transitions_and_rewards_for_clusters()

            agent.use_clusters = True
            plot_rewards(episode_rewards=episode_rewards)
            episode_rewards = []
            episodes = -1
            finished_training = True
            action_rewards, action_weights = agent.compute_action_rewards(state, states_mean, states_std)

        elif episodes < training_time:
            model.update_model(states, actions, rewards)

        if episodes == testing_time and finished_training:
            plot_rewards(episode_rewards=episode_rewards)
        
        rewards = 0.
        actions.clear()
        states.clear()
        state, info = env_manager.reset()
        states.append(state)
        episodes += 1
