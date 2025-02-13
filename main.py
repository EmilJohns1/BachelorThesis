from env_manager import EnvironmentManager
from model import Model
from agent import Agent
import pygame
import numpy as np
import time
from util.reward_visualizer import plot_rewards, write_to_json
import random

#################################################
# These variables should be logged for each run
environment = "CartPole-v1"
discount_factor = 1
training_time = 150
testing_time = 100
training_rewards = []
testing_rewards = []
k = 3000
seed = random.randint(0, 2**32 - 1)
#################################################
from util.cluster_visualizer import ClusterVisualizer

episode_rewards = []
render_mode = None  # Set to None to run without graphics

env_manager = EnvironmentManager(render_mode=render_mode, environment=environment, seed=seed)
model = Model(action_space_n=env_manager.env.action_space.n, _discount_factor=discount_factor, _observation_space=env_manager.env.observation_space)
agent = Agent(model)

rewards = 0.
actions = []
states = []
state, info = env_manager.reset()
states.append(state)

episodes = 0
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
        print(f"rewards: {rewards}")

        episode_rewards.append(rewards)

        if episodes == training_time and not finished_training:
            episodes = 0
            end = time.time()
            print("Time :{}".format(end-start))
            
            model.run_k_means(k=k)
            model.update_transitions_and_rewards_for_clusters()

            print(f"States shape: {model.states.shape}")
            print(f"Rewards shape: {model.rewards.shape}")

            visualizer = ClusterVisualizer(model)

            # Plot clusters
            visualizer.plot_clusters()

            # Plot rewards
            visualizer.plot_rewards()

            agent.use_clusters = True
            plot_rewards(episode_rewards=episode_rewards)
            training_rewards = episode_rewards
            episode_rewards = []
            episodes = -1
            finished_training = True
            action_rewards, action_weights = agent.compute_action_rewards(state, states_mean, states_std)

        elif episodes < training_time and not finished_training and not finished_training:
            model.update_model(states, actions, rewards)

        if episodes == testing_time and finished_training:
            testing_rewards = episode_rewards

            
            data = {
                "environment" : environment,
                "discount_factor" : discount_factor,
                "k" : k,
                "seed" : seed,
                "training_time" : training_time,
                "testing_time" : testing_time,
                "training_rewards" : training_rewards,
                "testing_rewards" : testing_rewards
            }
            write_to_json(data)


            plot_rewards(episode_rewards=episode_rewards)
            env_manager.close()
            exit()
        
        rewards = 0.
        actions.clear()
        states.clear()
        state, info = env_manager.reset()
        states.append(state)
        episodes += 1
