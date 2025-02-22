import random
import time
import pygame
from agent import Agent
from env_manager import EnvironmentManager
from model import Model
from util.cluster_visualizer import ClusterVisualizer
from util.logger import write_to_json
from util.reward_visualizer import plot_rewards, plot_multiple_runs

""" plot_multiple_runs(folder_name="logs/online_clustering_1e-2_filter", title="1e-2", field="testing_rewards", block=False)
plot_multiple_runs(folder_name="logs/online_clustering_8e-1_filter", title="8e-1", field="testing_rewards", block=False)
plot_multiple_runs(folder_name="logs/online_clustering_5e-1_filter", title="5e-1", field="testing_rewards", block=False)
plot_multiple_runs(folder_name="logs/online_clustering_2e-1_filter", title="2e-1", field="testing_rewards", block=False)
plot_multiple_runs(folder_name="logs/weighted_online_clustering", title="1e-1", field="testing_rewards")
 """
import numpy as np
for i in range(1):
    print("--- Starting new run ---")
    #################################################
    # These variables should be logged for each run
    environment = "CartPole-v1"
    discount_factor = 1
    k = 4000
    gaussian_width_rewards = 0.2
    seed = random.randint(0, 2**32 - 1)
    comments = ""
    training_time = 100
    testing_time = 100
    training_rewards = []
    testing_rewards = []
    #################################################

    episode_rewards = []
    render_mode = None  # Set to None to run without graphics

    env_manager = EnvironmentManager(
    render_mode=render_mode, environment=environment, seed=seed
    )
    model = Model(
        action_space_n=env_manager.env.action_space.n,
        discount_factor=discount_factor,
        observation_space=env_manager.env.observation_space,
        K=k,
        sigma=gaussian_width_rewards
    )
    agent = Agent(model)

    rewards = 0.0
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

                model.cluster_states(k=k, gaussian_width=gaussian_width_rewards)

                # Disable for further training after clustering
                agent.testing = True
                finished_training = True

                training_rewards = episode_rewards
                episode_rewards = []
                episodes = -1

            elif episodes < training_time and not finished_training:
                model.update_model(states, actions, rewards)

            if episodes == testing_time and finished_training:
                testing_rewards = episode_rewards

                
                data = {
                    "environment" : environment,
                    "discount_factor" : discount_factor,
                    "k" : k,
                    "gaussian_width_rewards" : gaussian_width_rewards,
                    "seed" : seed,
                    "comments" : comments,
                    "training_time" : training_time,
                    "testing_time" : testing_time,
                    "training_rewards" : training_rewards,
                    "testing_rewards" : testing_rewards
                }
                write_to_json(data, "online_clustering_1e-2_filter")


                #plot_rewards(episode_rewards=episode_rewards)
                env_manager.close()
                break
                
            rewards = 0.
            actions.clear()
            states.clear()
            state, info = env_manager.reset()
            states.append(state)

            episodes += 1
