import random
import time
import pygame
from agent import Agent
from env_manager import EnvironmentManager
from model import Model

from util.cluster_visualizer import ClusterVisualizer
from util.clustering_alg import Clustering_Type
from util.logger import write_to_json
from util.reward_visualizer import plot_rewards


def train_model_based_agent(
    env_name, training_time, show_clusters_and_rewards, k, run_clustering
):
    for i in range(100):
        print("--- Starting new run ---")
        #################################################
        # These variables should be logged for each run
        environment = env_name if env_name else "CartPole-v1"
        discount_factor = 1.0
        epsilon_decay = 0.999
        gaussian_width_rewards = 0.5
        training_seed = random.randint(0, 2**32 - 1)
        testing_seed = random.randint(0, 2**32 - 1)
        comments = ""
        training_time = training_time if training_time else 100
        testing_time = 100
        training_rewards = []
        testing_rewards = []
        k = k
        run_clustering = run_clustering
        #################################################

        episode_rewards = []
        render_mode = None  # Set to None to run without graphics

        env_manager = EnvironmentManager(
            render_mode=render_mode, environment=environment, seed=training_seed
        )
        model = Model(
            action_space_n=env_manager.env.action_space.n,
            discount_factor=discount_factor,
            observation_space=env_manager.env.observation_space,
            k=k,
            sigma=gaussian_width_rewards,
        )
        agent = Agent(model=model, gaussian_width=gaussian_width_rewards)

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
                    if event.type == pygame.KEYDOWN and (
                        event.key == pygame.K_ESCAPE or event.key == pygame.K_q
                    ):
                        env_manager.close()
                        exit()

            states_mean, states_std = agent.normalize_states()
            action_rewards, action_weights = agent.compute_action_rewards(
                state, states_mean, states_std
            )
            # agent.exploration_rate = max(0.05, 0.30 * (epsilon_decay**episodes))

            action = agent.get_action(action_rewards, action_weights)

            actions.append(action)
            prev_state = state.copy()
            state, reward, terminated, truncated, info = env_manager.step(action)

            actual_delta = state - prev_state
            agent.update_approximation(action, actual_delta)

            states.append(state)
            rewards += float(reward)

            if terminated or truncated:
                print(f"rewards: {rewards}")

                episode_rewards.append(rewards)

                if episodes == training_time and not finished_training:
                    episodes = 0
                    end = time.time()
                    print("Time :{}".format(end - start))

                    if run_clustering:
                        model.cluster_states(
                            k=k,
                            gaussian_width=gaussian_width_rewards,
                            cluster_type=Clustering_Type.K_Means,
                        )

                    agent.testing = True

                    if show_clusters_and_rewards:
                        plot_rewards(episode_rewards=episode_rewards)

                        cluster_visualizer = ClusterVisualizer(model=model)

                        # cluster_visualizer.plot_clusters()
                        # cluster_visualizer.plot_reward_distribution_per_cluster()
                        cluster_visualizer.plot_rewards_before_clustering()
                        cluster_visualizer.plot_rewards_after_clustering()

                    training_rewards = episode_rewards
                    episode_rewards = []
                    episodes = -1
                    finished_training = True

                    env_manager = EnvironmentManager(
                        render_mode="human", environment=environment, seed=testing_seed
                    )

                elif episodes < training_time and not finished_training:
                    model.update_model(states, actions, rewards)

                if episodes == testing_time and finished_training:
                    testing_rewards = episode_rewards

                    data = {
                        "environment": environment,
                        "discount_factor": discount_factor,
                        "epsilon_decay": epsilon_decay,
                        "k": k,
                        "gaussian_width_rewards": gaussian_width_rewards,
                        "training_seed": training_seed,
                        "testing_seed": testing_seed,
                        "comments": comments,
                        "training_time": training_time,
                        "testing_time": testing_time,
                        "training_rewards": training_rewards,
                        "testing_rewards": testing_rewards,
                    }
                    write_to_json(data)

                    if show_clusters_and_rewards:
                        plot_rewards(episode_rewards=episode_rewards)
                    env_manager.close()
                    break

                rewards = 0.0
                actions.clear()
                states.clear()
                state, info = env_manager.reset()
                states.append(state)
                episodes += 1
