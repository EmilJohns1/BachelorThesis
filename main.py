from env_manager import EnvironmentManager
from model import Model
from agent import Agent
import pygame

render_mode = "human"  # Set to None to run without graphics

env_manager = EnvironmentManager(render_mode=render_mode, seed=0)
model = Model(action_space_n=env_manager.env.action_space.n)
agent = Agent(model)

rewards = 0.
actions = []
states = []
state, info = env_manager.reset()
states.append(state)

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
        model.update_model(states, actions, rewards)
        rewards = 0.
        actions.clear()
        states.clear()
        state, info = env_manager.reset()
        states.append(state)
