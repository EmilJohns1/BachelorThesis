import argparse


from train_model_based import train_model_based_agent

# Model-free agents
from modelfree.dqn_learning_agent import train_dqn
from modelfree.dqn_learning_encoder import train_dqn_encoder
from modelfree.q_learning_agent import train_q_learning
from modelfree.q_learning_encoder import train_rbf_q_learning




from util.reward_visualizer import plot_multiple_runs

#plot_multiple_runs(folder_name="logs/gaussian_width_action_reward_2_0/clustering_width_0_5", title="0.5", field="testing_rewards", block=False)
#plot_multiple_runs(folder_name="logs/gaussian_width_action_reward_2_0/clustering_width_2_0", title="2.0", field="testing_rewards", block=False)
#plot_multiple_runs(folder_name="logs/gaussian_width_action_reward_2_0/clustering_width_3_0", title="3.0", field="testing_rewards", block=False)
#plot_multiple_runs(folder_name="logs/gaussian_width_action_reward_2_0/clustering_width_5_0", title="5.0", field="testing_rewards")

def main(args):
    if args.agent == "q-learning":
        train_q_learning(env_name=args.env, episodes=args.training_time)
    elif args.agent == "encoder-q-learning":
        train_rbf_q_learning(env_name=args.env, episodes=args.training_time)
    elif args.agent == "dqn-encoder":
        train_dqn_encoder(env_name=args.env, episodes=args.training_time, use_encoder=True)
    elif args.agent == "dqn":
        train_dqn(env_name=args.env, episodes=args.training_time)
    elif args.agent == "model-based":
        if args.find_optimal_k:
            find_k = True
            lower_k, upper_k, step = args.find_optimal_k
        else:
            find_k = False
            lower_k, upper_k, step = None, None, None
        train_model_based_agent(
            args.env,
            args.training_time,
            args.show_clusters_and_rewards,
            find_k,
            lower_k,
            upper_k,
            step,
        )
    else:
        raise ValueError("Agent type not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["q-learning", "encoder-q-learning", "dqn","dqn-encoder", "model-based"],
        required=True,
        help="Choose the RL agent",
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Choose the environment"
    )
    parser.add_argument(
        "--training_time", type=int, default=100, help="Training time in episodes"
    )
    parser.add_argument(
        "--show_clusters_and_rewards",
        action="store_true",
        help="Include this if you want to show clusters and rewards",
    )
    parser.add_argument(
        "--find_optimal_k",
        type=int,
        nargs=3,
        metavar=("LOWER_K", "UPPER_K", "STEP"),
        help="Find optimal k with range: lower_k, upper_k, step",
    )

    args = parser.parse_args()
    main(args)
# Run by executing etc: python main.py --agent model-based --env CartPole-v1 --training_time 100 --show_clusters_and_rewards --find_optimal_k 15000 25000 500
# Simple run: python main.py --agent model-based --env CartPole-v1 --training_time 100
# python main.py --agent q-learning --env CartPole-v1 --training_time 100
# python main.py --agent encoder-q-learning --env CartPole-v1 --training_time 100