import argparse
from modelfree.dqn_learning_agent import train_dqn
from modelfree.q_learning_agent import train_q_learning
from modelfree.q_learning_encoder import train_rbf_q_learning
from train_model_based import train_model_based_agent


def main(args):
    if args.agent == "q-learning":
        train_q_learning(env_name=args.env, episodes=args.training_time)
    elif args.agent == "encoder-q-learning":
        train_rbf_q_learning(env_name=args.env, episodes=args.training_time)
    elif args.agent == "dqn-encoder":
        train_dqn(env_name=args.env, episodes=args.training_time, use_encoder=True)
    elif args.agent == "dqn":
        train_dqn(env_name=args.env)
    elif args.agent == "model-based":
        if args.run_clustering:
            k = args.run_clustering
            run_clustering = True
        else:
            k = 3500
            run_clustering = False
        train_model_based_agent(
            args.env,
            args.training_time,
            args.show_clusters_and_rewards,
            k,
            run_clustering,
        )
    else:
        raise ValueError("Agent type not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=[
            "q-learning",
            "encoder-q-learning",
            "dqn",
            "dqn-encoder",
            "model-based",
        ],
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
        "--run_clustering",
        type=int,
        help="Enable clustering and specify the number of clusters (k)",
    )

    args = parser.parse_args()
    main(args)

# Run by executing etc: python main.py --agent model-based --env CartPole-v1 --training_time 100 --show_clusters_and_rewards --run_clustering 3500
# Simple run: python main.py --agent model-based --env CartPole-v1 --training_time 100
# python main.py --agent q-learning --env CartPole-v1 --training_time 100
# python main.py --agent encoder-q-learning --env CartPole-v1 --training_time 100
