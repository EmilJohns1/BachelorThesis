import argparse
from train_model_based import train_model_based_agent


def main(args):
    # if args.agent == "q-learning":
    #     train_q_learning(args.env)
    # elif args.agent == "dqn":
    #     train_dqn(args.env)
    if args.agent == "model-based":
        train_model_based_agent(args.env, args.training_time)
    else:
        raise ValueError("Agent type not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["q-learning", "dqn", "model-based"],
        required=True,
        help="Choose the RL agent",
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Choose the environment"
    )
    parser.add_argument(
        "--training_time", type=int, default=100, help="Training time in episodes"
    )

    args = parser.parse_args()
    main(args)

# Run by executing etc: python main.py --agent model-based --env CartPole-v1 --training_time 100
