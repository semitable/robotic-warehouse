import gym, ray
import pprint
from ray.rllib.agents import dqn
from ray import tune
from ray.tune.registry import register_env

from utils import ENVIRONMENT, parse, env_creator, extract_spaces, extract_num_agents

register_env(f"ray-{ENVIRONMENT}", env_creator)

if __name__ == "__main__":
    args = parse()

    num_agents = extract_num_agents(ENVIRONMENT)
    obs_space, act_space = extract_spaces(ENVIRONMENT)

    if args.redis_pwd is not None and args.ip_head is not None:
        ray.init(address=args.ip_head, redis_password=args.redis_pwd)
    else:
        ray.init()

    tune.run(
        "DQN",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={
            "episode_reward_mean": args.stop_reward,
            "training_iteration": args.num_iters,
        },
        config={
            "env": f"ray-{ENVIRONMENT}",

            # === Model ===
            # Number of atoms for representing the distribution of return. When
            # this is greater than 1, distributional Q-learning is used.
            # the discrete supports are bounded by v_min and v_max
            "num_atoms": 1,
            "v_min": -10.0,
            "v_max": 10.0,
            # Whether to use noisy network
            "noisy": False,
            # control the initial value of noisy nets
            "sigma0": 0.5,
            # Whether to use dueling dqn
            "dueling": True,
            # Whether to use double dqn
            "double_q": True,

            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 500,
            # Use softmax for sampling actions. Required for off policy estimation.
            "soft_q": False,

            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": 50000,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            # Alpha parameter for prioritized replay buffer.
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,

            # === Optimization ===
            # Learning rate for adam optimizer
            "lr": 5e-4, # tune.grid_search([0.01, 0.001, 5e-4, 0.0001]),
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 32,

            # === Parallelism ===
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": 5,
            "eager": False,
            "multiagent": {
                "policies": {
                    f"agent_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)
                },
                "policy_mapping_fn": lambda agent_name: agent_name
            }
        },
    )
