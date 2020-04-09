import gym, ray
import pprint
from ray.rllib.agents import a3c
from ray import tune
from ray.tune.registry import register_env

from utils import ENVIRONMENT, NUM_SEEDS, parse, env_creator, extract_spaces, extract_num_agents

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
        "A2C",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={
            "timesteps_total": args.timesteps_total,
            # "episode_reward_mean": args.stop_reward,
            # "training_iteration": args.num_iters,
        },
        config={
            "env": f"ray-{ENVIRONMENT}",
            "eager": False,
            # Size of rollout batch
            "sample_batch_size": 10,
            # Learning rate
            "lr": 1e-4, # tune.grid_search([5e-4, 2e-4, 1e-4, 5e-5]),
            # Seeds
            "seed": tune.grid_search([i for i in range(NUM_SEEDS)]),
            # Entropy coefficient
            "entropy_coeff": 0.001, # tune.grid_search([0.1, 0.01, 0.001]),
            # Workers sample async. Note that this increases the effective
            # sample_batch_size by up to 5x due to async buffering of batches.
            "sample_async": True,
            # multiagent policy for each agent
            "multiagent": {
                "policies": {
                    f"agent_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)
                },
                "policy_mapping_fn": lambda agent_name: agent_name
            }
        },
    )
