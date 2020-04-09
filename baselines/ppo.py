import gym, ray
import pprint
from ray.rllib.agents import ppo
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
        "PPO",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={
            "episode_reward_mean": args.stop_reward,
            "training_iteration": args.num_iters,
        },
        config={
            "env": f"ray-{ENVIRONMENT}",
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": 5,
            "lr": 1e-4, # tune.grid_search([5e-4, 2e-4, 1e-4, 5e-5]),
            # Seeds
            "seed": tune.grid_search([i for i in range(NUM_SEEDS)]),
            "eager": False,
            "multiagent": {
                "policies": {
                    f"agent_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)
                },
                "policy_mapping_fn": lambda agent_name: agent_name
            }
        },
    )
