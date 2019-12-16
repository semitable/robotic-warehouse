from robotic_warehouse import Warehouse, RewardType
from robotic_warehouse.utils.wrappers import DictAgents
import baselines.ppo

from ray.rllib.rollout import create_parser, run

ENVIRONMENT = "rware-1x3-2ag-10req-indiv-v0"
parser = create_parser()
args = parser.parse_args()
run(args, parser)
