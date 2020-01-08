# Run as "python3 a3c_rollout.py --run A3C ray_results/A3C/A3C_ray-rware-1x3-2ag-10req-indiv-v0_4b37f0e8_2020-01-07_16-53-05kgah2lt5/checkpoint_1000/checkpoint-1000"

from robotic_warehouse import Warehouse, RewardType
from robotic_warehouse.utils.wrappers import DictAgents
import baselines.a3c

from ray.rllib.rollout import create_parser, run

ENVIRONMENT = "rware-1x3-2ag-10req-indiv-v0"
parser = create_parser()
args = parser.parse_args()
run(args, parser)
