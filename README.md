# Multi-Robot Warehouse Environment (RWARE)
### A multi-agent environment for Reinforcement Learning 

# Please cite:
If you use this environment, consider citing
1. A comperative evaluation of MARL algorithms that includes this environment
```
@article{papoudakis2020comparative,
  title={Comparative Evaluation of Multi-Agent Deep Reinforcement Learning Algorithms},
  author={Papoudakis, Georgios and Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2006.07869},
  year={2020}
}
```
2. A method that achieves state-of-the-art performance in the robotic warehouse task
```
@article{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2006.07169},
  year={2020}
}
```

# Installation:
```sh
git clone https://github.com/semitable/robotic-warehouse.git
cd robotic-warehouse
pip install -e .
```

# Usage
To use in Python, import it and use it as a gym environment:
```python
import robotic_warehouse
import gym

env = gym.make("rware-tiny-2ag-v1")
n_obs = env.reset()
n_obs, n_reward, n_done, info = env.step(n_action)
```
where `n_*` is a list of length `len(agents)`

