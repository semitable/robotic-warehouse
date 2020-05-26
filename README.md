# Robotic Warehouse Environment 
### A multi-agent reinforcement learning environment

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

env = gym.make("rware-tiny-2ag-v0")
n_obs = env.reset()
n_obs, n_reward, n_done, info = env.step(n_action)
```
where `n_*` is a list of length `len(agents)`

