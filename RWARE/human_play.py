# Based on: https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/interactive_rendering.py
"""
Use this script to interactively play RWARE

You can control the interaction with the following keys:
- Up Arrow keys: move current agent forward
- Left/ Right Arrow keys: rotate current agent left/ right
- P/ L: pickup/ drop shelf
- SPACE: do nothing
- TAB: change the current agent
- R: reset the environment
- H: show help
- D: display agent info (per step)
- ESC: exit
"""
from argparse import ArgumentParser
import warnings

import numpy as np
import gymnasium as gym

from rware.warehouse import Action


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="rware-tiny-2ag-v2",
        help="Environment to use",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--display_info",
        action="store_true",
        help="Display agent info per step",
    )
    return parser.parse_args()


class InteractiveRWAREEnv:
    """Use this script to interactively play RWARE"""

    def __init__(
        self,
        env: str,
        max_steps,
        display_info: bool = True,
    ):
        self.env = gym.make(env, render_mode="human", max_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        self.running = True
        self.current_agent_index = 0
        self.current_action = None

        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.reset = False

        self.display_info = display_info

        obss, _ = self.env.reset()
        self.env.render()
        self.env.unwrapped.renderer.window.on_key_press = self._key_press

        if self.display_info:
            self._display_info(obss, [0] * self.n_agents, False)

        self._cycle()

    def _help(self):
        print("Use the up arrow key to move the current agent forward")
        print("Use the left/ right arrow keys to rotate the current agent left/ right")
        print("Press P or L to pickup/ drop shelf")
        print("Use the SPACE key to do nothing")
        print("Press TAB to change the current agent")
        print("Press R to reset the environment")
        print("Press H to show help")
        print("Press D to display agent info")
        print("Press ESC to exit")
        print()

    def _get_current_agent_info(self):
        agent_carrying = self.env.unwrapped.agents[self.current_agent_index].carrying_shelf
        agent_x = self.env.unwrapped.agents[self.current_agent_index].x
        agent_y = self.env.unwrapped.agents[self.current_agent_index].y
        agent_str = f"Agent {self.current_agent_index + 1} (at row {agent_y + 1}, col {agent_x + 1}"
        if agent_carrying:
            agent_str += ", carrying shelf)"
        else:
            agent_str += ")"
        return agent_str
        

    def _display_info(self, obss, rews, done):
        print(f"Step {self.t}:")
        print(f"\tSelected: {self._get_current_agent_info()}")
        print(f"\tObs: {obss[self.current_agent_index]}")
        print(f"\tRew: {round(rews[self.current_agent_index], 3)}")
        print(f"\tDone: {done}")
        print()

    def _increment_current_agent_index(self, index: int):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _key_press(self, k, mod):
        from pyglet.window import key

        if k == key.LEFT:
            self.current_action = Action.LEFT
        elif k == key.RIGHT:
            self.current_action = Action.RIGHT
        elif k == key.UP:
            self.current_action = Action.FORWARD
        elif k == key.P or k == key.L:
            self.current_action = Action.TOGGLE_LOAD
        elif k == key.SPACE:
            self.current_action = Action.NOOP
        elif k == key.TAB:
            self.current_action = None
            self.current_agent_index = self._increment_current_agent_index(
                self.current_agent_index
            )
            if self.display_info:
                print(f"Now selected: {self._get_current_agent_info()}")
        elif k == key.R:
            self.current_action = None
            self.reset = True
        elif k == key.H:
            self.current_action = None
            self._help()
        elif k == key.D:
            self.current_action = None
            self.display_info = not self.display_info
        elif k == key.ESCAPE:
            self.running = False
        else:
            self.current_action = None
            warnings.warn(f"Key {k} not recognized")

    def _cycle(self):
        while self.running:
            if self.reset:
                if self.display_info:
                    print(f"Finished episode with episodic returns: {[round(ret, 3) for ret in self.ep_returns]}")
                    print()
                obss, _ = self.env.reset()
                self.reset = False
                self.ep_returns = np.zeros(self.n_agents)
                self.t = 0

                if self.display_info:
                    self._display_info(obss, [0] * self.n_agents, False)

            if self.current_action is not None:
                actions = [Action.NOOP] * self.n_agents
                actions[self.current_agent_index] = self.current_action
                obss, rews, done, trunc, info = self.env.step([act.value for act in actions])
                self.ep_returns += np.array(rews)
                self.t += 1

                if self.display_info:
                    self._display_info(obss, rews, done or trunc)

                if done or trunc:
                    self.reset = True

                self.current_action = None
            self.env.render()
        self.env.close()



if __name__ == "__main__":
    args = parse_args()
    InteractiveRWAREEnv(env=args.env, max_steps=args.max_steps, display_info=args.display_info)
