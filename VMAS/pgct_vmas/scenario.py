"""Team Discovery: a multi-team variant of the VMAS ``discovery`` scenario for PGCT (needs ``vmas``).

VMAS discovery (Bettini et al., VMAS 2022; github.com/proroklab/VectorizedMultiAgentSimulator): holonomic
point-mass agents with LIDARs in a bounded 2D arena; a target is covered when at least ``agents_per_target``
agents are within ``covering_range`` of it; every covering agent receives ``covering_rew_coeff`` (individual
reward, shared_reward=False) and covered targets respawn at random free positions.

Extension for latent teams: agents belong to hidden teams and every target has a type (one per team). Physics
and covering ignore teams (any agents count towards covering), but a covering agent is paid only for targets of
its own team's type. Each agent carries one LIDAR per target type (so types are visible, like coloured
targets) plus an agent LIDAR. The observation is VMAS discovery's (own position, own velocity, LIDAR
distances); which type pays is never observed. No termination: episodes end at the time limit (truncation).

This file imports VMAS (GPLv3). VMAS is an optional dependency and is not vendored in this repository.
"""
from __future__ import annotations

from typing import Dict

import torch
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils

TYPE_COLORS = (Color.GREEN, Color.RED, Color.BLUE, Color.YELLOW)


class TeamDiscovery(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.team_ids = [int(t) for t in kwargs.pop("team_ids")]          # hidden assignment, one per agent
        self.n_types = int(kwargs.pop("n_types", max(self.team_ids) + 1))
        self.targets_per_type = int(kwargs.pop("targets_per_type", 3))
        self.x_semidim = float(kwargs.pop("x_semidim", 1.0))
        self.y_semidim = float(kwargs.pop("y_semidim", 1.0))
        self._min_dist_between_entities = float(kwargs.pop("min_dist_between_entities", .2))
        self._lidar_range = float(kwargs.pop("lidar_range", .35))
        self._covering_range = float(kwargs.pop("covering_range", .25))
        self.n_lidar_rays_targets = int(kwargs.pop("n_lidar_rays_targets", 12))
        self.n_lidar_rays_agents = int(kwargs.pop("n_lidar_rays_agents", 12))
        self._agents_per_target = int(kwargs.pop("agents_per_target", 1))
        self.covering_rew_coeff = float(kwargs.pop("covering_rew_coeff", 1.0))
        self.agent_collision_penalty = float(kwargs.pop("agent_collision_penalty", 0.0))
        self.time_penalty = float(kwargs.pop("time_penalty", 0.0))
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.n_agents = len(self.team_ids)
        self.min_collision_distance = .005
        self.agent_radius = self.target_radius = .05

        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim, collision_force=500,
                      substeps=2, drag=.25)
        for i in range(self.n_agents):
            sensors = [Lidar(world, n_rays=self.n_lidar_rays_targets, max_range=self._lidar_range,
                             entity_filter=(lambda e, t=t: e.name.startswith(f"target{t}_")))
                       for t in range(self.n_types)]
            sensors.append(Lidar(world, angle_start=.05, angle_end=2 * torch.pi + .05, n_rays=self.n_lidar_rays_agents,
                                 max_range=self._lidar_range, entity_filter=lambda e: e.name.startswith("agent")))
            agent = Agent(name=f"agent_{i}", collide=True, shape=Sphere(radius=self.agent_radius), sensors=sensors)
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)
        self._targets, types = [], []
        for t in range(self.n_types):
            for k in range(self.targets_per_type):
                target = Landmark(name=f"target{t}_{k}", collide=True, movable=False,
                                  shape=Sphere(radius=self.target_radius), color=TYPE_COLORS[t % len(TYPE_COLORS)])
                world.add_landmark(target)
                self._targets.append(target)
                types.append(t)
        self.target_types = torch.tensor(types, device=device)
        own = torch.tensor(self.team_ids, device=device)[:, None] == self.target_types[None, :]   # [agents, targets]
        self._own_mask = own
        zeros = torch.zeros(batch_dim, self.n_agents, device=device)
        self.covering_rew, self.covered_own, self.covered_other = zeros.clone(), zeros.clone(), zeros.clone()
        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(entities=self._targets + self.world.agents, world=self.world,
                                              env_index=env_index,
                                              min_dist_between_entities=self._min_dist_between_entities,
                                              x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                                              y_bounds=(-self.world.y_semidim, self.world.y_semidim))

    def reward(self, agent: Agent):
        index = self.world.agents.index(agent)
        if index == 0:
            agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)        # [B, n, 2]
            targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)          # [B, m, 2]
            within = torch.cdist(agents_pos, targets_pos) < self._covering_range           # [B, n, m]
            covered = within.sum(dim=1) >= self._agents_per_target                          # [B, m]
            hits = within & covered[:, None, :]
            self.covered_own = (hits & self._own_mask[None]).sum(-1).float()
            self.covered_other = (hits & ~self._own_mask[None]).sum(-1).float()
            self.covering_rew = self.covered_own * self.covering_rew_coeff
            self._agents_pos, self._covered = agents_pos, covered
        agent.collision_rew[:] = 0
        if self.agent_collision_penalty:
            for other in self.world.agents:
                if other is not agent:
                    close = self.world.get_distance(other, agent) < self.min_collision_distance
                    agent.collision_rew[close] += self.agent_collision_penalty
        if index == self.n_agents - 1:
            self._respawn_covered()
        return self.covering_rew[:, index] + agent.collision_rew + self.time_penalty

    def _respawn_covered(self):
        for i, target in enumerate(self._targets):
            others = [o.state.pos.unsqueeze(1) for o in self._targets if o is not target]
            occupied = torch.cat([self._agents_pos] + others, dim=1)
            pos = ScenarioUtils.find_random_pos_for_entity(
                occupied, env_index=None, world=self.world, min_dist_between_entities=self._min_dist_between_entities,
                x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                y_bounds=(-self.world.y_semidim, self.world.y_semidim))
            target.state.pos[self._covered[:, i]] = pos[self._covered[:, i]].squeeze(1)

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel] + [sensor.measure() for sensor in agent.sensors], dim=-1)

    def info(self, agent: Agent) -> Dict[str, torch.Tensor]:
        index = self.world.agents.index(agent)
        return {"covered_own": self.covered_own[:, index], "covered_other": self.covered_other[:, index]}

    def done(self):
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
