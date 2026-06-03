from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl_traffic.config import TrainingConfig


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_action_mask: np.ndarray | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 7) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    name = "dqn"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainingConfig,
        seed: int = 7,
        trainable: bool = True,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.trainable = trainable
        self.device = self._resolve_device(config.device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.policy = DQNNetwork(state_dim, action_dim, config.hidden_size).to(self.device)
        self.target = DQNNetwork(state_dim, action_dim, config.hidden_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_size, seed=seed)
        self.steps = 0
        self.losses: list[float] = []

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        epsilon = self.epsilon if self.trainable else self.config.eval_epsilon
        self.steps += 1
        valid_actions = self._valid_actions(info)
        if random.random() < epsilon:
            return int(random.choice(valid_actions))
        with torch.no_grad():
            state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy(state).squeeze(0)
            mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
            mask[torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)] = True
            q_values = q_values.masked_fill(~mask, -1e9)
            return int(torch.argmax(q_values).item())

    def observe(self, transition: dict[str, Any]) -> None:
        if not self.trainable:
            return
        self.replay.append(
            Transition(
                state=np.asarray(transition["state"], dtype=np.float32),
                action=int(transition["action"]),
                reward=float(transition["reward"]),
                next_state=np.asarray(transition["next_state"], dtype=np.float32),
                done=bool(transition["done"]),
                next_valid_action_mask=(
                    None
                    if transition.get("next_valid_action_mask") is None
                    else np.asarray(transition["next_valid_action_mask"], dtype=bool)
                ),
            )
        )
        if len(self.replay) >= max(self.config.min_replay_size, self.config.batch_size):
            try:
                self.optimize()
            except Exception as error:
                if not self._is_cuda_oom(error) or self.device.type != "cuda":
                    raise
                print("CUDA OOM during DQN optimize; switching agent to CPU.", flush=True)
                self._move_to_cpu()
                self.optimize()
        if self.steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

    @property
    def epsilon(self) -> float:
        progress = min(self.steps / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def optimize(self) -> None:
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states)
            next_masks_raw = [item.next_valid_action_mask for item in batch]
            if all(mask is not None for mask in next_masks_raw):
                next_masks = torch.as_tensor(
                    np.stack(
                        [
                            np.asarray(mask, dtype=bool)
                            for mask in next_masks_raw
                        ]
                    ),
                    dtype=torch.bool,
                    device=self.device,
                )
                empty_rows = ~next_masks.any(dim=1)
                if empty_rows.any():
                    next_masks[empty_rows] = True
                next_q = next_q.masked_fill(~next_masks, -1e9)
            next_q_values = next_q.max(dim=1).values
            targets = rewards + self.config.gamma * next_q_values * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        self.losses.append(float(loss.item()))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "policy_state_dict": self.policy.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps": self.steps,
                "config": self.config.__dict__,
            },
            output,
        )

    def clone_for_eval(self) -> "DQNAgent":
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        agent = DQNAgent(self.state_dim, self.action_dim, self.config, trainable=False)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        agent.policy.load_state_dict(self.policy.state_dict())
        agent.target.load_state_dict(self.target.state_dict())
        agent.steps = self.steps
        return agent

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: TrainingConfig,
        seed: int = 7,
        trainable: bool = False,
    ) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu")
        agent = cls(
            int(payload["state_dim"]),
            int(payload["action_dim"]),
            config,
            seed=seed,
            trainable=trainable,
        )
        agent.policy.load_state_dict(payload["policy_state_dict"])
        agent.target.load_state_dict(payload.get("target_state_dict", payload["policy_state_dict"]))
        if "optimizer_state_dict" in payload:
            agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.steps = int(payload.get("steps", 0))
        return agent

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _move_to_cpu(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = torch.device("cpu")
        self.policy.to(self.device)
        self.target.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    @staticmethod
    def _is_cuda_oom(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "cuda" in message
            and (
                "out of memory" in message
                or "cudaerrormemoryallocation" in message
                or "memory allocation" in message
            )
        )

    def _valid_actions(self, info: dict[str, Any]) -> list[int]:
        mask = info.get("valid_action_mask")
        if mask is None:
            return list(range(self.action_dim))
        mask_array = np.asarray(mask, dtype=bool)
        valid_actions = np.flatnonzero(mask_array).tolist()
        if not valid_actions:
            return list(range(self.action_dim))
        return [int(action) for action in valid_actions]
