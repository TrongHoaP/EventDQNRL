# EventDrivenRLController

## Modular SUMO RL Traffic Control

The RL traffic-control stack lives in `src/rl_traffic/`. It separates the SUMO
environment, safety layer, rewards, metrics, baseline controllers, and DQN
training code so each part can be changed without editing the core environment.

Default config:

```powershell
src/configs/rl_traffic_control.json
```

Smoke-test the Gymnasium environment with a random agent:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.smoke_env --episode-seconds 120
```

Run one baseline:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.run_baseline --controller fixed_time --episodes 1
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.run_baseline --controller actuated --episodes 1
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.run_baseline --controller max_pressure --episodes 1
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.run_baseline --controller random --episodes 1
```

Train DQN:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.train_dqn --episodes 5 --run-name dqn
```

Evaluate baselines plus a trained DQN checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.evaluate --checkpoint runs\dqn\checkpoints\best.pt --run-name evaluation
```

Important config knobs:

- `control.decision_interval_seconds`: default `10`, controls the action interval.
- `control.min_green_seconds` and `control.max_green_seconds`: enforced by the safe layer.
- `safety.enabled`: hard safety gate between controller actions and SUMO phase changes.
- `state.mode`: `lane_level` or `approach_level`.
- `reward`: all reward weights are editable without changing the environment.

Outputs are written under `runs/<run-name>/metrics/` and
`runs/<run-name>/checkpoints/`.
