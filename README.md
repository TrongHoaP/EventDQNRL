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

Train DQN with stability-first checkpointing:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.train_dqn --config src\configs\rl_traffic_control_stability.json --run-name dqn_stability
```

Train DQN with spike-aware reward and validation:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.train_dqn --config src\configs\rl_traffic_control_spike_005.json --run-name dqn_spike_005_50ep
```

Evaluate baselines plus a trained DQN checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.evaluate --checkpoint runs\dqn\checkpoints\best.pt --run-name evaluation
```

Evaluate DQN with seed-varying demand for robustness:

```powershell
.\.venv\Scripts\python.exe -m src.rl_traffic.runners.evaluate --config src\configs\rl_traffic_control_masked_approach_dqn_eval_robust.json --checkpoint runs\dqn_stability_50ep\checkpoints\best_validation.pt --run-name eval_robust
```

Important config knobs:

- `control.decision_interval_seconds`: default `10`, controls the action interval.
- `control.min_green_seconds` and `control.max_green_seconds`: enforced by the safe layer.
- `safety.enabled`: hard safety gate between controller actions and SUMO phase changes.
- `state.mode`: `lane_level` or `approach_level`.
- `reward`: all reward weights are editable without changing the environment.
- `reward.queue_spike_*` and `reward.tail_queue_spike_*`: thresholded penalties for reducing queue spikes.
- `sumo.randomize_demand` and `sumo.demand_time_jitter_seconds`: vary spawn times by episode seed for robustness evaluation.
- `training.checkpoint_window`: moving-average window for `best_stable.pt`.
- `training.validation_interval` and `training.validation_episodes`: optional validation checkpointing.

Outputs are written under `runs/<run-name>/metrics/` and
`runs/<run-name>/checkpoints/`. Training writes `best_train.pt` for single-episode
reward, `best_stable.pt` for moving-average reward, and `best_validation.pt`
when validation checkpointing is enabled.
