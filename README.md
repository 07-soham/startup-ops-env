# StartupOpsEnv

> A deterministic, modular Reinforcement Learning environment that simulates
> the daily operations of an early-stage startup.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-StartupOpsEnv-blue)](https://huggingface.co/spaces)

---

## Overview

**StartupOpsEnv** models a startup's operational challenges as a sequential
decision-making problem.  Each timestep the agent chooses one action from:

| Action         | Target          | Effect                                 |
|---             |---              |---                                     |
| `reply_email`  | `email_N`       | Removes email, boosts satisfaction     |
| `ignore_email` | `email_N`       | Penalty if urgency = high              |    
| `assign_task`  | `task_N`        | Consumes team_hours, marks task done   |
| `accept_offer` | `negotiation_N` | Spends budget cost, adds revenue       |
| `reject_offer` | `negotiation_N` | Discards negotiation                   |
| `negotiate`    | `negotiation_N` | Lowers offer price by 10 %             |
| `wait`         | —               | No-op; dynamics still advance          |

Invalid actions (unknown ID, wrong type, missing value) return current state
and apply a **−1.0 penalty** without crashing.

---

## Project Structure

```
startupOps/
├── app.py                  # Gradio UI + HF Spaces entry-point
├── main.py                 # CLI runner
├── openenv.yaml            # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
│
├── configs/
│   └── config.yaml         # All environment hyper-parameters
│
├── env/
│   ├── __init__.py
│   ├── core.py             # StartupOpsEnv class (reset / step / _get_obs)
│   ├── models.py           # Pydantic models: Email, Task, Negotiation, EnvState, Observation
│   ├── dynamics.py         # Deadline counting, miss detection, expiry
│   ├── reward.py           # Centralised reward function
│   ├── grader.py           # Episode grader (email/task/negotiation scores)
│   └── generator.py        # Deterministic probabilistic event generator
│
└── agents/
    ├── __init__.py
    └── baseline.py         # Priority-based heuristic agent
```

---

## Quick Start

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio UI
python app.py

# Run headless CLI (verbose step log)
python main.py --verbose

# Override seed or episode length
python main.py --seed 99 --steps 100
```

### Docker

```bash
docker build -t startup-ops-env .
docker run -p 7860:7860 startup-ops-env
```

Then open [http://localhost:7860](http://localhost:7860).

---

## Configuration (`configs/config.yaml`)

| Key | Default | Description |
|---|---|---|
| `seed` | 42 | RNG seed — all randomness routes through `random.Random(seed)` |
| `max_steps` | 50 | Episode length |
| `initial_budget` | 100 000 | Starting budget ($) |
| `initial_satisfaction` | 0.7 | Starting customer satisfaction [0, 1] |
| `initial_team_hours` | 160 | Available team hours per episode |
| `max_emails` | 5 | Maximum emails in inbox at once |
| `max_tasks` | 5 | Maximum tasks on the board at once |
| `max_negotiations` | 3 | Maximum open negotiations at once |
| `email_gen_prob` | 0.4 | Probability a new email arrives each step |
| `task_gen_prob` | 0.3 | Probability a new task appears each step |
| `negotiation_gen_prob` | 0.2 | Probability a new negotiation opens each step |
| `reply_satisfaction_boost` | 0.05 | Satisfaction gain per replied email |
| `high_urgency_ignore_penalty` | −2.0 | Reward penalty for ignoring urgent email |
| `task_miss_penalty` | −3.0 | Reward penalty per missed task |
| `accept_offer_budget_cost` | 5 000 | Budget consumed when accepting a deal |
| `negotiate_adjustment` | 0.9 | Factor applied to offer_price on `negotiate` |
| `step_survival_reward` | 0.1 | Small reward for each step survived |

---

## Design Contracts

1. **Internal state** — stored as `EnvState` (Pydantic model), never raw dicts.
2. **Observation** — derived via `_get_obs()`, returned as `Observation` (Pydantic model).
3. **Logs** — `env.logs: List[dict]` with keys `step / action / reward / budget / efficiency`.
4. **Termination** — `done = (time_step >= max_steps)`.
5. **RNG** — `self.rng = random.Random(config["seed"])` only; no global `random` calls.
6. **Reward** — computed entirely in `reward.py`; `core.py` only calls `calculate_reward(...)`.

---

## Grader

```python
from env.grader import grade_episode

grades = grade_episode(
    logs=env.logs,
    total_emails_created=totals["total_emails_created"],
    total_tasks_created=totals["total_tasks_created"],
    total_negotiations_created=totals["total_negotiations_created"],
    state=env.state,
)
# grades keys: email_score, task_score, negotiation_score, overall_score, total_reward, summary
```

Scoring weights: **30 % email · 40 % task · 30 % negotiation**.

---

## Hugging Face Spaces Deployment

1. Create a new Space (type: **Docker** or **Gradio**).
2. Push this repository as-is.
3. The entry-point `app.py` is auto-detected by HF Spaces.
4. No environment variables or secrets required.

---

## License

MIT
