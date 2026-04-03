
# 🚀 StartupOpsEnv

> A **deterministic, modular Reinforcement Learning environment** that simulates  
> the real-world daily operations of an early-stage startup.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-StartupOpsEnv-blue)](https://huggingface.co/spaces)

---

## 📌 Why StartupOpsEnv?

Most RL environments (grid worlds, Atari, etc.) fail to model **real-world decision complexity**.

**StartupOpsEnv bridges that gap** by introducing:
- 📊 Multi-objective optimization (budget, satisfaction, efficiency)
- ⏳ Time-sensitive decisions (deadlines, urgency)
- 🤝 Strategic trade-offs (negotiation vs rejection vs delay)
- 🔁 Deterministic reproducibility (seed-controlled simulation)

👉 This makes it ideal for:
- Reinforcement Learning experimentation  
- Decision system prototyping  
- AI agent benchmarking  
- Operations & strategy simulation  

---

## 🧠 Environment Overview

At each timestep, the agent selects an action:

| Action         | Target          | Effect                                 |
|---             |---              |---                                     |
| `reply_email`  | `email_N`       | Removes email, boosts satisfaction     |
| `ignore_email` | `email_N`       | Penalty if urgency = high              |    
| `assign_task`  | `task_N`        | Consumes team_hours, marks task done   |
| `accept_offer` | `negotiation_N` | Spends budget, adds revenue            |
| `reject_offer` | `negotiation_N` | Discards negotiation                   |
| `negotiate`    | `negotiation_N` | Lowers offer price by 10%              |
| `wait`         | —               | No-op; environment still evolves       |

⚠️ Invalid actions:
- Do **not crash the system**
- Return current state + **−1.0 penalty**

---

## 🎯 Example Simulation

Step 1:
Action: reply_email(email_2)
Reward: +0.15
Satisfaction: 0.70 → 0.75

Step 2:
Action: assign_task(task_1)
Reward: +0.10
Team Hours: 160 → 150

Step 3:
Action: ignore_email(email_3)
Reward: −2.0 (high urgency missed)

---

## 🏗️ Project Structure

startupOps/
├── app.py                  # Gradio UI + HF Spaces entry-point
├── main.py                 # CLI runner
├── openenv.yaml            # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
│
├── configs/
│   └── config.yaml         # Environment parameters
│
├── env/
│   ├── core.py             # Env class (reset, step)
│   ├── models.py           # Pydantic models
│   ├── dynamics.py         # Time + event evolution
│   ├── reward.py           # Reward logic
│   ├── grader.py           # Episode evaluation
│   └── generator.py        # Event generation
│
└── agents/
    └── baseline.py         # Heuristic agent

---

## ⚡ Quick Start

### 🔹 Local Setup

pip install -r requirements.txt

# Run UI
python app.py

# CLI simulation
python main.py --verbose

# Custom config
python main.py --seed 99 --steps 100

---

### 🔹 Docker

docker build -t startup-ops-env .
docker run -p 7860:7860 startup-ops-env

Open → http://localhost:7860

---

## ⚙️ Configuration

Defined in `configs/config.yaml`

| Key | Description |
|-----|------------|
| `seed` | Controls full determinism |
| `max_steps` | Episode length |
| `initial_budget` | Starting capital |
| `initial_satisfaction` | Customer satisfaction |
| `email_gen_prob` | Email arrival probability |
| `task_gen_prob` | Task generation probability |
| `negotiation_gen_prob` | Deal generation probability |

👉 All randomness is routed through:
random.Random(seed)

---

## 📊 Baseline Agent Performance

| Metric | Score |
|------|------|
| Email Handling | 0.82 |
| Task Completion | 0.76 |
| Negotiation Strategy | 0.68 |
| **Overall Score** | **0.75** |

👉 Heuristic: priority-based decision making

---

## 🧪 Design Principles

- ✅ Strict state modeling (Pydantic, no raw dicts)
- 🔁 Fully deterministic execution
- 🧩 Modular architecture
- 🎯 Centralized reward system
- 📈 Step-wise logging for analysis

---

## 📉 Reward System

- Email replies → positive satisfaction boost  
- Missed tasks → penalties  
- Ignored urgent emails → heavy penalties  
- Deal acceptance → budget vs revenue trade-off  
- Survival → small reward per step  

👉 Defined entirely in:
env/reward.py

---

## 🧮 Grading System

from env.grader import grade_episode

Returns:
- email_score
- task_score
- negotiation_score
- overall_score
- total_reward

📊 Weighting:
- Email: 30%  
- Task: 40%  
- Negotiation: 30%  

---

## 🌐 Hugging Face Deployment

1. Create a Space (Gradio or Docker)  
2. Push this repo  
3. app.py auto-detected  

### Deployment Features:
- Stateless execution  
- Deterministic rollouts  
- Interactive simulation UI  

---

## 💡 Use Cases

- Reinforcement Learning research  
- Multi-objective optimization problems  
- Startup decision modeling  
- AI agent benchmarking  
- Educational simulations  

---

## 🔮 Future Work

- 🤖 LLM-based decision agents  
- 👥 Multi-agent environments (teams, competitors)  
- 📊 Real-time analytics dashboard  
- 🧠 Curriculum learning scenarios  
- 🌍 Integration with real-world datasets  

---

## 🏁 Key Highlights

- Deterministic RL environment (rare in student projects)
- Real-world inspired simulation (not toy problem)
- Clean modular architecture
- Deployable + interactive (Gradio + HF Spaces)

---

## 📜 License

MIT License
