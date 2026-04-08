---
title: StartupOps
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_file: app.py
short_description: AI-powered startup ops simulator
---

# 🚀 StartupOps AI Simulator

> A **full-stack AI-powered simulation platform** where LLMs convert unstructured startup communication into structured signals, and a deterministic RL environment evaluates decision-making under realistic operational pressure.

---

## 📌 Why This Project Matters

Real-world decision-making is messy:
- Emails are unstructured  
- Priorities shift dynamically  
- Resources are limited  
- Decisions have trade-offs  

Most RL environments **do not capture this complexity**.

👉 **StartupOps AI Simulator solves this gap** by combining:
- 🧠 LLM-based understanding (emails → structured data)
- ⚙️ Deterministic RL environment (decision evaluation)
- 📊 Multi-objective optimization (budget, time, satisfaction)

This creates a **realistic testbed for AI agents and decision systems**.

---

## 🧠 Core Idea

Turn messy startup operations into a **structured, learnable decision problem**:
Unstructured Emails → LLM Parsing → Structured State → RL Decisions → Performance Score

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React/Next.js │────▶│   FastAPI       │────▶│   RL Environment│
│   Frontend      │     │   API Layer     │     │   + LLM Parser  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │              ┌────────┴──────────┐            │
        │              │  Scenario Lib     │            │
        │              │  - investor_pressure           │
        │              │  - vendor_delay   │            │
        │              │  - customer_churn │            │
        │              │  - hiring_crunch  │            │
        │              └───────────────────┘            │
        │                                               │
        │              ┌─────────────────┐              │
        └──────────────│  LLM Parser     │──────────────┘
                       │  - Claude API   │
                       │  - OpenAI API   │
                       │  - Fallback     │
                       └─────────────────┘
```


---

## ⚡ Key Features

### 🧠 AI-Powered Understanding
- LLM-based email parsing (Claude / OpenAI)
- Extracts:
  - urgency
  - sentiment
  - required actions
- Fallback parser ensures reliability without API

---

### 🎯 Realistic Simulation
- Scenario-driven environments:
  - Investor pressure
  - Vendor delays
  - Customer churn
  - Hiring crunch

- Context-aware threads + escalation logic
- Urgency overrides based on follow-ups

---

### ⚙️ Deterministic RL Engine
- Same seed + same inputs → identical outputs
- Fully reproducible experiments
- Modular environment design

---

### 🔄 Dual Execution Modes
- **Auto Mode** → scenario-based simulation  
- **Manual Mode** → custom inputs for testing  

---

## 🎮 Example Flow
```
Email: "We need this ASAP!"

        ↓ (LLM Parser)

{
        urgency: "high",
        requires_action: true
}

        ↓ (RL Decision)

Action: assign_task(task_1)

        ↓ (Environment)

Reward: +0.10
Budget: unchanged
Satisfaction: increased
```
---

## Overview

**StartupOpsEnv** models a startup's operational challenges as a sequential
decision-making problem with AI-powered email parsing and context-aware escalation.

### Actions

| Action         | Target          | Effect                                 |
|---             |---              |---                                     |
| `reply_email`  | `email_N`       | Removes email, boosts satisfaction     |
| `ignore_email` | `email_N`       | Penalty if urgency = high              |
| `assign_task`  | `task_N`        | Consumes team_hours, marks task done   |
| `accept_offer` | `negotiation_N` | Spends budget, adds revenue            |
| `reject_offer` | `negotiation_N` | Discards negotiation                   |
| `negotiate`    | `negotiation_N` | Lowers offer price by 10%              |
| `wait`         | —               | No-op; environment still evolves       |

### Features

- **Scenario Library**: 4 realistic startup scenarios (investor pressure, vendor delay, customer churn, hiring crunch)
- **LLM Email Parsing**: Claude or OpenAI API integration with keyword fallback
- **Context-Aware Threads**: Thread history passed to parser for better context
- **Escalation Logic**: Automatic escalation detection from follow-ups and sentiment
- **Urgency Override**: Level 1→min medium, Level 2+→high
- **Dual Mode**: Auto (scenario-based) or Manual (user-defined inputs)
- **Full Determinism**: Same seed + same input = same output

---

## 🎯 Example Simulation
```
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
```
---

## 🏗️ Project Structure
```
startupOps/
├── api.py                  # FastAPI backend entry-point
├── main.py                 # CLI runner
├── openenv.yaml            # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile              # Backend container
├── docker-compose.yml      # Full stack deployment
├── run.sh                  # Quick start script
│
├── configs/
│   └── config.yaml         # Environment hyper-parameters
│
├── env/                    # RL Environment
│   ├── __init__.py
│   ├── core.py             # StartupOpsEnv class
│   ├── models.py           # Pydantic models
│   ├── dynamics.py         # Deadline counting, miss detection
│   ├── reward.py           # Reward function
│   ├── grader.py           # Episode scoring
│   ├── generator.py        # Event generator (auto/manual modes)
│   ├── scenarios.py        # Scenario library
│   └── llm_parser.py       # LLM + fallback parser
│
├── agents/                 # RL Agents
│   ├── __init__.py
│   └── baseline.py         # Priority-based heuristic
│
├── tests/                  # Test suite
│   └── test_env.py
│
└── frontend/               # Next.js React Frontend
    ├── components/         # React components
    ├── pages/             # Next.js pages
    ├── lib/               # API client
    └── styles/            # Tailwind CSS
```
---

## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- (Optional) Anthropic API key or OpenAI API key for LLM parsing

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python api.py

# Or use the run script
./run.sh
```

API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be at http://localhost:3000

### CLI (Optional)

```bash
# Run headless CLI
python main.py --verbose

# Override seed or episode length
python main.py --seed 99 --steps 100
```

---

## Docker Deployment

### Docker Compose (Recommended)

```bash
# Set your API key (optional - supports Anthropic or OpenAI)
export ANTHROPIC_API_KEY=your_key_here
# OR
export OPENAI_API_KEY=your_key_here

# Run everything
docker-compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:3000

### Individual Containers

```bash
# Build and run backend (with Anthropic)
docker build -t startupops-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key startupops-api

# Or with OpenAI
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key startupops-api

# Build and run frontend
cd frontend
docker build -t startupops-frontend .
docker run -p 3000:3000 startupops-frontend
```

---

## API Endpoints

### GET `/`
Health check

### GET `/scenarios`
List all available scenarios

### POST `/parse-email`
Parse a single email with LLM

**Request:**
```json
{
  "text": "We need this ASAP!",
  "sender": "boss@company.com",
  "subject": "Urgent request"
}
```

**Response:**
```json
{
  "urgency": "high",
  "sentiment": "neutral",
  "requires_action": true,
  "confidence": 0.95
}
```

### POST `/run-simulation`
Run full simulation

**Auto Mode:**
```json
{
  "mode": "auto",
  "scenario": "investor_pressure",
  "seed": 42,
  "max_steps": 50
}
```

**Manual Mode:**
```json
{
  "mode": "manual",
  "emails": [
    {
      "text": "We need this ASAP!",
      "sender": "boss@company.com",
      "subject": "Urgent request",
      "thread_id": "thread_1"
    }
  ],
  "tasks": [
    {
      "name": "Fix critical bug",
      "hours_required": 4,
      "deadline": 2,
      "priority": "high",
      "effort": 3,
      "impact": 2.0
    }
  ],
  "seed": 42
}
```

---

## ⚙️ Configuration

| Key | Default | Description |
|---|---|---|
| `seed` | 42 | RNG seed — all randomness routes through `random.Random(seed)` |
| `max_steps` | 50 | Episode length |
| `initial_budget` | 100 000 | Starting budget ($) |
| `initial_satisfaction` | 0.8 | Starting customer satisfaction [0, 1] |
| `initial_team_hours` | 100 | Available team hours per episode |
| `max_emails` | 20 | Maximum emails in inbox at once |
| `max_tasks` | 20 | Maximum tasks on the board at once |
| `max_negotiations` | 10 | Maximum open negotiations at once |
| `email_gen_prob` | 0.0 | Probability a new email arrives each step |
| `task_gen_prob` | 0.0 | Probability a new task appears each step |
| `negotiation_gen_prob` | 0.0 | Probability a new negotiation opens each step |
| `reply_satisfaction_boost` | 0.1 | Satisfaction gain per replied email |
| `high_urgency_ignore_penalty` | −2.0 | Reward penalty for ignoring urgent email |
| `task_miss_penalty` | −5.0 | Reward penalty per missed task |
| `accept_offer_budget_cost` | 1000 | Budget consumed when accepting a deal |
| `negotiate_adjustment` | 0.9 | Factor applied to offer_price on `negotiate` |
| `step_survival_reward` | 0.1 | Small reward for each step survived |

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

## Testing

```bash
# Backend tests
pytest tests/test_env.py -v

# Frontend tests (in frontend/)
cd frontend
npm test
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for LLM parsing (optional) |
| `OPENAI_API_KEY` | OpenAI API key for LLM parsing (optional, alternative to Anthropic) |
| `NEXT_PUBLIC_API_URL` | Frontend API URL (default: http://localhost:8000) |

**Note:** If both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set, Anthropic will be used by default.

---
## 💡 Use Cases

- RL research & experimentation  
- AI decision system benchmarking  
- Startup strategy simulation  
- Multi-objective optimization problems  
- Hackathon AI demos  

---

## 🔮 Future Improvements

- Multi-agent simulation (team vs competitors)  
- LLM-driven autonomous agents  
- Real-time analytics dashboard  
- Adaptive difficulty scenarios  

---

## 🏁 What Makes This Stand Out

- Combines **LLM + RL (rare combo)**  
- Fully **deterministic + reproducible**  
- Real-world inspired (not toy problem)  
- Full-stack (Frontend + Backend + AI)  
- Production-ready architecture  

---

## Licenses

MIT
