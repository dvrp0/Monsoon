# Background & Objectives

Digital collectible card games (CCGs) pose unique AI challenges—hidden information, randomness, combinatorial action spaces, evolving card pools—and require agents that can generalize across decks and strategies. This methodology uses a **data‑driven heuristic agent** whose decision‐making is parameterized by a tunable weight vector, and applies a **competitive coevolutionary Evolutionary Strategy** to optimize those weights without external opponents .

The goal of this PRD is to guide the implementation of this methodology in **any** turn‑based card‑game environment, delivering:

* A **modular, simulator‑agnostic agent** framework
* A **reusable evolutionary optimizer** for self‑learning via coevolution
* A **robust evaluation harness** with logging, checkpoints, and reproducibility

---

## 1. Scope

* **In scope:**

  * Abstract interfaces for game state and action simulation
  * Heuristic agent with configurable weight vector
  * Evolutionary Strategy (µ + λ) with self‑adaptive mutation
  * Competitive coevolution fitness evaluation
  * Configuration management, logging, and results analysis

* **Out of scope:**

  * Game‑specific card implementations
  * User interfaces beyond basic dashboards
  * Integration with production matchmaking or networking

---

## 2. Functional Requirements

### 2.1 Simulator Abstraction

1. **GameState API**:

   * Query current resources (e.g., health, mana), board entities, hand, etc.
   * Clone & advance state by applying an `Action`.

2. **Action Enumeration**:

   * Given a `GameState`, return the set of legal `Action` objects.

3. **Simulation Engine**:

   * Run playouts deterministically from a `GameState` + `Action` to next state.

### 2.2 Heuristic Agent

1. **Weight Vector**:

   * A real‐valued vector **W** = ⟨w₁,…,wₙ⟩, configurable in \[0,1].

2. **Scoring Function**:

   * For each candidate action a in state S, compute

     ```
     Δ(a,S) = Δ_state(enemy) – Δ_state(agent) – Δ_resource
     ```

     where each Δ\_⋆ is a weighted sum of state‐feature differences .

3. **ValueOf(entity)**:

   * Generic feature vector for board entities (e.g., creatures) with weights.

4. **Action Selection**:

   * Greedy: pick the action maximizing Δ(a,S), repeating until “end‑turn.”

### 2.3 Evolutionary Optimizer

1. **Individual Representation**:

   * Genome = **W** plus per‐gene mutation strengths σᵢ (self‑adaptation).

2. **Initialization**:

   * Randomly sample µ parent individuals.

3. **Variation**:

   * For each parent, generate λ offspring by mutating σᵢ via

     ```
     σᵢ' = max(σᵢ · exp(τ'·N(0,1) + τ·Nᵢ(0,1)), ε)
     wᵢ' = clamp(wᵢ + N(0,σᵢ'), 0,1)
     ```

     with τ, τ′ = learning rates .

4. **Fitness Evaluation (Competitive Coevolution)**:

   * For all (µ + λ) individuals, run **g** simulated games against one another across **D** deck configurations; fitness = total wins .

5. **Selection (µ + λ)**:

   * Union of parents & offspring; keep top µ by fitness.

6. **Termination**:

   * After G generations or plateau of fitness.

---

## 3. Non‑Functional Requirements

* **Modularity & Reusability**: clear separation between simulator, agent, optimizer.
* **Scalability**: support parallel simulation and evaluation across CPU cores.
* **Configurability**: hyperparameters (µ, λ, G, g, D) exposed via config files.
* **Reproducibility**: fixed random seeds, versioned checkpoints every K generations.
* **Logging & Monitoring**: record fitness distributions, weight histograms, convergence metrics.
* **Robustness**: graceful handling of simulation errors; timeouts for playouts.

---

## 4. System Architecture

```text
+-----------------------+
|  Configuration Loader |
+-----------+-----------+
            |
            v
+-----------------------+         +-----------------------+
|  Evolutionary Driver  | <-----> |     Logging & DB      |
+-----------+-----------+         +-----------------------+
            |
            v
+-----------------------+
|    Population Pool    |
+-----------+-----------+
            |
            | mutate / select
            v
+-----------------------+         +-----------------------+
|    Fitness Evaluator  | <-----> | Simulation Orchestrator|
+-----------+-----------+         +-----------------------+
            |
            v
+-----------------------+
|   Heuristic Agent(s)  |
+-----------+-----------+
            |
            v
+-----------------------+
|  Game Simulator API   |
+-----------------------+
```

---

## 5. Component Breakdown

### 5.1 Configuration Loader

* Parse YAML/JSON for hyperparameters, deck definitions, random seeds.

### 5.2 HeuristicAgent Module

* Implements `scoreAction(action, state, W)` per Section 3.1 .
* Exposes `selectBestAction(state)`.

### 5.3 EvolutionaryDriver Module

* Manages population, applies ES operators, orchestrates generations.

### 5.4 FitnessEvaluator Module

* Schedules round‐robins among population using `g` games × configurations in D.
* Aggregates wins into fitness scores.

### 5.5 SimulationOrchestrator

* Batches simulation calls, handles parallelism, ensures deterministic seeds.

### 5.6 Logging & DB

* Stores per‐generation metrics, weight distributions, best‐agent snapshots.

---

## 6. Data & Interfaces

* **Deck Definition**: list of card IDs, initial hand/mulligan rules.
* **State Serialization**: JSON schema for `GameState`.
* **Action Representation**: type, parameters (e.g., card index, target ID).
* **Results Schema**: `{ agentId, opponentId, deckPair, outcome, seed }`.

---

## 7. Training & Evaluation Pipeline

1. **Setup**: ingest simulator, decks, config.
2. **Run ES**: for each generation

   * Mutate → Evaluate → Select → Log → Checkpoint
3. **Post‑Training**:

   * Analyze clusters of solutions (e.g., with Ward’s method) .
   * Export top N weight vectors.
4. **Deploy Agent**: integrate best weights into production agent.

---

## 8. Hyperparameters & Defaults

| Parameter | Default | Description            |
| :-------: | :-----: | :--------------------- |
|     µ     |    10   | Parent population size |
|     λ     |    10   | Offspring size         |
|     G     |   100   | Generations            |
|     g     |    20   | Games per deck pairing |
|     D     |    3    | Deck configurations    |
|     ε     |   1e‑5  | Minimum σᵢ             |

---

## 9. Testing & Validation

* **Unit Tests**: scoring function consistency, simulator interface mocks.
* **Integration Tests**: small‐scale ES run with µ=3, λ=3, G=5; verify non‑zero convergence.
* **Performance Tests**: measure games/sec, scale to target hardware.

---

## 10. Timeline & Milestones

| Phase                  | Duration | Deliverables                               |
| ---------------------- | -------- | ------------------------------------------ |
| 1. Setup & Interfaces  | 1 week   | Simulator adapter, data schemas            |
| 2. Agent Framework     | 2 weeks  | Heuristic agent module + unit tests        |
| 3. ES & Coevolution    | 3 weeks  | EvolutionaryDriver + FitnessEvaluator      |
| 4. Logging & CI        | 1 week   | Dashboards, CI pipelines, checkpoints      |
| 5. Experiments         | 2 weeks  | Training runs, hyperparam sweeps, analysis |
| 6. Reporting & Handoff | 1 week   | Final report, best‑agent snapshots         |

---

By following this PRD, development teams can systematically implement and extend the evolutionary optimization methodology—originally demonstrated in a Hearthstone context—to **any** turn‑based card game, ensuring a **general**, **detailed**, and **robust** AI pipeline.
