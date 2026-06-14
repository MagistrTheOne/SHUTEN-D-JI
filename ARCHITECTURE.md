# SHUTEN-DŌJI: Synthetic Strategic Intelligence Factory

## Architecture Document v0.1

---

## PART 1: CORE PRINCIPLES EXTRACTED FROM KIMI K2

### What Kimi K2 Actually Did

Kimi K2 is not a model. It is a **production system** that happens to output a model. The key insight:

1. **Synthetic data is the product, not the model.** The model is a compression artifact of the data factory.
2. **Environments are generators.** Real and simulated environments produce training signal at scale.
3. **Verification closes the loop.** Without verifiable outcomes, RL collapses to noise.
4. **Self-critique extends beyond verification.** When you can't verify, you teach the model to judge itself — grounded in verifiable signal from other domains.
5. **Token efficiency is the real scaling law.** Not "more tokens" but "more intelligence per token."
6. **Trajectories > demonstrations.** Multi-step interaction sequences with outcomes are worth more than static input-output pairs.
7. **The critic and the actor co-evolve.** The judge improves alongside the generator in a closed loop.

### Generalized Principles for SHUTEN-DŌJI

| Kimi K2 Principle | SHUTEN-DŌJI Generalization |
|---|---|
| Synthetic tool-use trajectories | Synthetic strategic decision trajectories |
| Simulated tool execution environment | Simulated world-state environments |
| Rubric-based task generation | Objective-function-based scenario generation |
| Quality filtering via judge | Outcome verification via simulation rollout |
| RLVR on coding/math | RLVR on planning/prediction/analysis |
| Self-critique reward | Strategic self-evaluation with hindsight |
| Agent diversification | Role-specialized cognitive agents |
| Domain evolution for tools | Domain evolution for world-states |
| Partial rollout for long trajectories | Partial rollout for long-horizon decisions |
| MoE for token efficiency | MoE for cognitive specialization |

---

## PART 2: WHY "SYNTHETIC STRATEGIC INTELLIGENCE FACTORY" NOT "LLM"

### The Fundamental Reframe

An LLM is a compression of text. SHUTEN-DŌJI is a **factory that produces intelligence artifacts** — trained models are one output among many.

```
LLM Paradigm:
  Internet Text → Compression → Model → Inference

SHUTEN-DŌJI Paradigm:
  World States → Simulation → Trajectories → Evaluation → Training Signal
       ↑                                                        ↓
       └────────────── Continuous Improvement Loop ─────────────┘
```

### Why This Matters

1. **The factory is the asset, not the weights.** Weights depreciate. The factory appreciates.
2. **You can produce specialized models on demand.** Need a geopolitical analyst? Run the factory with geopolitical scenarios. Need a logistics optimizer? Run it with supply chain states.
3. **The data never runs out.** Synthetic generation is bounded only by compute and scenario diversity.
4. **Verification is built in.** Every training example has an outcome. Every outcome can be scored.
5. **You can iterate the factory independently of the model.** Improve scenario generation without retraining from scratch.

### What SHUTEN-DŌJI Produces

| Output | Description |
|---|---|
| Strategic trajectory datasets | State → Analysis → Decision → Outcome sequences |
| Trained model checkpoints | Compressed intelligence from trajectories |
| Evaluation environments | Reusable scenario simulators |
| Reward functions | Learned and programmatic scoring systems |
| Critic models | Self-evaluation capabilities |
| Domain libraries | Parameterized world-state generators |

---

## PART 3: THE DATA FACTORY

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SHUTEN-DŌJI DATA FACTORY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │    STATE      │──▶│    SCENARIO       │──▶│     AGENT         │   │
│  │  GENERATOR    │   │    GENERATOR      │   │   SIMULATOR       │   │
│  └──────────────┘   └──────────────────┘   └───────────────────┘   │
│         │                    │                        │               │
│         ▼                    ▼                        ▼               │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │ ENVIRONMENT   │──▶│    OUTCOME        │──▶│   TRAJECTORY      │   │
│  │  SIMULATOR    │   │   EVALUATOR       │   │     STORE         │   │
│  └──────────────┘   └──────────────────┘   └───────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 STATE GENERATOR

**Purpose:** Generate realistic, machine-readable world states that serve as initial conditions for strategic scenarios.

#### Schema

```yaml
WorldState:
  id: uuid
  timestamp: ISO-8601
  domain: enum[business, geopolitics, logistics, markets, technology, military, organizational]
  
  entities:
    - id: uuid
      type: enum[organization, person, system, resource, location, asset]
      attributes: dict
      capabilities: list[str]
      constraints: list[str]
      objectives: list[Objective]
      
  relationships:
    - source: entity_id
      target: entity_id
      type: enum[controls, depends_on, competes_with, allies_with, supplies, reports_to]
      strength: float[0,1]
      volatility: float[0,1]
      
  events:
    - id: uuid
      type: enum[disruption, opportunity, threat, deadline, transition]
      timestamp: ISO-8601
      affected_entities: list[entity_id]
      probability: float[0,1]
      impact_magnitude: float[-1,1]
      
  constraints:
    - type: enum[resource, temporal, regulatory, physical, informational]
      scope: list[entity_id]
      description: str
      severity: float[0,1]
      
  risks:
    - id: uuid
      source: str
      probability: float[0,1]
      impact: float[0,1]
      affected: list[entity_id]
      mitigatable: bool
      
  objectives:
    - id: uuid
      owner: entity_id
      description: str
      priority: float[0,1]
      deadline: Optional[ISO-8601]
      measurable_criteria: list[str]
      conflicts_with: list[objective_id]
      
  information_state:
    known_to_all: list[fact_id]
    known_to_subset: dict[entity_id, list[fact_id]]
    unknown_but_discoverable: list[fact_id]
    fundamentally_uncertain: list[distribution]
```

#### Generation Strategy

1. **Seed from real data:** Extract entity-relationship structures from public sources (SEC filings, news corpora, supply chain databases, organizational charts).
2. **Parameterize and perturb:** Change names, scales, timing. Add/remove entities. Alter relationships.
3. **Complexity curriculum:** Start simple (3-5 entities, 1 objective) → scale to complex (50+ entities, conflicting objectives, hidden information).
4. **Domain evolution:** Start with base domains, then generate hybrid domains (e.g., "technology company in a logistics crisis during regulatory change").
5. **Consistency validation:** Run constraint propagation to ensure generated states are internally consistent.

#### Quality Controls

- **Realizability check:** Can the state actually exist? Are the constraints satisfiable?
- **Complexity scoring:** Entropy of the entity-relationship graph. Reject trivial or degenerate states.
- **Diversity scoring:** Embedding distance from all previously generated states. Reject near-duplicates.

---

### 3.2 SCENARIO GENERATOR

**Purpose:** Given a world state, generate plausible future branches with causal chains and uncertainty estimates.

#### Output Schema

```yaml
ScenarioBundle:
  initial_state: WorldState
  time_horizon: duration
  scenarios:
    - id: uuid
      label: str  # "Scenario A", "Scenario B", etc.
      probability: float[0,1]  # prior probability
      
      causal_chain:
        - step: int
          event: Event
          cause: str
          mechanism: str
          preconditions: list[condition]
          
      branch_points:
        - timestamp: ISO-8601
          decision_required: bool
          options: list[Action]
          information_available: list[fact_id]
          
      terminal_state: WorldState
      outcome_metrics:
        - metric: str
          value: float
          relative_to_baseline: float
          
      uncertainty_decomposition:
        aleatoric: float  # irreducible randomness
        epistemic: float  # reducible with more information
        decision_dependent: float  # depends on choices made
        
    # Minimum 3 scenarios per bundle, maximum 7
    
  critical_uncertainties:
    - description: str
      scenarios_affected: list[scenario_id]
      resolvability: enum[observable, partially_observable, unobservable]
```

#### Generation Strategy

1. **Identify critical uncertainties** in the initial state (events with high impact × moderate probability).
2. **Branch on decisions and uncertainties.** Each combination produces a scenario path.
3. **Forward-simulate causal chains** using rules + LLM reasoning.
4. **Prune implausible branches** via consistency checking.
5. **Assign probabilities** based on base rates, state conditions, and historical analogies.
6. **Generate at least 3 scenarios per state:** optimistic, pessimistic, and surprising.

#### Verification

- **Causal consistency:** Each effect must have a cause. No effect precedes its cause.
- **Probability coherence:** Scenario probabilities must sum to ≤ 1.0 (with residual for unmodeled outcomes).
- **Outcome differentiation:** Scenarios must be meaningfully different (measured by terminal state divergence).

---

### 3.3 AGENT SIMULATOR

**Purpose:** Create synthetic cognitive agents that interact with world states, producing decision trajectories.

#### Agent Types

| Role | Function | Evaluation Criteria |
|---|---|---|
| **Analyst** | Decompose state into factors, identify patterns, assess risks | Completeness, accuracy, insight novelty |
| **Planner** | Generate action sequences toward objectives | Feasibility, optimality, robustness |
| **Researcher** | Gather information, reduce uncertainty | Efficiency, relevance, depth |
| **Critic** | Identify flaws, risks, blind spots in plans/analyses | Detection rate, false positive rate, constructiveness |
| **Executor** | Implement plans step-by-step, handle contingencies | Completion rate, adaptation quality, resource efficiency |
| **Forecaster** | Predict outcomes, estimate probabilities | Calibration, resolution, discrimination |
| **Negotiator** | Manage multi-party interactions toward outcomes | Agreement rate, value captured, relationship preservation |

#### Agent Configuration

```yaml
Agent:
  id: uuid
  role: enum[analyst, planner, researcher, critic, executor, forecaster, negotiator]
  
  cognitive_profile:
    risk_tolerance: float[0,1]
    time_preference: float[0,1]  # discount rate
    information_seeking: float[0,1]
    confidence_calibration: float[0,1]
    creativity: float[0,1]
    
  knowledge_state:
    domain_expertise: dict[domain, float]
    information_access: list[fact_id]
    beliefs: dict[proposition, float]  # subjective probabilities
    
  behavioral_constraints:
    max_actions_per_turn: int
    available_actions: list[ActionType]
    communication_style: str
    
  system_prompt: str  # role-specific instructions
```

#### Trajectory Generation

1. **Initialize** agent with role, knowledge state, and objectives.
2. **Present** world state (filtered by agent's information access).
3. **Agent produces** analysis/plan/decision at each timestep.
4. **Environment responds** with outcomes, new information, other agent actions.
5. **Continue** until terminal condition (objective achieved, deadline reached, failure detected).

#### Multi-Agent Interactions

- **Collaborative:** Analyst feeds Planner, Critic reviews Plan, Executor implements.
- **Adversarial:** Competing planners, red-team vs blue-team.
- **Hierarchical:** Senior analyst supervises junior analysts, aggregates findings.

#### Quality Filtering

- Trajectories scored by outcome quality (did the agent achieve its objective?).
- Trajectories scored by process quality (was reasoning sound, even if outcome was unlucky?).
- Only top-quartile trajectories retained for training.
- Failed trajectories with informative mistakes retained separately for critique training.

---

### 3.4 ENVIRONMENT SIMULATOR

**Purpose:** Create dynamic environments that respond to agent actions with realistic consequences.

#### Environment Types

| Domain | Key Dynamics | State Variables |
|---|---|---|
| **Business** | Market share, revenue, competition, hiring | Financials, headcount, product pipeline, customer base |
| **Operations** | Throughput, bottlenecks, failures, maintenance | Capacity, queue lengths, equipment state, staffing |
| **Logistics** | Routes, delays, costs, demand fluctuation | Inventory, transit times, supplier status, demand forecast |
| **Markets** | Price, volume, sentiment, regulation | Order book, volatility, correlation, liquidity |
| **Projects** | Tasks, dependencies, resources, risks | Gantt state, budget, team allocation, scope |
| **Organizations** | Morale, politics, capabilities, culture | Org chart, skill matrix, satisfaction, turnover |

#### Environment Interface

```python
class StrategicEnvironment:
    def reset(self, world_state: WorldState) -> Observation
    def step(self, action: Action) -> tuple[Observation, Reward, Done, Info]
    def get_verifiable_metrics(self) -> dict[str, float]
    def get_ground_truth(self) -> WorldState  # hidden from agent
    def simulate_forward(self, steps: int) -> list[WorldState]  # for evaluation
    def inject_event(self, event: Event) -> None  # for difficulty control
```

#### Realism Mechanisms

1. **Stochastic outcomes:** Actions succeed probabilistically based on difficulty, preparation, and context.
2. **Delayed feedback:** Consequences of decisions emerge over multiple timesteps.
3. **Hidden state:** Agents observe partial information. Ground truth maintained by environment.
4. **Multi-agent effects:** Other entities in the environment act according to their own objectives.
5. **Regime changes:** The rules of the environment can shift (market crash, regulation change, technology disruption).
6. **Resource consumption:** Every action costs something (time, money, attention, political capital).

#### Difficulty Curriculum

- **Level 1:** Single objective, full information, deterministic outcomes.
- **Level 2:** Single objective, partial information, stochastic outcomes.
- **Level 3:** Multiple objectives, partial information, competing agents.
- **Level 4:** Conflicting objectives, hidden information, adversarial agents, regime changes.
- **Level 5:** Open-ended strategic situations with no clear right answer, only better/worse tradeoffs.

---

### 3.5 OUTCOME EVALUATOR

**Purpose:** Produce verifiable scores for trajectories. This is the critical component that enables RL.

#### Evaluation Dimensions

```yaml
TrajectoryEvaluation:
  trajectory_id: uuid
  
  primary_outcome:
    objective_achieved: float[0,1]  # degree of success
    efficiency: float[0,1]  # resources consumed vs minimum possible
    time_to_outcome: float  # normalized
    
  secondary_outcomes:
    side_effects:
      - description: str
        valence: float[-1,1]  # positive or negative
        magnitude: float[0,1]
        was_anticipated: bool
        
    risks_realized:
      - risk_id: uuid
        was_mitigated: bool
        damage_if_realized: float[0,1]
        
    opportunities_captured:
      - description: str
        value: float[0,1]
        was_created_by_agent: bool
        
  process_quality:
    information_utilization: float[0,1]  # did agent use available information?
    reasoning_soundness: float[0,1]  # were conclusions justified by evidence?
    calibration: float[0,1]  # were confidence levels appropriate?
    adaptability: float[0,1]  # did agent adjust when conditions changed?
    
  counterfactual_analysis:
    best_possible_outcome: float  # from oracle with full information
    worst_possible_outcome: float  # from adversarial policy
    agent_outcome_percentile: float[0,1]  # where agent landed
    
  composite_score: float  # weighted combination, primary training signal
```

#### Verification Methods

| Method | Applicable When | Strength |
|---|---|---|
| **Simulation rollout** | Outcomes can be computed by running the environment forward | Ground truth, no ambiguity |
| **Rule-based checking** | Success criteria are formal (e.g., profit > threshold) | Fast, reliable |
| **Hindsight evaluation** | After trajectory completes, compare to known-good strategies | Objective but requires reference solutions |
| **Monte Carlo comparison** | Sample many alternative action sequences, compare outcomes | Statistical, handles stochasticity |
| **Critic model** | Process quality assessment, reasoning soundness | Scales to subjective dimensions |
| **Ensemble agreement** | Multiple evaluators must agree on score | Reduces evaluator bias |

#### Reward Signal Design

```
R_total = w1 * R_outcome + w2 * R_process + w3 * R_efficiency - w4 * R_risk_penalty

Where:
  R_outcome:     Verifiable from environment terminal state
  R_process:     Assessed by critic model (calibrated against verifiable tasks)
  R_efficiency:  Token budget utilization and action economy
  R_risk_penalty: Unrealized risks created by agent's actions
```

---

### 3.6 TRAJECTORY STORE

**Purpose:** Persistent storage of all generated trajectories with full metadata for training data curation.

#### Storage Schema

```yaml
Trajectory:
  id: uuid
  created_at: ISO-8601
  version: int
  
  context:
    initial_state: WorldState
    scenario: ScenarioBundle
    agent_config: Agent
    environment_config: EnvironmentConfig
    difficulty_level: int
    domain: str
    
  sequence:
    - turn: int
      observation: Observation
      reasoning: str  # agent's internal reasoning (chain of thought)
      action: Action
      outcome: StepOutcome
      
  evaluation: TrajectoryEvaluation
  
  metadata:
    tokens_consumed: int
    wall_time: duration
    was_truncated: bool
    generation_model: str
    filter_status: enum[accepted, rejected, borderline]
    rejection_reason: Optional[str]
    
  training_signals:
    sft_quality: float  # usable for supervised fine-tuning?
    rl_quality: float   # usable for RL?
    critique_quality: float  # usable for critic training?
    diversity_contribution: float  # how different from existing data?
```

#### Curation Policies

1. **Accept** trajectories with composite_score > 0.7 for SFT.
2. **Accept** trajectories with composite_score in [0.3, 0.9] for RL (need variance).
3. **Accept** failed trajectories with clear diagnostic value for critique training.
4. **Reject** degenerate trajectories (repetitive actions, trivial scenarios, inconsistent reasoning).
5. **Deduplicate** by semantic similarity of trajectory embeddings.
6. **Balance** across domains, difficulty levels, and agent roles.

#### Scale Targets

| Phase | Trajectories | Tokens | Domains |
|---|---|---|---|
| MVP | 100K | ~5B | 3 |
| Phase 1 | 1M | ~50B | 7 |
| Phase 2 | 10M | ~500B | 15+ |
| Phase 3 | 100M | ~5T | open-ended |

---

## PART 4: MODEL ARCHITECTURE

### Derivation from Training Objective

The training objective is NOT next-token prediction of text.

The training objective IS:

```
Given: World State + Objective + Constraints + Information
Produce: Analysis + Simulation + Plan + Prediction + Execution Steps
Evaluated by: Verifiable Outcomes in Simulated Environments
```

This objective naturally implies:

1. **Multiple cognitive modes** (analysis vs planning vs simulation) → distinct expert groups
2. **State maintenance across long horizons** → explicit memory architecture
3. **Heterogeneous token types** → specialized token vocabularies
4. **Variable compute per subtask** → adaptive routing

### Token Taxonomy

SHUTEN-DŌJI operates with typed tokens that signal cognitive mode:

| Token Type | Purpose | Example |
|---|---|---|
| `<state>` | Encode world-state representations | Entity attributes, relationships |
| `<analysis>` | Decomposition and pattern recognition | "Revenue declining because..." |
| `<simulation>` | Forward modeling and counterfactuals | "If action X, then likely Y..." |
| `<planning>` | Action sequence generation | "Step 1: ... Step 2: ..." |
| `<prediction>` | Probability estimation | "P(success) = 0.7 because..." |
| `<memory>` | Retrieve/store long-term context | Cross-turn state maintenance |
| `<tool>` | External system interaction | API calls, data retrieval |
| `<role>` | Agent identity and constraints | "As analyst, I focus on..." |
| `<critique>` | Self-evaluation and correction | "This plan fails because..." |

### MoE Architecture: Yes, With Cognitive Specialization

**Should SHUTEN-DŌJI use MoE?** Yes, decisively.

**Rationale:**

Strategic intelligence requires fundamentally different computations for different subtasks:
- Analysis requires pattern matching across large context
- Simulation requires sequential forward modeling
- Planning requires search over action spaces
- Prediction requires calibrated probability estimation
- Critique requires adversarial reasoning

A dense model wastes capacity routing everything through the same parameters. MoE allows **cognitive specialization** — different experts handle different thinking modes.

### Proposed Architecture

```
SHUTEN-DŌJI Model Specifications
═══════════════════════════════════

Total Parameters:        132B
Active Parameters:       14B
Expert Count:            128 (routed) + 2 (shared)
Experts Active/Token:    6
Sparsity Ratio:          ~21x
Layers:                  48
Hidden Dimension:        5120
Expert Hidden Dimension: 2048
Attention Heads:         48
KV Heads:                8 (GQA)
Context Length:          128K (base), 512K (extended via YaRN)
Memory Slots:            256 persistent slots per sequence
Vocabulary:              128K tokens (including special typed tokens)
```

### Rationale for Each Choice

**Total 132B / Active 14B:**
- Sweet spot for single-node inference on 4×H100 or 2×H200.
- Sufficient capacity for strategic reasoning without requiring multi-node inference.
- Active params comparable to Qwen2.5-14B class — proven sufficient for complex reasoning when well-trained.
- 10x cheaper to iterate than 1T models. We are building the factory, not the monument.

**128 experts, 6 active:**
- Sparsity ~21x provides extreme parameter efficiency.
- 6 active experts gives enough combinatorial coverage for diverse cognitive modes.
- Fewer than Kimi K2's 384 because we're smaller scale and want faster iteration.

**Routing Strategy: Typed Token Routing**

```
Standard MoE:  router(hidden_state) → top-k experts
SHUTEN-DŌJI:   router(hidden_state, token_type) → top-k experts

The token type biases the routing toward expert groups specialized for that cognitive mode.
Expert groups are not hard-partitioned — an analysis token CAN route to a planning expert
if the router learns that's useful — but initialization and auxiliary loss encourage specialization.
```

**Expert Groups (soft, not hard):**

| Group | Experts | Specialization |
|---|---|---|
| State Encoding | 1-20 | World state compression and representation |
| Analysis | 21-45 | Pattern recognition, decomposition, comparison |
| Simulation | 46-70 | Forward modeling, causal reasoning, counterfactuals |
| Planning | 71-95 | Action generation, sequencing, optimization |
| Evaluation | 96-115 | Scoring, calibration, critique |
| General | 116-128 | Overflow, cross-domain, novel combinations |
| Shared | S1, S2 | Always active, provide baseline capabilities |

**Context Length 128K:**
- Strategic scenarios involve long state descriptions + multi-turn trajectories.
- 128K handles most scenarios without truncation.
- YaRN extension to 512K for extreme cases (full project histories, organizational analyses).

**Memory Architecture:**

```
256 persistent memory slots:
  - Written to by <memory> tokens during generation
  - Read from at each layer via cross-attention
  - Persist across turns in multi-turn interactions
  - Enable "working memory" for complex multi-step reasoning
  - Gated write: model decides what to store
  - Decaying read: older memories accessed with lower attention weight
```

This is NOT retrieval augmentation. This is **learned working memory** — the model decides what to write and when to read, trained end-to-end.

**Attention: GQA with 48 heads, 8 KV heads:**
- 6:1 ratio gives good quality/inference tradeoff.
- Cheaper than MLA (simpler implementation, no latent attention complexity).
- Sufficient for 128K context with RoPE.

---

## PART 5: TRAINING PIPELINE

### Phase 0: Synthetic Factory Bootstrap

**Objective:** Build and validate the data generation infrastructure. No model training yet.

**Activities:**
- Implement State Generator with 3 initial domains (business, logistics, markets)
- Implement Scenario Generator with rule-based + LLM-assisted branching
- Implement Environment Simulator for the 3 domains
- Implement Outcome Evaluator with simulation-based verification
- Generate 10K test trajectories using an existing open-source model (Qwen3-32B or similar) as the agent
- Validate trajectory quality manually on 500 samples
- Establish quality baselines and rejection criteria

**Datasets produced:** 10K validated trajectories (for sanity checking only)

**Compute:** 128 H100 GPU-hours for trajectory generation + 32 H100 GPU-hours for evaluation

**Expected outcome:** Working pipeline that can generate trajectories at scale. No model improvement yet.

**Duration:** 4-6 weeks

---

### Phase 1: Supervised Trajectory Learning

**Objective:** Train the base model to produce well-formed strategic reasoning by imitating high-quality trajectories.

**Activities:**
- Pre-train 132B MoE from scratch on 2T tokens of general text (code, math, knowledge, reasoning)
  - OR: Start from open-source base (Qwen3-32B, Llama-4-Scout) and expand to MoE via upcycling
- Generate 1M high-quality trajectories across all domains using the factory
- Filter to top 30% by composite score (300K trajectories, ~15B tokens)
- Fine-tune on trajectory data with typed token formatting
- Multi-task: mix trajectory SFT with general instruction following

**Datasets:**
- 2T tokens general pre-training (if from scratch)
- 15B tokens strategic trajectories (SFT)
- 5B tokens general instruction data (knowledge retention)

**Compute:**
- Pre-training from scratch: ~50K H100 GPU-hours (2T tokens at ~40K tokens/sec on 256 GPUs)
- SFT: ~5K H100 GPU-hours
- If starting from open-source: skip pre-training, spend 10K H100 GPU-hours on MoE upcycling + SFT

**Expected capability gain:**
- Model produces well-structured analyses, plans, and scenarios
- Format compliance > 95%
- Reasoning quality comparable to teacher model on in-distribution tasks
- No generalization beyond training distribution yet

**Duration:** 8-12 weeks (from scratch) or 4-6 weeks (from open-source base)

---

### Phase 2: Tool-Use Learning

**Objective:** Teach the model to interact with external systems — data retrieval, computation, environment interaction.

**Activities:**
- Define tool interface for strategic environments (query databases, run simulations, check constraints)
- Generate 500K tool-use trajectories using the factory
- Tools include: data lookup, calculation, web search simulation, document retrieval, simulation execution
- Train on tool-use demonstrations with format: `<tool>call</tool>` → result → continue reasoning
- Validate on held-out tool configurations (novel tools not seen in training)

**Datasets:**
- 500K tool-use trajectories (~25B tokens)
- 5K distinct tool specifications (real + synthetic)
- Multi-turn: average 8 tool calls per trajectory

**Compute:** ~8K H100 GPU-hours (generation + training)

**Expected capability gain:**
- Model correctly invokes tools > 90% of the time
- Novel tool generalization > 70% (tools not seen in training)
- Meaningful improvement in analysis depth (can now "look things up")
- Reduction in hallucinated facts

**Duration:** 3-4 weeks

---

### Phase 3: Scenario Simulation Learning

**Objective:** Train the model to generate and evaluate multiple future scenarios, with calibrated uncertainty.

**Activities:**
- Generate 2M scenario bundles (state → multiple futures)
- For each bundle, simulate all scenarios forward and compute ground-truth outcomes
- Train the model to:
  1. Generate plausible scenarios given a state
  2. Estimate probabilities for each scenario
  3. Identify critical uncertainties
  4. Recommend information-gathering actions to reduce uncertainty
- Evaluate calibration: predicted probabilities vs actual frequencies

**Datasets:**
- 2M scenario bundles × 3-5 scenarios each = 6-10M scenario paths
- ~100B tokens of simulation data
- Calibration labels from Monte Carlo rollouts

**Compute:** ~20K H100 GPU-hours (heavy simulation + training)

**Expected capability gain:**
- Model generates diverse, plausible futures (diversity score > 0.7)
- Probability calibration within 15% (Brier score improvement)
- Uncertainty decomposition accuracy > 60%
- Can identify information that would most reduce uncertainty

**Duration:** 6-8 weeks

---

### Phase 4: RLVR (Reinforcement Learning with Verifiable Rewards)

**Objective:** Improve planning and decision-making quality through outcome-based reinforcement.

**Activities:**
- Deploy model as agent in simulated environments
- Model makes decisions, environment returns outcomes
- Score trajectories using Outcome Evaluator (ground truth from simulation)
- Apply RL (policy optimization variant similar to K2's approach):
  - Sample K responses per prompt
  - Compute rewards from environment outcomes
  - Optimize toward higher-reward trajectories
- Verifiable domains:
  - Planning tasks with measurable success (resource allocation, scheduling)
  - Prediction tasks with known outcomes (historical scenarios replayed)
  - Analysis tasks with verifiable conclusions (can be checked against ground truth)

**Datasets:**
- Online generation during RL: ~50B tokens of rollouts
- Prompt set: 100K diverse strategic scenarios with verifiable outcomes
- Moderate difficulty selection: 20-80% pass rate on current policy

**Compute:** ~30K H100 GPU-hours (inference-heavy RL loop)

**Expected capability gain:**
- Planning success rate +15-25% over SFT baseline
- Prediction accuracy +10-15%
- Reduction in degenerate strategies (repetitive, overly conservative)
- Emergence of multi-step planning behaviors not present in SFT data
- Token efficiency improvement (concise but effective reasoning)

**Duration:** 8-12 weeks

---

### Phase 5: Self-Critique Reward

**Objective:** Extend RL to non-verifiable domains using self-evaluation grounded in verifiable signal.

**Activities:**
- Train a critic capability within the model:
  1. On verifiable tasks: critic learns to predict outcome evaluator scores
  2. On non-verifiable tasks: critic applies learned judgment
- Self-critique rubric design:
  - Strategic soundness (does the plan account for key risks?)
  - Information completeness (has the analysis considered all available evidence?)
  - Calibration quality (are confidence levels justified?)
  - Adversarial robustness (does the plan survive hostile conditions?)
  - Creativity (does the approach find non-obvious solutions?)
- Closed-loop: RL on verifiable tasks continuously calibrates the critic
- Apply critic-scored RL on:
  - Open-ended strategic analysis
  - Creative problem solving
  - Negotiation scenarios
  - Organizational design

**Datasets:**
- Mix of verifiable (60%) and non-verifiable (40%) tasks during RL
- Critic training data: 1M trajectory pairs with preference labels
- ~30B tokens of self-play and critique

**Compute:** ~25K H100 GPU-hours

**Expected capability gain:**
- Quality improvement on open-ended tasks +20-30% (by human evaluation)
- Critic accuracy > 75% agreement with human judges
- Reduced reward hacking (critic detects degenerate strategies)
- Generalization to novel strategic domains not in training

**Duration:** 6-8 weeks

---

### Phase 6: Continuous Self-Improvement Loop

**Objective:** Establish a perpetual improvement cycle where the model generates its own training data at increasing quality.

**Activities:**
- Model acts as agent in new environments → generates trajectories
- Outcome evaluator scores trajectories → training signal
- Best trajectories become new SFT data → model improves
- Improved model generates better trajectories → positive feedback loop
- Critic improves alongside → catches new failure modes
- New domains and environments added continuously
- Difficulty curriculum auto-advances as model capability grows

**Architecture:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Current    │────▶│  Environment │────▶│  Evaluator   │
│    Model     │     │  Rollouts    │     │   Scoring    │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                        │
       │                                        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Updated    │◀────│   Training   │◀────│   Filtered   │
│    Model     │     │    Loop      │     │    Data      │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Safeguards against collapse:**
- Diversity enforcement: reject trajectories too similar to recent training data
- Out-of-distribution testing: always evaluate on held-out domains
- Human spot-checking: random sample reviewed by domain experts monthly
- Capability regression tests: fixed benchmarks that must not degrade
- Critic calibration monitoring: track critic-vs-verifiable agreement over time

**Compute:** Ongoing, ~10K H100 GPU-hours/month

**Expected outcome:** Continuous capability improvement at decreasing marginal cost per capability unit.

---

## PART 6: BRUTAL CRITIQUE

### What Is Unrealistic

1. **The "world model" claim is mostly fake.** A language model simulating a business environment is not a world model — it's a language model generating plausible-sounding text about business environments. It has no grounded understanding of physics, economics, or human behavior. It pattern-matches from training data. Call it what it is: structured hallucination with quality filtering.

2. **Calibrated uncertainty from neural networks is extremely hard.** Claiming the model will produce calibrated probability estimates is aspirational. Every attempt at neural network calibration shows overconfidence or underconfidence that's domain-dependent and fragile. Temperature scaling helps. Proper scoring rules help. But "calibrated strategic forecasting" is a multi-decade research problem, not a training phase.

3. **The 132B parameter budget may be too small.** Strategic reasoning requires holding many factors in mind simultaneously. 14B active parameters may not be enough for complex multi-entity, multi-objective scenarios. Kimi K2 uses 32B active for simpler tasks (coding, tool use). Strategic reasoning arguably requires MORE working memory, not less.

4. **MoE upcycling from dense models often underperforms.** Training MoE from scratch vs upcycling is a real engineering tradeoff. Upcycling saves compute but the expert specialization is often weaker. If you want true cognitive specialization, you may need from-scratch training — which is expensive.

### What Is Likely to Fail

1. **Synthetic environment fidelity.** The biggest risk. If your simulated business environments don't capture real dynamics, the model learns to exploit simulator artifacts, not real strategic principles. This is the "sim-to-real gap" problem from robotics, applied to cognition. **Mitigation:** Continuous comparison with historical real-world outcomes. But this limits you to domains with available historical data.

2. **Reward hacking in RL.** The model WILL find ways to achieve high reward scores without actually being strategically intelligent. Examples:
   - Generating plans that look good by the metrics but are nonsensical
   - Exploiting regularities in the environment simulator
   - Producing verbose reasoning that scores well on "process quality" while being vacuous
   - Gaming the critic by producing outputs that match the critic's biases
   
   **Mitigation:** Diverse evaluation, adversarial testing, human spot-checks. But this is an arms race.

3. **Trajectory quality collapse at scale.** As you generate millions of trajectories, maintaining quality becomes exponentially harder. The generator model's weaknesses propagate. If the teacher model (generating trajectories) has systematic blind spots, the student inherits them. This is the "distillation ceiling" — you cannot exceed your teacher's quality through SFT alone.

4. **The critic problem.** Self-critique is only useful if the critic is better calibrated than the actor. But they're the same model (or close). The danger: the critic learns to approve the actor's failures, creating a mutual delusion. **Mitigation:** Ground critic in verifiable signal. But verifiable domains may not transfer to non-verifiable ones.

### Where Synthetic Data Can Collapse

1. **Mode collapse in state generation.** If the State Generator converges to a small set of archetypes (the "generic tech startup", the "supply chain disruption"), all downstream diversity is an illusion. Different names for the same structure.

2. **Difficulty miscalibration.** If scenarios are too easy, the model learns nothing. If too hard, reward signal is too sparse. The sweet spot shifts as the model improves, requiring constant recalibration.

3. **Causal chain fabrication.** LLM-generated causal chains often violate actual causal structure. "A caused B" in the synthetic data may reflect linguistic patterns, not genuine causality. The model learns convincing-sounding causality, not actual causality.

4. **Outcome evaluator gaming.** If the evaluator can be predicted, the model optimizes for the evaluator, not for the task. This is Goodhart's Law applied to training data.

### Where World-Model Claims Become Fake

1. **Every time the model extrapolates beyond training distribution.** If no business in the training data faced X situation, the model's "simulation" of X is confabulation dressed up as forecasting.

2. **Every time counterfactuals are generated.** "What would have happened if..." requires genuine causal understanding. The model does correlation-based counterfactuals, which are often wrong.

3. **Every time long chains are involved.** Errors compound exponentially. A 10-step causal chain with 90% accuracy per step is only 35% accurate overall. Claim careful about multi-step predictions.

4. **Every time domain transfer is assumed.** "Good at simulating business environments" does not imply "good at simulating military operations" even if the structures look similar at an abstract level.

### Where RL Can Overfit

1. **To the environment distribution.** RL optimizes for the environments it trains in. Novel environments not represented in training will see degraded performance. This is the core generalization problem of RL.

2. **To the reward function.** If the reward function has any systematic bias, RL will exploit it. For example: if the evaluator slightly favors verbose explanations, the model will learn to be verbose regardless of quality.

3. **To the prompt distribution.** RL is extremely sensitive to prompt formatting. A model that performs well on factory-generated prompts may fail on real user prompts with different structure.

4. **To the exploration policy.** If early RL samples a limited strategy space, the model can converge to local optima and never discover better strategies. The temperature decay from Kimi K2 helps, but doesn't solve this fundamentally.

### Where Simulation Becomes Hallucination

**This is the central existential risk of the project.**

The boundary is: if you cannot verify the output against a ground truth, you cannot distinguish simulation from hallucination. The model itself cannot tell the difference — it's just generating tokens.

**The only honest framing:**
- In domains with verifiable outcomes (backtesting against historical data, formal specifications, mathematical proofs): the model's output is testable.
- In truly open domains (novel geopolitical situations, unprecedented market conditions, creative strategy): the model's output is **plausible confabulation**, not simulation.

**What this means for deployment:**
- SHUTEN-DŌJI should be marketed as "structured reasoning assistance" not "strategic simulation" in open domains.
- Every output should carry an epistemic confidence marker indicating how close to verifiable ground truth the reasoning is.
- Users must understand: the model generates hypotheses, not predictions.

---

## FINAL DELIVERABLE

### 1. Intelligence Factory Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                   SHUTEN-DŌJI INTELLIGENCE FACTORY                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT LAYER                                                         │
│  ├── Domain Specifications (parameterized world types)              │
│  ├── Objective Functions (what "good" means per domain)             │
│  ├── Real-World Anchors (historical data for calibration)           │
│  └── Difficulty Curriculum (auto-adjusting complexity)               │
│                                                                      │
│  GENERATION LAYER                                                    │
│  ├── State Generator (world states from domain specs)               │
│  ├── Scenario Generator (future branches from states)               │
│  ├── Agent Simulator (cognitive agents from role specs)              │
│  └── Environment Simulator (dynamic response systems)               │
│                                                                      │
│  EVALUATION LAYER                                                    │
│  ├── Outcome Evaluator (verifiable scoring)                         │
│  ├── Critic Model (process quality assessment)                      │
│  ├── Calibration Monitor (probability accuracy tracking)            │
│  └── Diversity Auditor (prevent mode collapse)                      │
│                                                                      │
│  TRAINING LAYER                                                      │
│  ├── SFT Pipeline (trajectory imitation)                            │
│  ├── RL Pipeline (outcome optimization)                             │
│  ├── Critic Training (self-evaluation improvement)                  │
│  └── Continuous Loop (self-improvement cycle)                       │
│                                                                      │
│  OUTPUT LAYER                                                        │
│  ├── Model Checkpoints (versioned, specialized)                     │
│  ├── Trajectory Datasets (curated, scored)                          │
│  ├── Evaluation Benchmarks (domain-specific)                        │
│  └── Quality Reports (automated + human audit)                      │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

### 2. Data Generation Architecture

See Part 3 above. Summary:

- **Input:** Domain specs + complexity parameters + seed states from real data
- **Pipeline:** State → Scenario → Agent Interaction → Environment Response → Outcome
- **Output:** Scored trajectories with full metadata
- **Scale:** 100K (MVP) → 100M (frontier) trajectories
- **Quality:** Multi-stage filtering with diversity enforcement

### 3. Model Architecture

See Part 4 above. Summary:

- **Type:** Mixture-of-Experts Transformer
- **Scale:** 132B total / 14B active / 128 experts / 6 active per token
- **Key innovations:** Typed token routing, persistent memory slots, cognitive expert groups
- **Context:** 128K base, 512K extended
- **Attention:** GQA 48 heads / 8 KV heads
- **Special tokens:** state, analysis, simulation, planning, prediction, memory, tool, role, critique

### 4. Training Architecture

See Part 5 above. Summary:

| Phase | Objective | Compute (H100-hrs) | Duration |
|---|---|---|---|
| 0 | Factory validation | 160 | 4-6 weeks |
| 1 | Trajectory SFT | 5K-50K | 4-12 weeks |
| 2 | Tool-use learning | 8K | 3-4 weeks |
| 3 | Simulation learning | 20K | 6-8 weeks |
| 4 | RLVR | 30K | 8-12 weeks |
| 5 | Self-critique | 25K | 6-8 weeks |
| 6 | Continuous loop | 10K/month | Ongoing |

### 5. Evaluation Architecture

```
EVALUATION FRAMEWORK
════════════════════

Tier 1: Automated Verifiable (run continuously)
  ├── Planning success rate on held-out scenarios
  ├── Prediction calibration (Brier score)
  ├── Analysis completeness (rubric-based)
  ├── Tool-use accuracy
  └── Format compliance

Tier 2: Automated Non-Verifiable (run weekly)
  ├── Critic-scored reasoning quality
  ├── Diversity of generated strategies
  ├── Robustness to perturbations
  └── Cross-domain transfer

Tier 3: Human Evaluation (run monthly)
  ├── Expert domain review (100 trajectories/domain)
  ├── Side-by-side comparison with baselines
  ├── Novel scenario stress testing
  └── Adversarial red-teaming

Tier 4: Real-World Anchoring (run quarterly)
  ├── Backtest against historical events
  ├── Comparison with published expert analyses
  ├── Deployment feedback from pilot users
  └── Failure mode documentation
```

### 6. Compute Scaling Roadmap

| Milestone | Total Compute | Hardware | Timeline |
|---|---|---|---|
| Factory MVP | 500 H100-hrs | 8 H100s × 3 days | Month 1-2 |
| Phase 1 complete | 55K H100-hrs | 64 H100s × 5 weeks | Month 2-5 |
| Phase 2-3 complete | 83K H100-hrs | 128 H100s × 4 weeks | Month 5-8 |
| Phase 4-5 complete | 138K H100-hrs | 256 H100s × 3 weeks | Month 8-12 |
| Continuous ops | 10K H100-hrs/month | 64 H100s continuous | Month 12+ |
| **TOTAL YEAR 1** | **~250K H100-hrs** | | |

**Cost estimate (at $2.50/H100-hr spot):** ~$625K for Year 1.

**Comparison:** Kimi K2 used thousands of H800s for weeks. SHUTEN-DŌJI is ~50-100x cheaper by targeting smaller active params and focusing compute on the factory rather than raw pre-training.

### 7. MVP Version

**SHUTEN-DŌJI MVP: "Strategic Analyst"**

**Scope:**
- 3 domains only: business strategy, project management, market analysis
- 14B active parameter MoE (fine-tuned from open-source Qwen3-32B via expert splitting)
- 100K trajectories in training
- Basic tool use (data lookup, calculation)
- Scenario generation (3 scenarios per state)
- No continuous self-improvement loop yet

**What it can do:**
- Given a business situation description, produce structured analysis
- Generate 3 plausible future scenarios with rough probabilities
- Create action plans with success criteria
- Identify top risks and mitigation strategies
- Use tools to look up reference data

**What it cannot do:**
- Novel domains it hasn't been trained on
- Calibrated predictions (probabilities will be rough)
- Multi-agent strategic interaction
- Long-horizon (>20 step) planning
- Real-time environment interaction

**Timeline:** 4-5 months from project start
**Compute:** ~10K H100-hours total
**Cost:** ~$25K

### 8. Long-Term Frontier Version

**SHUTEN-DŌJI Frontier: "Strategic Intelligence Engine"**

**Target (18-24 months):**
- 500B total / 32B active parameters
- 512 experts with learned cognitive specialization
- 512K context with persistent memory
- 100+ domains with continuous expansion
- 100M+ training trajectories
- Self-improving at ~5% capability gain per month
- Calibrated forecasting within 10% on trained domains
- Multi-agent strategic simulation
- Real-time tool integration with live data sources
- Deployment as both standalone system and API service

**Architecture evolution:**
- Phase 1 model: 132B/14B (MVP)
- Phase 2 model: 250B/20B (expanded after factory produces sufficient data)
- Phase 3 model: 500B/32B (frontier, once continuous loop is stable)

**Key capability targets:**
- Beat human analyst teams on structured strategic tasks (where verification is possible)
- Produce scenario analyses comparable to top consulting firms on historical cases
- Demonstrate genuine generalization to novel domains via transfer learning
- Maintain calibration across domains (measured, not claimed)

### 9. Brutal Risk Review

| Risk | Severity | Likelihood | Mitigation | Residual Risk |
|---|---|---|---|---|
| Synthetic data mode collapse | Critical | High | Diversity metrics, seed from real data, human audit | Medium |
| Sim-to-real gap in environments | Critical | High | Historical backtesting, progressive realism | High |
| Reward hacking in RL | High | Very High | Multi-evaluator, adversarial testing, human spot-check | Medium |
| Critic learns to approve garbage | Critical | Medium | Ground in verifiable signal, monitor calibration drift | Medium |
| Compute budget insufficient | High | Medium | Start from open-source, aggressive efficiency | Low |
| Causal reasoning is fake | High | Very High | Never claim causal; frame as "plausible reasoning" | High (inherent) |
| Probability calibration is poor | Medium | High | Proper scoring rules, calibration training, honest reporting | Medium |
| Expert specialization doesn't emerge | Medium | Medium | Pre-train from scratch if upcycling fails | Low |
| Hallucination in novel domains | Critical | Very High | Epistemic confidence markers, refuse to answer out-of-domain | High (inherent) |
| Model exploits evaluator weaknesses | High | Very High | Rotating evaluators, evaluator ensemble, evaluator improvement | Medium |

**The two risks you cannot mitigate:**
1. **Causal reasoning is fake.** The model does statistical pattern matching, not causal inference. No amount of training changes this fundamental limitation. You can only be honest about it.
2. **Novel domain hallucination.** When the model encounters truly novel situations, its outputs are confabulation. The only defense is knowing when this is happening (which is itself hard to verify).

**The honest assessment:**

SHUTEN-DŌJI will be very good at:
- Structured reasoning in domains with sufficient training data
- Generating diverse hypotheses for human consideration
- Identifying risks and considerations a human might miss
- Maintaining consistency across long reasoning chains
- Producing well-formatted, actionable strategic documents

SHUTEN-DŌJI will be unreliable at:
- Predicting the actual future
- Providing calibrated probabilities for unprecedented events
- Reasoning causally about complex systems
- Performing better than domain experts in their own domains
- Generating truly novel strategic insights (vs recombining known patterns)

**Build the factory anyway.** The output is still enormously valuable — structured reasoning assistance that saves analysts hundreds of hours per project. Just don't call it intelligence. Call it what it is: an extremely sophisticated reasoning scaffold.

---

## APPENDIX: IMPLEMENTATION PRIORITY

```
WEEK 1-2:   State Generator prototype (3 domains)
WEEK 3-4:   Environment Simulator prototype (1 domain deep)
WEEK 5-6:   Trajectory generation with existing model (Qwen3-32B)
WEEK 7-8:   Outcome Evaluator implementation + validation
WEEK 9-12:  Scale trajectory generation to 100K
WEEK 13-16: MoE model training (Phase 1 SFT)
WEEK 17-20: Tool-use + scenario learning (Phase 2-3)
WEEK 21-28: RL pipeline (Phase 4-5)
WEEK 29+:   Continuous improvement loop

First usable checkpoint: Week 16
First competitive checkpoint: Week 28
```

---

*Document version: 0.1*
*Status: Architecture proposal — not implementation*
*Next step: Implement State Generator prototype for business domain*
