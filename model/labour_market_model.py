"""
Labour Market ABM with AI automation - model definition.

Based on:
  - Task-based production
  - Productivity functions inspired by Upreti & Sridhar
  - Kaleckian two-class demand feedback loop
  - Leontief technology at the firm level

Import this module and instantiate LabourMarketModel to run simulations.

=============================================================================
COMPONENT MAP  (see Chapter 4 - Implementation)
=============================================================================
The conceptual model consists of seven components (C1-C7) plus one
macro-feedback wiring layer (M1). Every section banner and method below is
tagged with its component ID, so the source can be read component by
component. Components do not map one-to-one onto files or classes: in an
agent-based model the behaviour of a component is distributed over the agent
class(es) that carry it. The map below lists, per component, where its code
lives.

  C1  Production            Leontief task technology and productivity.
                            leontief_output(), leontief_bottlenecks(), Task,
                            Producer: task construction, productivity_human(),
                            productivity_ai(), produce(), trim_ai_capacity().
  C2  Labour Market         Hiring, matching, separations, vacancy handling.
                            Worker; Producer.on_worker_separated(),
                            hiring_requests(), should_produce_more();
                            LabourMarketModel.separate_workers(),
                            match_and_hire(), hire_worker_to_task(),
                            _warm_start().
  C3  Wage-setting          Wage-curve supply with partial adjustment.
                            LabourMarketModel.update_wages(),
                            calculate_wage_bill().
  C4  AI Adoption           ULC and NPV automation decision (four modes).
                            Producer.unit_cost_human(), unit_cost_ai(),
                            preferred_input_for_task(), calculate_npv(),
                            should_automate(), record_automation_decision().
  C5  Learning Curve        AI rental cost k_ai(t), a closed-form decay.
                            LabourMarketModel.ai_cost_at().
  C6  Demand                Goods market and Kaleckian two-class consumption.
                            LabourMarketModel.calculate_profit_income(),
                            calculate_market_price().
  C7  Employment Protection Contracts, severance, ketenregeling (chain rule).
                            Producer.severance_cost_for_task();
                            LabourMarketModel.process_chain_limit().
  M1  Macro-feedback        Step ordering that wires C1-C7 into one loop.
                            LabourMarketModel.step() and per-step data
                            collection (_compute_all_stats()).
=============================================================================
"""
from __future__ import annotations

from numbers import Number
from pathlib import Path
import random as _stdlib_random
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

# Path to the Gmyrek 4-digit automation score dataset (relative to this file)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TASKS_BY_4DIGIT = _DATA_DIR / "tasks_by_4digit.xlsx"


# =============================================================================
# [C1 Production] Producer heterogeneity - routine-task share (alpha) sampling
# =============================================================================
def _load_alpha_population() -> np.ndarray:
    """Load mean automation scores from tasks_by_4digit.xlsx as a numpy array."""
    df = pd.read_excel(_TASKS_BY_4DIGIT, usecols=["mean_automation_score"])
    return df["mean_automation_score"].dropna().to_numpy(dtype=float)


def sample_alphas(
    n: int,
    alpha_source: str = "uniform",
    alpha_min: float = 0.2,
    alpha_max: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw n alpha values for Producer initialisation.

    alpha_source = "uniform"  → draw from U(alpha_min, alpha_max)
    alpha_source = "data"     → filter the Gmyrek 4-digit mean_automation_score
                                distribution to [alpha_min, alpha_max] and sample
                                n values from that filtered set with replacement.
                                Falls back to uniform if no values lie in the range.
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    if alpha_min > alpha_max:
        alpha_min, alpha_max = alpha_max, alpha_min

    if alpha_source == "uniform":
        return rng.uniform(alpha_min, alpha_max, size=n)

    if alpha_source == "data":
        population = _load_alpha_population()
        # Filter to values within the requested range
        mask = (population >= alpha_min) & (population <= alpha_max)
        filtered = population[mask]
        if len(filtered) == 0:
            # Fallback: no data in range → use uniform
            return rng.uniform(alpha_min, alpha_max, size=n)
        return rng.choice(filtered, size=n, replace=True)

    raise ValueError(f"alpha_source must be 'uniform' or 'data', got '{alpha_source}'")


# =============================================================================
# [C1 Production] Leontief production function
# =============================================================================

def leontief_output(u: Dict[int, Number], c: Dict[int, Number]) -> float:
    """
    Computes Leontief output:
        U = min_x ( u_x / c_x )

    where u_x is the quantity of task x produced and c_x is the
    required amount of task x per unit of final output (input coefficient).
    Output is limited by the scarcest task relative to its requirement.
    """
    if not u:
        return 0.0
    best = float("inf")
    for x, ux in u.items():
        cx = c.get(x, 1.0)
        if cx <= 0:
            raise ValueError(f"c[{x}] must be > 0")
        if ux < 0:
            raise ValueError(f"u[{x}] must be >= 0")
        best = min(best, ux / cx)
    return float(best)


def leontief_bottlenecks(
    u: Dict[int, Number], c: Dict[int, Number], eps: float = 1e-12
) -> Tuple[float, List[int]]:
    """
    Returns the Leontief output level and the list of bottleneck task IDs.
    Bottleneck tasks are those achieving the minimum u_x / c_x ratio.
    Increasing output requires expanding all bottleneck tasks simultaneously.
    """
    U = leontief_output(u, c)
    bottlenecks = [x for x, ux in u.items() if abs((ux / c.get(x, 1.0)) - U) <= eps]
    return U, bottlenecks


# =============================================================================
# [C1 Production] Task dataclass - the unit of production and of automation
# =============================================================================

@dataclass
class Task:
    """
    Represents a single production task within a Producer firm.

    task_type is either 'routine' or 'non_routine'. The firm's alpha parameter
    sets the SHARE of a firm's tasks that are routine (the first alpha*n_tasks
    tasks are routine; the rest are non-routine).
    complexity_index runs from 1 (simplest) to n (most complex) within each task type.
    A task can be performed by human workers (employees list) or by AI (n_ai units).
    When automated=True, the task is handled by AI and employees is empty.
    """
    task_id: int
    task_type: str                              # 'routine' or 'non_routine'
    producer: "Producer"
    complexity_index: int = 1                   # 1 = simple, n = complex (within task type)
    automated: bool = False                     # True if the task is performed by AI
    employees: List["Worker"] = field(default_factory=list)
    n_ai: int = 0                               # number of AI units assigned
    investment_cost: float = 0.0               # upfront AI investment cost for this task
    ai_investment_step: Optional[int] = None    # step at which AI was installed (None = never/ULC mode)
    ai_original_step: Optional[int] = None      # step of the FIRST automation decision (never overwritten on renewal)
    ai_investment_npv: Optional[float] = None   # NPV at the moment of investment (NPV modes only)
    _cached_npv: Optional[float] = None         # last NPV computed for this task (diagnostic cache)


# =============================================================================
# [C2 Labour Market] Worker agent
# =============================================================================

class Worker(Agent):
    """
    Represents a worker in the labour market.

    Workers are either 'high' or 'low' skilled. They are passive:
    all employment decisions (hiring, firing, automation) are made by Producers
    and wage/employment state is updated centrally by LabourMarketModel.
    """

    def __init__(self, unique_id: int, model: "LabourMarketModel", skill_level: str):
        super().__init__(unique_id, model)
        self.skill_level = skill_level  # 'high' or 'low'
        self.employed: bool = False
        self.employer: Optional["Producer"] = None
        self.task: Optional["Task"] = None
        self.wage: float = 0.0
        self.tenure: int = 0               # steps employed at current employer
        self.contract_type: str = "flex"   # "flex" or "vast" (permanent)


# =============================================================================
# Producer agent
# Carries the firm-side code of four components:
#   C1 Production         - task setup, productivity, produce()
#   C4 AI Adoption        - unit costs, NPV/ULC decision
#   C2 Labour Market      - separations and hiring (firm side)
#   C7 Employment Prot.   - severance cost of a task
# =============================================================================

class Producer(Agent):
    """
    Represents a firm that hires workers and/or AI to produce output.

    Each firm has n_tasks tasks, split into 'routine' and 'non_routine' according to alpha.
    Within each type, tasks are indexed by complexity (1 = simple, n = complex).
    Production follows a Leontief technology: output = min over tasks of (task output / coefficient).
    Productivity follows Upreti & Sridhar: exponential in complexity for humans,
    declining in complexity for AI - creating a natural gradient of automation thresholds.
    """

    def __init__(self, unique_id: int, model: "LabourMarketModel", alpha: float, n_tasks: int = 20):
        super().__init__(unique_id, model)

        self.employees: List["Worker"] = []
        self.alpha = alpha        # share of tasks that are routine (automation-prone)
        self.n_tasks = n_tasks

        # Create tasks with complexity indices within each type
        n_routine = round(alpha * n_tasks)
        n_nonroutine = n_tasks - n_routine

        self.tasks: List[Task] = (
            [Task(task_id=i, task_type="routine", producer=self,
                  complexity_index=i)                           # 1..n_routine
             for i in range(1, n_routine + 1)]
            + [Task(task_id=j, task_type="non_routine", producer=self,
                    complexity_index=j - n_routine)             # 1..n_nonroutine
               for j in range(n_routine + 1, n_tasks + 1)]
        )

        # Leontief input coefficients: c_x = 1.0 means one unit of each task per unit of output
        self.c: Dict[int, float] = {t.task_id: 1.0 for t in self.tasks}

        self.output = 0.0

        # Queue of pending hiring requests (filled by on_worker_separated and hiring_requests)
        self._pending_requests: List[tuple["Producer", Task, str]] = []

    # ----- [C1 Production] Convenience properties: task views & cost accounting -----

    @property
    def task_overview(self) -> "pd.DataFrame":
        """
        Returns a DataFrame with one row per task:
          task_type         - 'routine' or 'non_routine'
          complexity_index  - task complexity within its type (1 = simple)
          automated         - True if the task is currently run by AI
          workers_high      - number of high-skilled workers assigned
          workers_low       - number of low-skilled workers assigned
          n_ai              - number of AI units assigned
          productivity      - total task output
        Indexed by task_id.
        """
        rows = []
        for t in self.tasks:
            n_high = sum(1 for w in t.employees if w.skill_level == "high")
            n_low  = sum(1 for w in t.employees if w.skill_level == "low")
            if t.automated:
                total_prod = t.n_ai * self.productivity_ai(t)
            else:
                total_prod = sum(
                    self.productivity_human(t, w.skill_level)
                    for w in t.employees
                )
            rows.append({
                "task_id":          t.task_id,
                "task_type":        t.task_type,
                "complexity_index": t.complexity_index,
                "automated":        t.automated,
                "workers_high":     n_high,
                "workers_low":      n_low,
                "n_ai":             t.n_ai,
                "productivity":     round(total_prod, 3),
            })
        return pd.DataFrame(rows).set_index("task_id")

    @property
    def routine_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.task_type == "routine"]

    @property
    def nonroutine_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.task_type == "non_routine"]

    @property
    def n_automated(self) -> int:
        """Number of tasks currently automated by AI."""
        return sum(1 for t in self.tasks if t.automated)

    @property
    def n_routine(self) -> int:
        return len(self.routine_tasks)

    @property
    def n_nonroutine(self) -> int:
        return len(self.nonroutine_tasks)

    @property
    def ai_costs(self) -> float:
        """Total AI rental costs: sum of (n_ai * k_ai) over all automated tasks."""
        return sum(t.n_ai * self.model.k_ai for t in self.tasks if t.automated)

    @property
    def labour_costs(self) -> float:
        """Total wage bill: sum of wages of all workers on non-automated tasks."""
        total = 0.0
        for t in self.tasks:
            if not t.automated:
                for w in t.employees:
                    total += w.wage
        return total

    @property
    def total_costs(self) -> float:
        """Total production costs = labour costs + AI rental costs."""
        return self.labour_costs + self.ai_costs

    # ----- [C1 Production] Productivity functions (Upreti & Sridhar) -----

    def productivity_human(self, task: Task, skill_level: str) -> float:
        """
        Exponential productivity in task complexity (Upreti & Sridhar style).
        Routine tasks are skill-neutral in productivity; non-routine tasks
        favour high-skilled workers, while low-skilled non-routine assignments
        are dampened by xi (0 <= xi <= 1):

            routine,    low:   exp(a*x)
            routine,    high:  exp(a*x)
            non_routine, high: exp(a*x)
            non_routine, low:  exp(xi*a*x)

        This keeps routine productivity identical across skill groups while
        preserving a high-skill advantage on non-routine tasks (comparative advantage).
        """
        x  = task.complexity_index
        a  = self.model.a_prod
        xi = self.model.xi_prod
        if task.task_type == "routine":
            return np.exp(a * x)
        # non_routine
        return np.exp(a * x) if skill_level == "high" else np.exp(xi * a * x)

    def productivity_ai(self, task: Task) -> float:
        """
        AI productivity declines with task complexity - machines excel at simple, standardised tasks.
        A non-routine multiplier allows AI to be weaker or stronger on non-routine tasks.
        The combination creates a natural automation gradient: low-complexity routine tasks
        are automated first; high-complexity and non-routine tasks resist automation longest.
        """
        x    = task.complexity_index
        base = self.model.phi_base * np.exp(-self.model.phi_decay * x)
        return base if task.task_type == "routine" else base * self.model.nr_multiplier

    # ----- [C1 Production] Leontief output & AI capacity sizing -----

    def produce(self) -> Tuple[float, List[Task]]:
        """
        Compute Leontief output across all tasks and store it on self.output.

        For automated tasks: contribution = n_ai * productivity_ai(task).
        For human tasks:     contribution = sum of productivity_human(task, skill)
                             over the task's workers.
        Returns (output level U, list of bottleneck Task objects).
        """
        u: Dict[int, float] = {}
        for t in self.tasks:
            if t.automated:
                u[t.task_id] = t.n_ai * self.productivity_ai(t)
            else:
                total = 0.0
                for w in t.employees:
                    total += self.productivity_human(t, w.skill_level)
                u[t.task_id] = total

        U, bottleneck_ids = leontief_bottlenecks(u, self.c)
        bottlenecks = [t for t in self.tasks if t.task_id in bottleneck_ids]

        self.output = U
        return U, bottlenecks

    def trim_ai_capacity(self):
        """
        Right-size n_ai on all automated tasks to the minimum needed for current output.
        Prevents excess AI rental costs when a task is over-provisioned relative to
        the Leontief bottleneck. Always call after produce() so self.output is current.
        """
        U = self.output
        for t in self.tasks:
            if t.automated and t.n_ai > 0:
                minimum = int(np.ceil(U * self.c[t.task_id] / self.productivity_ai(t)))
                t.n_ai = max(1, minimum)

    # ----- [C4 AI Adoption] Unit-cost primitives (basis for every adoption decision) -----

    def unit_cost_human(self, task: Task, wage: float, skill_level: str) -> float:
        """Cost per unit of task output for a human worker: wage / productivity."""
        prod = self.productivity_human(task, skill_level)
        if prod <= 0:
            return float("inf")
        return wage / prod

    def unit_cost_ai(self, task: Task, k_ai: float) -> float:
        """Cost per unit of task output for an AI unit: k_ai / productivity."""
        prod = self.productivity_ai(task)
        if prod <= 0:
            return float("inf")
        return k_ai / prod

    def preferred_input_for_task(self, task: Task) -> str:
        """
        Returns the cheapest input type ('high', 'low', or 'ai') for the given task,
        based on current market wages and AI rental cost.
        With task-specific productivity, the preferred input varies by complexity index.
        """
        w_h = self.model.w_h
        w_l = self.model.w_l

        uc_h  = self.unit_cost_human(task, w_h, "high")
        uc_l  = self.unit_cost_human(task, w_l, "low")
        uc_ai = self.unit_cost_ai(task, self.model.k_ai)
        options = [("high", uc_h), ("low", uc_l), ("ai", uc_ai)]

        return min(options, key=lambda x: x[1])[0]

    def _dismiss_workers_from_task(self, task: Task):
        """[C2 Labour Market] Remove all workers from a task; mark them unemployed."""
        for w in task.employees:
            w.employed = False
            w.employer = None
            w.task = None
            if w in self.employees:
                self.employees.remove(w)
        task.employees = []

    def cheapest_human_for_task(self, task: Task) -> str:
        """Returns 'high' or 'low' - whichever skill type is cheaper for this task."""
        uc_h = self.unit_cost_human(task, self.model.w_h, "high")
        uc_l = self.unit_cost_human(task, self.model.w_l, "low")
        return "high" if uc_h <= uc_l else "low"

    # ----- [C4 AI Adoption] NPV / ULC automation decision (four adoption modes) -----

    def calculate_npv(self, task: Task) -> float:
        """
        Calculate the Net Present Value of automating a given task.

        NPV = -(I + severance) + sum_{t=1}^{T} [ delta_C_t / (1+r)^t ]

        where:
          I        = upfront investment cost (complexity-adjusted)
          severance = expected severance payout for dismissed vast workers on this task
                      (0 when employment_protection is off)
          delta_C  = per-period unit-cost saving = unit_cost_human_t - unit_cost_ai_t
                     (expressed in cost-per-unit-of-task-output terms)
          r        = discount rate per period
          T        = planning horizon (model steps)

        Wage expectations depend on self.model.adoption_mode:
          - "npv_naive":      current wages persist unchanged over the horizon
          - "npv_adaptive":   smoothed wage TREND extrapolated linearly over horizon
          - "npv_mean_field": projection using the wage supply curve applied to an
                              employment level adjusted for the smoothed displacement flow

        AI cost expectations: firms anticipate the economy-wide learning curve for k_ai.

        The per-period human cost uses the cheaper of the two skill groups at each step.
        Returns the NPV; the caller compares it against eta_hurdle * investment_cost.
        """
        m = self.model
        T = m.T_horizon
        ts = np.arange(1, T + 1)  # [1, 2, ..., T]

        # --- Productivity (task-specific, constant over horizon) ---
        prod_h = self.productivity_human(task, "high")
        prod_l = self.productivity_human(task, "low")
        prod_ai = self.productivity_ai(task)
        prod_h  = max(prod_h,  1e-9)
        prod_l  = max(prod_l,  1e-9)
        prod_ai = max(prod_ai, 1e-9)

        # --- AI cost vector: C5 learning curve evaluated over the horizon ---
        future_steps = m.current_step + ts
        future_k_ai = m.ai_cost_at(future_steps)   # [C5] vectorised over the T-step horizon
        ai_unit_costs = future_k_ai / prod_ai  # shape (T,)

        # --- Human cost vector: depends on expectation mode ---
        if m.adoption_mode == "npv_naive":
            # Wages stay flat — broadcast scalars
            uc_h = m.w_h / prod_h
            uc_l = m.w_l / prod_l
            human_unit_costs = np.full(T, min(uc_h, uc_l))

        elif m.adoption_mode == "npv_adaptive":
            floor_h = max(m.w_min, m.a_h)
            floor_l = max(m.w_min, m.a_l)
            exp_w_h = np.maximum(floor_h, m.w_h + ts * m.dw_h_smoothed)
            exp_w_l = np.maximum(floor_l, m.w_l + ts * m.dw_l_smoothed)
            human_unit_costs = np.minimum(exp_w_h / prod_h, exp_w_l / prod_l)

        elif m.adoption_mode == "npv_mean_field":
            # Use step_stats cache to avoid re-summing workers inside calculate_npv
            L_h_now = m._step_stats.get("_L_h")
            L_l_now = m._step_stats.get("_L_l")
            if L_h_now is None or L_l_now is None:
                # Fallback if called before _compute_all_stats (e.g. during warm-start)
                L_h_now = sum(1 for w in m.workers if w.employed and w.skill_level == "high")
                L_l_now = sum(1 for w in m.workers if w.employed and w.skill_level == "low")
            total = L_h_now + L_l_now
            share_h = L_h_now / total if total > 0 else 0.5
            L_h_proj = np.maximum(0.0, L_h_now - ts * m.displacement_flow_smoothed * share_h)
            L_l_proj = np.maximum(0.0, L_l_now - ts * m.displacement_flow_smoothed * (1 - share_h))
            exp_w_h = np.maximum(m.w_min, m.a_h + m.b_h * L_h_proj)
            exp_w_l = np.maximum(m.w_min, m.a_l + m.b_l * L_l_proj)
            human_unit_costs = np.minimum(exp_w_h / prod_h, exp_w_l / prod_l)

        else:
            raise ValueError(
                f"calculate_npv called in mode '{m.adoption_mode}' - only valid in NPV modes"
            )

        # --- Discount factors and NPV ---
        discount_factors = (1 + m.r_discount) ** ts
        # Severance cost (transitievergoeding) is treated as an additional upfront cost:
        # only permanent workers generate severance; flex workers have no statutory entitlement.
        severance = self.severance_cost_for_task(task)
        npv = -(task.investment_cost + severance) + float(np.sum((human_unit_costs - ai_unit_costs) / discount_factors))
        task._cached_npv = npv  # cache for _compute_all_stats diagnostics
        return npv

    def should_automate(self, task: Task) -> Tuple[bool, Optional[float]]:
        """
        Returns (should_automate, npv) where npv is None in ULC mode.

        In "ulc" mode: True if AI unit cost < cheapest human unit cost (instantaneous comparison).
        In NPV modes: True if NPV > eta_hurdle * I (forward-looking investment evaluation).

        The NPV is returned so callers can pass it directly to record_automation_decision,
        avoiding a redundant second call to calculate_npv.
        """
        m = self.model

        if m.adoption_mode == "ulc":
            pref = self.preferred_input_for_task(task)
            return pref == "ai", None
        else:
            npv = self.calculate_npv(task)
            return npv > m.eta_hurdle * task.investment_cost, npv

    def record_automation_decision(self, task: Task, npv: Optional[float] = None):
        """
        Store the step at which the task was automated and, in NPV modes, the
        NPV at the moment the automation decision is made. This helps distinguish
        between:
          - remaining non-automated tasks, which may look unattractive on average
          - tasks that were actually attractive enough to automate

        ai_original_step is set in all modes (including ULC) so the
        new_automations_this_step counter is consistent across adoption modes.

        Pass npv to avoid recomputing it when should_automate() already calculated it.
        """
        if task.ai_original_step is None:  # only record the very first automation
            task.ai_original_step = self.model.current_step

        if self.model.adoption_mode == "ulc":
            task.ai_investment_step = None
            task.ai_investment_npv = None
        else:
            task.ai_investment_step = self.model.current_step
            task.ai_investment_npv = npv if npv is not None else self.calculate_npv(task)

    def _log_evaluation(
        self,
        task: "Task",
        do_automate: bool,
        npv: Optional[float],
        evaluated: bool,
        channel: str,
        departed_worker: Optional["Worker"] = None,
    ) -> None:
        """
        Append one automation-evaluation record to model.investment_log.
        Only called when model.log_decisions is True.

        channel: "reactive"         – full vacancy caused by a worker separation
                 "reactive_partial" – separation left the task partly staffed
                 "proactive"        – proactive replacement / bottleneck expansion
                                      on a still-human-staffed task
                 "empty_slot"       – bottleneck on a task that was already empty
        """
        m = self.model

        prod_h_high = max(self.productivity_human(task, "high"), 1e-9)
        prod_h_low  = max(self.productivity_human(task, "low"),  1e-9)
        prod_ai     = max(self.productivity_ai(task),            1e-9)
        uc_human_now = min(m.w_h / prod_h_high, m.w_l / prod_h_low)
        uc_ai_now    = m.k_ai / prod_ai

        record: dict = {
            # ── Identity ──────────────────────────────────────────────
            "step":         m.current_step,
            "mode":         m.adoption_mode,
            "channel":      channel,
            "evaluated":    evaluated,
            "decision":     bool(evaluated and do_automate),
            # ── Task ──────────────────────────────────────────────────
            "producer_id":  self.unique_id,
            "alpha":        round(self.alpha, 3),
            "task_id":      task.task_id,
            "task_type":    task.task_type,
            "complexity":   task.complexity_index,
            # ── Current market conditions ─────────────────────────────
            "wage_h_now":   round(m.w_h, 4),
            "wage_l_now":   round(m.w_l, 4),
            "k_ai_now":     round(m.k_ai, 4),
            # ── Productivities ────────────────────────────────────────
            "prod_h_high":  round(prod_h_high, 4),
            "prod_h_low":   round(prod_h_low,  4),
            "prod_ai":      round(prod_ai,     4),
            # ── Unit costs at this step ───────────────────────────────
            "uc_human_now": round(uc_human_now, 4),
            "uc_ai_now":    round(uc_ai_now,    4),
        }

        if m.adoption_mode == "ulc":
            if not evaluated:
                record["decision_reason"] = "not evaluated (p_evaluate miss)"
            elif do_automate:
                record["decision_reason"] = "AI cheaper (ULC) -> automated"
            else:
                record["decision_reason"] = "human cheaper (ULC) -> not automated"

        else:  # NPV modes
            inv_cost = task.investment_cost
            sev_cost = self.severance_cost_for_task(task)
            hurdle   = m.eta_hurdle * inv_cost
            # Discounted savings = NPV + upfront costs (NPV = -upfront + savings)
            disc_sav = (npv + inv_cost + sev_cost) if npv is not None else float("nan")

            # AI unit cost at end of horizon (shows C5 learning-curve benefit)
            k_end = m.ai_cost_at(m.current_step + m.T_horizon)   # [C5]
            uc_ai_horizon = round(float(k_end) / prod_ai, 4)

            record.update({
                "npv":                round(npv, 4) if npv is not None else None,
                "investment_cost":    round(inv_cost, 4),
                "severance_cost":     round(sev_cost, 4),
                "total_upfront":      round(inv_cost + sev_cost, 4),
                "hurdle_threshold":   round(hurdle, 4),
                "npv_minus_hurdle":   round(npv - hurdle, 4) if npv is not None else None,
                "discounted_savings": round(disc_sav, 4),
                "uc_ai_at_horizon":   uc_ai_horizon,
                "T_horizon":          m.T_horizon,
                "r_discount":         m.r_discount,
                "eta_hurdle":         m.eta_hurdle,
            })

            # Expected wage summary (mean over planning horizon)
            ts = np.arange(1, m.T_horizon + 1)
            if m.adoption_mode == "npv_naive":
                exp_wh_mean = m.w_h
                exp_wl_mean = m.w_l
                record["expectation_detail"] = "static wages"
            elif m.adoption_mode == "npv_adaptive":
                floor_h = max(m.w_min, m.a_h)
                floor_l = max(m.w_min, m.a_l)
                exp_wh_mean = float(np.mean(np.maximum(floor_h, m.w_h + ts * m.dw_h_smoothed)))
                exp_wl_mean = float(np.mean(np.maximum(floor_l, m.w_l + ts * m.dw_l_smoothed)))
                record["expectation_detail"] = (
                    f"adaptive EMA: Δw_h={m.dw_h_smoothed:.4f}, Δw_l={m.dw_l_smoothed:.4f}"
                )
            elif m.adoption_mode == "npv_mean_field":
                L_h = m._step_stats.get("_L_h")
                L_l = m._step_stats.get("_L_l")
                if L_h is None or L_l is None:
                    L_h = sum(1 for w in m.workers if w.employed and w.skill_level == "high")
                    L_l = sum(1 for w in m.workers if w.employed and w.skill_level == "low")
                total_L = L_h + L_l
                sh = L_h / total_L if total_L > 0 else 0.5
                L_h_proj = np.maximum(0.0, L_h - ts * m.displacement_flow_smoothed * sh)
                L_l_proj = np.maximum(0.0, L_l - ts * m.displacement_flow_smoothed * (1 - sh))
                exp_wh_mean = float(np.mean(np.maximum(m.w_min, m.a_h + m.b_h * L_h_proj)))
                exp_wl_mean = float(np.mean(np.maximum(m.w_min, m.a_l + m.b_l * L_l_proj)))
                record["expectation_detail"] = (
                    f"mean-field: disp_flow={m.displacement_flow_smoothed:.2f},"
                    f" L_h={L_h}, L_l={L_l}"
                )
            else:
                exp_wh_mean = m.w_h
                exp_wl_mean = m.w_l
                record["expectation_detail"] = ""

            record["exp_wage_h_mean"] = round(exp_wh_mean, 4)
            record["exp_wage_l_mean"] = round(exp_wl_mean, 4)

            # Decision reason string
            if not evaluated:
                record["decision_reason"] = "not evaluated (p_evaluate miss)"
            elif npv is not None and npv > hurdle:
                record["decision_reason"] = (
                    f"NPV={npv:.3f} > hurdle={hurdle:.3f} -> automated"
                )
            else:
                record["decision_reason"] = (
                    f"NPV={npv:.3f} <= hurdle={hurdle:.3f} -> not automated"
                )

        # ── Employment protection detail ───────────────────────────────
        if m.employment_protection:
            vast_ws = [w for w in task.employees if w.contract_type == "vast"]
            flex_ws = [w for w in task.employees if w.contract_type == "flex"]
            record["n_vast_on_task"] = len(vast_ws)
            record["n_flex_on_task"] = len(flex_ws)
            if departed_worker is not None:
                record["departed_contract"] = departed_worker.contract_type
                record["departed_tenure_y"]  = round(
                    departed_worker.tenure / max(1, m.steps_per_year), 2
                )
            sev_parts = []
            for w in vast_ws:
                ty  = w.tenure / max(1, m.steps_per_year)
                sev = m.severance_rate * w.wage * ty
                sev_parts.append(f"W{w.unique_id}(vast, {ty:.1f}y, sev={sev:.2f})")
            record["severance_detail"] = "; ".join(sev_parts) if sev_parts else "—"

        m.investment_log.append(record)

    # ----- [C2 Labour Market] Separations & hiring (firm side) -----

    def on_worker_separated(self, task: Task, departed_worker: "Worker"):
        """
        Reactive callback: a worker was separated from this task.

        Two cases are handled differently:

        Full vacancy (task is now completely empty):
          - Evaluate automation (p_evaluate roll).
          - If automating  → automate immediately.
          - If not         → add an unconditional hiring request; the task cannot
                             produce anything at all without workers.

        Partial vacancy (task still has other workers after this separation):
          - Evaluate automation (p_evaluate roll).
          - If automating  → dismiss remaining workers and automate.
          - If not         → do NOT add a hiring request here.  The proactive
                             channel (hiring_requests / bottleneck detection) decides
                             whether the understaffing actually limits output and
                             requests a replacement only if it does.  This prevents
                             unconditional refilling that would keep employment at
                             ~100% regardless of production conditions.

        departed_worker is passed so n_ai sizing covers full pre-automation output.
        """
        is_full_vacancy = not task.employees   # True when last worker just left

        evaluate = self.random.random() < self.model.p_evaluate

        do_automate, npv = self.should_automate(task)

        channel = "reactive" if is_full_vacancy else "reactive_partial"

        if self.model.log_decisions:
            self._log_evaluation(task, do_automate, npv, evaluate,
                                  channel=channel, departed_worker=departed_worker)

        if evaluate and do_automate:
            # Drop stale requests: automating, so no human replacement needed.
            self._pending_requests = [r for r in self._pending_requests if r[1] is not task]

            # Size n_ai to cover the output of ALL workers (remaining + departed)
            # so Leontief output does not drop after automation.
            departed_output  = self.productivity_human(task, departed_worker.skill_level)
            remaining_output = sum(
                self.productivity_human(task, w.skill_level) for w in task.employees
            )
            pre_output = departed_output + remaining_output

            # Dismiss any workers still on the task (displaced by AI).
            if task.employees:
                self.model.new_displaced_workers_this_step += len(task.employees)
                self._dismiss_workers_from_task(task)

            ai_prod = self.productivity_ai(task)
            task.automated = True
            task.n_ai = max(1, int(np.ceil(pre_output / ai_prod)) if ai_prod > 0 else 1)
            self.model.new_reactive_automations_this_step += 1
            self.record_automation_decision(task, npv)
            return

        # Human replacement path.
        if is_full_vacancy:
            # Task is completely empty — must hire immediately regardless of bottleneck
            # status, because zero workers means zero output on this task.
            # Do NOT clear existing requests: two workers leaving the same task in the
            # same step should each generate their own replacement request.
            best = self.cheapest_human_for_task(task)
            self._pending_requests.append((self, task, best))
        # else: partial vacancy, not automating → let proactive bottleneck detection
        # decide whether a replacement is needed this step.

    def hiring_requests(self) -> List[tuple["Producer", Task, str]]:
        """
        Determine which bottleneck tasks need additional input this step.
        Returns a list of (producer, task, input_type) hiring requests.

        For each bottleneck task (after a p_evaluate roll at firm level):
          - Automated task, ULC mode or ai_irreversible=False:
              AI still cheapest  -> scale n_ai to produce target output + 1.
              Human now cheaper -> request human placement (free de-automation).
          - Automated task, NPV mode with ai_irreversible=True:
              Investment horizon expired -> re-evaluate at current costs; either
                  request human replacement or renew AI (reset the clock).
              Otherwise              -> cannot de-automate; just scale n_ai.
          - Human-staffed task, AI now clears hurdle -> automate (dismiss workers).
          - Human-staffed task, AI does not clear hurdle -> request extra worker.
          - Empty task (edge case) -> automate or request hire depending on preference.

        Pending requests from on_worker_separated() are always flushed (even when unprofitable).
        Proactive bottleneck expansion is skipped when marginal cost >= market price.
        """
        _, bottlenecks = self.produce()

        requests = list(self._pending_requests)
        self._pending_requests = []

        # Always return reactive (separation-triggered) requests even when unprofitable.
        # Only generate proactive bottleneck-expansion requests if MC < market price.
        if not self.should_produce_more():
            return requests

        # Single evaluate roll per firm per step
        evaluate_this_tick = self.random.random() < self.model.p_evaluate

        for t in bottlenecks:
            # Skip if a request for this task was already queued by on_worker_separated
            if any(r[1] is t for r in requests):
                continue

            if t.automated:
                # --- Already automated task is a bottleneck ---
                if not evaluate_this_tick:
                    continue

                # Free reversal: ULC mode always allows it; NPV modes allow it when
                # ai_irreversible=False (the explicit reversibility toggle).
                if self.model.adoption_mode == "ulc" or not self.model.ai_irreversible:
                    pref = self.preferred_input_for_task(t)
                    if pref == "ai":
                        # AI still cheapest -> scale up
                        target = self.output + 1
                        t.n_ai = int(np.ceil(target * self.c[t.task_id] / self.productivity_ai(t)))
                    else:
                        # Humans cheaper -> request replacement (free de-automation)
                        requests.append((self, t, pref))
                else:
                    # NPV mode with irreversibility - only reconsider after investment horizon
                    if (t.ai_investment_step is not None and
                            self.model.current_step - t.ai_investment_step >= self.model.T_horizon):
                        # Investment fully depreciated - re-evaluate with current costs
                        uc_h = self.unit_cost_human(t, self.model.w_h, "high")
                        uc_l = self.unit_cost_human(t, self.model.w_l, "low")
                        uc_ai = self.unit_cost_ai(t, self.model.k_ai)
                        if min(uc_h, uc_l) < uc_ai:
                            best = self.cheapest_human_for_task(t)
                            requests.append((self, t, best))
                        else:
                            # AI still cheaper - renew (reset clock, no new investment cost)
                            t.ai_investment_step = self.model.current_step
                    else:
                        # Investment not yet depreciated - scale AI if needed, don't de-automate
                        target = self.output + 1
                        t.n_ai = int(np.ceil(target * self.c[t.task_id] / self.productivity_ai(t)))

            elif len(t.employees) > 0:
                # --- Human-staffed task is a bottleneck ---
                if not evaluate_this_tick:
                    continue

                do_automate, npv = self.should_automate(t)
                if self.model.log_decisions:
                    self._log_evaluation(t, do_automate, npv, True, channel="proactive")
                if do_automate:
                    # Compute pre-automation task output so AI can be sized to cover it.
                    # With >1 worker, 1 AI unit might produce less than the dismissed
                    # team; we set n_ai to exactly cover the pre-automation contribution
                    # so Leontief output does not drop transiently. trim_ai_capacity()
                    # will right-size downward on the next produce() call if over-provisioned.
                    pre_output = sum(
                        self.productivity_human(t, w.skill_level)
                        for w in t.employees
                    )
                    self.model.new_proactive_automations_this_step += 1
                    self.model.new_displaced_workers_this_step += len(t.employees)
                    self._dismiss_workers_from_task(t)
                    t.automated = True
                    ai_prod = self.productivity_ai(t)
                    t.n_ai = max(1, int(np.ceil(pre_output / ai_prod)) if ai_prod > 0 else 1)
                    self.record_automation_decision(t, npv)
                else:
                    # Adoption criterion not met -> request additional human
                    best = self.cheapest_human_for_task(t)
                    requests.append((self, t, best))

            else:
                # --- Empty task slot (edge case) ---
                do_automate, npv = self.should_automate(t)
                if self.model.log_decisions:
                    self._log_evaluation(t, do_automate, npv, evaluate_this_tick,
                                          channel="empty_slot")
                if evaluate_this_tick and do_automate:
                    t.automated = True
                    t.n_ai = 1
                    self.model.new_proactive_automations_this_step += 1
                    self.record_automation_decision(t, npv)
                else:
                    best = self.cheapest_human_for_task(t)
                    requests.append((self, t, best))

        return requests

    def should_produce_more(self) -> bool:
        """
        Returns True when the firm is currently profitable (price > average cost),
        meaning it is worth proactively relaxing bottlenecks to expand output.

        In a Leontief technology, all 20 tasks are complementary inputs that together
        produce one unit of composite output.  Comparing the raw factor cost of a
        single bottleneck worker to the product price conflates two different units
        (cost-per-worker vs price-per-composite-output) and fires far too early.
        The economically correct expansion signal is: does the current price cover
        average total costs?  If not, proactive hiring deepens losses; if yes,
        producing more is profitable.

        Side effect: when AC >= price the firm is counted in
        model.n_gate_blocks_this_step for diagnostic purposes.
        """
        if self.output <= 0:
            return True   # always try to produce something when output is zero
        average_cost = self.total_costs / self.output
        price = self.model.p
        if average_cost > price:
            self.model.n_gate_blocks_this_step += 1
        return average_cost < price

    # ----- [C7 Employment Protection] Severance cost of a task -----

    def severance_cost_for_task(self, task: Task) -> float:
        """
        Total severance (transitievergoeding) that would be owed if every worker
        currently assigned to this task were dismissed.

        Used both as an upfront cost term inside calculate_npv and as a
        diagnostic metric. Only permanent ('vast') workers generate severance;
        flex workers have no statutory severance entitlement.

        Per-worker formula (Dutch Wet arbeidsmarkt in balans):
            severance = severance_rate × wage × (tenure_in_steps / steps_per_year)

        Returns 0.0 when employment_protection is disabled or no vast workers
        are on the task.
        """
        if not self.model.employment_protection:
            return 0.0
        total = 0.0
        for w in task.employees:
            if w.contract_type == "vast":
                tenure_years = w.tenure / self.model.steps_per_year
                total += self.model.severance_rate * w.wage * tenure_years
        return total


# =============================================================================
# LabourMarketModel
# Carries the market-level code and wires every component into the step loop:
#   C5 Learning Curve     - ai_cost_at()
#   C3 Wage-setting       - calculate_wage_bill(), update_wages()
#   C6 Demand             - calculate_profit_income(), calculate_market_price()
#   C7 Employment Prot.   - process_chain_limit()
#   C2 Labour Market      - separate_workers(), _warm_start(), match_and_hire()
#   M1 Macro-feedback     - step() ordering and _compute_all_stats()
# =============================================================================

class LabourMarketModel(Model):
    """
    Agent-based model of a labour market with AI automation.

    Demand side: inverse demand curve p = A - B*Q, where A shifts upward with
    aggregate factor income. Following the post-Keynesian / Kaleckian two-class
    consumption function (Kaldor 1955; Pasinetti 1962; Kalecki 1971;
    Bhaduri & Marglin 1990), demand depends on BOTH wage income W and
    profit income Pi, with potentially different marginal propensities to
    consume (gamma > gamma_pi gives a wage-led tilt: wage income transmits to demand more strongly than profit income):
        A = A_base + gamma * W_{t-1} + gamma_pi * Pi_{t-1}
    where Pi_{t-1} = sum_i (P - AC_i) * Q_i is aggregate operating profit
    in the previous step.
    Supply side: producers with Leontief technology, hiring workers or AI per task.
    Wage setting: upward-sloping wage curve (Blanchflower-Oswald style) per skill group.
    """

    def __init__(
            self,
            n_producers: int = 20,
            n_high_skilled: int = 200,
            n_low_skilled: int = 250,
            delta: float = 0.005,     # exogenous separation rate (probability of job loss per step)
            k_ai: float = 24.0,       # AI rental cost per unit (contestable with labour)
            p_evaluate: float = 0.15, # probability that a firm evaluates AI adoption this step
            A_base: float = 10.0,     # autonomous demand level
            gamma: float = 0.003,     # wage-income demand weight (wage-led channel; not a literal MPC)
            gamma_pi: float = 0.0015, # profit-income demand weight (Kaleckian capital-income channel);
                                      # empirically gamma_pi < gamma — capitalists save more than workers
                                      # (Onaran & Galanis 2014; Storm & Naastepad 2012)
            seed: int = 42,
            B: float = 0.5,           # price sensitivity of demand (inverse demand slope)
            lam: float = 0.25,        # wage adjustment speed (0 = none, 1 = instant to target)
            a_h: float = 2.0,         # high-skilled wage intercept
            b_h: float = 0.14,        # high-skilled wage slope
            a_l: float = 1.0,         # low-skilled wage intercept
            b_l: float = 0.10,        # low-skilled wage slope
            w_min: float = 1.0,       # explicit minimum wage floor (minimumloon); 0 = inactive
            # U&S productivity parameters (Upreti & Sridhar 2023)
            a_prod: float = 0.1,      # exponential human productivity scale over task complexity
            xi_prod: float = 0.5,     # low-skill non-routine productivity dampening (0 < xi < 1)
            phi_base: float = 3.0,    # baseline AI productivity (before complexity decay)
            phi_decay: float = 0.04,  # AI productivity decay rate with task complexity
            nr_multiplier: float = 0.85,  # AI productivity multiplier on non-routine tasks
            # --- Matching: cross-skill fallback tolerance ---
            # Firms request preferred skill first. If unavailable, a worker of the other
            # skill type is hired when their unit cost <= mismatch_theta x preferred unit cost.
            # 1.0 = no cross-skill hiring; large value ~ fully free substitution.
            mismatch_theta: float = 1.5,
            # --- Adoption mechanism ---
            # "ulc"          = instantaneous unit-cost comparison (Upreti & Sridhar baseline)
            # "npv_naive"    = NPV with naive wage expectations (current wages persist)
            # "npv_adaptive" = NPV with adaptive wage expectations (weighted avg of recent wages)
            # "npv_mean_field" = NPV with mean-field wage expectations (wage-curve projection)
            adoption_mode: str = "ulc",
            # --- AI reversibility ---
            # True  (default for NPV modes): once invested, AI cannot be replaced by
            #        humans until the planning horizon T_horizon expires.
            # False: free de-automation at any step (like ULC mode), even in NPV modes.
            ai_irreversible: bool = False,
            # --- NPV investment parameters (ignored when adoption_mode == "ulc") ---
            I_base: float = 60.0,             # upfront AI investment cost per task
            complexity_scaling: float = 0.15, # cost multiplier: I(task) = I_base * (1 + cs * x)
            r_discount: float = 0.05,         # discount rate per period
            T_horizon: int = 20,              # planning horizon in model steps
            eta_hurdle: float = 0.10,         # hurdle rate: NPV must exceed eta * I for adoption
            # --- Adaptive expectations (only used in npv_adaptive mode) ---
            omega_adapt: float = 0.3,         # weight on most recent wage observation
            # --- Learning curve for AI rental cost (active in ALL modes) ---
            k_ai_decay: float = 0.020,        # speed of cost decline
            k_ai_floor: float = 18.0,         # minimum AI rental cost
            # --- Employment protection (ontslagbescherming) ---
            employment_protection: bool = False,   # master switch; False = identical to old behaviour
            steps_per_year: int = 12,              # time-scale: 1 step = 1 month
            severance_rate: float = 1 / 3,         # transitievergoeding: monthly salaries per service year
            chain_limit: int = 36,                 # ketenregeling: flex → conversion/non-renewal after this many steps
            p_convert: float = 0.5,                # probability that a flex worker is converted to vast at chain_limit
            init_share_vast: float = 0.6,          # fraction of employed workers given a vast contract at warm-start
            init_tenure_max_years: float = 10.0,   # maximum tenure in years drawn at warm-start
            # --- Decision logging (opt-in; set True from dashboard) ---
            log_decisions: bool = False,           # when True every automation evaluation is appended to investment_log
            # --- Alpha distribution (routine-task share per firm) ---
            # "uniform": draw from U(alpha_min, alpha_max)
            # "data":    sample from Gmyrek 4-digit mean_automation_score distribution
            alpha_source: str = "uniform",
            alpha_min: float = 0.7,
            alpha_max: float = 1.0,
                 ):
        super().__init__()
        self._seed = seed          # stored so _warm_start can use its own isolated RNG
        self.random.seed(seed)
        np.random.seed(seed)

        self.delta = delta
        self.p_evaluate = p_evaluate

        # Keynesian / Kaleckian demand parameters.
        # A_t = A_base + gamma * W_{t-1} + gamma_pi * Pi_{t-1}
        #   W  = aggregate wage income (labour share of GDP)
        #   Pi = aggregate operating profit income (capital share of GDP)
        # Setting gamma_pi = 0 reproduces the original pure wage-led specification.
        self.A_base   = A_base    # anchor for autonomous demand
        self.gamma    = gamma     # wage-income demand weight
        self.gamma_pi = gamma_pi  # profit-income demand weight (Kaleckian extension)
        self.A  = A_base   # dynamic demand parameter; starts at A_base
        self.Y  = 0.0      # lagged wage bill;    starts at 0 -> A = A_base in step 1
        self.Pi = 0.0      # lagged profit income; starts at 0 -> A = A_base in step 1

        agent_id = 0

        # Store alpha distribution settings (used by external analysis tools)
        self.alpha_source = alpha_source
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # Create producers; alpha drawn according to alpha_source / range
        _alpha_rng = np.random.default_rng(seed)
        _alphas = sample_alphas(n_producers, alpha_source, alpha_min, alpha_max, rng=_alpha_rng)
        self.producers: List[Producer] = []
        for a in _alphas:
            p = Producer(agent_id, self, alpha=float(a))
            self.producers.append(p)
            agent_id += 1

        # Create workers by skill group
        self.workers: List[Worker] = []
        for _ in range(n_high_skilled):
            w = Worker(agent_id, self, skill_level="high")
            self.workers.append(w)
            agent_id += 1

        for _ in range(n_low_skilled):
            w = Worker(agent_id, self, skill_level="low")
            self.workers.append(w)
            agent_id += 1

        # Wage supply curves: w = a + b * L  (upward-sloping wage curve, Blanchflower-Oswald style)
        self.a_h, self.b_h = a_h, b_h     # high-skilled wage intercept and slope
        self.a_l, self.b_l = a_l, b_l     # low-skilled wage intercept and slope
        self.w_min = w_min                 # explicit minimum wage floor (0 = inactive)
        self.B = B
        self.w_h = max(self.w_min, self.a_h)  # initialise respecting w_min
        self.w_l = max(self.w_min, self.a_l)
        self.lam = lam

        self.k_ai = k_ai

        # Cross-skill fallback tolerance
        self.mismatch_theta = mismatch_theta

        # Adoption mode
        if adoption_mode not in ("ulc", "npv_naive", "npv_adaptive", "npv_mean_field"):
            raise ValueError(f"adoption_mode must be one of 'ulc', 'npv_naive', 'npv_adaptive', 'npv_mean_field', got '{adoption_mode}'")
        self.adoption_mode = adoption_mode

        # AI reversibility flag
        self.ai_irreversible = ai_irreversible

        # NPV investment parameters
        self.I_base            = I_base
        self.complexity_scaling = complexity_scaling
        self.r_discount        = r_discount
        self.T_horizon         = T_horizon
        self.eta_hurdle        = eta_hurdle
        self.omega_adapt       = omega_adapt
        for producer in self.producers:
            for task in producer.tasks:
                task.investment_cost = self.I_base * (
                    1.0 + self.complexity_scaling * task.complexity_index
                )

        # Learning curve state
        self.k_ai_0     = k_ai        # store initial AI cost
        self.k_ai_floor = k_ai_floor
        self.k_ai_decay = k_ai_decay
        self.current_step = 0          # track model step count

        # Adaptive expectations: smooth the wage CHANGE (trend), not the level
        self.dw_h_smoothed = 0.0   # smoothed wage change per period (high-skilled)
        self.dw_l_smoothed = 0.0   # smoothed wage change per period (low-skilled)
        self.w_h_prev = self.w_h   # previous-step wages for delta calculation
        self.w_l_prev = self.w_l

        # Cached stats computed once per step via _compute_all_stats().
        # DataCollector lambdas do a simple dict lookup instead of re-iterating workers/tasks.
        self._step_stats: dict = {}
        self._npv_stats:  dict = {}

        # Per-step automation and gate diagnostics (reset at the start of each step)
        self.new_reactive_automations_this_step = 0
        self.new_proactive_automations_this_step = 0
        self.new_displaced_workers_this_step = 0
        self.n_gate_blocks_this_step = 0
        self.n_human_cheaper_tasks_this_step = 0
        # Mean-field expectations: smoothed worker displacement flow per step.
        self.displacement_flow_smoothed = 0.0

        # Employment protection (ontslagbescherming)
        self.employment_protection = employment_protection
        self.steps_per_year        = steps_per_year
        self.severance_rate        = severance_rate
        self.chain_limit           = chain_limit
        self.p_convert             = p_convert
        self.init_share_vast       = init_share_vast
        self.init_tenure_max_years = init_tenure_max_years
        # Per-step event counters (reset at the start of each step)
        self._conversions_this_step  = 0
        self._non_renewals_this_step = 0
        # Decision logging (populated only when log_decisions=True)
        self.log_decisions:   bool       = log_decisions
        self.investment_log:  list[dict] = []

        # U&S productivity parameters
        self.a_prod     = a_prod
        self.xi_prod    = xi_prod
        self.phi_base   = phi_base
        self.phi_decay  = phi_decay
        self.nr_multiplier = nr_multiplier

        # Inverse demand: p = A - B*Q  (A is dynamic via Keynesian loop)
        self.p = self.A

        self.datacollector = DataCollector(
            model_reporters={
                # All metrics below read from _step_stats / _npv_stats, which are computed
                # once per step in a single pass by _compute_all_stats(). This avoids
                # the DataCollector calling 20+ separate list comprehensions that each
                # re-iterate the full worker and task lists independently.
                "price":                          lambda m: m.p,
                "wage_high":                      lambda m: m.w_h,
                "wage_low":                       lambda m: m.w_l,
                "skill_wage_premium":             lambda m: m._step_stats.get("skill_wage_premium", float("nan")),
                "wage_bill_lagged":               lambda m: m.Y,
                "profit_income_lagged":           lambda m: m.Pi,
                "wage_demand_contribution":       lambda m: m.gamma    * m.Y,
                "profit_demand_contribution":     lambda m: m.gamma_pi * m.Pi,
                "A_demand":                       lambda m: m.A,
                "k_ai_current":                   lambda m: m.k_ai,
                "adoption_mode":                  lambda m: m.adoption_mode,
                "ai_irreversible":                lambda m: m.ai_irreversible,
                "w_min":                          lambda m: m.w_min,
                "new_reactive_automations_this_step":  lambda m: m.new_reactive_automations_this_step,
                "new_proactive_automations_this_step": lambda m: m.new_proactive_automations_this_step,
                "new_displaced_workers_this_step":     lambda m: m.new_displaced_workers_this_step,
                "n_gate_blocks":                       lambda m: m.n_gate_blocks_this_step,
                "n_human_cheaper_tasks":               lambda m: m.n_human_cheaper_tasks_this_step,

                # Cached in _step_stats (single worker+task pass):
                "employment_rate_high":           lambda m: m._step_stats.get("employment_rate_high", float("nan")),
                "employment_rate_low":            lambda m: m._step_stats.get("employment_rate_low",  float("nan")),
                "ai_adoption_rate":               lambda m: m._step_stats.get("ai_adoption_rate",     float("nan")),
                "gini_income":                    lambda m: m._step_stats.get("gini_income",          float("nan")),
                "reactive_automation_share_this_step":  lambda m: m._step_stats.get("reactive_automation_share_this_step",  float("nan")),
                "proactive_automation_share_this_step": lambda m: m._step_stats.get("proactive_automation_share_this_step", float("nan")),
                "total_output":                   lambda m: m._step_stats.get("total_output",          float("nan")),
                "wage_bill":                      lambda m: m._step_stats.get("wage_bill",             float("nan")),
                "labour_share":                   lambda m: m._step_stats.get("labour_share",          float("nan")),
                "ULC":                            lambda m: m._step_stats.get("ULC",                   float("nan")),
                "input_cost_ai_routine":          lambda m: m._step_stats.get("input_cost_ai_routine",          float("nan")),
                "input_cost_ai_nonroutine":       lambda m: m._step_stats.get("input_cost_ai_nonroutine",       float("nan")),
                "input_cost_human_high_routine":  lambda m: m._step_stats.get("input_cost_human_high_routine",  float("nan")),
                "input_cost_human_low_routine":   lambda m: m._step_stats.get("input_cost_human_low_routine",   float("nan")),
                "input_cost_human_high_nonroutine": lambda m: m._step_stats.get("input_cost_human_high_nonroutine", float("nan")),
                "input_cost_human_low_nonroutine":  lambda m: m._step_stats.get("input_cost_human_low_nonroutine",  float("nan")),
                "Avg labour cost per producer":   lambda m: m._step_stats.get("avg_labour_cost",       float("nan")),
                # "Avg profit per producer" = average OPERATING profit (revenue - labour costs - AI rental).
                # Excludes upfront AI investment and severance.
                "Avg profit per producer":        lambda m: m._step_stats.get("avg_profit",            float("nan")),
                "new_automations_this_step":      lambda m: m._step_stats.get("new_automations_this_step", float("nan")),

                # Cached in _npv_stats (single NPV pass):
                "avg_npv_routine":                lambda m: m._npv_stats.get("avg_npv_routine",             float("nan")),
                "avg_npv_nonroutine":             lambda m: m._npv_stats.get("avg_npv_nonroutine",          float("nan")),
                "share_adoptable_npv_routine":    lambda m: m._npv_stats.get("share_adoptable_npv_routine",    float("nan")),
                "share_adoptable_npv_nonroutine": lambda m: m._npv_stats.get("share_adoptable_npv_nonroutine", float("nan")),
                "avg_realized_npv_this_step":     lambda m: m._npv_stats.get("avg_realized_npv_this_step",  float("nan")),
                "avg_realized_npv_cumulative":    lambda m: m._npv_stats.get("avg_realized_npv_cumulative", float("nan")),

                # Employment protection metrics (nan when employment_protection=False)
                "share_vast_high":           lambda m: m._step_stats.get("share_vast_high",        float("nan")),
                "share_vast_low":            lambda m: m._step_stats.get("share_vast_low",         float("nan")),
                "share_flex_high":           lambda m: m._step_stats.get("share_flex_high",        float("nan")),
                "share_flex_low":            lambda m: m._step_stats.get("share_flex_low",         float("nan")),
                "avg_tenure_years":          lambda m: m._step_stats.get("avg_tenure_years",       float("nan")),
                "avg_tenure_vast_years":     lambda m: m._step_stats.get("avg_tenure_vast_years",  float("nan")),
                "avg_tenure_flex_years":     lambda m: m._step_stats.get("avg_tenure_flex_years",  float("nan")),
                "avg_severance_per_task":    lambda m: m._step_stats.get("avg_severance_per_task", float("nan")),
                "conversions_this_step":     lambda m: m._conversions_this_step  if m.employment_protection else float("nan"),
                "non_renewals_this_step":    lambda m: m._non_renewals_this_step if m.employment_protection else float("nan"),
            }
        )
        self._warm_start()
        self._collect_initial_state()

    # ----- [C5 Learning Curve] AI rental-cost path -----

    def ai_cost_at(self, step):
        """
        [C5] AI rental cost k_ai at a given model step - a Wright's-law style
        learning curve in closed form:

            k_ai(t) = k_ai_floor + (k_ai_0 - k_ai_floor) * exp(-k_ai_decay * t)

        The cost decays geometrically from the initial level k_ai_0 towards the
        long-run floor k_ai_floor. `step` may be a scalar (the current step,
        used by M1 in step()) or a NumPy array (a whole planning horizon, used
        by C4 in calculate_npv() so firms can anticipate future AI costs). This
        method is the single source of truth for the AI cost path.
        """
        return self.k_ai_floor + (self.k_ai_0 - self.k_ai_floor) * np.exp(
            -self.k_ai_decay * step
        )

    # ----- [M1 Macro-feedback] Per-step observation / data collection -----

    def _compute_all_stats(self):
        """
        Compute all DataCollector aggregate metrics in a single combined pass.

        Two passes — one over workers, one over tasks — replace the ~20 separate
        list comprehensions that the DataCollector lambdas would otherwise run
        independently each step. Results are stored in self._step_stats and
        self._npv_stats; the lambdas read from those dicts directly.

        Called once per step (and once at t=0) before datacollector.collect().
        """
        # --- Worker pass ---
        n_high = n_high_employed = 0
        n_low  = n_low_employed  = 0
        wage_bill = 0.0
        # Income distribution across ALL workers (unemployed count as 0 income).
        # Used to compute the Gini coefficient of worker income.
        incomes: list[float] = []
        # Employment-protection accumulators (only populated when enabled)
        vast_high = vast_low = flex_high = flex_low = 0
        tenure_all: list[float] = []
        tenure_vast: list[float] = []
        tenure_flex: list[float] = []
        for w in self.workers:
            incomes.append(w.wage if w.employed else 0.0)
            if w.skill_level == "high":
                n_high += 1
                if w.employed:
                    n_high_employed += 1
                    wage_bill += w.wage
                    if self.employment_protection:
                        t_yrs = w.tenure / self.steps_per_year
                        tenure_all.append(t_yrs)
                        if w.contract_type == "vast":
                            vast_high += 1
                            tenure_vast.append(t_yrs)
                        else:
                            flex_high += 1
                            tenure_flex.append(t_yrs)
            else:
                n_low += 1
                if w.employed:
                    n_low_employed += 1
                    wage_bill += w.wage
                    if self.employment_protection:
                        t_yrs = w.tenure / self.steps_per_year
                        tenure_all.append(t_yrs)
                        if w.contract_type == "vast":
                            vast_low += 1
                            tenure_vast.append(t_yrs)
                        else:
                            flex_low += 1
                            tenure_flex.append(t_yrs)

        # --- Producer / task pass ---
        total_output   = 0.0
        total_lc       = 0.0   # total labour costs
        # Operating profit: revenue - (labour costs + AI rental costs).
        # Excludes upfront AI investment costs and severance — those are
        # one-off cash flows tracked separately in the NPV / investment log.
        total_profit   = 0.0
        n_tasks        = 0
        n_automated    = 0
        new_auto       = 0     # automations this step

        uc_ai_r,    uc_ai_nr    = [], []
        uc_hh_r,   uc_hh_nr    = [], []
        uc_hl_r,   uc_hl_nr    = [], []

        # NPV accumulators
        npv_r, npv_nr           = [], []
        adopt_r, adopt_nr       = [], []
        realized_this, realized_all = [], []

        is_npv = self.adoption_mode != "ulc"

        # Employment-protection: severance diagnostic
        severance_per_task: list[float] = []

        for p in self.producers:
            total_output += p.output
            total_lc     += p.labour_costs
            total_profit += p.output * self.p - p.total_costs
            for t in p.tasks:
                n_tasks += 1
                if t.automated:
                    n_automated += 1
                if t.ai_original_step == self.current_step:
                    new_auto += 1

                # Input cost diagnostics (both routine and non-routine tasks)
                uc_ai = p.unit_cost_ai(t, self.k_ai)
                uc_hh = p.unit_cost_human(t, self.w_h, "high")
                uc_hl = p.unit_cost_human(t, self.w_l, "low")
                if t.task_type == "routine":
                    uc_ai_r.append(uc_ai); uc_hh_r.append(uc_hh); uc_hl_r.append(uc_hl)
                else:
                    uc_ai_nr.append(uc_ai); uc_hh_nr.append(uc_hh); uc_hl_nr.append(uc_hl)

                # Crossover counter: cheapest human < AI on this task
                if min(uc_hh, uc_hl) < uc_ai:
                    self.n_human_cheaper_tasks_this_step += 1

                # Employment-protection: severance cost diagnostic for human-staffed tasks
                if self.employment_protection and not t.automated and t.employees:
                    severance_per_task.append(p.severance_cost_for_task(t))

                # NPV diagnostics (NPV modes only) — use cached value from last should_automate call
                # to avoid recomputing NPV for all tasks. Cache may be one step stale for tasks
                # that were not evaluated this step (p_evaluate gating), which is fine for
                # diagnostic averages.
                if is_npv:
                    if not t.automated and t._cached_npv is not None:
                        npv = t._cached_npv
                        threshold = self.eta_hurdle * t.investment_cost
                        if t.task_type == "routine":
                            npv_r.append(npv); adopt_r.append(float(npv > threshold))
                        else:
                            npv_nr.append(npv); adopt_nr.append(float(npv > threshold))
                    if t.ai_investment_npv is not None:
                        realized_all.append(t.ai_investment_npv)
                        if t.ai_investment_step == self.current_step:
                            realized_this.append(t.ai_investment_npv)

        total_revenue = total_output * self.p

        # Gini coefficient of worker income (employed wage; unemployed = 0).
        # 0 = perfect equality, 1 = maximum inequality. NaN when no workers or total income is zero.
        gini_income = float("nan")
        if incomes:
            arr = np.sort(np.asarray(incomes, dtype=float))
            n = arr.size
            s = arr.sum()
            if s > 0.0:
                # Standard Gini formula: G = (2 * sum(i * x_i) - (n+1) * sum(x)) / (n * sum(x))
                idx = np.arange(1, n + 1, dtype=float)
                gini_income = float((2.0 * np.sum(idx * arr) - (n + 1) * s) / (n * s))

        self._step_stats = {
            "_L_h":                 n_high_employed,
            "_L_l":                 n_low_employed,
            # Per-skill employment rates: share of the (fixed) worker
            # population of each skill group currently assigned to a task.
            # A worker counts as unemployed when w.employed is False; there is
            # no search/participation margin, so the denominator is the whole
            # population. The aggregate unemployment rate reported in the
            # thesis is derived from these two rates as
            #   u = 1 - (n_high*e_h + n_low*e_l) / (n_high + n_low)
            # in multirun_experiments.add_unemployment_rate(); with 450
            # workers and 400 tasks the starting distribution is ~11% unemployed.
            "employment_rate_high": n_high_employed / max(1, n_high),
            "employment_rate_low":  n_low_employed  / max(1, n_low),
            "skill_wage_premium":   self.w_h / self.w_l if self.w_l > 0 else float("nan"),
            "wage_bill":            wage_bill,
            "labour_share":         wage_bill / max(0.01, total_revenue) if total_output > 0 else 0.0,
            "total_output":         total_output,
            "ULC":                  total_lc / max(0.01, total_output),
            "avg_labour_cost":      total_lc / max(1, len(self.producers)),
            "avg_profit":           total_profit / max(1, len(self.producers)),
            "ai_adoption_rate":     n_automated / max(1, n_tasks),
            "gini_income":          gini_income,
            "reactive_automation_share_this_step":  self.new_reactive_automations_this_step  / max(1, n_tasks),
            "proactive_automation_share_this_step": self.new_proactive_automations_this_step / max(1, n_tasks),
            "new_automations_this_step": float(new_auto),
            "input_cost_ai_routine":           float(np.mean(uc_ai_r))  if uc_ai_r  else float("nan"),
            "input_cost_ai_nonroutine":        float(np.mean(uc_ai_nr)) if uc_ai_nr else float("nan"),
            "input_cost_human_high_routine":   float(np.mean(uc_hh_r))  if uc_hh_r  else float("nan"),
            "input_cost_human_low_routine":    float(np.mean(uc_hl_r))  if uc_hl_r  else float("nan"),
            "input_cost_human_high_nonroutine":float(np.mean(uc_hh_nr)) if uc_hh_nr else float("nan"),
            "input_cost_human_low_nonroutine": float(np.mean(uc_hl_nr)) if uc_hl_nr else float("nan"),
            # Employment-protection stats (nan when disabled)
            "share_vast_high":       (vast_high / max(1, n_high_employed)) if self.employment_protection else float("nan"),
            "share_vast_low":        (vast_low  / max(1, n_low_employed))  if self.employment_protection else float("nan"),
            "share_flex_high":       (flex_high / max(1, n_high_employed)) if self.employment_protection else float("nan"),
            "share_flex_low":        (flex_low  / max(1, n_low_employed))  if self.employment_protection else float("nan"),
            "avg_tenure_years":      float(np.mean(tenure_all))  if tenure_all  else float("nan"),
            "avg_tenure_vast_years": float(np.mean(tenure_vast)) if tenure_vast else float("nan"),
            "avg_tenure_flex_years": float(np.mean(tenure_flex)) if tenure_flex else float("nan"),
            "avg_severance_per_task":float(np.mean(severance_per_task)) if severance_per_task else float("nan"),
        }

        if is_npv:
            self._npv_stats = {
                "avg_npv_routine":                float(np.mean(npv_r))         if npv_r        else float("nan"),
                "avg_npv_nonroutine":             float(np.mean(npv_nr))        if npv_nr       else float("nan"),
                "share_adoptable_npv_routine":    float(np.mean(adopt_r))       if adopt_r      else float("nan"),
                "share_adoptable_npv_nonroutine": float(np.mean(adopt_nr))      if adopt_nr     else float("nan"),
                "avg_realized_npv_this_step":     float(np.mean(realized_this)) if realized_this else float("nan"),
                "avg_realized_npv_cumulative":    float(np.mean(realized_all))  if realized_all  else float("nan"),
            }
        else:
            self._npv_stats = {k: float("nan") for k in (
                "avg_npv_routine", "avg_npv_nonroutine",
                "share_adoptable_npv_routine", "share_adoptable_npv_nonroutine",
                "avg_realized_npv_this_step", "avg_realized_npv_cumulative",
            )}

    def _collect_initial_state(self):
        """
        Record a true t=0 state after warm-start but before the first model step.

        By this point, workers have already been matched and wages have already
        been set directly to the level implied by the warm-start allocation.
        This t=0 record therefore reflects the intended initial economy, while
        later steps again use lam-driven wage adjustment.
        """
        for p in self.producers:
            p.produce()
            p.trim_ai_capacity()
        self.p = self.calculate_market_price()
        self._compute_all_stats()
        self.datacollector.collect(self)

    # ----- [C3 Wage-setting] Wage-curve supply & wage bill -----

    def calculate_wage_bill(self) -> float:
        """
        Aggregate labour income of all currently employed workers.
        Called at the START of each step (before any wage / employment
        updates this step) so that the value captured represents the
        end-of-previous-step wage bill -- the "lagged" wage bill Y_{t-1}
        used in the Keynesian demand equation this step.
        """
        return sum(w.wage for w in self.workers if w.employed)

    def update_wages(self, instant: bool = False):
        """
        Update market wages for each skill group using an upward-sloping wage curve:
            w_target = a + b * L
        Wages adjust gradually toward the target via partial adjustment speed lam.
        When instant=True, wages jump directly to the target. This is used during
        warm-start so the initial economy begins at the wage level implied by the
        initial worker allocation.
        """
        Lh = sum(1 for w in self.workers if w.employed and w.skill_level == "high")
        Ll = sum(1 for w in self.workers if w.employed and w.skill_level == "low")

        # Explicit minimum wage floor: w_min acts like the Dutch minimumloon.
        # When w_min=0 (default) this has no effect - the natural floor is a_h / a_l.
        target_h = max(self.w_min, self.a_h + self.b_h * Lh)
        target_l = max(self.w_min, self.a_l + self.b_l * Ll)

        if instant:
            self.w_h = target_h
            self.w_l = target_l
        else:
            self.w_h = max(self.w_min, self.lam * target_h + (1 - self.lam) * self.w_h)
            self.w_l = max(self.w_min, self.lam * target_l + (1 - self.lam) * self.w_l)

    # ----- [C6 Demand] Goods market & Kaleckian two-class consumption -----

    def calculate_profit_income(self) -> float:
        """
        Aggregate operating profit income across all producers:
            Pi = sum_i (P - AC_i) * Q_i = sum_i (P * Q_i - total_costs_i)

        This is the capital-share counterpart to calculate_wage_bill() — together
        they form the two factor-income components that drive demand in the
        Kaleckian two-class consumption function:
            A_t = A_base + gamma * W_{t-1} + gamma_pi * Pi_{t-1}

        Operating profit excludes upfront AI investment cost and severance
        payments (one-off cash flows), matching the convention used by the
        'Avg profit per producer' diagnostic. Called at the START of each step
        so the value captured represents the end-of-previous-step profit
        income — the lagged Pi_{t-1} that enters the demand equation this step.

        Rationale (Servaas Storm, supervisor feedback, May 2026):
        AI automation redistributes income from wages to profits. With only
        wage income feeding demand, every euro of automation-driven wage
        savings vanishes from aggregate demand, generating an underconsumption
        spiral (P falls below AC, profits turn negative). Adding the profit-
        income channel closes the income identity W + Pi = P*Q and prevents
        the spurious "missing demand" effect, while gamma > gamma_pi preserves
        the empirical wage-led tilt of the Dutch demand regime
        (Storm & Naastepad 2012; Onaran & Galanis 2014).
        """
        # P is self.p (set at end of previous step in step() Step 8 / collect_initial_state).
        # output and total_costs reflect end-of-previous-step state.
        return sum(p.output * self.p - p.total_costs for p in self.producers)

    def calculate_market_price(self) -> float:
        """
        Inverse demand: p = A - B*Q.

        A is updated via the Kaleckian two-class consumption function:
            A = A_base + gamma * Y + gamma_pi * Pi
        where
            Y  = lagged wage bill                  (labour-income channel)
            Pi = lagged aggregate operating profit (capital-income channel)
        gamma and gamma_pi are the marginal propensities to consume out of
        wage and profit income respectively. Empirically gamma > gamma_pi:
        capitalists save more than workers, so income redistribution from
        labour to capital still reduces aggregate demand — but by less than
        in the pure wage-led specification (gamma_pi = 0), and crucially the
        accounting identity W + Pi = P*Q is now respected.
        """
        Q = sum(p.output for p in self.producers)
        self.A = self.A_base + self.gamma * self.Y + self.gamma_pi * self.Pi
        return max(0.01, self.A - self.B * Q)

    # ----- [C7 Employment Protection] Ketenregeling (chain rule) -----

    def process_chain_limit(self):
        """
        Ketenregeling (chain clause): flex workers whose tenure reaches chain_limit
        are either converted to a permanent ('vast') contract or not renewed.

        Non-renewal procedure:
          - The flex worker is separated from their employer without triggering
            on_worker_separated(), so NO automation evaluation occurs.
          - A plain hiring request for a new flex worker is queued directly on
            the firm's _pending_requests, picked up by match_and_hire() later.
          - If no unemployed workers are available as replacements, the worker
            is converted instead (the firm cannot leave the task empty).

        Conversion: contract_type flips to 'vast'; worker stays in place.

        Does nothing when employment_protection=False.
        """
        if not self.employment_protection:
            return

        # Count available unemployed workers for replacement decisions.
        n_unemployed = sum(1 for w in self.workers if not w.employed)

        for w in list(self.workers):
            if not w.employed or w.contract_type != "flex":
                continue
            if w.tenure < self.chain_limit:
                continue

            # This flex worker has reached the chain limit.
            if self.random.random() < self.p_convert:
                # Conversion: worker stays, contract flips to permanent.
                w.contract_type = "vast"
                self._conversions_this_step += 1
                continue

            # Non-renewal desired — only possible if a replacement can be found.
            if n_unemployed <= 0:
                # No replacement available: forced conversion.
                w.contract_type = "vast"
                self._conversions_this_step += 1
                continue

            # Non-renewal: separate worker without automation callback.
            task     = w.task
            employer = w.employer

            if task is not None and w in task.employees:
                task.employees.remove(w)
            if employer is not None and w in employer.employees:
                employer.employees.remove(w)
            w.employed = False
            w.employer = None
            w.task     = None
            w.wage     = 0.0
            w.tenure   = 0

            n_unemployed -= 1
            self._non_renewals_this_step += 1

            # Queue a plain hiring request for a new flex worker (no automation eval).
            if task is not None and employer is not None:
                best_skill = employer.cheapest_human_for_task(task)
                employer._pending_requests.append((employer, task, best_skill))

    # ----- [C2 Labour Market] Separations, warm-start & matching -----

    def separate_workers(self):
        """
        Exogenous job separation: each employed worker faces probability delta of being laid off.
        After separation, the firm immediately decides what to do with the empty task slot
        via on_worker_separated() (which may trigger automation or queue a hiring request).
        """
        for w in self.workers:
            if w.employed and self.random.random() < self.delta:
                task = w.task
                employer = w.employer

                # Decouple worker from task and employer
                if task is not None and w in task.employees:
                    task.employees.remove(w)
                if employer is not None and w in employer.employees:
                    employer.employees.remove(w)
                w.employed = False
                w.employer = None
                w.task = None
                w.wage = 0.0

                # Callback: firm decides what to do with the empty task slot.
                # Pass w so n_ai sizing can use the departed worker's productivity.
                if task is not None and employer is not None:
                    employer.on_worker_separated(task, w)

    def _warm_start(self):
        """
        Distribute workers across firms so that each task has exactly one worker at t=0.

        Uses the same economic preference rule as the ongoing matching: each task is
        assigned its cheapest skill type first (via cheapest_human_for_task), with
        unconditional fallback to the other pool when the preferred pool is exhausted.
        No theta check at warm-start - we always want all tasks filled at t=0.

        Wages and the initial wage bill are set consistently with initial employment.

        Isolation: all shuffles and tenure draws inside _warm_start use a dedicated
        RNG seeded from self._seed so that changing any model parameter (e.g. w_min)
        does not shift the warm-start random state and confound experiment comparisons.
        """
        _ws_rng = _stdlib_random.Random(self._seed)

        available_high = [w for w in self.workers if w.skill_level == "high"]
        available_low  = [w for w in self.workers if w.skill_level == "low"]
        _ws_rng.shuffle(available_high)
        _ws_rng.shuffle(available_low)

        for p in self.producers:
            for t in p.tasks:
                preferred = p.cheapest_human_for_task(t)
                if preferred == "high":
                    pool, fallback_pool = available_high, available_low
                else:
                    pool, fallback_pool = available_low, available_high

                if pool:
                    w = pool.pop()
                elif fallback_pool:
                    w = fallback_pool.pop()
                else:
                    continue
                self.hire_worker_to_task(w, p, t)

        # Set wages directly to the level implied by the warm-start allocation.
        # Only from the first actual model step onward do we use partial
        # adjustment via lam.
        self.update_wages(instant=True)
        for w in self.workers:
            if w.employed:
                w.wage = self.w_h if w.skill_level == "high" else self.w_l
        self.w_h_prev = self.w_h
        self.w_l_prev = self.w_l
        self.Y = self.calculate_wage_bill()

        # Initialise tenure and contract types for employed workers at warm-start
        if self.employment_protection:
            employed = [w for w in self.workers if w.employed]
            n_vast = int(self.init_share_vast * len(employed))
            _ws_rng.shuffle(employed)
            for i, w in enumerate(employed):
                if i < n_vast:
                    w.contract_type = "vast"
                    min_tenure = self.chain_limit
                    max_tenure = int(self.init_tenure_max_years * self.steps_per_year)
                    w.tenure = _ws_rng.randint(min_tenure, max(min_tenure, max_tenure))
                else:
                    w.contract_type = "flex"
                    w.tenure = _ws_rng.randint(0, max(1, self.chain_limit - 1))

    def match_and_hire(self):
        """
        Collect hiring requests from all firms and match them to unemployed workers.
        Uses two-pass matching to allow a controlled cross-skill fallback:

        Pass 1 - preferred skill:
            Each request is matched to an unemployed worker of the requested skill type.
            Unmatched requests are carried forward.

        Pass 2 - cross-skill fallback bounded by mismatch_theta:
            For each still-unmatched request, the alternative skill pool is tried.
            A fallback worker is hired only if:
                unit_cost_fallback <= mismatch_theta * unit_cost_preferred
            This preserves skill segmentation while preventing Leontief output losses
            caused purely by matching rigidity when the preferred pool is temporarily empty.
            mismatch_theta = 1.0 disables fallback; large values approach free substitution.
        """
        all_requests = []
        for p in self.producers:
            all_requests.extend(p.hiring_requests())

        unemployed_high = [w for w in self.workers if (not w.employed) and w.skill_level == "high"]
        unemployed_low  = [w for w in self.workers if (not w.employed) and w.skill_level == "low"]

        self.random.shuffle(all_requests)

        # Pass 1: preferred skill
        unmatched = []
        for producer, task, req_type in all_requests:
            if req_type == "high" and unemployed_high:
                w = unemployed_high.pop()
                self.hire_worker_to_task(w, producer, task)
            elif req_type == "low" and unemployed_low:
                w = unemployed_low.pop()
                self.hire_worker_to_task(w, producer, task)
            else:
                unmatched.append((producer, task, req_type))

        # Pass 2: cross-skill fallback (economically bounded by mismatch_theta)
        for producer, task, req_type in unmatched:
            fallback = "low" if req_type == "high" else "high"
            fallback_pool = unemployed_low if fallback == "low" else unemployed_high
            if not fallback_pool:
                continue
            # Economic acceptability check: mismatch cost must not exceed theta x preferred cost
            w_pref = self.w_h if req_type == "high" else self.w_l
            w_fall = self.w_l if fallback == "low" else self.w_h
            uc_pref = producer.unit_cost_human(task, w_pref, req_type)
            uc_fall = producer.unit_cost_human(task, w_fall, fallback)
            if uc_fall <= self.mismatch_theta * uc_pref:
                w = fallback_pool.pop()
                self.hire_worker_to_task(w, producer, task)

    def hire_worker_to_task(self, w: Worker, p: Producer, task: Task):
        """
        Place worker w on task within firm p.
        If the task was automated, de-automate it first (AI is replaced by the worker).
        In NPV modes, de-automation should only happen after investment horizon expiry.
        """
        if task.automated:
            # When ai_irreversible=True, de-automation is only allowed after the
            # investment horizon expires (or if investment_step was never recorded).
            if self.ai_irreversible and not (
                task.ai_investment_step is None
                or self.current_step - task.ai_investment_step >= self.T_horizon
            ):
                raise RuntimeError(
                    f"Bug: de-automating task {task.task_id} before investment horizon expired"
                )
            task.automated = False
            task.n_ai = 0
            task.ai_investment_step = None
            task.ai_original_step = None
            task.ai_investment_npv = None

        w.employed = True
        w.employer = p
        w.wage = self.w_h if w.skill_level == "high" else self.w_l
        task.employees.append(w)
        w.task = task
        if w not in p.employees:
            p.employees.append(w)
        # New hires always start as flex with zero tenure when employment protection is on
        if self.employment_protection:
            w.contract_type = "flex"
            w.tenure = 0

    # ----- [M1 Macro-feedback] Simulation loop: wires C1-C7 into one step -----

    def step(self):
        """
        One model step (M1 macro-feedback). The sub-steps run in a fixed
        order; each is tagged with the component (C1-C7) whose mechanism it
        triggers. Some sub-steps call a dedicated method, others are short
        inline blocks - M1 is the wiring, not a component in its own right.
           0  [M1/C5] Advance the step counter, reset per-step counters, and
                      advance the AI learning curve k_ai(t) via ai_cost_at().
           1  [C6]    Capture the lagged wage bill Y and lagged profit income
                      Pi that enter the demand equation this step.
           2  [C1]    Produce output at all firms; trim AI capacity.
           3  [C6]    Set the market price via inverse demand
                      p = max(0.01, A - B*Q), with A = A_base + gamma*Y
                      + gamma_pi*Pi (Kaleckian two-class consumption).
           4  [C7]    Ketenregeling: convert or non-renew flex workers whose
                      tenure reached chain_limit (skipped when protection off).
           5  [C2]    Exogenous separations at rate delta, with reactive
                      automation callbacks via on_worker_separated().
           6  [C3]    Update wage targets and apply partial adjustment.
           7  [C3]    Update adaptive wage expectations (smoothed dw_h, dw_l).
           8  [C2/C4] Match unemployed workers to hiring requests (two passes).
                      hiring_requests() also carries the proactive automation
                      channel, gated by should_produce_more().
           9  [C7]    Increment tenure for all currently employed workers.
          10  [C4]    Update the smoothed displacement flow (mean-field NPV).
          11  [C3]    Propagate updated wage levels to all worker objects.
          12  [C1/C6] Recompute output and price for end-of-step consistency.
          13  [M1]    Compute aggregate statistics and collect data.
        """
        # Step 0 [M1]: advance step counter, reset per-step counters
        self.current_step += 1
        self.new_reactive_automations_this_step = 0
        self.new_proactive_automations_this_step = 0
        self.new_displaced_workers_this_step = 0
        self._conversions_this_step = 0
        self._non_renewals_this_step = 0
        self.n_gate_blocks_this_step = 0
        self.n_human_cheaper_tasks_this_step = 0
        # Step 0 [C5]: advance the AI learning curve (active in all modes)
        self.k_ai = self.ai_cost_at(self.current_step)

        # Step 1 [C6]: lagged factor incomes used in Kaleckian demand this step.
        # Pi MUST be captured BEFORE produce() is called so it reflects
        # end-of-previous-step revenue and costs (one-period lag, matching Y).
        self.Y  = self.calculate_wage_bill()
        self.Pi = self.calculate_profit_income()

        # Steps 2-3 [C1 produce / C6 price]: production and pricing
        for p in self.producers:
            p.produce()
            p.trim_ai_capacity()   # right-size AI units before hiring decisions
        self.p = self.calculate_market_price()

        # Steps 4-8 [C7 chain rule / C2 separations / C3 wages / C2-C4 matching]
        self.process_chain_limit()   # ketenregeling before exogenous separations
        self.separate_workers()
        self.update_wages()
        # Adaptive expectations: smooth the wage RATE OF CHANGE
        dw_h_now = self.w_h - self.w_h_prev
        dw_l_now = self.w_l - self.w_l_prev
        self.dw_h_smoothed = self.omega_adapt * dw_h_now + (1 - self.omega_adapt) * self.dw_h_smoothed
        self.dw_l_smoothed = self.omega_adapt * dw_l_now + (1 - self.omega_adapt) * self.dw_l_smoothed
        self.w_h_prev = self.w_h
        self.w_l_prev = self.w_l
        self.match_and_hire()

        # Increment tenure for all currently employed workers
        if self.employment_protection:
            for w in self.workers:
                if w.employed:
                    w.tenure += 1

        # Track worker displacement flow for mean-field wage expectations.
        self.displacement_flow_smoothed = (
            self.omega_adapt * self.new_displaced_workers_this_step
            + (1 - self.omega_adapt) * self.displacement_flow_smoothed
        )

        # Step 11 [C3]: propagate updated wage levels to worker objects
        for w in self.workers:
            if w.employed:
                w.wage = self.w_h if w.skill_level == "high" else self.w_l

        # Step 12 [C1/C6]: re-run production and pricing so collected data is end-of-step consistent
        for p in self.producers:
            p.produce()
            p.trim_ai_capacity()   # keep AI capacity consistent in reported data
        self.p = self.calculate_market_price()

        # Step 13 [M1]: collect data
        self._compute_all_stats()   # single combined pass before data collection
        self.datacollector.collect(self)
