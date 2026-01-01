#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 23:20:21 2026

Withdrawal Strategy Test Rig (67..95) — 2 assets (Safe + Equity)

Implements:
A) Withdrawal rules incl. Guardrails (simplified Guyton-style)
B) Minimal tax logic: tax on realized gains portion of withdrawals (cost basis tracking)
C) DIN 77230 "input mask" checklist dataclasses: completeness warnings, not a corset

Model (intentionally simple, but auditable):
- Monthly correlated log-returns for Safe & Equity
- Monthly rebalancing to target mix
- Monthly fee drag
- Withdrawals applied end-of-month
- If withdrawal_is_real: maintain purchasing power unless adjusted by guardrails
- Ruin: wealth <= 0

Outputs per mix:
- success_prob (funding to age_end)
- ruin_rate, median ruin month (if any)
- terminal wealth quantiles (real terms)
- max drawdown quantiles (real terms)
- effective average tax paid (optional)

Notes:
- "median_annual" refers to median SIMPLE return; converted to monthly log-median.
- vol_annual interpreted as LOG-return volatility (approx). Keep consistent.
- Tax model: only on taxable fraction of account, only on realized gains at withdrawal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Callable, Literal
import json, math, hashlib, datetime as dt

import numpy as np
import pandas as pd


# -------------------------
# Reproducibility helpers
# -------------------------

def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def sha256_of_dict(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# -------------------------
# C) DIN 77230-ish input mask (freedom-enabling, not a corset)
# -------------------------

@dataclass
class DIN77230Profile:
    """
    Minimal structure that helps not forgetting important fields.
    You can extend this to match your DIN workflow.
    """
    # Identity & horizon
    age: Optional[int] = None
    retirement_age: int = 67
    life_expectancy_age: int = 95

    # Household context
    household_size: Optional[int] = None
    dependents: Optional[int] = None

    # Income streams (monthly, nominal)
    statutory_pension_monthly: float = 0.0
    occupational_pension_monthly: float = 0.0
    other_secure_income_monthly: float = 0.0  # rents, annuities etc.

    # Expenses (monthly, nominal)
    essential_expenses_monthly: Optional[float] = None
    discretionary_expenses_monthly: Optional[float] = None

    # Liquidity & safety
    liquidity_buffer_months: Optional[float] = None
    insured_health: Optional[bool] = None
    insured_liability: Optional[bool] = None
    long_term_care_plan: Optional[str] = None  # "insured", "self-funded", "unknown"

    # Assets & liabilities (high level)
    investable_assets: Optional[float] = None
    real_estate_value: Optional[float] = None
    mortgage_debt: Optional[float] = None
    other_debt: Optional[float] = None

    # Values & constraints (Guide-first)
    life_priorities: Tuple[str, ...] = ()
    nonnegotiables: Tuple[str, ...] = ()
    ethical_exclusions: Tuple[str, ...] = ()
    willingness_to_adjust_lifestyle: Optional[str] = None  # "high/medium/low/unknown"

    def missing_fields(self) -> List[str]:
        required = [
            ("age", self.age),
            ("household_size", self.household_size),
            ("dependents", self.dependents),
            ("essential_expenses_monthly", self.essential_expenses_monthly),
            ("discretionary_expenses_monthly", self.discretionary_expenses_monthly),
            ("liquidity_buffer_months", self.liquidity_buffer_months),
            ("insured_health", self.insured_health),
            ("insured_liability", self.insured_liability),
            ("long_term_care_plan", self.long_term_care_plan),
            ("investable_assets", self.investable_assets),
        ]
        missing = [name for name, val in required if val is None]
        return missing

    def guide_prompts(self) -> List[str]:
        # These are deliberately provocative; you can change tone.
        return [
            "Welches Leben willst du als Rentner führen – und welches nicht?",
            "Willst du den Lebensstandard halten, senken oder bewusst steigern? Warum?",
            "Welche Ausgaben sind Ausdruck von Würde und welche sind Gewohnheit?",
            "Was ist dein Exit-Plan, wenn Märkte 2 Jahre schlecht laufen?",
            "Welche Risiken willst du NICHT versichern, sondern tragen – und warum?"
        ]


# -------------------------
# A) Withdrawal rules
# -------------------------

WithdrawalMode = Literal["fixed_nominal", "fixed_real", "guardrails"]

@dataclass(frozen=True)
class GuardrailsRule:
    """
    Simplified guardrails (Guyton-style) evaluated annually.

    Let W_real be current monthly withdrawal in real terms (year_0 purchasing power).
    Compute current withdrawal rate each anniversary:
        rate = (W_real * 12) / current_real_wealth

    Let initial_rate = (W_real_initial * 12) / initial_real_wealth

    Guardrails:
        upper = initial_rate * (1 + upper_band)
        lower = initial_rate * (1 - lower_band)

    If rate > upper: cut W_real by cut_pct
    If rate < lower: raise W_real by raise_pct

    Additionally clamp W_real to [floor_mult, ceiling_mult] * W_real_initial
    """
    upper_band: float = 0.20    # 20% above initial rate
    lower_band: float = 0.20    # 20% below initial rate
    cut_pct: float = 0.10       # cut by 10%
    raise_pct: float = 0.10     # raise by 10%
    floor_mult: float = 0.70    # never go below 70% of initial real withdrawal
    ceiling_mult: float = 1.30  # never go above 130% of initial real withdrawal
    eval_every_months: int = 12 # annual review


# -------------------------
# B) Minimal tax logic
# -------------------------

@dataclass(frozen=True)
class TaxModel:
    """
    Minimal taxable account model with cost basis tracking.

    - taxable_fraction: share of portfolio subject to this tax logic (0..1)
      (e.g., part in taxable brokerage vs. pension wrapper)
    - cap_gains_tax_rate: effective tax on realized gains at withdrawal (0..1)
    - ignore_tax_loss_harvest: True => gains portion cannot be negative for tax
    """
    taxable_fraction: float = 1.0
    cap_gains_tax_rate: float = 0.25
    ignore_tax_loss_harvest: bool = True


# -------------------------
# Simulation inputs
# -------------------------

@dataclass(frozen=True)
class WithdrawalPlan:
    age_start: int = 67
    age_end: int = 95

    # Desired monthly withdrawal from portfolio (currency). Interpretation depends on mode:
    # - fixed_real: this is real (year_0 purchasing power)
    # - fixed_nominal: this is nominal constant
    # - guardrails: this is initial real withdrawal (year_0 purchasing power)
    monthly_withdrawal: float = 2500.0

    withdrawal_mode: WithdrawalMode = "guardrails"
    guardrails: GuardrailsRule = GuardrailsRule()

    inflation_annual: float = 0.02
    annual_fee_rate: float = 0.002

    start_wealth: float = 500_000.0

    # Simulation controls
    n_sims: int = 50_000
    seed: int = 42

    @property
    def months(self) -> int:
        return (self.age_end - self.age_start) * 12  # 28*12=336


@dataclass(frozen=True)
class TwoAssetParams:
    # Annual median SIMPLE returns
    median_safe: float = 0.02
    median_equity: float = 0.06

    # Annual volatilities of LOG returns (approx)
    vol_safe: float = 0.01
    vol_equity: float = 0.18

    # Dependence
    corr: float = 0.0


@dataclass(frozen=True)
class Mix:
    name: str
    w_equity: float  # 0..1


# -------------------------
# Return conversion utilities
# -------------------------

def annual_median_simple_to_monthly_log_mu(median_annual_simple: float) -> float:
    gross = 1.0 + median_annual_simple
    if gross <= 0:
        raise ValueError("Median annual gross return must be > 0.")
    mu_annual_log = math.log(gross)  # median of lognormal
    return mu_annual_log / 12.0

def annual_log_vol_to_monthly_log_sigma(vol_annual_log: float) -> float:
    if vol_annual_log < 0:
        raise ValueError("Volatility must be non-negative.")
    return vol_annual_log / math.sqrt(12.0)

def monthly_inflation_from_annual(infl_annual: float) -> float:
    return (1.0 + infl_annual) ** (1.0 / 12.0) - 1.0


# -------------------------
# Core simulation
# -------------------------

def correlated_monthly_log_returns(
    rng: np.random.Generator,
    plan: WithdrawalPlan,
    params: TwoAssetParams,
) -> Tuple[np.ndarray, np.ndarray]:
    n, T = plan.n_sims, plan.months

    mu_s = annual_median_simple_to_monthly_log_mu(params.median_safe)
    mu_e = annual_median_simple_to_monthly_log_mu(params.median_equity)
    sig_s = annual_log_vol_to_monthly_log_sigma(params.vol_safe)
    sig_e = annual_log_vol_to_monthly_log_sigma(params.vol_equity)

    rho = float(params.corr)
    rho = max(min(rho, 0.999), -0.999)

    cov = np.array([
        [sig_s**2, rho*sig_s*sig_e],
        [rho*sig_s*sig_e, sig_e**2]
    ])
    mean = np.array([mu_s, mu_e])

    z = rng.multivariate_normal(mean=mean, cov=cov, size=(n, T))
    safe_log = z[:, :, 0]
    equity_log = z[:, :, 1]
    return safe_log, equity_log

def compute_max_drawdown(paths: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(paths, axis=1)
    dd = (paths - peak) / np.where(peak == 0, np.nan, peak)
    max_dd = np.nanmin(dd, axis=1)
    return max_dd

def simulate_withdrawal(
    plan: WithdrawalPlan,
    params: TwoAssetParams,
    mix: Mix,
    tax: TaxModel,
) -> Dict[str, float]:
    """
    Vectorized across sims, loop over months (336).
    Tracks:
    - wealth (nominal)
    - cost basis for taxable fraction (nominal)
    - withdrawal rule (fixed / guardrails)
    """
    rng = np.random.default_rng(plan.seed)
    n, T = plan.n_sims, plan.months

    w_e = float(mix.w_equity)
    w_s = 1.0 - w_e

    safe_log, equity_log = correlated_monthly_log_returns(rng, plan, params)
    safe_gross = np.exp(safe_log)
    equity_gross = np.exp(equity_log)

    # fees applied monthly (approx)
    fee_m = 1.0 - (plan.annual_fee_rate / 12.0)

    # inflation index for real/nominal conversion
    infl_m = monthly_inflation_from_annual(plan.inflation_annual)

    wealth = np.full((n,), plan.start_wealth, dtype=float)

    # Track a real-terms shadow wealth for guardrails decisions (deflate)
    # We'll maintain a running nominal->real conversion factor.
    infl_factor = 1.0

    # Taxable basis tracking for the taxable fraction of the portfolio
    taxable_frac = float(np.clip(tax.taxable_fraction, 0.0, 1.0))
    basis = np.full((n,), plan.start_wealth * taxable_frac, dtype=float)  # cost basis (nominal)

    # Withdrawal level in REAL terms (year_0 purchasing power)
    W0_real = float(plan.monthly_withdrawal)
    W_real = np.full((n,), W0_real, dtype=float)

    # Guardrails initial rate (computed on initial real wealth)
    initial_real_wealth = plan.start_wealth  # at t=0, nominal==real
    initial_rate = (W0_real * 12.0) / max(initial_real_wealth, 1e-9)

    ruined = np.zeros((n,), dtype=bool)
    ruin_month = np.full((n,), np.nan)
    taxes_paid_total = np.zeros((n,), dtype=float)

    # For max drawdown in REAL terms, store path (real)
    # memory: n_sims x (T+1). For very large n_sims, this is heavy.
    # Keep it optional by computing with a thinner approach: store monthly real wealth for drawdown.
    real_path = np.zeros((n, T + 1), dtype=float)
    real_path[:, 0] = wealth  # t=0 real

    def clamp_w_real(x: np.ndarray) -> np.ndarray:
        g = plan.guardrails
        lo = g.floor_mult * W0_real
        hi = g.ceiling_mult * W0_real
        return np.clip(x, lo, hi)

    for t in range(T):
        # Apply portfolio return (rebalance each month by using weighted gross)
        port_gross = w_s * safe_gross[:, t] + w_e * equity_gross[:, t]
        wealth = wealth * port_gross

        # Apply fees
        wealth = wealth * fee_m

        # Update inflation factor (end of month)
        if t > 0 or True:
            infl_factor *= (1.0 + infl_m)

        # Determine nominal withdrawal for this month
        if plan.withdrawal_mode == "fixed_nominal":
            W_nominal = np.full((n,), plan.monthly_withdrawal, dtype=float)
        elif plan.withdrawal_mode == "fixed_real":
            W_nominal = np.full((n,), plan.monthly_withdrawal * infl_factor, dtype=float)
        elif plan.withdrawal_mode == "guardrails":
            # Nominal withdrawal is current real W_real indexed by inflation
            W_nominal = W_real * infl_factor
        else:
            raise ValueError(f"Unknown withdrawal_mode: {plan.withdrawal_mode}")

        # --- Taxes on withdrawal (minimal cost-basis logic) ---
        # Withdrawal draws proportionally from taxable part (taxable_frac of wealth).
        # Realized gains fraction in taxable account = max(wealth_taxable - basis, 0)/wealth_taxable
        wealth_taxable = wealth * taxable_frac
        # Protect from division by zero
        gains = wealth_taxable - basis
        if tax.ignore_tax_loss_harvest:
            gains = np.maximum(gains, 0.0)
        gains_frac = np.where(wealth_taxable > 1e-12, gains / wealth_taxable, 0.0)

        W_taxable = W_nominal * taxable_frac
        realized_gains = W_taxable * gains_frac
        tax_due = realized_gains * tax.cap_gains_tax_rate

        taxes_paid_total += np.where(~ruined, tax_due, 0.0)

        # Reduce basis by principal portion of taxable withdrawal
        principal_withdrawn = W_taxable - realized_gains
        basis = basis - np.where(~ruined, principal_withdrawn, 0.0)
        basis = np.maximum(basis, 0.0)

        # Apply withdrawal + tax from wealth
        wealth = wealth - W_nominal - tax_due

        # Mark ruin
        just_ruined = (~ruined) & (wealth <= 0.0)
        ruined[just_ruined] = True
        ruin_month[just_ruined] = t + 1
        wealth = np.maximum(wealth, 0.0)

        # Real wealth for drawdown & guardrail checks
        real_wealth = wealth / infl_factor
        real_path[:, t + 1] = real_wealth

        # --- Guardrails annual evaluation ---
        if plan.withdrawal_mode == "guardrails":
            g = plan.guardrails
            if (t + 1) % g.eval_every_months == 0:
                # compute current withdrawal rate using current real wealth
                # rate = annual real withdrawal / current real wealth
                denom = np.maximum(real_wealth, 1e-9)
                rate = (W_real * 12.0) / denom

                upper = initial_rate * (1.0 + g.upper_band)
                lower = initial_rate * (1.0 - g.lower_band)

                # Apply cuts/raises only for those not ruined
                adj = np.ones((n,), dtype=float)
                adj = np.where((~ruined) & (rate > upper), 1.0 - g.cut_pct, adj)
                adj = np.where((~ruined) & (rate < lower), 1.0 + g.raise_pct, adj)

                W_real = W_real * adj
                W_real = clamp_w_real(W_real)

    success = ~ruined
    success_prob = float(np.mean(success))
    ruin_rate = float(np.mean(ruined))
    ruin_median = float(np.nanmedian(ruin_month)) if np.any(ruined) else float("nan")

    terminal_real = real_path[:, -1]
    q05, q50, q95 = np.quantile(terminal_real, [0.05, 0.50, 0.95])

    max_dd = compute_max_drawdown(real_path)  # negative values
    dd_q05, dd_q50 = np.quantile(max_dd, [0.05, 0.50])

    avg_tax_paid = float(np.mean(taxes_paid_total))
    avg_tax_paid_success = float(np.mean(taxes_paid_total[success])) if np.any(success) else float("nan")

    return {
        "success_prob": success_prob,
        "ruin_rate": ruin_rate,
        "ruin_median_month": ruin_median,
        "terminal_real_q05": float(q05),
        "terminal_real_q50": float(q50),
        "terminal_real_q95": float(q95),
        "max_drawdown_q05": float(dd_q05),
        "max_drawdown_median": float(dd_q50),
        "avg_tax_paid_total": avg_tax_paid,
        "avg_tax_paid_total_success": avg_tax_paid_success,
    }

def run_grid(
    plan: WithdrawalPlan,
    params: TwoAssetParams,
    mixes: List[Mix],
    tax: TaxModel,
) -> pd.DataFrame:
    rows = []
    for m in mixes:
        metrics = simulate_withdrawal(plan, params, m, tax)
        rows.append({"mix": m.name, "w_equity": m.w_equity, **metrics})
    return pd.DataFrame(rows).sort_values("w_equity")


# -------------------------
# Convenience: scenario builder
# -------------------------

def default_mixes() -> List[Mix]:
    return [
        Mix("100% Equity", 1.00),
        Mix("80/20", 0.80),
        Mix("60/40", 0.60),
        Mix("40/60", 0.40),
        Mix("20/80", 0.20),
        Mix("0% Equity", 0.00),
    ]

def print_din_check(profile: DIN77230Profile) -> None:
    missing = profile.missing_fields()
    print("\nDIN-ish completeness check")
    print("-------------------------")
    if missing:
        print("Missing fields (fill them or justify why not):")
        for f in missing:
            print(" -", f)
    else:
        print("Core fields present.")
    print("\nGuide prompts (decide before you model):")
    for q in profile.guide_prompts():
        print(" *", q)


# -------------------------
# Example run (edit these)
# -------------------------

def main():
    # C) DIN-ish input mask example:
    profile = DIN77230Profile(
        age=67,
        household_size=1,
        dependents=0,
        statutory_pension_monthly=1800.0,
        occupational_pension_monthly=300.0,
        other_secure_income_monthly=0.0,
        essential_expenses_monthly=2200.0,
        discretionary_expenses_monthly=800.0,
        liquidity_buffer_months=6,
        insured_health=True,
        insured_liability=True,
        long_term_care_plan="unknown",
        investable_assets=650_000.0,
        life_priorities=("Zeit", "Gesundheit", "Autonomie"),
        nonnegotiables=("keine existenzielle Angst", "Wohnsicherheit"),
        willingness_to_adjust_lifestyle="medium",
    )
    print_din_check(profile)

    # The *portfolio withdrawal* you want to test should be net of secure incomes.
    # Example: you want 3,000/month total; secure incomes cover 2,100 => from portfolio: 900/month (real).
    desired_total = 3000.0
    secure_income = profile.statutory_pension_monthly + profile.occupational_pension_monthly + profile.other_secure_income_monthly
    portfolio_needed = max(desired_total - secure_income, 0.0)

    # A) Withdrawal plan with Guardrails (real)
    plan = WithdrawalPlan(
        age_start=67,
        age_end=95,
        monthly_withdrawal=portfolio_needed,         # initial real withdrawal from portfolio
        withdrawal_mode="guardrails",
        guardrails=GuardrailsRule(
            upper_band=0.20,
            lower_band=0.20,
            cut_pct=0.10,
            raise_pct=0.10,
            floor_mult=0.75,
            ceiling_mult=1.25,
            eval_every_months=12
        ),
        inflation_annual=0.02,
        annual_fee_rate=0.003,
        start_wealth=float(profile.investable_assets or 0.0),
        n_sims=40_000,
        seed=7
    )

    # Scenario set: choose medians/vols/corr
    params = TwoAssetParams(
        median_safe=0.02,
        median_equity=0.06,
        vol_safe=0.01,
        vol_equity=0.18,
        corr=0.10
    )

    # B) Minimal tax model
    tax = TaxModel(
        taxable_fraction=0.7,        # e.g. 70% taxable brokerage, 30% wrapper
        cap_gains_tax_rate=0.25
    )

    mixes = default_mixes()
    df = run_grid(plan, params, mixes, tax)

    meta = {
        "timestamp": now_iso(),
        "profile": asdict(profile),
        "plan": asdict(plan),
        "params": asdict(params),
        "tax": asdict(tax),
        "mixes": [asdict(m) for m in mixes]
    }
    fingerprint = sha256_of_dict(meta)

    print("\nRun fingerprint:", fingerprint)
    print("\nResults (real terms):")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.to_string(index=False))

    # Persist for reproducibility
    df.to_csv("withdrawal_grid_results.csv", index=False)
    with open("withdrawal_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nWrote: withdrawal_grid_results.csv")
    print("Wrote: withdrawal_run_meta.json")


if __name__ == "__main__":
    main()
