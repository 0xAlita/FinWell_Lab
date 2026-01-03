# Withdrawal Strategy Test Rig (67..95) — 2 assets (Safe + Equity)

This Python script implements a **Monte Carlo simulation framework** to evaluate withdrawal strategies for retirement portfolios with a **two-asset allocation** (Safe and Equity). It supports:

- **Flexible withdrawal rules**, including **Guardrails** (Guyton-style)
- **Minimal tax logic** with cost basis tracking
- **DIN 77230 input mask** for completeness checks and guiding decision-making
- **Reproducible simulations** with detailed output metrics

---

## 📌 Overview

This tool simulates retirement portfolio performance over time under various withdrawal strategies and asset mixes, while accounting for:

- **Market risk** (correlated log-returns)
- **Inflation**
- **Fees**
- **Tax on realized gains**
- **Ruin probability**
- **Terminal wealth and drawdown statistics**

---

## 🧠 Key Concepts

### 1. **Withdrawal Rules**

You can choose one of three modes:

| Mode             | Description |
|------------------|-------------|
| `fixed_nominal`  | Constant nominal withdrawal (e.g., €2500/month) |
| `fixed_real`     | Constant real withdrawal indexed by inflation |
| `guardrails`     | Real withdrawal with annual adjustments based on current portfolio value (Guyton-style) |

Guardrails allow for dynamic adjustment of withdrawals to prevent ruin while maintaining purchasing power.

### 2. **Tax Model**

Minimal cost-basis tracking for taxable portions of withdrawals:

- Tracks **cost basis**
- Applies tax only on **realized gains**
- Supports configurable **taxable fraction** of the portfolio

### 3. **DIN 77230 Input Mask**

A dataclass-based structure to collect important retirement planning inputs:

- Age, household size, dependents
- Income and expense streams
- Liquidity buffer, insurance status
- Ethical exclusions, lifestyle preferences

> **Note**: This is a "freedom-enabling" input mask — not a corset. You can extend it to match your own workflow.

---

## 🧪 Simulation Inputs

### 📦 `WithdrawalPlan`

Defines the withdrawal strategy and simulation parameters:

| Field                        | Description |
|-----------------------------|-------------|
| `monthly_withdrawal`        | Initial withdrawal amount (real terms) |
| `withdrawal_mode`           | One of `fixed_nominal`, `fixed_real`, or `guardrails` |
| `guardrails`                | Parameters for guardrail logic |
| `inflation_annual`          | Annual inflation rate |
| `annual_fee_rate`           | Annual fee rate (e.g., 0.2%) |
| `start_wealth`              | Initial portfolio value |
| `n_sims`                    | Number of simulations |
| `seed`                      | Random seed for reproducibility |

### 📦 `TwoAssetParams`

Describes the market model:

| Field             | Description |
|-------------------|-------------|
| `median_safe`     | Safe asset median annual return |
| `median_equity`   | Equity asset median annual return |
| `vol_safe`        | Safe asset volatility (log-return) |
| `vol_equity`      | Equity asset volatility (log-return) |
| `corr`            | Correlation between safe and equity returns |

### 📦 `Mix`

Defines a portfolio mix (name, equity weight):

| Field     | Description |
|-----------|-------------|
| `name`    | Name of the mix (e.g., "60/40") |
| `w_equity`| Weight of equity in the portfolio (0.0 to 1.0) |

---

## 📊 Simulation Outputs

Each simulation returns a dictionary with the following metrics:

| Metric                          | Description |
|---------------------------------|-------------|
| `success_prob`                  | Probability of not running out of money |
| `ruin_rate`                     | Fraction of simulations that went bankrupt |
| `ruin_median_month`             | Median month of ruin (if any) |
| `terminal_real_q05`, `q50`, `q95` | 5th, 50th, 95th quantiles of terminal real wealth |
| `max_drawdown_q05`, `q50`       | 5th and median quantiles of max drawdown |
| `avg_tax_paid_total`            | Average total tax paid across all simulations |
| `avg_tax_paid_total_success`    | Average tax paid only for successful simulations |

---

## 📦 Example Run

```bash
python withdrawal_strategy.py
```

This will:

1. Print a **DIN 77230 completeness check**
2. Run simulations with default mixes (0% to 100% equity)
3. Output results in tabular form
4. Save results to:
   - `withdrawal_grid_results.csv`
   - `withdrawal_run_meta.json` (for reproducibility)

---

## 🧪 Customization

You can customize:

- **Withdrawal strategy**
- **Asset mix**
- **Market parameters**
- **Tax model**
- **Simulation settings**

Example:
```python
plan = WithdrawalPlan(
    monthly_withdrawal=2500.0,
    withdrawal_mode="guardrails",
    guardrails=GuardrailsRule(upper_band=0.15, lower_band=0.15),
    ...
)
```

---

## 🔐 Reproducibility

- All simulations are seeded
- Meta data (inputs, settings) are saved to `withdrawal_run_meta.json`
- A SHA256 fingerprint of the run is printed for easy tracking

---

## 🧰 Requirements

- Python 3.8+
- `numpy`
- `pandas`

Install dependencies:
```bash
pip install numpy pandas
```

---

## 📁 Files Generated

| File                      | Description |
|---------------------------|-------------|
| `withdrawal_grid_results.csv` | Simulation results |
| `withdrawal_run_meta.json`    | Full run metadata for reproducibility |

---

## 📚 References

- Guyton, K. (2003). *A Common Sense Approach to Financial Planning*
- DIN 77230 – Retirement Planning Input Mask (German standard)

---

## 🧑‍💻 Author

**0xAlita**  


---

## 📌 License

MIT License

--- 

*This tool is for educational and planning purposes only. Not financial advice.*
