# Cargill Ocean Transportation Datathon 2026 - Submission

**Team:** Tall Cone
**Date:** January 2026

## Team Contributions:
Justin Theodorus: Built the ML/risk adjustment component by engineering features from voyage outputs and scenario inputs, training and comparing models, and producing risk-adjusted profitability signals used in final recommendations. Integrated results back into the overall workflow, ensured reproducibility, and consolidated the report/README so judges could quickly follow the methodology and verify outputs.

Justin Dalva Wicent: Designed the vessel–cargo allocation strategy by generating feasible pairings and selecting assignments that maximize profitability under practical constraints. Iterated on ranking/selection logic, stress-tested decisions under scenarios, and generated the final allocation deliverables in a submission-ready format.

Calvin Ang: Implemented the core voyage calculation engine to compute distance/time components, fuel usage, costs, revenue, and profitability. Modularized the logic into reusable Python code and produced detailed calculation outputs (CSV artifacts) that powered both the analysis and the allocation step.

Zoe Dharmadi: Interpreted the datathon brief and translated it into an end-to-end pipeline (data prep → voyage calculations → profitability metric → allocation → risk adjustment). Defined the key business metric(s) used for decision-making (e.g., TCE / risk-adjusted TCE) and aligned all modeling and selection choices to optimize that objective.

Felicia Joyvina Handoyo: Organized raw inputs into a reproducible project structure and handled data cleaning/standardization across port, vessel, and cargo fields. Managed missing or inconsistent values through documented assumptions and validation checks so the downstream steps were built on reliable inputs.

---

## Quick Start (For Judges)

### Insert API Key under (9) in test_run.ipynb

### Option 1: View Results Immediately

Open `report.md` for the complete 5-page analysis with all findings.

### Option 2: Run the Notebook

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook test_run.ipynb

# 3. Run all cells
# Kernel -> Restart & Run All
```

### Option 3: Try the AI Chatbot

**With OpenAI API key (GPT-powered):**
```bash
export OPENAI_API_KEY=your-key-here
python chatbot.py
```

**Without API key (rule-based fallback):**
```bash
python chatbot.py
```

The chatbot works in both modes. The rule-based fallback provides the same analysis data without requiring an external API.

---

## File Structure

```
cargill-datathon/
  test_run.ipynb                    Main analysis notebook (Run All Cells)
  report.md                         5-page report
  chatbot.py                        AI chatbot (OpenAI GPT + Gradio UI + fallback)
  freight_calculator.py             Core voyage calculation engine
  vessel_cargo_data.py              All vessel & cargo data (15 vessels, 11 cargoes)
  main_analysis.py                  Analysis functions (combinations, optimization)
  test_data_completion.py           Data validation tests
  requirements.txt                  Python dependencies
  Port_Distances.csv                Port-to-port distance matrix
  DATATHON_BRIEF.MD                 Competition brief reference
  optimal_allocation.csv            Final allocation output (3 rows)
  voyage_calculations_detailed.csv  All 165 combination calculations
  scenario_port_delays.csv          Port delay scenario results
  scenario_bunker_prices.csv        Bunker price scenario results
  ml_risk_analysis.csv              ML risk analysis output
```

---

## Key Results

### Optimal Allocation (Total Profit: $5,169,313)

| # | Vessel | Cargo Route | TCE ($/day) | Profit |
|---|--------|-------------|-------------|--------|
| 1 | OCEAN HORIZON | Coal: Taboneo -> Krishnapatnam | $71,597 | $1,127,451 |
| 2 | NAVIS PRIDE | Bauxite: Kamsar -> Mangalore | $70,717 | $2,324,441 |
| 3 | IRON CENTURY | Iron Ore: Ponta da Madeira -> Caofeidian | $50,631 | $1,717,421 |

12 vessels remain unallocated and available for committed cargoes.

### Scenario Thresholds

- **Port delays (China):** Robust up to 15 days; max loss $240,000 (4.6%)
- **Bunker price increase:** 50% increase causes $801,000 loss (15.5%)

### ML Model

- **Gradient Boosting:** R2 = 0.99, MAE = $857/day (10-Fold CV)
- **Top risk factors:** Route length, TCE margin, bunker cost ratio
- **165 combinations** analyzed with 14 domain-knowledge features

---

## Innovation Highlights

1. **AI Chatbot** - OpenAI GPT with function calling for on-demand scenario calculations
2. **Gradio Web UI** - Interactive chat interface with example queries
3. **ML Risk Model** - 14 domain features, SHAP interpretability analysis
4. **Natural Portfolio Hedging** - 2 of 3 allocations avoid China ports, reducing congestion risk

---

## Verification

```bash
# Run data validation tests
python test_data_completion.py

# Verify outputs
python -c "
import pandas as pd
alloc = pd.read_csv('optimal_allocation.csv')
assert len(alloc) == 3, 'Should have 3 allocations'
print('All checks passed')
"
```
