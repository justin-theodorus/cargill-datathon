# Cargill Ocean Transportation Datathon 2026 - Submission

**Team:** Tall Cone
**Date:** January 2026

---

## Quick Start (For Judges)

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
