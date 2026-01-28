# Cargill Ocean Transportation Datathon 2026
## Voyage Optimization & Risk Analysis Report

**Team:** Tall Cone
**Date:** January 2026

---

## 1. Executive Summary

We evaluated **165 vessel-cargo combinations** (15 Capesize vessels x 11 cargoes) to identify the portfolio of voyages that maximizes total profit. Our freight calculator engine computes all revenue and cost components --- hire, bunker consumption, port costs, and commissions --- and derives Time Charter Equivalent (TCE) as the primary decision metric.

### Recommended Allocation

| # | Vessel | Cargo | Route | TCE ($/day) | Profit ($) | Days |
|---|--------|-------|-------|-------------|------------|------|
| 1 | OCEAN HORIZON | Coal (Taboneo-Krishnapatnam) | Map Ta Phut -> Taboneo -> Krishnapatnam | 71,597 | 1,127,451 | 20.2 |
| 2 | NAVIS PRIDE | Bauxite (Kamsar-Mangalore) | Mundra -> Kamsar -> Mangalore | 70,717 | 2,324,441 | 42.6 |
| 3 | IRON CENTURY | Iron Ore (Ponta da Madeira-Caofeidian) | Port Talbot -> Ponta da Madeira -> Caofeidian | 50,631 | 1,717,421 | 49.6 |

**Total Portfolio Profit: $5,169,313**

The remaining 12 vessels are available for Cargill's three committed cargoes and additional market opportunities. Our analysis reveals that market cargo arbitrage opportunities currently offer superior returns compared to committed cargoes, which is a key commercial insight for fleet deployment.

### Key Thresholds

- **Port delays in China:** Portfolio is robust up to 15 days of additional delay, with only $240,000 (4.6%) profit reduction, because only 1 of 3 voyages discharges in China.
- **Bunker price increases:** A 50% increase reduces profit by $801,000 (15.5%). The degradation is linear, with no abrupt tipping point.
- **Risk-adjusted allocation:** ML risk simulation confirms the same allocation remains optimal after accounting for weather, congestion, and mechanical risks.

---

## 2. Methodology

### Freight Calculator Framework

We developed a Python-based freight calculator that evaluates complete voyage economics for every vessel-cargo combination:

**Revenue:** Freight rate ($/MT) x cargo quantity, minus broker/charterer commissions (1.25%--3.75%).

**Costs:**
- *Hire cost:* Daily hire rate x total voyage duration (sea days + port days)
- *Bunker cost:* Fuel consumption (VLSF + MGO) at sea and in port, priced using the February 2026 bunker forward curve by region (Singapore, Qingdao, Rotterdam, Durban, etc.)
- *Port cost:* Loading and discharge port costs as specified per cargo

**Key Metric --- Time Charter Equivalent (TCE):**

> TCE = (Net Revenue - Bunker Cost - Port Cost) / Total Voyage Days

TCE normalizes profitability per day, enabling fair comparison across voyages of different durations and routes.

### Optimization Approach

We apply a greedy allocation algorithm: sort all 165 combinations by TCE descending, then assign each cargo to the highest-TCE vessel, ensuring no vessel or cargo is double-allocated. This produces the globally near-optimal solution for our portfolio size.

### Machine Learning Risk Model

To quantify operational risks, we trained a **Gradient Boosting** model with **14 domain-knowledge features** to predict risk-adjusted TCE:

- *Geographic:* China discharge ports, South America loading, West Africa anchorage, Australia terminals
- *Route:* Total route length (nm), ballast-to-total distance ratio
- *Seasonal:* Cyclone season (Jun-Nov), winter monsoon (Dec-Feb)
- *Economic:* Bunker cost ratio, hire cost ratio, TCE margin
- *Cargo/Vessel:* Commodity type (iron ore, bauxite, coal), vessel bunker levels

**Model validation:** 10-Fold Cross-Validation yields R^2 = 0.99 and MAE = $857/day, with consistent performance across all folds. We compared five models (Random Forest, Gradient Boosting, Linear Regression, Ridge Regression, Decision Tree) and selected Gradient Boosting for its balance of accuracy and interpretability.

### Key Assumptions

1. All voyages use **economical speeds** (vs. warranted) for fuel efficiency
2. Bunker prices based on **February 2026 forward curve** from the brief
3. **Port distance proxies** used where exact distances are unavailable (e.g., Kamsar proxied by Conakry, Map Ta Phut by Laem Chabang)
4. Market vessel **hire rates estimated** from FFA 5TC average ($15,700--$16,300/day)
5. Market cargo **freight rates estimated** from FFA C3/C5 equivalents

---

## 3. Results & Analysis

### Why Each Allocation is Optimal

**1. OCEAN HORIZON -> Coal (Taboneo-Krishnapatnam) --- $1,127,451 profit**

This allocation achieves the highest TCE ($71,597/day) due to an exceptionally short voyage (20.2 days). OCEAN HORIZON is positioned at Map Ta Phut, Thailand --- only a short ballast to Taboneo, Indonesia. The coal cargo to Krishnapatnam, India avoids China port congestion entirely. Despite lower absolute profit than other voyages, the high daily earnings rate makes this the most efficient use of vessel time.

**2. NAVIS PRIDE -> Bauxite (Kamsar-Mangalore) --- $2,324,441 profit**

The highest absolute profit in the portfolio. NAVIS PRIDE at Mundra, India has a manageable ballast to Kamsar, Guinea (West Africa). The strong freight rate ($22/MT) on the bauxite route, combined with delivery to Mangalore, India (not China), keeps costs controlled. This voyage earns $70,717/day TCE over 42.6 days, making it both highly profitable and operationally efficient.

**3. IRON CENTURY -> Iron Ore (Ponta da Madeira-Caofeidian) --- $1,717,421 profit**

The longest route in the portfolio (49.6 days), carrying Vale's iron ore from Brazil to China. IRON CENTURY is positioned at Port Talbot, Wales, requiring a transatlantic ballast to Brazil. The C3-equivalent freight rate ($19/MT) on 190,000 MT generates strong revenue. This is the only China-discharge voyage, providing targeted exposure to that market while the other two voyages hedge the portfolio against China congestion.

### Commercial Insight: Market Arbitrage

A significant finding is that all three optimal allocations involve **market cargoes**, not Cargill's committed cargoes. This suggests the current market freight rates for these routes offer superior economics. The three committed Cargill cargoes (Bauxite Kamsar-Qingdao, Iron Ore Hedland-Lianyungang, Iron Ore Itaguai-Qingdao) should be carried by the remaining unallocated vessels, with market vessels available for hire if needed.

---

## 4. Scenario Analysis & ML Insights

### Scenario 1: Port Delays in China

We tested 0--15 days of additional port delay at Chinese discharge ports.

| Delay (days) | Portfolio Profit | Loss | Reduction |
|:---:|---:|---:|---:|
| 0 | $5,169,313 | $0 | 0.0% |
| 6 | $5,073,313 | $96,000 | 1.9% |
| 10 | $5,009,313 | $160,000 | 3.1% |
| 15 | $4,929,313 | $240,000 | 4.6% |

**Finding:** The portfolio is inherently resilient to China port delays because only **1 of 3 allocated voyages** (IRON CENTURY to Caofeidian) discharges in China. Even 15 days of delay produces only a 4.6% profit reduction. This natural hedging is a strength of the allocation --- the two India-discharge voyages are unaffected.

### Scenario 2: Bunker Price Increases

We tested 0--50% increases in bunker prices at all ports.

| Increase | Portfolio Profit | Loss | Reduction |
|:---:|---:|---:|---:|
| 0% | $5,169,313 | $0 | 0.0% |
| 10% | $5,009,124 | $160,189 | 3.1% |
| 20% | $4,848,935 | $320,378 | 6.2% |
| 30% | $4,688,746 | $480,567 | 9.3% |
| 50% | $4,368,368 | $800,945 | 15.5% |

**Finding:** Profit degrades linearly with bunker price increases. There is no sharp tipping point --- the allocation remains optimal across all tested levels. The most exposed voyage is IRON CENTURY (Brazil-China), which has the longest route and highest absolute bunker consumption.

### ML Feature Importance

The Gradient Boosting model identifies these top predictors of risk-adjusted TCE:

1. **Route length (nm):** Most important factor. Longer routes expose voyages to more weather variability and amplify bunker price sensitivity.
2. **TCE margin:** Higher base TCE provides a buffer to absorb unexpected costs from delays or fuel price spikes.
3. **Bunker cost ratio:** Voyages where fuel exceeds 30% of revenue are disproportionately sensitive to price volatility.
4. **China discharge ports:** Historical congestion patterns add 2--3 days on average.
5. **Ballast positioning ratio:** Long ballast legs reduce earning efficiency and increase the cost of vessel positioning.

**SHAP analysis** confirms that route length and bunker cost ratio interact --- long routes with high fuel ratios carry compounded risk, while short routes are inherently more resilient.

**Risk-adjusted allocation comparison:** Running the optimizer with risk-adjusted TCE (incorporating simulated weather delays, port congestion, and mechanical risks) produces the **same allocation**, confirming that our base case recommendation is robust to operational uncertainties.

---

## 5. Recommendations & Risk Mitigation

### Strategic Recommendations

**1. Execute the three-vessel allocation ($5.17M portfolio profit)**

The recommended allocation maximizes TCE across the available fleet and cargoes. Each vessel is matched to its best available cargo based on positioning, route economics, and freight rates.

**2. Deploy remaining vessels for committed cargoes**

With 12 vessels unallocated, Cargill has ample capacity to serve the three committed cargoes (Bauxite Kamsar-Qingdao at $23/MT, Iron Ore Hedland-Lianyungang at $9/MT, Iron Ore Itaguai-Qingdao at $22.30/MT). We recommend assigning these based on vessel positioning and laycan compatibility.

**3. Active risk monitoring**

- *Bunker prices:* Monitor VLSF forward curve. Consider bunker hedging (collar options) for the IRON CENTURY Brazil-China voyage, which has approximately $794,000 in bunker exposure.
- *China port congestion:* Track Caofeidian wait times. If delays consistently exceed 5 days, evaluate whether to divert IRON CENTURY to an alternative Chinese port (Qingdao, Tianjin) on the same TCE basis as permitted in the cargo terms.
- *Weather windows:* The March-April loading period avoids peak cyclone season (Jun-Nov), reducing weather risk for all three voyages.

### Innovation: AI Chatbot

We developed an AI-powered chatbot that allows trading managers to interactively query our analysis:

- **OpenAI GPT backend** with function calling for natural language understanding
- **Three callable tools** for on-demand scenario calculations, vessel comparisons, and voyage detail lookups
- **Gradio web interface** with example queries for easy exploration
- **Rule-based fallback** ensures the chatbot works even without an API key

This enables rapid what-if analysis during commercial negotiations --- e.g., "What if bunker prices rise 15%?" or "Compare vessel A vs vessel B for this cargo."

### Future Enhancements

1. **Real-time data integration:** API feeds for bunker prices, port congestion, and weather forecasts to enable dynamic re-optimization
2. **Multi-period optimization:** Extend from single-voyage to fleet scheduling over multiple periods, accounting for vessel repositioning between voyages
3. **Portfolio diversification metrics:** Quantify concentration risk (currently 67% iron ore, significant China exposure in committed cargoes) and recommend commodity/geographic diversification targets

### Conclusion

Our analysis recommends a three-vessel allocation generating $5.17M in portfolio profit. The allocation is robust to moderate market disruptions: resilient to 15 days of China port delays (4.6% impact) and 50% bunker price increases (15.5% impact). The ML risk model validates the allocation under operational uncertainties, and the AI chatbot enables ongoing interactive exploration of scenarios and alternatives.

---

*Appendix A: Full calculation methodology available in `freight_calculator.py`*
*Appendix B: ML model validation results in notebook Section 5*
*Appendix C: All 165 combination details exported to `voyage_calculations_detailed.csv`*
