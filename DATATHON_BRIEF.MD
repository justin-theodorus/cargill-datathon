# Cargill Ocean Transportation Datathon 2026

## About Cargill Ocean Transportation

We are a leading mover of dry and wet bulk cargo and one of the world's largest traders of freight markets. With an average daily fleet of about 640 chartered vessels worldwide in 2024 and 70% of our activities executed for our third-party customers, we have been operating in the industry since 1956.

Every day, we move our customers' bulk goods around the world so they can help keep the global economy running.

---

## Event Description

The Cargill-SMU Data-thon 2026 challenges participants to step into the role of a **Dry Bulk Trader** in Cargill Ocean Transportation and make data-driven decisions on vessel employment and voyage optimization.

Using realistic commercial and operational data, your team will:

1. **Build a freight calculator** to evaluate multiple voyage options based on vessel positions and market opportunities. This may include the use of machine learning models to account for shipping risks that could impact the profitability of each voyage.

2. **Run scenario analyses** to understand how changes in market conditions affect vessel employment recommendations.

3. **Enhance model interactivity** by developing an interactive chatbot powered by Generative AI, enabling your analytical models and tools to come alive.

This Data-thon bridges data science, maritime economics, and Generative AI technology, enabling participants to solve real-world challenges that freight traders face daily where **the right decision can mean millions in profit or loss**.

---

## Scenario

You are a **Dry Bulk Trader** at Cargill Ocean Transportation Singapore, managing a fleet of Capesize vessels (large bulk carriers) and committed to move bulk cargoes such as iron ore and bauxite across global trade routes for our customers.

### Your Task

**Four of your Capesize vessels are about to complete discharge operations.** Your task is to:
1. Determine their next employment deciding where each vessel should sail next and which cargo it should load
2. Identify the most suitable vessel to carry each of Cargill's three committed cargoes

### Market Opportunities

Additionally, the freight market offers a range of opportunities. You can optimize or arbitrage your position by:
- Engaging market vessels to carry Cargill's committed cargoes, OR
- Bidding for market cargoes to be carried by Cargill vessels

**Your goal:** Evaluate available cargo options, vessel positions, and prevailing freight rates to determine the most effective employment strategy.

---

## Required Outputs

### 1. Your Freight Calculator

Build and present your Freight Calculator, a model or tool that recommends a trader decide the best next voyage for each of the four Capesize vessels and three committed Cargill cargoes to maximize the portfolio profit.

Your recommendation should provide an outcome which:
- **Include all revenue and cost parameters**
- Use Freight Calculator in excel format provided as a reference
- Use distance table provided – if distances between any 2 ports are not available, make necessary assumptions

**Provide a voyage recommendation** for each of Cargill's vessels and committed cargoes. If applicable:
- State the hire offer for market 3rd party vessels, OR
- State the freight quotation for market 3rd party cargoes to support your recommendation

**Use machine learning techniques** wherever applicable to better simulate shipping risks such as adverse weather and port congestion.

**Perform an explainability/interpretability analysis** on the ML model output: parameter importance, SHAP values, coefficients, etc.

### 2. Scenario Analysis

Start with your base case voyage recommendation. Adjust either bunker price or port delay to find the point where another option overtakes your current best choice in terms of total voyage profit or TCE (Time Charter Equivalent). Record the threshold and describe how it changes your recommendation.

| Scenario | Parameter Tested | Task |
|----------|------------------|------|
| 1 | **Port Delay in China** (Delay duration in days) | Find the number of additional port delay days that would make your current recommendation no longer optimal |
| 2 | **Bunker Price Increase at all ports** (VLSF price % change) | Identify the fuel price increase (%) at which your current recommendation becomes less profitable than the next best option |

### 3. Conversational Chatbot

Bring your model to life through a simple AI chatbot that allows users to interact with your voyage recommendation model.

- Use any AI tools or resources you are familiar with to build the chatbot
- If required, reach out to hector_ordonez@cargill.com to request for an OpenAI API key

**Minimum Requirement (Expected):**
- Prompt your model output (vessel-cargo recommendation, Time Charter Equivalent (TCE), profit, etc.) into the chatbot
- Make the chatbot respond like an AI assistant giving a recommendation
- Example: "Based on current freight rates, Vessel A should carry Cargo 2 for the best TCE."

**Higher Achievement (To Excel):**
Teams aiming for a higher score can:
- Build features that makes the chatbot experience more professional (e.g., side-by-side voyage comparison, sliders, dropdowns or charts)
- Enable the chatbot to summarize trade-offs in different scenarios clearly, e.g.: "If bunker prices rise above USD 700/MT in Singapore, Vessel B becomes more cost-effective to carry Cargo 2."

---

## Submission Requirements

Each team must submit their final solution in three components:

### 1. Voyage Recommendation Report

A short report (maximum 5 pages) summarizing:
- Your recommended vessel–cargo allocation and rationale
- Key assumptions used
- Results of your scenario analyses
- Threshold insights (e.g., "Bunker price above $700/MT changes our recommendation to Vessel B")

**The report should read as if you are presenting your findings to a freight trading manager.**

### 2. Model & Chatbot Submission

Teams may choose one of the following formats:

**Option A: Jupyter Notebook (Recommended)**
- Combine Python code, charts, and explanations in one file
- Include:
  - Model logic for freight and scenario calculations
  - Chatbot prompt examples (e.g., "What if bunker +10%?" → system response)
- Notebook must run from start to finish with no errors

**Option B: Chatbot Demo / UI (Bonus Marks)**
- Optional enhancement for higher scores
- Create a user interface that:
  - Accepts user inputs (vessel, cargo, bunker price)
  - Displays voyage recommendations interactively
- Include a short screen capture or link (if hosted locally)

### 3. Documentation & Repository

- Submit all source files via a zipped repository
- Include the Jupyter notebook, data files, and readme instructions
- Provide a PDF documentation that:
  - Summarizes file structure and how to reproduce results
  - Lists team members and responsibilities

---

## Cargill's Capesize Vessels

| Vessel Name | DWT (MT) | Hire Rate (USD/day) | Warranted Speed (Laden/Ballast) | Economical Speed (Laden/Ballast) | Port Consumption (MGO) | Voyage / Position Update (ETD) | Bunker Remaining on Board |
|-------------|----------|---------------------|----------------------------------|-----------------------------------|------------------------|-------------------------------|---------------------------|
| **ANN BELL** | 180,803 | 11,750 | **Laden:** 13.5 kn / 60 mt VLSF + 2.0 mt MGO<br>**Ballast:** 14.5 kn / 55 mt VLSF + 2.0 mt MGO | **Laden:** 12.0 kn / 42 mt VLSF + 2.0 mt MGO<br>**Ballast:** 12.5 kn / 38 mt VLSF + 2.0 mt MGO | **Idle:** 2.0 mt VLSF/day<br>**Working:** 3.0 mt VLSF/day | Discharging **Qingdao, China**<br>ETD: 25 Feb 2026 | 401.3 mt VLSF<br>45.1 mt MGO |
| **OCEAN HORIZON** | 181,550 | 15,750 | **Laden:** 13.8 kn / 61 mt VLSF + 1.8 mt MGO<br>**Ballast:** 14.8 kn / 56.5 mt VLSF + 1.8 mt MGO | **Laden:** 12.3 kn / 43.0 mt VLSF + 1.8 mt MGO<br>**Ballast:** 12.8 kn / 39.5 mt VLSF + 1.8 mt MGO | **Idle:** 1.8 mt VLSF/day<br>**Working:** 3.2 mt VLSF/day | Discharging **Map Ta Phut, Thailand**<br>ETD: 1 Mar 2026 | 265.8 mt VLSF<br>64.3 mt MGO |
| **PACIFIC GLORY** | 182,320 | 14,800 | **Laden:** 13.5 kn / 59 mt VLSF + 1.9 mt MGO<br>**Ballast:** 14.2 kn / 54 mt VLSF + 1.9 mt MGO | **Laden:** 12.2 kn / 44 mt VLSF + 1.9 mt MGO<br>**Ballast:** 12.7 kn / 40 mt VLSF + 1.9 mt MGO | **Idle:** 2.0 mt VLSF/day<br>**Working:** 3.0 mt VLSF/day | Discharging **Gwangyang, S. Korea**<br>ETD: 10 Mar 2026 | 601.9 mt VLSF<br>98.1 mt MGO |
| **GOLDEN ASCENT** | 179,965 | 13,950 | **Laden:** 13.0 kn / 58 mt VLSF + 2.0 mt MGO<br>**Ballast:** 14.0 kn / 53 mt VLSF + 2.0 mt MGO | **Laden:** 11.8 kn / 41 mt VLSF + 2.0 mt MGO<br>**Ballast:** 12.3 kn / 37 mt VLSF + 2.0 mt MGO | **Idle:** 1.9 mt VLSF/day<br>**Working:** 3.1 mt VLSF/day | Discharging **Fangcheng, China**<br>ETD: 8 Mar 2026 | 793.3 mt VLSF<br>17.1 mt MGO |

---

## Cargill Committed Cargoes

| Route | Customer | Commodity | Quantity | Laycan | Freight Rate | Load Port | Loading Terms | Discharge Port | Discharge Terms | Port Cost | Commission |
|-------|----------|-----------|----------|--------|--------------|-----------|---------------|----------------|-----------------|-----------|------------|
| **West Africa – China** (Bauxite) | EGA | Bauxite | 180,000 MT +/- 10% Owners' Option | 2–10 April 2026 | **$23 PMT** | Kamsar, Guinea (Anchorage Loading) | 30,000 MT PWWD SHINC<br>12 hr turn time | Qingdao, China (other Chinese ports allowed on same TCE basis) | 25,000 MT PWWD SHINC<br>12 hr turn time | Nil, borne by Charterer | **1.25%** due to broker |
| **Australia – China** (Iron ore) | BHP | Iron Ore | 160,000 MT +/- 10%<br>(half freight applies to any cargo loaded in excess of 176,000 MT) | 7–11 March 2026 | **$9 PMT** | Port Hedland | 80,000 MT PWWD SHINC<br>12 hr turn time | Lianyungang, China | 30,000 MT PWWD SHINC<br>24 hr turn time | **USD 260K** at load port<br>**USD 120K** at discharge port | **3.75%** due to charterer |
| **Brazil – China** (Iron ore) | CSN | Iron Ore | 180,000 MT +/- 10% MOLOO | 1–8 April 2026 | **$22.30 PMT** | Itaguai, Brazil (freight offer base 17.8M draft) | 60,000 MT PWWD SHINC + 6 hr turn time | Qingdao or other ports in the Far East, including river ports (freight base Qingdao) | 30,000 MT PWWD SHINC + 24 hr turn time | **USD 75K** at load port<br>**USD 90K** at discharge port | **3.75%** due to charterer |

**Terms Glossary:**
- **PMT** = Per Metric Ton
- **PWWD** = Per Weather Working Day
- **SHINC** = Sundays and Holidays Included
- **MOLOO** = More or Less Owner's Option

---

## Market Vessels (Available for Hire)

| Vessel Name | DWT (MT) | Warranted Speed (Laden/Ballast) | Economical Speed (Laden/Ballast) | Port Consumption (MGO) | Voyage / Position Update (ETD) | Bunker Remaining |
|-------------|----------|----------------------------------|-----------------------------------|------------------------|-------------------------------|------------------|
| **ATLANTIC FORTUNE** | 181,200 | Laden: 13.8 kn / 60 mt VLSFO + 2.0 mt MGO<br>Ballast: 14.6 kn / 56 mt VLSFO + 2.0 mt MGO | Laden: 12.3 kn / 43 mt VLSFO + 2.0 mt MGO<br>Ballast: 12.9 kn / 39.5 mt VLSFO + 2.0 mt MGO | Idle: 2.0 mt VLSFO/day<br>Working: 3.0 mt VLSFO/day | Discharging **Paradip, India**<br>ETD: 2 Mar 2026 | 512.4 mt VLSFO<br>38.9 mt MGO |
| **PACIFIC VANGUARD** | 182,050 | Laden: 13.6 kn / 59 mt VLSFO + 1.9 mt MGO<br>Ballast: 14.3 kn / 54 mt VLSFO + 1.9 mt MGO | Laden: 12.0 kn / 42 mt VLSFO + 1.9 mt MGO<br>Ballast: 12.5 kn / 38 mt VLSFO + 1.9 mt MGO | Idle: 1.9 mt VLSFO/day<br>Working: 3.0 mt VLSFO/day | Discharging **Caofeidian, China**<br>ETD: 26 Feb 2026 | 420.3 mt VLSFO<br>51.0 mt MGO |
| **CORAL EMPEROR** | 180,450 | Laden: 13.4 kn / 58 mt VLSFO + 2.0 mt MGO<br>Ballast: 14.1 kn / 53 mt VLSFO + 2.0 mt MGO | Laden: 11.9 kn / 40 mt VLSFO + 2.0 mt MGO<br>Ballast: 12.3 kn / 36.5 mt VLSFO + 2.0 mt MGO | Idle: 2.0 mt VLSFO/day<br>Working: 3.1 mt VLSFO/day | Discharging **Rotterdam, Netherlands**<br>ETD: 5 Mar 2026 | 601.7 mt VLSFO<br>42.3 mt MGO |
| **EVEREST OCEAN** | 179,950 | Laden: 13.7 kn / 61 mt VLSFO + 1.8 mt MGO<br>Ballast: 14.5 kn / 56.5 mt VLSFO + 1.8 mt MGO | Laden: 12.4 kn / 43.5 mt VLSFO + 1.8 mt MGO<br>Ballast: 12.8 kn / 39 mt VLSFO + 1.8 mt MGO | Idle: 1.8 mt VLSFO/day<br>Working: 3.0 mt VLSFO/day | Discharging **Xiamen, China**<br>ETD: 3 Mar 2026 | 478.2 mt VLSFO<br>56.4 mt MGO |
| **POLARIS SPIRIT** | 181,600 | Laden: 13.9 kn / 62 mt VLSFO + 1.9 mt MGO<br>Ballast: 14.7 kn / 57 mt VLSFO + 1.9 mt MGO | Laden: 12.5 kn / 44 mt VLSFO + 1.9 mt MGO<br>Ballast: 13.0 kn / 40 mt VLSFO + 1.9 mt MGO | Idle: 2.0 mt VLSFO/day<br>Working: 3.1 mt VLSFO/day | Discharging **Kandla, India**<br>ETD: 28 Feb 2026 | 529.8 mt VLSFO<br>47.1 mt MGO |
| **IRON CENTURY** | 182,100 | Laden: 13.5 kn / 59 mt VLSFO + 2.1 mt MGO<br>Ballast: 14.2 kn / 54 mt VLSFO + 2.1 mt MGO | Laden: 12.0 kn / 41 mt VLSFO + 2.1 mt MGO<br>Ballast: 12.5 kn / 37.5 mt VLSFO + 2.1 mt MGO | Idle: 2.1 mt VLSFO/day<br>Working: 3.2 mt VLSFO/day | Discharging **Port Talbot, Wales**<br>ETD: 9 Mar 2026 | 365.6 mt VLSFO<br>60.7 mt MGO |
| **MOUNTAIN TRADER** | 180,890 | Laden: 13.3 kn / 58 mt VLSFO + 2.0 mt MGO<br>Ballast: 14.0 kn / 53 mt VLSFO + 2.0 mt MGO | Laden: 12.1 kn / 42 mt VLSFO + 2.0 mt MGO<br>Ballast: 12.6 kn / 38 mt VLSFO + 2.0 mt MGO | Idle: 2.0 mt VLSFO/day<br>Working: 3.1 mt VLSFO/day | Discharging **Gwangyang, South Korea**<br>ETD: 6 Mar 2026 | 547.1 mt VLSFO<br>32.4 mt MGO |
| **NAVIS PRIDE** | 181,400 | Laden: 13.8 kn / 61 mt VLSFO + 1.8 mt MGO<br>Ballast: 14.5 kn / 56 mt VLSFO + 1.8 mt MGO | Laden: 12.6 kn / 44 mt VLSFO + 1.8 mt MGO<br>Ballast: 13.0 kn / 39 mt VLSFO + 1.8 mt MGO | Idle: 1.8 mt VLSFO/day<br>Working: 3.0 mt VLSFO/day | Discharging **Mundra, India**<br>ETD: 27 Feb 2026 | 493.8 mt VLSFO<br>45.2 mt MGO |
| **AURORA SKY** | 179,880 | Laden: 13.4 kn / 58 mt VLSFO + 2.0 mt MGO<br>Ballast: 14.1 kn / 53 mt VLSFO + 2.0 mt MGO | Laden: 12.0 kn / 41 mt VLSFO + 2.0 mt MGO<br>Ballast: 12.5 kn / 37.5 mt VLSFO + 2.0 mt MGO | Idle: 2.0 mt VLSFO/day<br>Working: 3.1 mt VLSFO/day | Discharging **Jingtang, China**<br>ETD: 4 Mar 2026 | 422.7 mt VLSFO<br>29.8 mt MGO |
| **ZENITH GLORY** | 182,500 | Laden: 13.9 kn / 61 mt VLSFO + 1.9 mt MGO<br>Ballast: 14.6 kn / 56.5 mt VLSFO + 1.9 mt MGO | Laden: 12.4 kn / 43.5 mt VLSFO + 1.9 mt MGO<br>Ballast: 12.9 kn / 39 mt VLSFO + 1.9 mt MGO | Idle: 1.9 mt VLSFO/day<br>Working: 3.1 mt VLSFO/day | Discharging **Vizag, India**<br>ETD: 7 Mar 2026 | 502.3 mt VLSFO<br>44.6 mt MGO |
| **TITAN LEGACY** | 180,650 | Laden: 13.5 kn / 59 mt VLSFO + 2.0 mt MGO<br>Ballast: 14.2 kn / 54 mt VLSFO + 2.0 mt MGO | Laden: 12.2 kn / 42 mt VLSFO + 2.0 mt MGO<br>Ballast: 12.7 kn / 38 mt VLSFO + 2.0 mt MGO | Idle: 2.0 mt VLSFO/day<br>Working: 3.0 mt VLSFO/day | Discharging **Jubail, Saudi Arabia**<br>ETD: 1 Mar 2026 | 388.5 mt VLSFO<br>53.1 mt MGO |

---

## Market Cargoes (Available for Bidding)

| Route | Customer | Commodity | Quantity | Laycan | Load Port | Loading Terms | Discharge Port | Discharge Terms | Port Cost | Commission |
|-------|----------|-----------|----------|--------|-----------|---------------|----------------|-----------------|-----------|------------|
| **Australia – China** (Iron Ore) | Rio Tinto | Iron Ore | 170,000 MT +/- 10% MOLOO | 12–18 March 2026 | Dampier, Australia | 80,000 MT PWWD SHINC + 12 hr turn time | Qingdao, China | 30,000 MT PWWD SHINC + 24 hr turn time | USD 240K total (load & discharge) | 3.75% due to charterer |
| **Brazil – China** (Iron Ore) | Vale | Iron Ore | 190,000 MT +/- 10% MOLOO | 3–10 April 2026 | Ponta da Madeira, Brazil | 60,000 MT PWWD SHINC + 12 hr turn time | Caofeidian, China | 30,000 MT PWWD SHINC + 24 hr turn time | USD 75K load / USD 95K discharge | 3.75% due to charterer |
| **South Africa – China** (Iron Ore) | Anglo American | Iron Ore | 180,000 MT +/- 10% MOLOO | 15–22 March 2026 | Saldanha Bay, South Africa | 55,000 MT PWWD SHINC + 6 hr turn time | Tianjin, China | 25,000 MT PWWD SHINC + 24 hr turn time | USD 180K total | 3.75% due to charterer |
| **Indonesia – India** (Coal) | Adaro | Thermal Coal | 150,000 MT +/- 10% MOLOO | 10–15 April 2026 | Taboneo, Indonesia | 35,000 MT PWWD SHINC + 12 hr turn time | Krishnapatnam, India | 25,000 MT PWWD SHINC + 24 hr turn time | USD 90K total | 2.50% due to broker |
| **Canada – China** (Coking Coal) | Teck Resources | Coking Coal | 160,000 MT +/- 10% MOLOO | 18–26 March 2026 | Vancouver, Canada | 45,000 MT PWWD SHINC + 12 hr turn time | Fangcheng, China | 25,000 MT PWWD SHINC + 24 hr turn time | USD 180K at load / USD 110K at discharge | 3.75% due to charterer |
| **West Africa – India** (Bauxite) | Guinea Alumina Corp | Bauxite | 175,000 MT +/- 10% MOLOO | 10–18 April 2026 | Kamsar, Guinea (Anchorage Loading) | 30,000 MT PWWD SHINC | Mangalore, India | 25,000 MT PWWD SHINC + 12 hr turn time | USD 150K total | 2.50% due to broker |
| **Australia – South Korea** (Iron Ore) | BHP | Iron Ore | 165,000 MT +/- 10% MOLOO | 9–15 March 2026 | Port Hedland, Australia | 80,000 MT PWWD SHINC + 12 hr turn time | Gwangyang, South Korea | 30,000 MT PWWD SHINC + 24 hr turn time | USD 230K total | 3.75% due to charterer |
| **Brazil – Malaysia** (Iron Ore) | Vale Malaysia | Iron Ore | 180,000 MT +/- 10% MOLOO | 25 March – 2 April 2026 | Tubarão, Brazil | 60,000 MT PWWD SHINC + 6 hr turn time | Teluk Rubiah, Malaysia | 25,000 MT PWWD SHINC + 24 hr turn time | USD 85K load / USD 80K discharge | 3.75% due to charterer |

---

## Baltic Exchange Capesize FFA Report

Forward Freight Agreements (FFAs) are financial derivatives that allow market participants to hedge or speculate on future freight rates. The FFA rate reflects the market's expectation of future spot freight rates for specific routes and vessel types (e.g., Capesize, Panamax, Supramax).

| Route | Feb 26 | Mar 26 | Q4 25 | Q1 26 | Q2 26 | Q3 26 | Q4 26 | Q1 27 | Cal 26 | Cal 27 | Cal 28 | Cal 29 | Cal 30 | Cal 31 | Cal 32 |
|-------|--------|--------|-------|-------|-------|-------|-------|-------|--------|--------|--------|--------|--------|--------|--------|
| **5TC** (avg. of 5 Capesize T/C routes) | 14,157 | 18,454 | 24,336 | 16,746 | 22,436 | 25,146 | 25,418 | 16,339 | 22,437 | 21,714 | 20,289 | 19,404 | 18,943 | 18,775 | 18,682 |
| **C3** (Tubarao–Qingdao) | 17.833 | 20.908 | 22.819 | 19.456 | 21.475 | 23.192 | 23.592 | 19.408 | 21.929 | 20.197 | 19.988 | – | – | – | – |
| **C5** (West Australia–Qingdao) | 6.633 | 8.717 | 9.689 | 7.700 | 9.083 | 9.288 | 9.408 | 7.392 | 8.870 | – | – | – | – | – | – |
| **C7** (Bolivar–Rotterdam) | 10.625 | 11.821 | 13.157 | 11.219 | 12.210 | 12.610 | 12.986 | 11.190 | 12.256 | 11.940 | 11.540 | 11.000 | 10.900 | – | – |

---

## Bunker Forward Curve (USD/MT)

| Location / Grade | Feb-26 | Mar-26 | Apr-26 | May-26 | Jun-26 | Jul-26 | Aug-26 | Sep-26 | Oct-26 | Nov-26 | Dec-26 | Cal-27 |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Singapore VLSFO** | 491 | 490 | 489 | 489 | 487 | 484 | 482 | 480 | 479 | 476 | 474 | 470 |
| **Singapore MGO** | 654 | 649 | 642 | 639 | 637 | 637 | 637 | 637 | 637 | 637 | 637 | 637 |
| **Fujairah VLSFO** | 479 | 478 | 477 | 476 | 475 | 473 | 471 | 469 | 467 | 465 | 463 | 460 |
| **Fujairah MGO** | 640 | 638 | 636 | 633 | 632 | 631 | 630 | 629 | 628 | 626 | 624 | 620 |
| **Durban VLSFO** | 436 | 437 | 437 | 436 | 434 | 432 | 430 | 427 | 423 | 421 | 419 | 414 |
| **Durban MGO** | 511 | 510 | 509 | 510 | 507 | 505 | 502 | 501 | 499 | 496 | 493 | 484 |
| **Rotterdam VLSFO** | 468 | 467 | 466 | 464 | 463 | 461 | 460 | 458 | 456 | 454 | 453 | 450 |
| **Rotterdam MGO** | 615 | 613 | 610 | 608 | 606 | 604 | 603 | 601 | 600 | 598 | 597 | 595 |
| **Gibraltar VLSFO** | 475 | 474 | 473 | 472 | 470 | 468 | 466 | 464 | 462 | 460 | 458 | 455 |
| **Gibraltar MGO** | 625 | 623 | 621 | 619 | 617 | 615 | 614 | 612 | 610 | 608 | 606 | 604 |
| **Port Louis VLSFO** | 455 | 454 | 454 | 453 | 451 | 449 | 448 | 446 | 444 | 442 | 440 | 438 |
| **Port Louis MGO** | 585 | 583 | 581 | 580 | 579 | 578 | 577 | 576 | 575 | 574 | 573 | 570 |
| **Qingdao VLSFO** | 648 | 643 | 639 | 636 | 633 | 630 | 628 | 626 | 624 | 622 | 620 | 616 |
| **Qingdao MGO** | 838 | 833 | 828 | 825 | 823 | 822 | 821 | 820 | 818 | 817 | 816 | 815 |
| **Shanghai VLSFO** | 650 | 645 | 638 | 636 | 633 | 632 | 630 | 634 | 633 | 631 | 630 | 627 |
| **Shanghai MGO** | 841 | 836 | 829 | 826 | 824 | 823 | 822 | 822 | 820 | 818 | 816 | 818 |
| **Richards Bay VLSFO** | 442 | 441 | 441 | 440 | 438 | 436 | 434 | 432 | 430 | 428 | 426 | 423 |
| **Richards Bay MGO** | 520 | 519 | 518 | 516 | 514 | 512 | 510 | 508 | 506 | 504 | 502 | 500 |

---

## Datathon Scoring Guide (Grading Bands)

### Band Definitions

| Band | Label | Interpretation |
|------|-------|----------------|
| 4 | **Excellent** | Professional quality, robust, clearly above expectations |
| 3 | **Good** | Solid, correct, meets expectations |
| 2 | **Fair** | Partially correct, noticeable gaps |
| 1 | **Weak** | Major issues, incomplete or incorrect |

### Analytical Accuracy (25%)

| Band | Should meet |
|------|-------------|
| 4 | All calculations correct and internally consistent; assumptions clearly stated and justified; constraints correctly applied; effective and appropriate use of AI/ML models; no material logical errors. |
| 3 | Mostly correct calculations and logic; minor issues that do not materially affect results; AI/ML usage is sensible but not deeply optimized. |
| 2 | Several inconsistencies or oversimplified assumptions; partial or incorrect constraint handling; AI/ML applied superficially or incorrectly in places. |
| 1 | Major calculation errors; flawed logic or methodology; misuse or misunderstanding of AI/ML; results unreliable. |

### Commercial Insight (20%)

| Band | Should meet |
|------|-------------|
| 4 | Insights are actionable, commercially realistic, and clearly tied to data; recommendations directly address the core problem; strong understanding of trade-offs and implications. |
| 3 | Clear interpretation of results with reasonable recommendations; insights mostly relevant but may lack depth or prioritization. |
| 2 | Observations are descriptive rather than insightful; recommendations are generic or weakly supported by analysis. |
| 1 | Little or no business interpretation; conclusions disconnected from data; recommendations unclear or impractical. |

### Clarity & Storytelling (20%)

| Band | Should meet |
|------|-------------|
| 4 | Very clear structure and flow; effective visualizations; assumptions, methods, and insights explained concisely; presentation easy to follow for non-technical judges. |
| 3 | Generally clear narrative; visuals support the message; some sections could be better structured or explained. |
| 2 | Disorganized or overly technical; visuals unclear or poorly labeled; key ideas require effort to understand. |
| 1 | Confusing presentation; missing explanations; visuals misleading or absent; difficult to follow overall logic. |

### Scenario Sensitivity (15%)

| Band | Should meet |
|------|-------------|
| 4 | Multiple well-designed scenarios tested; clear comparison to baseline; strong explanation of drivers and implications for decisions. |
| 3 | At least one meaningful alternative scenario; results compared to baseline; reasonable interpretation of impacts. |
| 2 | Scenarios tested mechanically with limited explanation; weak linkage to decisions or strategy. |
| 1 | No meaningful scenario analysis; changes unexplained or incorrectly applied. |

### Innovation (20%)

| Band | Should meet |
|------|-------------|
| 4 | Highly creative approach; effective and thoughtful use of Generative AI or advanced methods; innovation clearly enhances insight or decision-making. |
| 3 | Some originality or novel features; AI/tools used appropriately but not deeply integrated. |
| 2 | Limited innovation; use of AI or tools is basic or cosmetic. |
| 1 | No evident innovation; tools or AI not used or used inappropriately. |

---

## Contact & Support

For questions or to request OpenAI API key:
**Email:** hector_ordonez@cargill.com

---

**Thank you and we look forward to your submission!** 🚢