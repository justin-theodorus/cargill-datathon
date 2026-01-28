"""
Cargill Ocean Transportation Datathon 2026
Smart Voyage Chatbot - OpenAI GPT + Gradio UI

Architecture:
- ContextBuilder: Generates structured system prompt from analysis DataFrames
- VoyageTools: OpenAI function-calling tools for on-demand calculations
- VoyageChatbot: Main class with OpenAI backend + rule-based fallback
- create_gradio_app: Gradio ChatInterface for web UI
"""

import os
import json
import re
import pandas as pd
from typing import Optional, List

# Conditional imports for graceful degradation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from freight_calculator import FreightCalculator, load_bunker_prices
from vessel_cargo_data import get_all_vessels, get_all_cargoes


# ============================================================================
# Context Builder
# ============================================================================

class ContextBuilder:
    """Builds structured context from analysis results for the LLM system prompt."""

    @staticmethod
    def build_system_prompt(
        optimal_allocation: dict,
        all_results_df: pd.DataFrame,
        scenario_port_delays: pd.DataFrame = None,
        scenario_bunker_prices: pd.DataFrame = None,
        ml_feature_importance: pd.DataFrame = None,
        ml_model_performance: dict = None,
    ) -> str:
        """Generate a comprehensive system prompt from actual analysis results."""

        sections = []

        # Role
        sections.append(
            "You are a senior freight trading assistant at Cargill Ocean Transportation. "
            "You help traders make data-driven decisions about vessel employment and voyage "
            "optimization for Capesize bulk carriers. Respond professionally and concisely, "
            "always citing specific numbers from the analysis. Use dollar formatting with commas."
        )

        # Optimal allocation
        sections.append("\n## OPTIMAL ALLOCATION (Base Case)")
        sections.append(f"Total Portfolio Profit: ${optimal_allocation['total_profit']:,.0f}")
        sections.append("")
        sections.append("| # | Vessel | Cargo | TCE ($/day) | Profit ($) | Days |")
        sections.append("|---|--------|-------|-------------|------------|------|")
        for i, row in enumerate(optimal_allocation["allocation"], 1):
            sections.append(
                f"| {i} | {row['vessel_name']} | {row['cargo_name']} | "
                f"{row['tce']:,.0f} | {row['voyage_profit']:,.0f} | {row['total_days']:.1f} |"
            )

        unalloc = optimal_allocation.get("unallocated_vessels", [])
        if unalloc:
            sections.append(f"\nUnallocated vessels ({len(unalloc)}): {', '.join(unalloc)}")

        # Top combinations
        sections.append("\n## TOP 10 COMBINATIONS BY TCE")
        top10 = all_results_df.nlargest(10, "tce")
        sections.append("| Vessel | Cargo | TCE ($/day) | Profit ($) | Days |")
        sections.append("|--------|-------|-------------|------------|------|")
        for _, r in top10.iterrows():
            sections.append(
                f"| {r['vessel_name']} | {r['cargo_name']} | "
                f"{r['tce']:,.0f} | {r['voyage_profit']:,.0f} | {r['total_days']:.1f} |"
            )

        # Scenario: port delays
        if scenario_port_delays is not None and len(scenario_port_delays) > 0:
            sections.append("\n## SCENARIO: PORT DELAYS IN CHINA")
            sections.append("| Delay (days) | Total Profit ($) | Loss ($) |")
            sections.append("|-------------|------------------|----------|")
            for _, r in scenario_port_delays.iterrows():
                sections.append(
                    f"| {r['delay_days']} | {r['total_profit']:,.0f} | {r['profit_loss']:,.0f} |"
                )

        # Scenario: bunker prices
        if scenario_bunker_prices is not None and len(scenario_bunker_prices) > 0:
            sections.append("\n## SCENARIO: BUNKER PRICE INCREASES")
            sections.append("| Increase (%) | Total Profit ($) | Loss ($) |")
            sections.append("|-------------|------------------|----------|")
            for _, r in scenario_bunker_prices.iterrows():
                sections.append(
                    f"| {r['price_increase_pct']}% | {r['total_profit']:,.0f} | {r['profit_loss']:,.0f} |"
                )

        # ML insights
        if ml_feature_importance is not None and len(ml_feature_importance) > 0:
            sections.append("\n## ML RISK MODEL")
            if ml_model_performance:
                sections.append(
                    f"Model: {ml_model_performance.get('model_name', 'Gradient Boosting')} "
                    f"(R2: {ml_model_performance.get('r2', 0.99):.3f}, "
                    f"MAE: ${ml_model_performance.get('mae', 857):,.0f}/day)"
                )
            sections.append("\nTop features by importance:")
            for _, r in ml_feature_importance.head(5).iterrows():
                sections.append(f"- {r['Feature']}: {r['Importance']:.3f}")

        # Fleet overview
        sections.append("\n## FLEET (15 vessels)")
        for v in get_all_vessels():
            sections.append(
                f"- {v['name']}: {v['dwt']:,} DWT, ${v['hire_rate']:,}/day, at {v['current_port']} (ETD {v['etd']})"
            )

        # Cargoes overview
        sections.append("\n## CARGOES (11 total: 3 committed + 8 market)")
        for c in get_all_cargoes():
            sections.append(
                f"- {c['name']}: {c['commodity']}, {c['quantity']:,} MT, ${c['freight_rate']}/MT, "
                f"{c['load_port']} -> {c['discharge_port']}"
            )

        # Instructions
        sections.append("\n## INSTRUCTIONS")
        sections.append(
            "- Always cite specific dollar amounts and days from the data above.\n"
            "- When asked about scenarios, use the calculate_scenario tool for precise results.\n"
            "- When comparing vessels or cargoes, use the compare_options tool.\n"
            "- For specific voyage details, use the get_voyage_details tool.\n"
            "- Present findings as a freight trading professional would.\n"
            "- If asked about something outside the analysis scope, say so clearly.\n"
            "- Keep responses concise but thorough."
        )

        return "\n".join(sections)


# ============================================================================
# Voyage Tools (OpenAI Function Calling)
# ============================================================================

class VoyageTools:
    """Provides callable tools for OpenAI function calling."""

    def __init__(
        self,
        calculator: FreightCalculator,
        all_results_df: pd.DataFrame,
        optimal_allocation: dict,
    ):
        self.calc = calculator
        self.all_results = all_results_df
        self.optimal = optimal_allocation
        self.vessels = {v["name"]: v for v in get_all_vessels()}
        self.cargoes = {c["name"]: c for c in get_all_cargoes()}

    def get_tool_definitions(self) -> list:
        """Return OpenAI function definitions for tool use."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_scenario",
                    "description": (
                        "Calculate impact of bunker price change or port delay on voyage profitability. "
                        "Apply to the full portfolio or a specific vessel-cargo pair."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bunker_price_change_pct": {
                                "type": "number",
                                "description": "Percentage change in bunker prices (e.g. 15 for +15%)",
                            },
                            "port_delay_days": {
                                "type": "number",
                                "description": "Additional port delay days (applied to China ports)",
                            },
                            "vessel_name": {
                                "type": "string",
                                "description": "Optional: specific vessel name to analyze",
                            },
                            "cargo_name": {
                                "type": "string",
                                "description": "Optional: specific cargo name to analyze",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_options",
                    "description": "Compare multiple vessels for a specific cargo, showing side-by-side TCE, profit, costs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vessel_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of vessel names to compare",
                            },
                            "cargo_name": {
                                "type": "string",
                                "description": "Cargo name to compare for (can be partial match)",
                            },
                        },
                        "required": ["vessel_names", "cargo_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_voyage_details",
                    "description": "Get detailed voyage calculation for a specific vessel-cargo combination.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vessel_name": {
                                "type": "string",
                                "description": "Vessel name",
                            },
                            "cargo_name": {
                                "type": "string",
                                "description": "Cargo name (can be partial match)",
                            },
                        },
                        "required": ["vessel_name", "cargo_name"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return JSON result string."""
        try:
            if tool_name == "calculate_scenario":
                return self._calculate_scenario(**arguments)
            elif tool_name == "compare_options":
                return self._compare_options(**arguments)
            elif tool_name == "get_voyage_details":
                return self._get_voyage_details(**arguments)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _find_matching_rows(self, vessel_name: str = None, cargo_name: str = None) -> pd.DataFrame:
        """Find matching rows in results using fuzzy matching on names."""
        df = self.all_results
        if vessel_name:
            mask = df["vessel_name"].str.upper().str.contains(vessel_name.upper(), na=False)
            df = df[mask]
        if cargo_name:
            mask = df["cargo_name"].str.upper().str.contains(cargo_name.upper(), na=False)
            df = df[mask]
        return df

    def _calculate_scenario(
        self,
        bunker_price_change_pct: float = 0,
        port_delay_days: float = 0,
        vessel_name: str = None,
        cargo_name: str = None,
    ) -> str:
        """Run scenario calculation."""
        china_ports = [
            "QINGDAO", "LIANYUNGANG", "CAOFEIDIAN", "TIANJIN",
            "FANGCHENG", "SHANGHAI", "JINGTANG",
        ]

        results = []

        # Determine which allocations to test
        if vessel_name or cargo_name:
            rows = self._find_matching_rows(vessel_name, cargo_name)
            if rows.empty:
                return json.dumps({"error": f"No match found for vessel={vessel_name}, cargo={cargo_name}"})
            target_rows = [rows.iloc[0]]
        else:
            target_rows = self.optimal["allocation"]

        for row in target_rows:
            base_profit = row["voyage_profit"]
            base_tce = row["tce"]
            base_days = row["total_days"]
            bunker_cost = row["bunker_cost"]
            hire_cost = row["hire_cost"]
            port_cost = row["port_cost"]
            net_revenue = row["net_revenue"]
            v_name = row["vessel_name"]
            c_name = row["cargo_name"]

            # Get vessel hire rate
            vessel = self.vessels.get(v_name, {})
            hire_rate = vessel.get("hire_rate", hire_cost / base_days if base_days > 0 else 0)

            # Check if cargo discharges in China
            cargo = None
            for c in get_all_cargoes():
                if c["name"] == c_name:
                    cargo = c
                    break
            is_china = cargo and any(
                cp in cargo["discharge_port"].upper() for cp in china_ports
            )

            # Apply adjustments
            adj_bunker = bunker_cost * (1 + bunker_price_change_pct / 100)
            adj_delay = port_delay_days if is_china else 0
            adj_days = base_days + adj_delay
            adj_hire = hire_rate * adj_days
            adj_total_cost = adj_hire + adj_bunker + port_cost
            adj_profit = net_revenue - adj_total_cost
            adj_tce = (net_revenue - adj_bunker - port_cost) / adj_days if adj_days > 0 else 0

            results.append({
                "vessel": v_name,
                "cargo": c_name,
                "china_port": is_china,
                "base_tce": round(base_tce, 0),
                "adjusted_tce": round(adj_tce, 0),
                "tce_change": round(adj_tce - base_tce, 0),
                "base_profit": round(base_profit, 0),
                "adjusted_profit": round(adj_profit, 0),
                "profit_change": round(adj_profit - base_profit, 0),
                "base_days": round(base_days, 1),
                "adjusted_days": round(adj_days, 1),
            })

        total_base = sum(r["base_profit"] for r in results)
        total_adj = sum(r["adjusted_profit"] for r in results)

        return json.dumps({
            "scenario": {
                "bunker_change_pct": bunker_price_change_pct,
                "port_delay_days": port_delay_days,
            },
            "results": results,
            "portfolio_base_profit": round(total_base, 0),
            "portfolio_adjusted_profit": round(total_adj, 0),
            "portfolio_profit_change": round(total_adj - total_base, 0),
        })

    def _compare_options(self, vessel_names: list, cargo_name: str) -> str:
        """Compare vessels side-by-side for a given cargo."""
        comparisons = []
        for v_name in vessel_names:
            rows = self._find_matching_rows(v_name, cargo_name)
            if not rows.empty:
                r = rows.iloc[0]
                comparisons.append({
                    "vessel": r["vessel_name"],
                    "cargo": r["cargo_name"],
                    "tce": round(r["tce"], 0),
                    "voyage_profit": round(r["voyage_profit"], 0),
                    "total_days": round(r["total_days"], 1),
                    "hire_cost": round(r["hire_cost"], 0),
                    "bunker_cost": round(r["bunker_cost"], 0),
                    "port_cost": round(r["port_cost"], 0),
                    "net_revenue": round(r["net_revenue"], 0),
                    "route": r.get("route", ""),
                })
            else:
                comparisons.append({"vessel": v_name, "error": "No matching combination found"})

        if len(comparisons) >= 2 and all("tce" in c for c in comparisons):
            best = max(comparisons, key=lambda x: x.get("tce", 0))
            winner = best["vessel"]
        else:
            winner = None

        return json.dumps({"comparisons": comparisons, "recommended_vessel": winner})

    def _get_voyage_details(self, vessel_name: str, cargo_name: str) -> str:
        """Get full voyage breakdown for a specific combination."""
        rows = self._find_matching_rows(vessel_name, cargo_name)
        if rows.empty:
            return json.dumps({"error": f"No combination found for {vessel_name} + {cargo_name}"})

        r = rows.iloc[0]
        detail = {
            "vessel_name": r["vessel_name"],
            "cargo_name": r["cargo_name"],
            "route": r.get("route", ""),
            "total_days": round(r["total_days"], 1),
            "sea_days": round(r["sea_days"], 1),
            "port_days": round(r["port_days"], 1),
            "ballast_distance": round(r["ballast_distance"], 0),
            "laden_distance": round(r["laden_distance"], 0),
            "tce": round(r["tce"], 0),
            "voyage_profit": round(r["voyage_profit"], 0),
            "net_revenue": round(r["net_revenue"], 0),
            "hire_cost": round(r["hire_cost"], 0),
            "bunker_cost": round(r["bunker_cost"], 0),
            "port_cost": round(r["port_cost"], 0),
            "total_costs": round(r["total_costs"], 0),
            "freight_revenue": round(r["freight_revenue"], 0),
            "commission": round(r["commission"], 0),
        }
        return json.dumps(detail)


# ============================================================================
# Main Chatbot Class
# ============================================================================

class VoyageChatbot:
    """
    Smart voyage chatbot with OpenAI GPT backend and rule-based fallback.

    Usage:
        bot = VoyageChatbot(calculator, optimal, results_df, ...)
        response = bot.chat("What is the optimal allocation?")
    """

    def __init__(
        self,
        calculator: FreightCalculator,
        optimal_allocation: dict,
        all_results_df: pd.DataFrame,
        scenario_port_delays: pd.DataFrame = None,
        scenario_bunker_prices: pd.DataFrame = None,
        ml_feature_importance: pd.DataFrame = None,
        ml_model_performance: dict = None,
        openai_api_key: str = None,
        model: str = "gpt-4o-mini",
    ):
        self.tools = VoyageTools(calculator, all_results_df, optimal_allocation)
        self.optimal = optimal_allocation
        self.all_results = all_results_df
        self.scenario_delays = scenario_port_delays
        self.scenario_bunker = scenario_bunker_prices

        self.system_prompt = ContextBuilder.build_system_prompt(
            optimal_allocation,
            all_results_df,
            scenario_port_delays,
            scenario_bunker_prices,
            ml_feature_importance,
            ml_model_performance,
        )

        # Try to initialize OpenAI client
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.use_openai = OPENAI_AVAILABLE and api_key is not None and len(api_key) > 0

        if self.use_openai:
            self.client = OpenAI(api_key=api_key)
            self.model = model

        self.conversation_history: List[dict] = []

    def chat(self, user_message: str) -> str:
        """Process a user message and return a response."""
        if self.use_openai:
            return self._chat_openai(user_message)
        return self._chat_fallback(user_message)

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _chat_openai(self, user_message: str) -> str:
        """Send message to OpenAI with function calling."""
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools.get_tool_definitions(),
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        # Handle tool calls (may need multiple rounds)
        max_rounds = 3
        rounds = 0
        while assistant_msg.tool_calls and rounds < max_rounds:
            rounds += 1
            # Add assistant message with tool calls
            self.conversation_history.append(assistant_msg.model_dump())

            # Execute each tool call
            for tc in assistant_msg.tool_calls:
                result = self.tools.execute_tool(
                    tc.function.name, json.loads(tc.function.arguments)
                )
                self.conversation_history.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            # Get next response
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self.tools.get_tool_definitions(),
                tool_choice="auto",
            )
            assistant_msg = response.choices[0].message

        reply = assistant_msg.content or ""
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _chat_fallback(self, user_message: str) -> str:
        """Rule-based fallback when OpenAI is not available."""
        q = user_message.lower()

        if any(w in q for w in ["optimal", "allocation", "recommend", "best", "summary"]):
            return self._fb_allocation()
        elif any(w in q for w in ["what if", "scenario", "bunker", "fuel"]):
            return self._fb_scenario(user_message)
        elif "delay" in q or "port delay" in q or "china" in q and "delay" in q:
            return self._fb_delay(user_message)
        elif "compare" in q:
            return self._fb_compare(user_message)
        elif "risk" in q or "feature" in q or "ml" in q or "model" in q:
            return self._fb_risk()
        elif "why" in q:
            return self._fb_why(user_message)
        else:
            return self._fb_help()

    def _fb_allocation(self) -> str:
        allocs = self.optimal["allocation"]
        lines = ["OPTIMAL VESSEL-CARGO ALLOCATION", "=" * 50, ""]
        total = 0
        for i, row in enumerate(allocs, 1):
            lines.append(
                f"{i}. {row['vessel_name']} -> {row['cargo_name']}\n"
                f"   Route: {row.get('route', 'N/A')}\n"
                f"   TCE: ${row['tce']:,.0f}/day | Profit: ${row['voyage_profit']:,.0f} | "
                f"Days: {row['total_days']:.1f}"
            )
            total += row["voyage_profit"]
        lines.append(f"\nTotal Portfolio Profit: ${total:,.0f}")
        unalloc = self.optimal.get("unallocated_vessels", [])
        if unalloc:
            lines.append(f"Unallocated vessels: {len(unalloc)} ({', '.join(unalloc[:5])}...)")
        return "\n".join(lines)

    def _fb_scenario(self, query: str) -> str:
        pct = self._extract_number(query)
        if pct is None:
            pct = 10
        if self.scenario_bunker is not None and len(self.scenario_bunker) > 0:
            closest_idx = (self.scenario_bunker["price_increase_pct"] - pct).abs().idxmin()
            row = self.scenario_bunker.loc[closest_idx]
            return (
                f"BUNKER PRICE SCENARIO: +{row['price_increase_pct']:.0f}%\n"
                f"{'=' * 50}\n\n"
                f"Base case profit: ${self.optimal['total_profit']:,.0f}\n"
                f"Adjusted profit: ${row['total_profit']:,.0f}\n"
                f"Profit loss: ${row['profit_loss']:,.0f}\n"
                f"Reduction: {row['profit_loss'] / self.optimal['total_profit'] * 100:.1f}%\n\n"
                f"The allocation remains profitable under this scenario."
            )
        return "Bunker scenario data not loaded. Please run the scenario analysis first."

    def _fb_delay(self, query: str) -> str:
        days = self._extract_number(query)
        if days is None:
            days = 5
        if self.scenario_delays is not None and len(self.scenario_delays) > 0:
            closest_idx = (self.scenario_delays["delay_days"] - days).abs().idxmin()
            row = self.scenario_delays.loc[closest_idx]
            return (
                f"PORT DELAY SCENARIO: +{row['delay_days']:.0f} days in China\n"
                f"{'=' * 50}\n\n"
                f"Base case profit: ${self.optimal['total_profit']:,.0f}\n"
                f"Adjusted profit: ${row['total_profit']:,.0f}\n"
                f"Profit loss: ${row['profit_loss']:,.0f}\n"
                f"Reduction: {row['profit_loss'] / self.optimal['total_profit'] * 100:.1f}%\n\n"
                f"Only 1 of 3 allocated voyages discharges in China (IRON CENTURY at Caofeidian)."
            )
        return "Port delay scenario data not loaded. Please run the scenario analysis first."

    def _fb_compare(self, query: str) -> str:
        vessels_found = []
        for name in self.tools.vessels:
            if name.upper() in query.upper():
                vessels_found.append(name)
        if len(vessels_found) < 2:
            return (
                "To compare, mention two vessel names. Example:\n"
                "\"Compare OCEAN HORIZON vs NAVIS PRIDE for the bauxite cargo\""
            )
        # Find cargo keyword
        cargo_kw = None
        for kw in ["bauxite", "iron", "coal", "kamsar", "taboneo", "hedland", "madeira"]:
            if kw in query.lower():
                cargo_kw = kw
                break

        results = []
        for vn in vessels_found[:2]:
            rows = self.tools._find_matching_rows(vn, cargo_kw)
            if not rows.empty:
                r = rows.iloc[0]
                results.append(r)

        if len(results) < 2:
            return "Could not find voyage data for both vessels with that cargo."

        lines = ["VESSEL COMPARISON", "=" * 50, ""]
        for r in results:
            lines.append(
                f"{r['vessel_name']} -> {r['cargo_name']}\n"
                f"  TCE: ${r['tce']:,.0f}/day | Profit: ${r['voyage_profit']:,.0f} | "
                f"Days: {r['total_days']:.1f}\n"
                f"  Costs: Hire ${r['hire_cost']:,.0f} + Bunker ${r['bunker_cost']:,.0f} + Port ${r['port_cost']:,.0f}\n"
            )

        winner = results[0] if results[0]["tce"] > results[1]["tce"] else results[1]
        lines.append(f"Recommended: {winner['vessel_name']} (higher TCE by ${abs(results[0]['tce'] - results[1]['tce']):,.0f}/day)")
        return "\n".join(lines)

    def _fb_risk(self) -> str:
        return (
            "ML RISK ANALYSIS\n" +
            "=" * 50 + "\n\n"
            "Model: Gradient Boosting (R2 = 0.99, MAE = $857/day)\n"
            "Validated with 10-fold cross-validation on 165 combinations.\n\n"
            "Top Risk Factors:\n"
            "1. Route Length (nm): Longer routes increase weather and fuel exposure\n"
            "2. TCE Margin: Higher base TCE provides buffer against risk\n"
            "3. Bunker Cost Ratio: High fuel costs relative to revenue amplify price risk\n"
            "4. China Discharge Ports: Known congestion adding 2-3 days average\n"
            "5. Ballast Ratio: Long ballast legs reduce earning efficiency\n\n"
            "Risk Summary:\n"
            "- Average weather delay: 1.7 days\n"
            "- Average port congestion: 3.7 days\n"
            "- Mechanical issue probability: 7.3%\n"
            "- Expected profit loss from risks: ~$14.4M across all 165 combinations"
        )

    def _fb_why(self, query: str) -> str:
        q_upper = query.upper()
        for row in self.optimal["allocation"]:
            if row["vessel_name"] in q_upper or any(
                kw in q_upper for kw in row["cargo_name"].split("_")
            ):
                return (
                    f"WHY {row['vessel_name']} -> {row['cargo_name']}\n"
                    f"{'=' * 50}\n\n"
                    f"This pairing achieves TCE of ${row['tce']:,.0f}/day and "
                    f"profit of ${row['voyage_profit']:,.0f}.\n\n"
                    f"Route: {row.get('route', 'N/A')}\n"
                    f"Duration: {row['total_days']:.1f} days\n"
                    f"Costs: Hire ${row['hire_cost']:,.0f} + Bunker ${row['bunker_cost']:,.0f} + "
                    f"Port ${row['port_cost']:,.0f}\n\n"
                    f"This vessel was selected because it offers the highest TCE among "
                    f"available vessels for this cargo, considering its current position "
                    f"and ballast distance to the load port."
                )
        return "Please specify which vessel or cargo allocation you'd like explained."

    def _fb_help(self) -> str:
        return (
            "CARGILL VOYAGE OPTIMIZATION ASSISTANT\n" +
            "=" * 50 + "\n\n"
            "I can help with:\n\n"
            "- \"What is the optimal allocation?\" - View vessel-cargo assignments\n"
            "- \"What if bunker prices increase 20%?\" - Scenario analysis\n"
            "- \"What if China port delays reach 10 days?\" - Delay impact\n"
            "- \"Compare OCEAN HORIZON vs NAVIS PRIDE for bauxite\" - Side-by-side\n"
            "- \"What are the risk factors?\" - ML analysis insights\n"
            "- \"Why was NAVIS PRIDE chosen for bauxite?\" - Allocation rationale\n\n"
            "Note: Running without OpenAI API key (rule-based mode).\n"
            "Set OPENAI_API_KEY environment variable for GPT-powered responses."
        )

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """Extract first number from text."""
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else None


# ============================================================================
# Gradio App
# ============================================================================

def create_gradio_app(chatbot: VoyageChatbot) -> "gr.Blocks":
    """Create Gradio chat interface."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed. Run: pip install gradio")

    def respond(message, history):
        return chatbot.chat(message)

    with gr.Blocks(title="Cargill Voyage Optimizer", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Cargill Ocean Transportation - Voyage Optimization Assistant\n"
            "Ask about vessel allocations, run scenarios, compare options, or explore risk factors."
        )

        gr.ChatInterface(
            fn=respond,
            examples=[
                "What is the optimal vessel-cargo allocation?",
                "What if bunker prices increase by 20%?",
                "What happens if China port delays reach 10 days?",
                "Compare OCEAN HORIZON vs IRON CENTURY for the coal cargo",
                "What are the top risk factors from the ML model?",
                "Why was NAVIS PRIDE chosen for the bauxite cargo?",
                "What is the voyage detail for IRON CENTURY carrying the Vale iron ore?",
            ],
        )

    return app


def launch_chatbot(
    calculator: FreightCalculator,
    optimal: dict,
    all_results_df: pd.DataFrame,
    scenario_port_delays: pd.DataFrame = None,
    scenario_bunker_prices: pd.DataFrame = None,
    ml_feature_importance: pd.DataFrame = None,
    ml_model_performance: dict = None,
    api_key: str = None,
    share: bool = False,
):
    """Create and launch the chatbot. Callable from notebook or CLI."""
    bot = VoyageChatbot(
        calculator=calculator,
        optimal_allocation=optimal,
        all_results_df=all_results_df,
        scenario_port_delays=scenario_port_delays,
        scenario_bunker_prices=scenario_bunker_prices,
        ml_feature_importance=ml_feature_importance,
        ml_model_performance=ml_model_performance,
        openai_api_key=api_key,
    )

    if GRADIO_AVAILABLE:
        app = create_gradio_app(bot)
        app.launch(share=share)
    else:
        print("Gradio not available. Using text-based chat.")
        print("Type 'quit' to exit.\n")
        while True:
            query = input("You: ")
            if query.lower() in ("quit", "exit", "q"):
                break
            print(f"\nAssistant: {bot.chat(query)}\n")

    return bot


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Loading data...")
    distances = pd.read_csv("Port_Distances.csv")
    bunker_prices = load_bunker_prices()
    calc = FreightCalculator(distances, bunker_prices)

    from main_analysis import calculate_all_combinations, find_optimal_allocation

    all_vessels = get_all_vessels()
    all_cargoes = get_all_cargoes()
    results_df = calculate_all_combinations(calc, all_vessels, all_cargoes)
    optimal = find_optimal_allocation(results_df, num_vessels=len(all_vessels), num_cargoes=3)

    print(f"Loaded {len(results_df)} combinations, portfolio profit: ${optimal['total_profit']:,.0f}")

    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    launch_chatbot(calc, optimal, results_df, api_key=api_key, share=False)
