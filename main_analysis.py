"""
Main Analysis Script
- Uses FreightCalculator to compute best vessel-cargo outcomes
- Optimizes assignment for Cargill committed cargoes (4 vessels x 3 cargoes)
- NO distance estimation: if distance missing in Port_Distances.csv -> infeasible
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import warnings

from freight_calculator import FreightCalculator, load_bunker_prices_flat
from vessel_cargo_data import (
    CARGILL_VESSELS,
    CARGILL_CARGOES,
    MARKET_VESSELS,
    MARKET_CARGOES,
)

warnings.filterwarnings("ignore")


def _select_best_result(results: List[Dict[str, Any]], qty_label: str = "base") -> Optional[Dict[str, Any]]:
    """
    Select best feasible result for a vessel-cargo pair:
    - Prefer P&L with max voyage_profit
    - QUOTE_* treated as 0 (break-even quote)
    """
    feasible = [r for r in results if r.get("feasible") and r.get("qty_label") == qty_label]
    if not feasible:
        return None

    def score(r: Dict[str, Any]) -> float:
        if r.get("mode") == "P&L":
            return float(r.get("voyage_profit", -1e30))
        if r.get("mode") in ("QUOTE_HIRE", "QUOTE_FREIGHT"):
            return 0.0
        return -1e30

    return max(feasible, key=score)


def build_combinations_df(
    calc: FreightCalculator,
    vessels: List[Dict[str, Any]],
    cargoes: List[Dict[str, Any]],
    qty_label: str = "base",
    try_both_speed_modes: bool = True,
    use_rob: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Compute best result per vessel-cargo pair (optionally across eco+war),
    returns a flat DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for v in vessels:
        for c in cargoes:
            candidates: List[Dict[str, Any]] = []
            if try_both_speed_modes:
                candidates.extend(calc.evaluate_vessel_cargo(v, c, use_economical_speed=True, use_rob=use_rob))
                candidates.extend(calc.evaluate_vessel_cargo(v, c, use_economical_speed=False, use_rob=use_rob))
            else:
                candidates.extend(calc.evaluate_vessel_cargo(v, c, use_economical_speed=True, use_rob=use_rob))

            best = _select_best_result(candidates, qty_label=qty_label)

            if best is None:
                issue = None
                for r in candidates:
                    if not r.get("feasible"):
                        issue = r.get("issue")
                        break

                rows.append({
                    "vessel_name": v.get("name"),
                    "cargo_name": c.get("name"),
                    "feasible": False,
                    "issue": issue or "no_feasible_result",
                })
                continue

            base_row = {
                "vessel_name": best.get("vessel"),
                "cargo_name": best.get("cargo"),
                "feasible": True,
                "mode": best.get("mode"),
                "speed_mode": best.get("speed_mode"),
                "qty_label": best.get("qty_label"),
                "qty": best.get("qty"),
                "total_days": best.get("total_days"),
                "tce": best.get("tce"),
                "voyage_profit": best.get("voyage_profit"),
                "net_revenue": best.get("net_revenue"),
                "bunker_cost": best.get("bunker_cost"),
                "hire_cost": best.get("hire_cost"),
                "port_cost": best.get("port_cost"),
                "required_freight_rate_usd_per_mt": best.get("required_freight_rate_usd_per_mt"),
                "required_hire_rate_usd_per_day": best.get("required_hire_rate_usd_per_day"),
                "ballast_nm": best.get("ballast_nm"),
                "laden_nm": best.get("laden_nm"),
                "waiting_days": best.get("waiting_days"),
                "bunker_region": best.get("bunker_region"),
                "vlsf_price": best.get("vlsf_price"),
                "mgo_price": best.get("mgo_price"),
            }
            rows.append(base_row)

    return pd.DataFrame(rows)


def main():
    print("Cargill Ocean Transportation Datathon 2026")
    print("Freight Calculator - Main Analysis\n")

    # Load inputs
    distances = pd.read_csv("Port_Distances.csv")
    bunker_prices = load_bunker_prices_flat()

    calc = FreightCalculator(distances, bunker_prices)
    print(f"✓ Loaded {len(distances)} distance rows")
    print(f"✓ Loaded bunker prices for {len(bunker_prices)} regions\n")

    # 1) Compute full table
    all_vessels = CARGILL_VESSELS + MARKET_VESSELS
    all_cargoes = CARGILL_CARGOES + MARKET_CARGOES

    print(f"Computing best outcome per vessel-cargo pair (no distance assumptions)...")
    combos_df = build_combinations_df(
        calc,
        vessels=all_vessels,
        cargoes=all_cargoes,
        qty_label="base",
        try_both_speed_modes=True,
        use_rob=None,
    )
    combos_df.to_csv("all_best_combinations.csv", index=False)
    print(f"✓ Saved: all_best_combinations.csv")
    print(f"  Feasible pairs: {int((combos_df['feasible'] == True).sum())} / {len(combos_df)}\n")

    # 2) Optimize
    print("Optimizing Cargill committed cargo assignment (brute-force)...")
    opt = calc.optimize_committed_cargo_assignment(
        vessels=CARGILL_VESSELS,
        committed_cargoes=CARGILL_CARGOES,
        qty_label="base",
        try_both_speed_modes=True,
        use_rob=None,
    )

    if not opt.get("feasible"):
        print("✗ No feasible committed assignment found.")
        print(opt)
        return combos_df, opt

    # Save optimal assignment
    alloc_df = pd.DataFrame(opt["assignments"])
    alloc_df.to_csv("optimal_committed_assignment.csv", index=False)
    print("✓ Saved: optimal_committed_assignment.csv")
    print(f"TOTAL PROFIT (committed): {opt['total_profit']:.2f}")
    print(f"Unused vessels: {', '.join(opt.get('unused_vessels', []))}\n")

    return combos_df, opt


if __name__ == "__main__":
    main()
