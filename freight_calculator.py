"""
Cargill Ocean Transportation Datathon 2026
Freight Calculator for Capesize Vessels
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from itertools import permutations, combinations

class FreightCalculator:
    def __init__(
        self,
        distances_df: pd.DataFrame,
        bunker_prices: Dict,
        bunker_forward_curve: Optional[Dict] = None,
        default_use_rob: bool = True,
    ):
        """
        Args:
            distances_df: DataFrame with PORT_NAME_FROM, PORT_NAME_TO, DISTANCE
            bunker_prices: fallback flat prices (region -> {VLSF, MGO})
            bunker_forward_curve: optional forward curve:
                {
                  "SINGAPORE": {"VLSF": {"2026-02": 491, "2026-03": 490, ...},
                                "MGO":  {"2026-02": 654, ...}},
                  ...
                }
            default_use_rob: apply ROB bunkers costing if vessel has bunker_vlsf/bunker_mgo
        """
        self.distances = distances_df.copy()
        self.bunker_prices = bunker_prices
        self.bunker_forward_curve = bunker_forward_curve
        self.default_use_rob = default_use_rob

        self._create_distance_lookup()
    
    def _create_distance_lookup(self):
        """Directed lookup: store exactly as table provides (no reverse). Also stores known ports for safe matching."""
        required = {"PORT_NAME_FROM", "PORT_NAME_TO", "DISTANCE"}
        missing = required - set(self.distances.columns)
        if missing:
            raise ValueError(f"distances_df missing columns: {missing}")

        self.distance_dict: Dict[Tuple[str, str], float] = {}

        df = self.distances[["PORT_NAME_FROM", "PORT_NAME_TO", "DISTANCE"]].copy()
        df["A"] = df["PORT_NAME_FROM"].astype(str).str.upper().str.strip()
        df["B"] = df["PORT_NAME_TO"].astype(str).str.upper().str.strip()
        df["D"] = pd.to_numeric(df["DISTANCE"], errors="coerce")

        df = df.dropna(subset=["A", "B", "D"])
        df = df[df["D"] > 0]
        df = df.drop_duplicates(subset=["A", "B"], keep="first")

        for a, b, d in df[["A", "B", "D"]].itertuples(index=False):
            self.distance_dict[(a, b)] = float(d)

        # Store known ports for safe matching (exact or unique-substring only)
        self._known_ports = sorted(set(df["A"]).union(set(df["B"])))
        self._known_port_set = set(self._known_ports)

    def _resolve_port_name(self, port: str) -> str:
        """
        Resolve user-given port string to a known port in the distance table.
        - Exact match wins.
        - If not exact, allow unique substring match (e.g., "QINGDAO" -> "QINGDAO, CHINA") ONLY if exactly one match.
        - Otherwise return the cleaned original (so lookup will fail and remain strict).
        """
        p = str(port).upper().strip()
        if not p:
            return p
        if p in self._known_port_set:
            return p

        matches = [x for x in self._known_ports if p in x]
        if len(matches) == 1:
            return matches[0]

        return p


    def get_distance(self, from_port: str, to_port: str, allow_reverse: bool = True) -> Optional[float]:
        a = self._resolve_port_name(from_port)
        b = self._resolve_port_name(to_port)
        d = self.distance_dict.get((a, b))
        if d is None and allow_reverse:
            d = self.distance_dict.get((b, a))
        return d
    
    @staticmethod
    def _parse_date(d: Any) -> Optional[datetime]:
        if d is None:
            return None
        if isinstance(d, datetime):
            return d
        s = str(d)
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def _month_key(dt: datetime) -> str:
        return f"{dt.year:04d}-{dt.month:02d}"
    
    def _get_bunker_price(self, region: str, grade: str, voyage_start: Optional[datetime]) -> float:
        """Uses forward curve if provided; otherwise uses flat bunker_prices."""
        region = str(region).upper()
        grade = str(grade).upper()

        if grade == "VLSFO":
            grade = "VLSF"

        if self.bunker_forward_curve and voyage_start:
            mk = self._month_key(voyage_start)
            reg = self.bunker_forward_curve.get(region)
            if reg:
                g = reg.get(grade)
                if isinstance(g, dict) and mk in g:
                    return float(g[mk])

        price = self.bunker_prices.get(region, {}).get(grade, None)
        return float(price) if price is not None else 0.0

    def _get_bunker_region(self, port_name: str) -> str:
        """Proxy mapping from port string to bunker hub/region (simplified)."""
        p = str(port_name).upper()

        if any(x in p for x in ["QINGDAO", "CAOFEIDIAN", "LIANYUNGANG", "FANGCHENG", "SHANGHAI", "JINGTANG", "TIANJIN", "XIAMEN"]):
            return "QINGDAO"

        if any(x in p for x in ["SINGAPORE", "MAP TA PHUT", "THAILAND", "TELUK RUBIAH", "MALAYSIA"]):
            return "SINGAPORE"

        if any(x in p for x in ["HEDLAND", "DAMPIER", "AUSTRALIA"]):
            return "SINGAPORE"

        if any(x in p for x in ["ITAGUAI", "BRAZIL", "TUBARAO", "PONTA DA MADEIRA", "PONTA"]):
            return "ROTTERDAM"

        if any(x in p for x in ["KAMSAR", "GUINEA", "SALDANHA", "RICHARDS"]):
            return "DURBAN"

        if any(x in p for x in ["JUBAIL", "FUJAIRAH", "GIBRALTAR"]):
            return "FUJAIRAH"

        return "SINGAPORE"
    
    @staticmethod
    def calculate_steaming_time(distance_nm: Optional[float], speed_kn: float) -> Optional[float]:
        if distance_nm is None or speed_kn is None or speed_kn <= 0:
            return None
        return float(distance_nm) / (float(speed_kn) * 24.0)

    @staticmethod
    def _safe_div(a: float, b: float) -> Optional[float]:
        if b is None or b <= 0:
            return None
        return float(a) / float(b)
    
    @staticmethod
    def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
        """Safe float conversion; returns default on None/invalid."""
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default

    def _infer_port_rates(self, vessel: Dict) -> Dict[str, float]:
        """
        Uses keys exactly from your dict:
          port_vlsf -> idle VLSFO/day
          port_mgo  -> working VLSFO/day (as per slide, despite key name)
        If you later add explicit MGO port rates, extend the dict and this function.
        """
        idle_vlsf = float(vessel.get("port_vlsf", 0.0))
        work_vlsf = float(vessel.get("port_mgo", idle_vlsf))

        return {
            "idle_vlsf": idle_vlsf,
            "work_vlsf": work_vlsf,
            "idle_mgo": 0.0,   # not provided
            "work_mgo": 0.0,   # not provided
        }
    
    def _compute_waiting_days(
        self,
        vessel_etd: Optional[datetime],
        ballast_days: float,
        laycan_start: Optional[datetime],
        laycan_end: Optional[datetime],
    ) -> Tuple[bool, float, Optional[datetime], Optional[datetime]]:
        """
        Returns:
          feasible, waiting_days, arrival_load_dt, load_start_dt
        """
        if vessel_etd is None or ballast_days is None:
            return False, 0.0, None, None

        arrival_load = vessel_etd + timedelta(days=float(ballast_days))

        if laycan_start is None or laycan_end is None:
            return True, 0.0, arrival_load, arrival_load

        if arrival_load > laycan_end:
            return False, 0.0, arrival_load, None

        waiting = 0.0
        load_start = arrival_load
        if arrival_load < laycan_start:
            waiting = (laycan_start - arrival_load).total_seconds() / 86400.0
            load_start = laycan_start

        return True, float(waiting), arrival_load, load_start
    
    def get_quantity_scenarios(self, cargo: Dict) -> List[Tuple[str, float]]:
        """
        NO assumption of +/- 10%.
        Only generates scenarios if cargo explicitly provides:
          qty_tolerance or quantity_tolerance
        Otherwise returns [("base", quantity)] only.
        """
        base = float(cargo.get("quantity", 0.0))
        tol = cargo.get("qty_tolerance", cargo.get("quantity_tolerance", None))
        if tol is None or base <= 0:
            return [("base", base)]

        tol = float(tol)
        if tol <= 0:
            return [("base", base)]

        return [
            ("min", base * (1.0 - tol)),
            ("base", base),
            ("max", base * (1.0 + tol)),
        ]
    
    def _freight_revenue(self, cargo: Dict, qty: float) -> Optional[float]:
        """
        Computes freight revenue ONLY if cargo has freight_rate.

        Special half-freight rule is ONLY applied if explicitly provided:
          half_freight_above_qty
          half_freight_factor
        """
        rate = cargo.get("freight_rate", None)
        if rate is None:
            return None
        rate = float(rate)

        threshold = cargo.get("half_freight_above_qty", None)
        factor = cargo.get("half_freight_factor", None)

        if threshold is None:
            return qty * rate

        threshold = float(threshold)
        factor = float(factor if factor is not None else 0.5)

        if qty <= threshold:
            return qty * rate

        main = threshold * rate
        excess = (qty - threshold) * rate * factor
        return main + excess

    def evaluate_vessel_cargo(
        self,
        vessel: Dict,
        cargo: Dict,
        use_economical_speed: bool = True,
        use_rob: Optional[bool] = None,
        ballast_nm: Optional[float] = None,
        laden_nm: Optional[float] = None,
    ) -> List[Dict]:
        """
        Returns a list of results (one per quantity scenario).
        - If distance missing => infeasible.
        - If cargo has no freight_rate => returns required freight quote.
        - If vessel has no hire_rate => returns required hire quote.
        """

        if use_rob is None:
            use_rob = self.default_use_rob

        vessel_pos = vessel.get("current_port")
        load_port = cargo.get("load_port")
        discharge_port = cargo.get("discharge_port")

        # Basic required fields (avoid silent None / crashes)
        if not vessel_pos or not load_port or not discharge_port:
            return [{
                "feasible": False,
                "issue": "missing_port_fields",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
                "current_port": vessel_pos,
                "load_port": load_port,
                "discharge_port": discharge_port,
            }]

        # Distances
        if ballast_nm is None:
            ballast_nm = self.get_distance(vessel_pos, load_port)
        if laden_nm is None:
            laden_nm = self.get_distance(load_port, discharge_port)

        # Still strict: if either missing after overrides/resolution -> infeasible
        if ballast_nm is None or laden_nm is None:
            return [{
                "feasible": False,
                "issue": "missing_distance",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
                "current_port": vessel_pos,
                "load_port": load_port,
                "discharge_port": discharge_port,
                "ballast_nm": ballast_nm,
                "laden_nm": laden_nm,
            }]


        # Speed & sea consumption
        if use_economical_speed:
            ballast_speed = vessel.get("eco_ballast_speed")
            laden_speed = vessel.get("eco_laden_speed")
            ballast_vlsf = vessel.get("eco_ballast_vlsf")
            ballast_mgo = vessel.get("eco_ballast_mgo")
            laden_vlsf = vessel.get("eco_laden_vlsf")
            laden_mgo = vessel.get("eco_laden_mgo")
            speed_mode = "eco"
        else:
            ballast_speed = vessel.get("war_ballast_speed")
            laden_speed = vessel.get("war_laden_speed")
            ballast_vlsf = vessel.get("war_ballast_vlsf")
            ballast_mgo = vessel.get("war_ballast_mgo")
            laden_vlsf = vessel.get("war_laden_vlsf")
            laden_mgo = vessel.get("war_laden_mgo")
            speed_mode = "war"

        required_nums = {
            "ballast_speed": ballast_speed,
            "laden_speed": laden_speed,
            "ballast_vlsf": ballast_vlsf,
            "ballast_mgo": ballast_mgo,
            "laden_vlsf": laden_vlsf,
            "laden_mgo": laden_mgo,
        }
        missing_or_bad = [k for k, v in required_nums.items() if v is None or (isinstance(v, (int, float)) and v < 0)]
        if missing_or_bad:
            return [{
                "feasible": False,
                "issue": "missing_vessel_performance_fields",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
                "speed_mode": speed_mode,
                "missing_fields": missing_or_bad,
            }]

        ballast_days = self.calculate_steaming_time(ballast_nm, float(ballast_speed))
        laden_days = self.calculate_steaming_time(laden_nm, float(laden_speed))
        if ballast_days is None or laden_days is None:
            return [{
                "feasible": False,
                "issue": "invalid_speed",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
            }]

        sea_days = float(ballast_days + laden_days)

        # Dates & laycan feasibility
        etd = self._parse_date(vessel.get("etd"))
        laycan_start = self._parse_date(cargo.get("laycan_start"))
        laycan_end = self._parse_date(cargo.get("laycan_end"))

        feasible_laycan, waiting_days, arrival_load_dt, _ = self._compute_waiting_days(
            etd, ballast_days, laycan_start, laycan_end
        )

        if not feasible_laycan:
            return [{
                "feasible": False,
                "issue": "missed_laycan",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
                "etd": vessel.get("etd"),
                "arrival_load": arrival_load_dt.strftime("%Y-%m-%d") if arrival_load_dt else None,
                "laycan_start": cargo.get("laycan_start"),
                "laycan_end": cargo.get("laycan_end"),
                "waiting_days": waiting_days,
            }]

        port_rates = self._infer_port_rates(vessel)

        bunker_region = self._get_bunker_region(load_port)
        price_dt = etd if etd else laycan_start
        vlsf_price = self._get_bunker_price(bunker_region, "VLSF", price_dt)
        mgo_price = self._get_bunker_price(bunker_region, "MGO", price_dt)

        if vlsf_price <= 0 or mgo_price <= 0:
            return [{
                "feasible": False,
                "issue": "missing_bunker_price",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
                "bunker_region": bunker_region,
                "vlsf_price": vlsf_price,
                "mgo_price": mgo_price,
            }]

        hire_rate = vessel.get("hire_rate", None)
        hire_rate = float(hire_rate) if hire_rate is not None else None

        commission_rate = float(cargo.get("commission_rate", 0.0))
        port_cost = float(cargo.get("port_cost", 0.0))

        results: List[Dict] = []

        for qty_label, qty in self.get_quantity_scenarios(cargo):
            qty = float(qty)

            load_rate = float(cargo.get("load_rate", 0.0))
            dis_rate = float(cargo.get("discharge_rate", 0.0))
            load_tt = float(cargo.get("load_tt", 0.0))
            dis_tt = float(cargo.get("discharge_tt", 0.0))

            load_days = self._safe_div(qty, load_rate)
            dis_days = self._safe_div(qty, dis_rate)
            if load_days is None or dis_days is None:
                results.append({
                    "feasible": False,
                    "issue": "invalid_port_rates",
                    "vessel": vessel.get("name"),
                    "cargo": cargo.get("name"),
                    "qty_label": qty_label,
                    "qty": qty,
                })
                continue

            working_port_days = float(load_days + load_tt + dis_days + dis_tt)
            idle_port_days = float(waiting_days)
            total_port_days = working_port_days + idle_port_days
            total_days = sea_days + total_port_days

            # Sea fuel
            vlsf_at_sea = float(ballast_days * float(ballast_vlsf) + laden_days * float(laden_vlsf))
            mgo_at_sea = float(ballast_days * float(ballast_mgo) + laden_days * float(laden_mgo))

            # Port fuel (idle vs working) - only VLSF supported by provided data
            vlsf_in_port = idle_port_days * port_rates["idle_vlsf"] + working_port_days * port_rates["work_vlsf"]
            mgo_in_port = 0.0

            total_vlsf = float(vlsf_at_sea + vlsf_in_port)
            total_mgo = float(mgo_at_sea + mgo_in_port)

            # ROB handling
            if use_rob:
                rob_vlsf = float(vessel.get("bunker_vlsf", 0.0))
                rob_mgo = float(vessel.get("bunker_mgo", 0.0))
                buy_vlsf = max(0.0, total_vlsf - rob_vlsf)
                buy_mgo = max(0.0, total_mgo - rob_mgo)
                end_rob_vlsf = max(0.0, rob_vlsf - total_vlsf)
                end_rob_mgo = max(0.0, rob_mgo - total_mgo)
            else:
                rob_vlsf = None
                rob_mgo = None
                buy_vlsf = total_vlsf
                buy_mgo = total_mgo
                end_rob_vlsf = None
                end_rob_mgo = None

            bunker_cost = buy_vlsf * vlsf_price + buy_mgo * mgo_price

            # Revenue
            freight_revenue = self._freight_revenue(cargo, qty)

            if freight_revenue is None:
                # Quote freight for market cargo
                if hire_rate is None:
                    results.append({
                        "feasible": False,
                        "issue": "cannot_quote_missing_hire_and_freight",
                        "vessel": vessel.get("name"),
                        "cargo": cargo.get("name"),
                        "qty_label": qty_label,
                        "qty": qty,
                    })
                    continue

                assumed_hire_cost = hire_rate * total_days
                total_costs_for_quote = assumed_hire_cost + bunker_cost + port_cost

                required_freight_rate = (
                    (total_costs_for_quote / qty) / (1.0 - commission_rate)
                    if qty > 0 and (1.0 - commission_rate) > 0
                    else None
                )

                results.append({
                    "feasible": True,
                    "mode": "QUOTE_FREIGHT",
                    "vessel": vessel.get("name"),
                    "cargo": cargo.get("name"),
                    "qty_label": qty_label,
                    "qty": qty,
                    "speed_mode": speed_mode,
                    "ballast_nm": ballast_nm,
                    "laden_nm": laden_nm,
                    "sea_days": sea_days,
                    "working_port_days": working_port_days,
                    "idle_port_days": idle_port_days,
                    "total_days": total_days,
                    "bunker_region": bunker_region,
                    "vlsf_price": vlsf_price,
                    "mgo_price": mgo_price,
                    "bunker_cost": bunker_cost,
                    "port_cost": port_cost,
                    "commission_rate": commission_rate,
                    "hire_rate": hire_rate,
                    "required_freight_rate_usd_per_mt": required_freight_rate,
                })
                continue


            commission = freight_revenue * commission_rate
            net_revenue = freight_revenue - commission

            if hire_rate is None:
                # Quote hire for market vessel
                required_hire_rate = ((net_revenue - bunker_cost - port_cost) / total_days) if total_days > 0 else None
                results.append({
                    "feasible": True,
                    "mode": "QUOTE_HIRE",
                    "vessel": vessel.get("name"),
                    "cargo": cargo.get("name"),
                    "qty_label": qty_label,
                    "qty": qty,
                    "speed_mode": speed_mode,
                    "total_days": total_days,
                    "net_revenue": net_revenue,
                    "bunker_cost": bunker_cost,
                    "port_cost": port_cost,
                    "commission_rate": commission_rate,
                    "required_hire_rate_usd_per_day": required_hire_rate,
                })
                continue

            hire_cost = hire_rate * total_days
            total_costs = hire_cost + bunker_cost + port_cost
            voyage_profit = net_revenue - total_costs
            tce = (net_revenue - bunker_cost - port_cost) / total_days if total_days > 0 else None

            results.append({
                "feasible": True,
                "mode": "P&L",
                "vessel": vessel.get("name"),
                "cargo": cargo.get("name"),
                "qty_label": qty_label,
                "qty": qty,
                "speed_mode": speed_mode,
                "etd": vessel.get("etd"),
                "arrival_load": arrival_load_dt.strftime("%Y-%m-%d") if arrival_load_dt else None,
                "laycan_start": cargo.get("laycan_start"),
                "laycan_end": cargo.get("laycan_end"),
                "waiting_days": waiting_days,
                "ballast_nm": ballast_nm,
                "laden_nm": laden_nm,
                "ballast_days": float(ballast_days),
                "laden_days": float(laden_days),
                "sea_days": sea_days,
                "working_port_days": working_port_days,
                "idle_port_days": idle_port_days,
                "total_days": total_days,
                "bunker_region": bunker_region,
                "vlsf_price": vlsf_price,
                "mgo_price": mgo_price,
                "total_vlsf": total_vlsf,
                "total_mgo": total_mgo,
                "rob_vlsf_start": rob_vlsf,
                "rob_mgo_start": rob_mgo,
                "buy_vlsf": buy_vlsf,
                "buy_mgo": buy_mgo,
                "end_rob_vlsf": end_rob_vlsf,
                "end_rob_mgo": end_rob_mgo,
                "bunker_cost": bunker_cost,
                "hire_rate": hire_rate,
                "hire_cost": hire_cost,
                "port_cost": port_cost,
                "freight_revenue": freight_revenue,
                "commission": commission,
                "net_revenue": net_revenue,
                "total_costs": total_costs,
                "voyage_profit": voyage_profit,
                "tce": tce,
            })

        return results
    
    def _best_result_for_pair(
        self,
        vessel: Dict,
        cargo: Dict,
        qty_label: str = "base",
        try_both_speed_modes: bool = True,
        use_rob: Optional[bool] = None,
    ) -> Optional[Dict]:
        """
        Returns the best feasible result dict for this vessel-cargo pair.
        - Prefers higher voyage_profit for mode=="P&L"
        - For QUOTE_HIRE / QUOTE_FREIGHT, treats as 0-profit (break-even quote)
        """
        candidates: List[Dict] = []

        speed_modes = [True, False] if try_both_speed_modes else [True]

        for use_eco in speed_modes:
            res_list = self.evaluate_vessel_cargo(
                vessel=vessel,
                cargo=cargo,
                use_economical_speed=use_eco,
                use_rob=use_rob,
            )
            for r in res_list:
                if not r.get("feasible"):
                    continue
                if r.get("qty_label") != qty_label:
                    continue
                candidates.append(r)

        if not candidates:
            return None

        def score(r: Dict) -> float:
            mode = r.get("mode")
            if mode == "P&L":
                return float(r.get("voyage_profit", -1e30))
            if mode in ("QUOTE_HIRE", "QUOTE_FREIGHT"):
                return 0.0
            return -1e30

        return max(candidates, key=score)

    def optimize_committed_cargo_assignment(
        self,
        vessels: List[Dict],
        committed_cargoes: List[Dict],
        qty_label: str = "base",
        try_both_speed_modes: bool = True,
        use_rob: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Brute-force optimizer for small N (your case: 4 vessels, 3 cargoes).
        Maximizes total voyage_profit across assigned committed cargoes,
        subject to 1 vessel -> 1 cargo, 1 cargo -> 1 vessel.

        Returns:
          {
            "feasible": bool,
            "total_profit": float,
            "assignments": [ { "vessel": ..., "cargo": ..., "result": {...} }, ... ],
            "unused_vessels": [vessel_name, ...]
          }
        """
        if not committed_cargoes:
            return {"feasible": True, "total_profit": 0.0, "assignments": [], "unused_vessels": [v.get("name") for v in vessels]}

        n_c = len(committed_cargoes)
        if len(vessels) < n_c:
            return {"feasible": False, "issue": "not_enough_vessels", "needed": n_c, "available": len(vessels)}

        best: Optional[Dict[str, Any]] = None

        for vessel_subset in combinations(vessels, n_c):
            for cargo_perm in permutations(committed_cargoes, n_c):
                total_profit = 0.0
                assignments: List[Dict[str, Any]] = []
                feasible = True

                for v, c in zip(vessel_subset, cargo_perm):
                    best_pair = self._best_result_for_pair(
                        vessel=v,
                        cargo=c,
                        qty_label=qty_label,
                        try_both_speed_modes=try_both_speed_modes,
                        use_rob=use_rob,
                    )
                    if best_pair is None:
                        feasible = False
                        break

                    mode = best_pair.get("mode")
                    profit = float(best_pair.get("voyage_profit", 0.0)) if mode == "P&L" else 0.0
                    total_profit += profit

                    assignments.append({
                        "vessel": v.get("name"),
                        "cargo": c.get("name"),
                        "mode": mode,
                        "profit": profit,
                        "result": best_pair,
                    })

                if not feasible:
                    continue

                if best is None or total_profit > float(best.get("total_profit", -1e30)):
                    used_names = {a["vessel"] for a in assignments}
                    unused = [v.get("name") for v in vessels if v.get("name") not in used_names]
                    best = {
                        "feasible": True,
                        "total_profit": float(total_profit),
                        "assignments": assignments,
                        "unused_vessels": unused,
                    }

        return best if best is not None else {"feasible": False, "issue": "no_feasible_assignment"}


def load_bunker_prices_flat() -> Dict:
    """Flat fallback prices."""
    return {
        "SINGAPORE": {"VLSF": 491, "MGO": 654},
        "FUJAIRAH": {"VLSF": 479, "MGO": 640},
        "DURBAN": {"VLSF": 436, "MGO": 511},
        "ROTTERDAM": {"VLSF": 468, "MGO": 615},
        "GIBRALTAR": {"VLSF": 475, "MGO": 625},
        "QINGDAO": {"VLSF": 648, "MGO": 838},
        "SHANGHAI": {"VLSF": 650, "MGO": 841},
        "RICHARDS BAY": {"VLSF": 442, "MGO": 520},
    }

if __name__ == "__main__":
    distances = pd.read_csv("Port_Distances.csv")
    bunker_prices = load_bunker_prices_flat()
    calc = FreightCalculator(distances, bunker_prices)

    print("FreightCalculator initialized.")
