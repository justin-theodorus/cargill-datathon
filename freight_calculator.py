"""
Cargill Ocean Transportation Datathon 2026
Freight Calculator for Capesize Vessels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class FreightCalculator:
    """Calculate voyage profitability for vessel-cargo combinations"""
    
    def __init__(self, distances_df: pd.DataFrame, bunker_prices: Dict, ffa_rates: Dict = None):
        """
        Initialize the calculator
        
        Args:
            distances_df: DataFrame with PORT_NAME_FROM, PORT_NAME_TO, DISTANCE
            bunker_prices: Dict with bunker prices by location and fuel type
            ffa_rates: Optional FFA rates for market pricing
        """
        self.distances = distances_df
        self.bunker_prices = bunker_prices
        self.ffa_rates = ffa_rates or {}
        
        self._create_distance_lookup()
    
    def _create_distance_lookup(self):
        """Create a lookup dictionary for fast distance queries"""
        self.distance_dict = {}
        
        for _, row in self.distances.iterrows():
            from_port = row['PORT_NAME_FROM'].upper()
            to_port = row['PORT_NAME_TO'].upper()
            distance = row['DISTANCE']
            
            self.distance_dict[(from_port, to_port)] = distance
            self.distance_dict[(to_port, from_port)] = distance
    
    def get_distance(self, from_port: str, to_port: str) -> float:
        """
        Get distance between two ports
        
        Returns: Distance in nautical miles, or None if not found
        """
        key = (from_port.upper(), to_port.upper())
        return self.distance_dict.get(key)
    
    def calculate_steaming_time(self, distance: float, speed: float) -> float:
        """
        Calculate steaming time in days
        
        Args:
            distance: Distance in nautical miles
            speed: Speed in knots
        
        Returns: Time in days
        """
        if speed <= 0:
            return 0
        return distance / (speed * 24)
    
    def calculate_fuel_consumption(self, time_days: float, consumption_rate: float) -> float:
        """
        Calculate fuel consumption
        
        Args:
            time_days: Time in days
            consumption_rate: Consumption in MT per day
        
        Returns: Total fuel consumption in MT
        """
        return time_days * consumption_rate
    
    def calculate_port_time(self, cargo_qty: float, load_rate: float, discharge_rate: float,
                           load_tt: float, discharge_tt: float, port_idle: float = 0) -> Dict:
        """
        Calculate time spent in ports
        
        Args:
            cargo_qty: Cargo quantity in MT
            load_rate: Loading rate in MT per day
            discharge_rate: Discharge rate in MT per day
            load_tt: Loading turn time in days
            discharge_tt: Discharge turn time in days
            port_idle: Additional idle time in days
        
        Returns: Dict with loading_days, discharge_days, total_port_days
        """
        loading_days = (cargo_qty / load_rate) + load_tt
        discharge_days = (cargo_qty / discharge_rate) + discharge_tt
        total_port_days = loading_days + discharge_days + port_idle
        
        return {
            'loading_days': loading_days,
            'discharge_days': discharge_days,
            'total_port_days': total_port_days
        }
    
    def calculate_voyage_costs(self, vessel: Dict, cargo: Dict, 
                              ballast_distance: float, laden_distance: float,
                              use_economical_speed: bool = True) -> Dict:
        """
        Calculate all voyage costs and profit
        
        Args:
            vessel: Dictionary with vessel details
            cargo: Dictionary with cargo details
            ballast_distance: Distance in ballast leg (NM)
            laden_distance: Distance in laden leg (NM)
            use_economical_speed: Use economical vs warranted speed
        
        Returns: Dictionary with all calculations
        """
        if use_economical_speed:
            ballast_speed = vessel['eco_ballast_speed']
            laden_speed = vessel['eco_laden_speed']
            ballast_vlsf = vessel['eco_ballast_vlsf']
            ballast_mgo = vessel['eco_ballast_mgo']
            laden_vlsf = vessel['eco_laden_vlsf']
            laden_mgo = vessel['eco_laden_mgo']
        else:
            ballast_speed = vessel['war_ballast_speed']
            laden_speed = vessel['war_laden_speed']
            ballast_vlsf = vessel['war_ballast_vlsf']
            ballast_mgo = vessel['war_ballast_mgo']
            laden_vlsf = vessel['war_laden_vlsf']
            laden_mgo = vessel['war_laden_mgo']
        
        ballast_days = self.calculate_steaming_time(ballast_distance, ballast_speed)
        laden_days = self.calculate_steaming_time(laden_distance, laden_speed)
        sea_days = ballast_days + laden_days

        port_time = self.calculate_port_time(
            cargo['quantity'],
            cargo['load_rate'],
            cargo['discharge_rate'],
            cargo['load_tt'],
            cargo['discharge_tt']
        )

        total_days = sea_days + port_time['total_port_days']
        
        vlsf_at_sea = (ballast_days * ballast_vlsf) + (laden_days * laden_vlsf)
        mgo_at_sea = (ballast_days * ballast_mgo) + (laden_days * laden_mgo)
        
        vlsf_in_port = port_time['total_port_days'] * vessel['port_vlsf']
        mgo_in_port = port_time['total_port_days'] * vessel['port_mgo']

        total_vlsf = vlsf_at_sea + vlsf_in_port
        total_mgo = mgo_at_sea + mgo_in_port
        
        load_port_region = self._get_bunker_region(cargo['load_port'])
        vlsf_price = self.bunker_prices.get(load_port_region, {}).get('VLSF', 500)
        mgo_price = self.bunker_prices.get(load_port_region, {}).get('MGO', 650)
        
        hire_cost = vessel['hire_rate'] * total_days
        bunker_cost = (total_vlsf * vlsf_price) + (total_mgo * mgo_price)
        port_cost = cargo.get('port_cost', 0)
        
        freight_revenue = cargo['quantity'] * cargo['freight_rate']
        
        address_commission = freight_revenue * cargo.get('commission_rate', 0)
        
        net_revenue = freight_revenue - address_commission
        
        total_costs = hire_cost + bunker_cost + port_cost
        
        voyage_profit = net_revenue - total_costs
        
        tce = (net_revenue - bunker_cost - port_cost) / total_days if total_days > 0 else 0
        
        return {
            'vessel_name': vessel['name'],
            'cargo_name': cargo['name'],
            'ballast_days': ballast_days,
            'laden_days': laden_days,
            'sea_days': sea_days,
            'port_days': port_time['total_port_days'],
            'total_days': total_days,
            'vlsf_consumption': total_vlsf,
            'mgo_consumption': total_mgo,
            'vlsf_price': vlsf_price,
            'mgo_price': mgo_price,
            'hire_cost': hire_cost,
            'bunker_cost': bunker_cost,
            'port_cost': port_cost,
            'total_costs': total_costs,
            'freight_revenue': freight_revenue,
            'commission': address_commission,
            'net_revenue': net_revenue,
            'voyage_profit': voyage_profit,
            'tce': tce,
            'ballast_distance': ballast_distance,
            'laden_distance': laden_distance
        }
    
    def _get_bunker_region(self, port_name: str) -> str:
        """Map port to bunker pricing region"""
        port_upper = port_name.upper()
        
        if any(x in port_upper for x in ['QINGDAO', 'CAOFEIDIAN', 'LIANYUNGANG', 
                                         'FANGCHENG', 'SHANGHAI', 'JINGTANG']):
            return 'QINGDAO'
        
        if any(x in port_upper for x in ['SINGAPORE', 'MAP TA PHUT', 'THAILAND']):
            return 'SINGAPORE'
        
        if any(x in port_upper for x in ['HEDLAND', 'DAMPIER', 'AUSTRALIA']):
            return 'SINGAPORE'
        
        if any(x in port_upper for x in ['ITAGUAI', 'BRAZIL', 'TUBARAO', 'MADEIRA']):
            return 'ROTTERDAM' 
        
        if any(x in port_upper for x in ['KAMSAR', 'GUINEA', 'SALDANHA', 'RICHARDS']):
            return 'DURBAN'
        
        return 'SINGAPORE'


def load_bunker_prices() -> Dict:
    """Load bunker prices from the presentation data"""
    return {
        'SINGAPORE': {'VLSF': 491, 'MGO': 654},
        'FUJAIRAH': {'VLSF': 479, 'MGO': 640},
        'DURBAN': {'VLSF': 436, 'MGO': 511},
        'ROTTERDAM': {'VLSF': 468, 'MGO': 615},
        'GIBRALTAR': {'VLSF': 475, 'MGO': 625},
        'QINGDAO': {'VLSF': 648, 'MGO': 838},
        'SHANGHAI': {'VLSF': 650, 'MGO': 841},
        'RICHARDS BAY': {'VLSF': 442, 'MGO': 520}
    }

if __name__ == "__main__":
    distances = pd.read_csv('Port_Distances.csv')
    bunker_prices = load_bunker_prices()

    calc = FreightCalculator(distances, bunker_prices)

    print("Freight Calculator Initialized!")
    print(f"Loaded {len(distances)} port distance records")
    print(f"Bunker prices for {len(bunker_prices)} regions")

    test_distance = calc.get_distance('PORT HEDLAND', 'QINGDAO')
    if test_distance:
        print(f"\nTest: Port Hedland to Qingdao = {test_distance:.2f} NM")
