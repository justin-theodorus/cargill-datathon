"""
Main Analysis Script
Calculate optimal vessel-cargo allocation
"""

import pandas as pd
import numpy as np
from freight_calculator import FreightCalculator, load_bunker_prices
from vessel_cargo_data import CARGILL_VESSELS, CARGILL_CARGOES, get_all_vessels, get_all_cargoes
from itertools import permutations, combinations
import warnings
warnings.filterwarnings('ignore')


def estimate_distance(from_port: str, to_port: str) -> float:
    """
    Estimate distance between two ports based on typical maritime routes
    """
    # Port region classifications
    regions = {
        'CHINA': ['QINGDAO', 'TIANJIN', 'CAOFEIDIAN', 'LIANYUNGANG', 'FANGCHENG',
                  'SHANGHAI', 'JINGTANG', 'HONG KONG', 'XIAMEN'],
        'AUSTRALIA': ['PORT HEDLAND', 'DAMPIER', 'BRISBANE', 'SYDNEY'],
        'BRAZIL': ['RIO DE JANEIRO', 'SANTOS', 'TUBARAO'],
        'INDIA': ['VISAKHAPATNAM', 'VIZAG', 'PARADIP', 'MUNDRA', 'COCHIN', 'KANDLA'],
        'WEST_AFRICA': ['CONAKRY', 'RICHARDS BAY'],
        'MIDDLE_EAST': ['DUBAI', 'JUBAIL'],
        'EUROPE': ['ROTTERDAM', 'ANTWERP'],
        'KOREA': ['BUSAN', 'INCHEON'],
        'SOUTHEAST_ASIA': ['SINGAPORE', 'BALIKPAPAN', 'LAEM CHABANG'],
        'NORTH_AMERICA': ['SEATTLE', 'VANCOUVER']
    }

    # Find regions for each port
    from_region = None
    to_region = None

    for region, ports in regions.items():
        if any(p in from_port.upper() for p in ports):
            from_region = region
        if any(p in to_port.upper() for p in ports):
            to_region = region

    # Typical inter-region distances (nautical miles)
    distance_matrix = {
        ('CHINA', 'AUSTRALIA'): 3700,
        ('CHINA', 'BRAZIL'): 12000,
        ('CHINA', 'INDIA'): 3500,
        ('CHINA', 'WEST_AFRICA'): 10500,
        ('CHINA', 'MIDDLE_EAST'): 4500,
        ('CHINA', 'EUROPE'): 11000,
        ('CHINA', 'KOREA'): 600,
        ('CHINA', 'SOUTHEAST_ASIA'): 1500,
        ('CHINA', 'NORTH_AMERICA'): 5500,
        ('AUSTRALIA', 'BRAZIL'): 9500,
        ('AUSTRALIA', 'INDIA'): 4800,
        ('AUSTRALIA', 'WEST_AFRICA'): 7500,
        ('AUSTRALIA', 'MIDDLE_EAST'): 5200,
        ('AUSTRALIA', 'EUROPE'): 11500,
        ('AUSTRALIA', 'KOREA'): 4000,
        ('AUSTRALIA', 'SOUTHEAST_ASIA'): 2500,
        ('AUSTRALIA', 'NORTH_AMERICA'): 7000,
        ('BRAZIL', 'INDIA'): 7500,
        ('BRAZIL', 'WEST_AFRICA'): 3500,
        ('BRAZIL', 'MIDDLE_EAST'): 8000,
        ('BRAZIL', 'EUROPE'): 5000,
        ('BRAZIL', 'KOREA'): 12500,
        ('BRAZIL', 'SOUTHEAST_ASIA'): 10000,
        ('BRAZIL', 'NORTH_AMERICA'): 5000,
        ('INDIA', 'WEST_AFRICA'): 4500,
        ('INDIA', 'MIDDLE_EAST'): 2000,
        ('INDIA', 'EUROPE'): 6500,
        ('INDIA', 'KOREA'): 4500,
        ('INDIA', 'SOUTHEAST_ASIA'): 2000,
        ('INDIA', 'NORTH_AMERICA'): 9000,
        ('WEST_AFRICA', 'MIDDLE_EAST'): 5500,
        ('WEST_AFRICA', 'EUROPE'): 4000,
        ('WEST_AFRICA', 'KOREA'): 11000,
        ('WEST_AFRICA', 'SOUTHEAST_ASIA'): 8500,
        ('WEST_AFRICA', 'NORTH_AMERICA'): 5500,
        ('MIDDLE_EAST', 'EUROPE'): 6000,
        ('MIDDLE_EAST', 'KOREA'): 5500,
        ('MIDDLE_EAST', 'SOUTHEAST_ASIA'): 3500,
        ('MIDDLE_EAST', 'NORTH_AMERICA'): 8500,
        ('EUROPE', 'KOREA'): 11500,
        ('EUROPE', 'SOUTHEAST_ASIA'): 9000,
        ('EUROPE', 'NORTH_AMERICA'): 3500,
        ('KOREA', 'SOUTHEAST_ASIA'): 1800,
        ('KOREA', 'NORTH_AMERICA'): 5000,
        ('SOUTHEAST_ASIA', 'NORTH_AMERICA'): 7500,
    }

    # Look up distance
    if from_region and to_region:
        key = tuple(sorted([from_region, to_region]))
        distance = distance_matrix.get(key)
        if distance:
            return distance

    # Same region - short distance
    if from_region == to_region and from_region is not None:
        return 500

    # Default fallback (conservative estimate for long routes)
    return 6000


def normalize_port_name(port: str) -> str:
    """Normalize port names for distance lookup"""
    port_mapping = {
        # Existing mappings
        'KAMSAR': 'CONAKRY',  # Use nearby port as proxy
        'MAP TA PHUT': 'LAEM CHABANG',  # Use nearby Thailand port
        'GWANGYANG': 'BUSAN',  # Use nearby Korean port
        'ITAGUAI': 'RIO DE JANEIRO',  # Use nearby Brazilian port
        'DAMPIER': 'PORT HEDLAND',  # Both are in Western Australia
        'PONTA DA MADEIRA': 'RIO DE JANEIRO',  # Brazilian port

        # New mappings for market cargoes
        'SALDANHA BAY': 'RICHARDS BAY',  # Both South Africa
        'TABONEO': 'BALIKPAPAN',  # Both Indonesia
        'KRISHNAPATNAM': 'VISAKHAPATNAM',  # Both India east coast
        'VANCOUVER': 'SEATTLE',  # Both Pacific Northwest
        'MANGALORE': 'COCHIN',  # Both India west coast
        'TUBARAO': 'RIO DE JANEIRO',  # Brazilian port
        'TELUK RUBIAH': 'SINGAPORE',  # Malaysian port near Singapore

        # Additional vessel current ports
        'CAOFEIDIAN': 'TIANJIN',  # Both North China
        'PARADIP': 'VISAKHAPATNAM',  # Both India east coast
        'KANDLA': 'MUNDRA',  # Both India west coast (Gujarat)
        'VIZAG': 'VISAKHAPATNAM',  # Short form
        'XIAMEN': 'XIAMEN',  # Keep Xiamen (it exists in table)
        'HONG KONG': 'QINGDAO',  # South China - use major port
        'JINGTANG': 'TIANJIN',  # Both North China
        'JUBAIL': 'JEBEL ALI',  # Middle East Gulf
        'PORT TALBOT': 'ROTTERDAM',  # Both Northwest Europe
        'TELUK RUBIAH': 'SINGAPORE',  # Malaysia - close to Singapore
        'BALIKPAPAN': 'SINGAPORE',  # Indonesia - regional hub
        'SEATTLE': 'VANCOUVER (CANADA)',  # Pacific Northwest
        'VANCOUVER': 'VANCOUVER (CANADA)',  # Ensure consistent naming
        'COCHIN': 'MUNDRA'  # India west coast
    }

    port_upper = port.upper()
    return port_mapping.get(port_upper, port_upper)


def calculate_all_combinations(calculator: FreightCalculator,
                               vessels: list,
                               cargoes: list,
                               use_economical: bool = True) -> pd.DataFrame:
    """
    Calculate voyage profit for all vessel-cargo combinations

    Returns: DataFrame with all calculations
    """
    results = []
    missing_routes = set()  # Track missing routes to report once

    for vessel in vessels:
        for cargo in cargoes:
            try:
                # Normalize port names
                vessel_port = normalize_port_name(vessel['current_port'])
                load_port = normalize_port_name(cargo['load_port'])
                discharge_port = normalize_port_name(cargo['discharge_port'])

                # Get distances
                ballast_dist = calculator.get_distance(vessel_port, load_port)
                laden_dist = calculator.get_distance(load_port, discharge_port)

                # Use estimated distances as fallback if not found
                if ballast_dist is None:
                    missing_routes.add(f"{vessel_port} → {load_port}")
                    # Estimate based on typical routes
                    ballast_dist = estimate_distance(vessel_port, load_port)

                if laden_dist is None:
                    missing_routes.add(f"{load_port} → {discharge_port}")
                    # Estimate based on typical routes
                    laden_dist = estimate_distance(load_port, discharge_port)
                
                # Calculate voyage economics
                result = calculator.calculate_voyage_costs(
                    vessel=vessel,
                    cargo=cargo,
                    ballast_distance=ballast_dist,
                    laden_distance=laden_dist,
                    use_economical_speed=use_economical
                )
                
                # Add additional info
                result['vessel_current_port'] = vessel['current_port']
                result['vessel_etd'] = vessel['etd']
                result['cargo_laycan'] = f"{cargo['laycan_start']} to {cargo['laycan_end']}"
                result['route'] = f"{vessel['current_port']} -> {cargo['load_port']} -> {cargo['discharge_port']}"
                
                results.append(result)
                
            except Exception as e:
                print(f"Error calculating {vessel['name']} x {cargo['name']}: {str(e)}")

    # Report missing routes summary
    if missing_routes:
        print(f"\nℹ️  Note: {len(missing_routes)} port-pair distances not in table (using estimates):")
        for route in sorted(list(missing_routes)[:10]):  # Show first 10
            print(f"   • {route}")
        if len(missing_routes) > 10:
            print(f"   ... and {len(missing_routes) - 10} more")
        print(f"\n✓ Used regional distance estimates for missing routes")

    return pd.DataFrame(results)


def find_optimal_allocation(results_df: pd.DataFrame, 
                           num_vessels: int = 4,
                           num_cargoes: int = 3) -> dict:
    """
    Find the optimal vessel-cargo allocation that maximizes total profit
    
    Uses a greedy approach: assign each cargo to the vessel with highest TCE,
    ensuring no vessel is used twice
    """
    # Sort by TCE descending
    sorted_results = results_df.sort_values('tce', ascending=False).copy()
    
    allocated_vessels = set()
    allocated_cargoes = set()
    allocation = []
    total_profit = 0
    
    # Greedy allocation
    for _, row in sorted_results.iterrows():
        vessel = row['vessel_name']
        cargo = row['cargo_name']
        
        # Check if vessel and cargo are still available
        if vessel not in allocated_vessels and cargo not in allocated_cargoes:
            allocation.append(row)
            allocated_vessels.add(vessel)
            allocated_cargoes.add(cargo)
            total_profit += row['voyage_profit']
            
            # Stop when all cargoes are allocated
            if len(allocated_cargoes) == num_cargoes:
                break
    
    return {
        'allocation': allocation,
        'total_profit': total_profit,
        'allocated_vessels': allocated_vessels,
        'unallocated_vessels': [v for v in set(results_df['vessel_name']) 
                               if v not in allocated_vessels]
    }


def print_allocation_summary(optimal: dict):
    """Print a summary of the optimal allocation"""
    print("\n" + "="*80)
    print("OPTIMAL VESSEL-CARGO ALLOCATION")
    print("="*80)
    
    for i, row in enumerate(optimal['allocation'], 1):
        print(f"\n{i}. {row['vessel_name']} → {row['cargo_name']}")
        print(f"   Route: {row['route']}")
        print(f"   Voyage Days: {row['total_days']:.1f} days")
        print(f"   TCE: ${row['tce']:,.0f}/day")
        print(f"   Voyage Profit: ${row['voyage_profit']:,.0f}")
        print(f"   Revenue: ${row['net_revenue']:,.0f}")
        print(f"   Total Costs: ${row['total_costs']:,.0f}")
        print(f"     - Hire: ${row['hire_cost']:,.0f}")
        print(f"     - Bunker: ${row['bunker_cost']:,.0f}")
        print(f"     - Port: ${row['port_cost']:,.0f}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL PORTFOLIO PROFIT: ${optimal['total_profit']:,.0f}")
    print(f"{'='*80}")
    
    if optimal['unallocated_vessels']:
        print(f"\nUnallocated Vessels: {', '.join(optimal['unallocated_vessels'])}")
        print("(These vessels can be considered for market cargo opportunities)")


def main():
    """Main execution function"""
    print("Cargill Ocean Transportation Datathon 2026")
    print("Freight Calculator Analysis\n")
    
    # Load data
    print("Loading data...")
    distances = pd.read_csv('Port_Distances.csv')
    bunker_prices = load_bunker_prices()
    
    # Initialize calculator
    calc = FreightCalculator(distances, bunker_prices)
    print(f"✓ Loaded {len(distances)} port distance records")
    print(f"✓ Loaded bunker prices for {len(bunker_prices)} regions\n")
    
    # Calculate all combinations (including market vessels and cargoes)
    print("Calculating all vessel-cargo combinations...")
    all_vessels = get_all_vessels()
    all_cargoes = get_all_cargoes()
    print(f"  - Vessels: {len(all_vessels)} ({len(CARGILL_VESSELS)} Cargill + {len(all_vessels) - len(CARGILL_VESSELS)} Market)")
    print(f"  - Cargoes: {len(all_cargoes)} ({len(CARGILL_CARGOES)} Committed + {len(all_cargoes) - len(CARGILL_CARGOES)} Market)")

    results_df = calculate_all_combinations(calc, all_vessels, all_cargoes)
    print(f"✓ Calculated {len(results_df)} combinations\n")

    # Find optimal allocation (only for Cargill committed cargoes)
    print("Finding optimal allocation for Cargill committed cargoes...")
    optimal = find_optimal_allocation(results_df,
                                     num_vessels=len(all_vessels),
                                     num_cargoes=len(CARGILL_CARGOES))
    
    # Print summary
    print_allocation_summary(optimal)
    
    # Save detailed results
    output_file = 'voyage_calculations_detailed.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed calculations saved to: {output_file}")

    # Save allocation summary
    allocation_df = pd.DataFrame(optimal['allocation'])
    summary_file = 'optimal_allocation.csv'
    allocation_df.to_csv(summary_file, index=False)
    print(f"✓ Optimal allocation saved to: {summary_file}")
    
    return results_df, optimal


if __name__ == "__main__":
    results_df, optimal = main()
