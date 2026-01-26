"""
Test data completeness after Priority 1 implementation
"""
from vessel_cargo_data import *

def test_vessel_count():
    """Test that we have the correct number of vessels"""
    assert len(CARGILL_VESSELS) == 4, f"Expected 4 Cargill vessels, got {len(CARGILL_VESSELS)}"
    assert len(MARKET_VESSELS) == 11, f"Expected 11 market vessels, got {len(MARKET_VESSELS)}"
    assert len(get_all_vessels()) == 15, f"Expected 15 total vessels, got {len(get_all_vessels())}"
    print("✓ Vessel count correct: 4 Cargill + 11 Market = 15 total")

def test_cargo_count():
    """Test that we have the correct number of cargoes"""
    assert len(CARGILL_CARGOES) == 3, f"Expected 3 Cargill cargoes, got {len(CARGILL_CARGOES)}"
    assert len(MARKET_CARGOES) == 8, f"Expected 8 market cargoes, got {len(MARKET_CARGOES)}"
    assert len(get_all_cargoes()) == 11, f"Expected 11 total cargoes, got {len(get_all_cargoes())}"
    print("✓ Cargo count correct: 3 Cargill + 8 Market = 11 total")

def test_vessel_structure():
    """Test that all vessels have required fields"""
    required_fields = [
        'name', 'dwt', 'hire_rate', 'current_port', 'etd',
        'war_laden_speed', 'war_laden_vlsf', 'war_laden_mgo',
        'war_ballast_speed', 'war_ballast_vlsf', 'war_ballast_mgo',
        'eco_laden_speed', 'eco_laden_vlsf', 'eco_laden_mgo',
        'eco_ballast_speed', 'eco_ballast_vlsf', 'eco_ballast_mgo',
        'port_vlsf', 'port_mgo',
        'bunker_vlsf', 'bunker_mgo'
    ]

    for vessel in get_all_vessels():
        for field in required_fields:
            assert field in vessel, f"{vessel['name']} missing {field}"

        # Check that hire_rate is not None
        assert vessel['hire_rate'] is not None, f"{vessel['name']} has None hire_rate"
        assert vessel['hire_rate'] > 0, f"{vessel['name']} has invalid hire_rate: {vessel['hire_rate']}"

    print(f"✓ All {len(get_all_vessels())} vessels have required fields and valid hire rates")

def test_cargo_structure():
    """Test that all cargoes have required fields"""
    required_fields = [
        'name', 'customer', 'commodity', 'quantity', 'freight_rate',
        'load_port', 'discharge_port', 'load_rate', 'discharge_rate',
        'load_tt', 'discharge_tt', 'port_cost', 'commission_rate',
        'laycan_start', 'laycan_end'
    ]

    for cargo in get_all_cargoes():
        for field in required_fields:
            assert field in cargo, f"{cargo['name']} missing {field}"

        # Check that freight_rate is not None
        assert cargo['freight_rate'] is not None, f"{cargo['name']} has None freight_rate"
        assert cargo['freight_rate'] > 0, f"{cargo['name']} has invalid freight_rate: {cargo['freight_rate']}"

    print(f"✓ All {len(get_all_cargoes())} cargoes have required fields and valid freight rates")

def test_combinations():
    """Test that we can generate the expected number of combinations"""
    vessels = get_all_vessels()
    cargoes = get_all_cargoes()
    expected_combos = len(vessels) * len(cargoes)

    assert expected_combos == 165, f"Expected 165 combinations (15×11), got {expected_combos}"
    print(f"✓ Expected combinations: {len(vessels)} vessels × {len(cargoes)} cargoes = {expected_combos} total")

def test_hire_rates():
    """Test that all market vessels have reasonable hire rates"""
    for vessel in MARKET_VESSELS:
        hire = vessel['hire_rate']
        assert 10000 <= hire <= 25000, f"{vessel['name']} has unreasonable hire rate: ${hire}/day"

    print(f"✓ All market vessel hire rates are in reasonable range ($10K-$25K/day)")

def test_freight_rates():
    """Test that all market cargoes have reasonable freight rates"""
    for cargo in MARKET_CARGOES:
        rate = cargo['freight_rate']
        assert 5 <= rate <= 30, f"{cargo['name']} has unreasonable freight rate: ${rate}/MT"

    print(f"✓ All market cargo freight rates are in reasonable range ($5-$30/MT)")

def main():
    """Run all tests"""
    print("="*80)
    print("TESTING DATA COMPLETENESS - PRIORITY 1")
    print("="*80)
    print()

    try:
        test_vessel_count()
        test_cargo_count()
        test_vessel_structure()
        test_cargo_structure()
        test_combinations()
        test_hire_rates()
        test_freight_rates()

        print()
        print("="*80)
        print("✅ ALL DATA COMPLETENESS TESTS PASSED!")
        print("="*80)
        print()
        print("Summary:")
        print(f"  • Vessels: {len(get_all_vessels())} total (4 Cargill + 11 Market)")
        print(f"  • Cargoes: {len(get_all_cargoes())} total (3 Committed + 8 Market)")
        print(f"  • Combinations: {len(get_all_vessels()) * len(get_all_cargoes())} total")
        print()
        print("Next step: Run 'python main_analysis.py' to test full analysis")
        print()

        return True

    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ TEST FAILED: {str(e)}")
        print("="*80)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
