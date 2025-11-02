from td.analysis import analyze_bit_timing

# --- Configuration ---
CCLASS_AVRO_PATH = 'data/cclass.avro'
SCLASS_AVRO_PATH = 'data/sclass.avro'
ROUTE_RULES_PATH = 'route_rules.txt'

def analyze_timing():
    """
    Analyze the timing relationship between bit changes and train movements.
    
    This helps distinguish:
    - Route setting bits: Set BEFORE train movements (negative offset)
    - Signal output bits: Change AFTER train movements (positive offset)
    """
    print("=== Route Bit Timing Analysis ===")
    print("\nThis analysis examines when bits change relative to train movements.")
    print("Route bits should be set BEFORE trains move.")
    print("Signal output bits typically change DURING or AFTER movements.\n")
    
    analyze_bit_timing(
        cclass_path=CCLASS_AVRO_PATH,
        sclass_path=SCLASS_AVRO_PATH,
        route_rules_path=ROUTE_RULES_PATH
    )

if __name__ == "__main__":
    analyze_timing()
