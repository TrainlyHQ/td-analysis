from td.analysis import run_analysis, analyze_route_bits

# --- Configuration ---
CCLASS_AVRO_PATH = 'data/cclass.avro'
SCLASS_AVRO_PATH = 'data/sclass.avro'

def analyze_routes():
    """
    Run analysis to find rules for routes based on 'step' events.
    
    IMPORTANT: Route setting bits are NOT the same as train movements (steps).
    Route setting bits indicate which route has been SELECTED by the signalman,
    and they have these characteristics:
    - Set in advance of train movements
    - May not clear every time a train passes
    - Where trains can be signaled in two ways, there will be TWO bits (one for each route)
    - Both bits at 0 = neither route selected
    - Both bits at 1 = likely indicates incorrect decode
    
    Note: Some "route rules" may actually be output signal bits rather than 
    true route setting bits. This analysis uses train step events as a proxy
    to infer which bits correlate with route selections.
    """
    run_analysis(
        cclass_path=CCLASS_AVRO_PATH,
        sclass_path=SCLASS_AVRO_PATH,
        output_rules_path='route_rules.txt',
        cclass_type='step', # Use step events to identify routes
        target_feature='route', # Predict the route (from->to combination)
        max_depth=12
    )
    
    # Run additional analysis to detect potential route bit pairs and conflicts
    print("\n--- Analyzing route bit patterns ---")
    analyze_route_bits(
        sclass_path=SCLASS_AVRO_PATH,
        route_rules_path='route_rules.txt'
    )

if __name__ == "__main__":
    analyze_routes()
