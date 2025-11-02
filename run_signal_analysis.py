from td.analysis import run_analysis

# --- Configuration ---
CCLASS_AVRO_PATH = 'data/cclass.avro'
SCLASS_AVRO_PATH = 'data/sclass.avro'

def analyze_signals():
    """Run analysis to find rules for signals based on 'step' events.
    
    Optimized configuration:
    - merge_direction='nearest': Captures signals triggered just before or after step events
    - merge_tolerance=10: Wider window to catch more correlations without too much noise
    - min_samples=3: Good balance between confidence and coverage
    """
    run_analysis(
        cclass_path=CCLASS_AVRO_PATH,
        sclass_path=SCLASS_AVRO_PATH,
        output_rules_path='signal_rules.txt',
        cclass_type='step',
        target_feature='from',  # Predict berth the train is leaving
        max_depth=10,
        min_samples=5,
        merge_direction='nearest',
        merge_tolerance=10
    )

if __name__ == "__main__":
    analyze_signals()
