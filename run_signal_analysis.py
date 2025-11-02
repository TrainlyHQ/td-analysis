from td.analysis import run_analysis

# --- Configuration ---
CCLASS_AVRO_PATH = 'data/cclass.avro'
SCLASS_AVRO_PATH = 'data/sclass.avro'

def analyze_signals():
    """Run analysis to find rules for signals based on 'step' events."""
    run_analysis(
        cclass_path=CCLASS_AVRO_PATH,
        sclass_path=SCLASS_AVRO_PATH,
        output_rules_path='signal_rules.txt',
        cclass_type='step',
        target_feature='from', # Predict berth the train is leaving
        max_depth=10
    )

if __name__ == "__main__":
    analyze_signals()
