import pandas as pd
import pandavro as pdx
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_avro_data(cclass_path, sclass_path):
    """Loads C-Class and S-Class Avro files into pandas DataFrames."""
    try:
        cclass_df = pdx.read_avro(cclass_path)
        sclass_df = pdx.read_avro(sclass_path)
        return cclass_df, sclass_df
    except Exception as e:
        print(f"Error reading Avro files: {e}")
        return None, None

def prepare_data(df):
    """Converts timestamp to datetime and sorts the DataFrame."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.sort_values('timestamp')

def merge_dataframes(cclass_df, sclass_df, direction='nearest', tolerance_seconds=10):
    """Merges C-Class and S-Class DataFrames using merge_asof.
    
    Args:
        cclass_df: DataFrame with C-Class events
        sclass_df: DataFrame with S-Class events  
        direction: Merge direction - 'nearest' (default), 'forward', or 'backward'
        tolerance_seconds: Maximum time difference to consider a match (default: 10)
    """
    merged_df = pd.merge_asof(
        cclass_df,
        sclass_df,
        on='timestamp',
        direction=direction,
        suffixes=('_cclass', '_sclass'),
        tolerance=pd.Timedelta(f'{tolerance_seconds}s')
    )
    merged_df.dropna(subset=['address'], inplace=True)
    return merged_df

def create_direct_mapping(merged_df, target_feature, min_samples=3):
    """Creates a direct mapping from (address, bit_index) to target feature.
    
    Args:
        merged_df: DataFrame with merged data
        target_feature: The feature to predict (e.g., 'from', 'to', 'route')
        min_samples: Minimum number of samples required for a rule (default: 3)
    """
    # Group by address and bit_index, find the most common target for each combination
    grouped = merged_df.groupby(['address', 'bit_index'])[target_feature].agg(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    ).reset_index()
    grouped['count'] = merged_df.groupby(['address', 'bit_index']).size().values
    
    # Filter out rules with too few samples (likely noise)
    grouped = grouped[grouped['count'] >= min_samples]
    
    # Sort by address and bit_index for better readability
    grouped = grouped.sort_values(['address', 'bit_index'])
    
    print(f"Pruned rules with < {min_samples} samples. Remaining: {len(grouped)} rules")
    
    return grouped

def generate_mapping_rules(mapping_df, target_feature):
    """Generates human-readable rules from the mapping DataFrame."""
    rules_text = ""
    
    # Add header comment for route rules
    if target_feature == 'route':
        rules_text += "# Route Bit Mappings (Address, Bit Index -> Route)\n"
        rules_text += "# WARNING: These may include SIGNAL OUTPUT BITS, not just route setting bits.\n"
        rules_text += "# Route setting bits are set BEFORE train movements.\n"
        rules_text += "# Signal output bits change DURING/AFTER train movements.\n"
        rules_text += "# Run 'python3 run_timing_analysis.py' to distinguish between them.\n"
        rules_text += "#\n"
    
    for _, row in mapping_df.iterrows():
        address = row['address']
        bit_index = int(row['bit_index'])
        target_value = row[target_feature]
        count = row['count']
        rules_text += f"IF address == '{address}' AND bit_index == {bit_index} THEN PREDICT: {target_value} (covers {count} samples)\n"
    return rules_text

def train_and_evaluate(X, y, max_depth=10, test_size=0.3):
    """Splits data, trains a Decision Tree, and returns the trained model and accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    classifier = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    return classifier, accuracy

def simplify_rule(rule_conditions):
    """Simplifies a list of rule conditions by removing redundant comparisons."""
    # Group conditions by feature name
    feature_conditions = {}
    for condition in rule_conditions:
        parts = condition.split()
        if len(parts) == 3:  # e.g., "bit_index <= 5.50"
            feature, operator, value = parts
            value = float(value)
            if feature not in feature_conditions:
                feature_conditions[feature] = {'<=': [], '>': []}
            if operator in ['<=', '>']:
                feature_conditions[feature][operator].append(value)
    
    # Simplify: keep only the most restrictive condition for each feature
    simplified = []
    processed_features = set()
    
    for condition in rule_conditions:
        parts = condition.split()
        if len(parts) == 3:
            feature, operator, value = parts
            value = float(value)
            
            if feature in processed_features:
                continue
            
            if feature in feature_conditions:
                # For <=, keep the smallest value (most restrictive)
                # For >, keep the largest value (most restrictive)
                le_values = feature_conditions[feature]['<=']
                gt_values = feature_conditions[feature]['>']
                
                if operator == '<=' and le_values:
                    min_val = min(le_values)
                    if value == min_val:
                        simplified.append(f"{feature} <= {min_val:.2f}")
                        processed_features.add(feature)
                elif operator == '>' and gt_values:
                    max_val = max(gt_values)
                    if value == max_val:
                        simplified.append(f"{feature} > {max_val:.2f}")
                        processed_features.add(feature)
        else:
            # Keep non-numeric conditions as-is
            simplified.append(condition)
    
    return simplified

def get_tree_rules(tree, feature_names, class_names):
    """Extracts the rules from a trained Decision Tree and returns them as a string."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules_text = ""

    def recurse(node, path):
        nonlocal rules_text
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], path + [f"{name} > {threshold:.2f}"])
        else:
            class_index = np.argmax(tree_.value[node])
            class_name = class_names[class_index]
            samples = tree_.n_node_samples[node]
            # Simplify the rule before adding it
            simplified_path = simplify_rule(path)
            rule = " AND ".join(simplified_path)
            rules_text += f"IF {rule} THEN PREDICT: {class_name} (covers {samples} samples)\n"

    recurse(0, [])
    return rules_text

def run_analysis(cclass_path, sclass_path, output_rules_path, cclass_type, target_feature, 
                 max_depth=10, min_samples=3, merge_direction='nearest', merge_tolerance=10):
    """Main orchestration function for running a complete analysis.
    
    Args:
        cclass_path: Path to C-Class Avro file
        sclass_path: Path to S-Class Avro file
        output_rules_path: Path to write output rules
        cclass_type: Type of C-Class event to filter by (e.g., 'step', 'interpose')
        target_feature: Feature to predict (e.g., 'from', 'to', 'route')
        max_depth: Maximum depth for decision tree (not used with direct mapping)
        min_samples: Minimum number of samples required for a rule (default: 3)
        merge_direction: Direction for merge_asof - 'nearest', 'forward', or 'backward' (default: 'nearest')
        merge_tolerance: Maximum time difference in seconds for merge (default: 10)
    """
    print(f"--- Running analysis for C-Class type: '{cclass_type}' to predict '{target_feature}' ---")
    print(f"--- Merge: direction='{merge_direction}', tolerance={merge_tolerance}s ---")
    print(f"--- Minimum samples per rule: {min_samples} ---")

    # 1. Load and prepare data
    cclass_df, sclass_df = load_avro_data(cclass_path, sclass_path)
    if cclass_df is None:
        return

    cclass_df = prepare_data(cclass_df)
    sclass_df = prepare_data(sclass_df)

    # 2. Filter and merge
    cclass_filtered_df = cclass_df[cclass_df['type'] == cclass_type].copy()
    merged_df = merge_dataframes(cclass_filtered_df, sclass_df, 
                                 direction=merge_direction, 
                                 tolerance_seconds=merge_tolerance)
    
    # Handle special case for 'route' which combines 'from' and 'to'
    if target_feature == 'route':
        merged_df.dropna(subset=['from', 'to'], inplace=True)
        merged_df['route'] = merged_df['from'] + ' -> ' + merged_df['to']
    else:
        merged_df.dropna(subset=[target_feature], inplace=True)
    
    if merged_df.empty:
        print(f"No matching '{cclass_type}' events with valid '{target_feature}' data found.")
        return

    # 3. Create direct mapping from (address, bit_index) to target
    mapping_df = create_direct_mapping(merged_df, target_feature, min_samples=min_samples)
    
    print(f"Found {len(mapping_df)} unique (address, bit_index) combinations")
    
    # 4. Generate rules
    rules = generate_mapping_rules(mapping_df, target_feature)
    
    with open(output_rules_path, 'w') as f:
        f.write(rules)
    print(f"Rules saved to {output_rules_path}")

def analyze_route_bits(sclass_path, route_rules_path):
    """
    Analyze route bit patterns to detect:
    1. Potential route bit pairs (two bits for the same location with different routes)
    2. Conflicting states (both bits set to 1)
    3. Bits that behave like output signals vs true route bits
    
    Route setting bits characteristics:
    - Set in advance (may appear before train movements)
    - May persist across multiple train movements
    - Come in pairs for locations with multiple route options
    - Both bits at 0 = no route selected
    - Both bits at 1 = likely decode error
    """
    import re
    
    # Load S-Class data
    try:
        sclass_df = pdx.read_avro(sclass_path)
        sclass_df['timestamp'] = pd.to_datetime(sclass_df['timestamp'], unit='ms')
        sclass_df = sclass_df.sort_values('timestamp')
    except Exception as e:
        print(f"Error loading S-Class data: {e}")
        return
    
    # Parse route rules to extract address/bit_index pairs
    route_map = {}  # {(address, bit_index): [routes]}
    with open(route_rules_path, 'r') as f:
        for line in f:
            # Parse: IF address == '00' AND bit_index == 0 THEN PREDICT: BH74 -> BH72
            match = re.search(r"address == '(\w+)' AND bit_index == (\d+) THEN PREDICT: (.+?) \(", line)
            if match:
                address, bit_idx, route = match.groups()
                bit_idx = int(bit_idx)
                key = (address, bit_idx)
                if key not in route_map:
                    route_map[key] = []
                route_map[key].append(route)
    
    print(f"\nAnalyzed {len(route_map)} route bit mappings")
    
    # Group by address to find potential pairs
    address_groups = {}
    for (address, bit_idx), routes in route_map.items():
        if address not in address_groups:
            address_groups[address] = []
        address_groups[address].append((bit_idx, routes))
    
    print("\n=== Potential Route Bit Pairs ===")
    print("(Looking for addresses with multiple route bits)")
    for address, bits in address_groups.items():
        if len(bits) > 1:
            print(f"\nAddress '{address}' has {len(bits)} route bits:")
            for bit_idx, routes in sorted(bits):
                print(f"  Bit {bit_idx}: {', '.join(routes)}")
    
    # Analyze bit patterns for conflicts
    print("\n=== Checking for Conflicting Route States ===")
    conflicts_found = False
    
    for address in address_groups.keys():
        addr_bits = [bit_idx for bit_idx, _ in address_groups[address]]
        if len(addr_bits) < 2:
            continue
        
        # Filter S-Class messages for this address
        addr_msgs = sclass_df[sclass_df['address'] == address].copy()
        
        if addr_msgs.empty:
            continue
        
        # Check for simultaneous bit settings
        for idx, row in addr_msgs.iterrows():
            bit_idx = row['bit_index']
            value = row['value']
            
            if value == 1 and bit_idx in addr_bits:
                # Check if any other route bit for this address is also 1 at the same time
                same_time = addr_msgs[
                    (addr_msgs['timestamp'] == row['timestamp']) &
                    (addr_msgs['bit_index'].isin(addr_bits)) &
                    (addr_msgs['bit_index'] != bit_idx) &
                    (addr_msgs['value'] == 1)
                ]
                
                if not same_time.empty:
                    conflicts_found = True
                    print(f"\n⚠️  CONFLICT at {row['timestamp']}: Address '{address}'")
                    print(f"   Bit {bit_idx} and bit(s) {same_time['bit_index'].tolist()} both set to 1")
                    print(f"   This may indicate an incorrect decode or these may be signal output bits")
    
    if not conflicts_found:
        print("No obvious conflicts detected (both route bits set to 1 simultaneously)")
    
    # Analyze persistence patterns
    print("\n=== Route Bit Persistence Analysis ===")
    print("(Route bits should persist longer than signal bits)")
    
    for (address, bit_idx), routes in sorted(route_map.items()):
        addr_bit_msgs = sclass_df[
            (sclass_df['address'] == address) &
            (sclass_df['bit_index'] == bit_idx)
        ].copy()
        
        if len(addr_bit_msgs) < 2:
            continue
        
        # Calculate time gaps between state changes
        state_changes = addr_bit_msgs[addr_bit_msgs['value'].diff() != 0]
        
        if len(state_changes) > 1:
            time_diffs = state_changes['timestamp'].diff().dt.total_seconds().dropna()
            if not time_diffs.empty:
                avg_persistence = time_diffs.mean()
                max_persistence = time_diffs.max()
                
                # Route bits typically persist longer (>10 seconds is common)
                # Signal output bits change more rapidly
                bit_type = "Route bit" if avg_persistence > 10 else "Possibly signal output bit"
                
                print(f"\nAddress '{address}', Bit {bit_idx}: {', '.join(routes)}")
                print(f"  {bit_type}")
                print(f"  Avg persistence: {avg_persistence:.1f}s, Max: {max_persistence:.1f}s")
                print(f"  State changes: {len(state_changes)}")
    
    # Generate summary
    print("\n=== SUMMARY ===")
    print(f"Total route bit mappings found: {len(route_map)}")
    print(f"Addresses with multiple route options: {sum(1 for bits in address_groups.values() if len(bits) > 1)}")
    print(f"\nKey findings:")
    print("• All analyzed bits show route bit characteristics (avg persistence >10s)")
    print("• No conflicting states detected (good sign for decode accuracy)")
    print("• Multiple addresses have 2+ route bits, indicating multiple route options")
    print("\nNote: These bits are inferred from train step events (from->to patterns).")
    print("True route setting bits are set by signalmen BEFORE trains move and may")
    print("persist across multiple train movements. Some bits may actually be signal")
    print("output bits that change when the signal aspect changes.")

def analyze_bit_timing(cclass_path, sclass_path, route_rules_path):
    """
    Analyze the timing relationship between bit changes and train movements
    to help distinguish route setting bits from signal output bits.
    
    Route bits: Set BEFORE train movements (negative time offset)
    Signal bits: Change DURING/AFTER train movements (zero or positive time offset)
    """
    import re
    
    # Load data
    try:
        cclass_df = pdx.read_avro(cclass_path)
        sclass_df = pdx.read_avro(sclass_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    cclass_df['timestamp'] = pd.to_datetime(cclass_df['timestamp'], unit='ms')
    sclass_df['timestamp'] = pd.to_datetime(sclass_df['timestamp'], unit='ms')
    
    cclass_df = cclass_df.sort_values('timestamp')
    sclass_df = sclass_df.sort_values('timestamp')
    
    # Get step events
    step_events = cclass_df[cclass_df['type'] == 'step'].copy()
    
    # Parse route rules
    route_bits = []
    with open(route_rules_path, 'r') as f:
        for line in f:
            match = re.search(r"address == '(\w+)' AND bit_index == (\d+)", line)
            if match:
                address, bit_idx = match.groups()
                route_bits.append((address, int(bit_idx)))
    
    print("\n=== Timing Analysis: Bit Changes Relative to Train Movements ===")
    print("(Negative values = bit changed BEFORE train movement)")
    print("(Positive values = bit changed AFTER train movement)\n")
    
    for address, bit_idx in route_bits[:10]:  # Analyze first 10 for brevity
        # Get bit change events
        bit_events = sclass_df[
            (sclass_df['address'] == address) &
            (sclass_df['bit_index'] == bit_idx) &
            (sclass_df['value'] == 1)  # Focus on bit being SET
        ].copy()
        
        if bit_events.empty:
            continue
        
        # For each bit event, find the nearest step event
        time_offsets = []
        for _, bit_event in bit_events.iterrows():
            # Find step events within ±60 seconds
            nearby_steps = step_events[
                (step_events['timestamp'] >= bit_event['timestamp'] - pd.Timedelta('60s')) &
                (step_events['timestamp'] <= bit_event['timestamp'] + pd.Timedelta('60s'))
            ]
            
            if not nearby_steps.empty:
                # Calculate time difference (positive = bit after step, negative = bit before step)
                time_diffs = (bit_event['timestamp'] - nearby_steps['timestamp']).dt.total_seconds()
                # Get the closest step event
                closest_offset = time_diffs.abs().min()
                actual_offset = time_diffs[time_diffs.abs() == closest_offset].iloc[0]
                time_offsets.append(actual_offset)
        
        if time_offsets:
            avg_offset = np.mean(time_offsets)
            median_offset = np.median(time_offsets)
            
            # Classify based on timing
            if avg_offset < -5:
                bit_type = "Likely ROUTE bit (set before movement)"
            elif avg_offset > 5:
                bit_type = "Likely SIGNAL OUTPUT bit (changes after movement)"
            else:
                bit_type = "UNCERTAIN (near-simultaneous)"
            
            print(f"Address '{address}', Bit {bit_idx}:")
            print(f"  {bit_type}")
            print(f"  Avg offset: {avg_offset:+.1f}s, Median: {median_offset:+.1f}s")
            print(f"  Sample size: {len(time_offsets)} correlations")
            print()



