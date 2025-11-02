# Train Describer Analysis

A Python tool for collecting and analyzing UK railway Train Describer (TD) data from Network Rail's STOMP feed. This project listens to real-time train movements (C-Class messages) and signaling state changes (S-Class messages), then uses machine learning to decode the relationship between signal bits and train routes/berths.

## Overview

The Train Describer system is used by Network Rail to track train movements across the UK railway network. This project:

1. **Collects Data**: Connects to Network Rail's live data feed via STOMP protocol to capture:
   - **C-Class Messages**: Train movements (steps, cancellations, interposes)
   - **S-Class Messages**: Signal/route bit state changes

2. **Stores Data**: Saves messages in efficient Apache Avro format for analysis

3. **Analyzes Patterns**: Uses Decision Tree machine learning to identify:
   - Which signal bits correspond to specific berths (locations)
   - Which bits represent route selections
   - Timing relationships between bit changes and train movements

## How It Works

The project includes three analysis scripts:

- **`run_signal_analysis.py`**: Identifies which signal bits correspond to train berths/locations
- **`run_route_analysis.py`**: Detects which bits represent route selections
- **`run_timing_analysis.py`**: Analyzes when bits change relative to train movements to distinguish route setting bits from signal output bits

Analysis uses Decision Trees to find patterns, outputting human-readable rules like:
```
IF address == 'EA' AND bit_index == 3 THEN PREDICT: 0015 (covers 45 samples)
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Network Rail data feed account (free registration required)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrainlyHQ/td-analysis
   cd td-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On macOS/Linux
   # or
   env\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (see below)

5. **Create required directories**
   ```bash
   mkdir -p data
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Network Rail Data Feed Credentials
NETWORK_RAIL_USERNAME=your_email@example.com
NETWORK_RAIL_PASSWORD=your_password
NETWORK_RAIL_HOST=publicdatafeeds.networkrail.co.uk
NETWORK_RAIL_PORT=61618

# Train Describer Feed Configuration
TD_FEED_TOPIC=TD_ALL_SIG_AREA
AREA_ID=M1 # Find at https://wiki.openraildata.com/index.php/List_of_Train_Describers

# Data File Paths (optional - defaults shown)
CCLASS_FILE=data/cclass.avro
SCLASS_FILE=data/sclass.avro
```

### Where to Get Credentials

1. **Register for a Network Rail Data Feed account**:
   - Visit: https://publicdatafeeds.networkrail.co.uk/
   - Click "Register" and complete the registration form
   - Accept the data feed license terms

2. **Get your credentials**:
   - Your username is your email address
   - Your password will be sent to you via email

3. **Choose your area ID**:
   - Area IDs correspond to Network Rail signaling areas (e.g., "M1", "GD", "PH")
   - See [Open Rail Data Wiki](https://wiki.openraildata.com/index.php/List_of_Train_Describers) for available area codes
   - Common examples: M1 (Liverpool), GD (Guildford), PH (Plymouth)

## Usage

### 1. Collect Live Data
```bash
cd td
python3 listener.py
```
This will connect to the Network Rail feed and start collecting data to `data/cclass.avro` and `data/sclass.avro`. Let it run for a while to collect sufficient data (at least several hours for meaningful analysis).

### 2. Run Analysis

**Analyze signal/berth mappings:**
```bash
python3 run_signal_analysis.py
```
Output: `signal_rules.txt`

**Analyze route selections:**
```bash
python3 run_route_analysis.py
```
Output: `route_rules.txt`

**Analyze bit timing patterns:**
```bash
python3 run_timing_analysis.py
```
Shows timing relationships between bit changes and train movements.

## Authors

- Woody Willis

## License

Please ensure compliance with Network Rail's data feed license terms when using this software.