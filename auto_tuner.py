"""
The Auto-Tuning Engine

Purpose:
- Reads the complete performance history from the journal file.
- Analyzes the profitability of trades under different parameter conditions.
- Determines the optimal thresholds for key strategy parameters (e.g., trend slope, entropy).
- Generates a new, optimized strategy profile JSON file.
- This closes the feedback loop, allowing the bot to learn from its own performance.

To Run:
- python auto_tuner.py
- This will generate 'optimized_profile.json'. The StrategyCore will automatically use this file on the next run.
"""
import pandas as pd
import numpy as np
import json
import os

# --- Configuration ---
JOURNAL_FILE = os.getenv("ANALYSIS_HISTORY_GOATPLUS_CSV", "analysis_history_goat_plus.csv")
BASE_PROFILE_FILE = "strategy_profiles.json"
OPTIMIZED_PROFILE_FILE = "optimized_profile.json"
SIMULATED_RR = 2.0  # Must match the value in journal_analyzer.py for consistent simulation

# --- Parameters to be Tuned ---
# We will find the optimal thresholds for these journal columns.
TUNING_CONFIG = {
    # Parameter Name in Profile: (Journal Column, Optimal Condition)
    "min_trend_slope": ("trend_slope", "greater_than"),
    "max_entropy": ("entropy", "less_than"),
    "max_congestion": ("congestion", "less_than"),
}

def simulate_trade_outcome(row):
    """Simulates trade outcome. Must be consistent with the Maestro's simulation."""
    if row['label_out'] not in ['Buy', 'Sell']: return 'Filtered'
    favorable_conditions = 0
    if pd.notna(row['trend_slope']) and abs(row['trend_slope']) > 0.5: favorable_conditions += 1
    if pd.notna(row['entropy']) and row['entropy'] < 0.8: favorable_conditions += 1
    if pd.notna(row['congestion']) and row['congestion'] < 0.5: favorable_conditions += 1
    if not any(b in str(row['blocked_reasons']) for b in ['sr_too_close', 'high_congestion']): favorable_conditions += 1
    return 'Win' if favorable_conditions >= 3 else 'Loss'

def calculate_pnl(row):
    """Calculates simulated P&L. Must be consistent with the Maestro's simulation."""
    if row['outcome'] == 'Win': return SIMULATED_RR
    elif row['outcome'] == 'Loss': return -1.0
    return 0.0

def find_optimal_threshold(df: pd.DataFrame, column: str, condition: str) -> Optional[float]:
    """
    Finds the best threshold for a given parameter by maximizing simulated P&L.
    """
    if column not in df.columns or df[column].isnull().all():
        return None

    best_pnl = -np.inf
    best_threshold = None
    
    # Iterate through potential thresholds (from 5th to 95th percentile)
    for percentile in np.linspace(0.05, 0.95, 20):
        threshold = df[column].quantile(percentile)
        if pd.isna(threshold):
            continue

        if condition == "greater_than":
            filtered_df = df[df[column] > threshold]
        elif condition == "less_than":
            filtered_df = df[df[column] < threshold]
        else:
            continue
            
        current_pnl = filtered_df['pnl'].sum()
        
        if current_pnl > best_pnl:
            best_pnl = current_pnl
            best_threshold = threshold
            
    return best_threshold

def run_auto_tuner():
    """Main function to run the tuning process."""
    print("🤖 Auto-Tuning Engine starting...")
    
    if not os.path.exists(JOURNAL_FILE):
        print(f"ERROR: Journal file '{JOURNAL_FILE}' not found. Cannot perform tuning.")
        return

    df = pd.read_csv(JOURNAL_FILE)
    if len(df) < 50:
        print(f"WARNING: Only {len(df)} records found. At least 50 are recommended for meaningful tuning. Proceeding with caution.")
    
    # Simulate P&L to evaluate performance
    df['outcome'] = df.apply(simulate_trade_outcome, axis=1)
    df['pnl'] = df.apply(calculate_pnl, axis=1)
    
    # --- Start Tuning Process ---
    optimized_params = {}
    print("Analyzing historical performance to find optimal parameters...")
    for param_name, (column, condition) in TUNING_CONFIG.items():
        print(f"  - Tuning for '{param_name}' based on '{column}'...")
        optimal_value = find_optimal_threshold(df, column, condition)
        
        if optimal_value is not None:
            optimized_params[param_name] = round(optimal_value, 4)
            print(f"    => Found optimal threshold: {optimal_value:.4f}")
        else:
            print(f"    => Could not determine optimal threshold for '{param_name}'.")

    if not optimized_params:
        print("No optimal parameters could be determined. Aborting profile generation.")
        return

    # --- Generate the new profile ---
    print("\nGenerating new 'optimized_v1' strategy profile...")
    if not os.path.exists(BASE_PROFILE_FILE):
        print(f"ERROR: Base profile '{BASE_PROFILE_FILE}' not found. Cannot create optimized profile.")
        return
        
    with open(BASE_PROFILE_FILE, 'r') as f:
        config = json.load(f)

    # Use the 'default' profile as a template for the new optimized one
    optimized_profile = config['profiles'].get('default', {}).copy()
    optimized_profile['description'] = f"Auto-tuned by The Maestro on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Add the new, machine-learned parameters
    optimized_profile.update(optimized_params)
    
    # Add the new profile to the main config structure
    config['profiles']['optimized_v1'] = optimized_profile
    
    # Save the complete new configuration
    with open(OPTIMIZED_PROFILE_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*50)
    print("✅ SUCCESS: Auto-Tuning Complete!")
    print(f"   New profile 'optimized_v1' has been generated.")
    print(f"   Saved to: '{OPTIMIZED_PROFILE_FILE}'")
    print("   The Guardian will automatically load these new settings on the next run.")
    print("="*50)

if __name__ == "__main__":
    run_auto_tuner()
