"""
The Journal Analyzer ("The Maestro")

Purpose:
- Reads the bot's detailed trade decision logs (analysis_history_goat_plus.csv).
- Simulates trade outcomes (Win/Loss) based on the recorded data.
- Generates a rich, interactive HTML performance dashboard using Plotly.
- Provides deep insights into what strategies, parameters, and market conditions lead to success.

Requires:
- pandas
- plotly

To run:
- python journal_analyzer.py
- This will generate a file named 'performance_dashboard.html' in the same directory.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Configuration ---
LOG_FILE = os.getenv("ANALYSIS_HISTORY_GOATPLUS_CSV", "analysis_history_goat_plus.csv")
OUTPUT_HTML_FILE = "performance_dashboard.html"
SIMULATED_RR = 2.0 # Assume a reward/risk ratio of 2:1 for outcome simulation

def simulate_trade_outcome(row):
    """
    Simulates if a trade would have been a 'Win' or 'Loss'.
    This is a placeholder until real trade outcomes are logged.
    
    Logic: A trade is a 'Win' if it was approved ('label_out' is Buy/Sell)
    and the underlying conditions were favorable.
    """
    if row['label_out'] not in ['Buy', 'Sell']:
        return 'Filtered' # The bot correctly filtered this trade out

    # Simulate wins based on strong underlying metrics
    # This is a proxy for a real win. A more advanced version would use backtesting.
    favorable_conditions = 0
    
    # High trend slope is a good sign
    if pd.notna(row['trend_slope']) and abs(row['trend_slope']) > 0.5:
        favorable_conditions += 1
        
    # Low entropy (clear trend) is good
    if pd.notna(row['entropy']) and row['entropy'] < 0.8:
        favorable_conditions += 1
        
    # Low congestion is good
    if pd.notna(row['congestion']) and row['congestion'] < 0.5:
        favorable_conditions += 1
        
    # No serious blocks
    if not any(blocker in str(row['blocked_reasons']) for blocker in ['sr_too_close', 'high_congestion', 'entry_drift_exceeds_R']):
        favorable_conditions += 1

    return 'Win' if favorable_conditions >= 3 else 'Loss'

def calculate_pnl(row):
    """Calculates a simulated Profit and Loss."""
    if row['outcome'] == 'Win':
        return SIMULATED_RR
    elif row['outcome'] == 'Loss':
        return -1.0
    return 0.0

def analyze_journal():
    print(f"Maestro is starting... Reading journal from '{LOG_FILE}'...")
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: Journal file not found at '{LOG_FILE}'. Please run the bot to generate some data.")
        return

    df = pd.read_csv(LOG_FILE)
    print(f"Found {len(df)} total decisions to analyze.")

    # --- Data Processing and Simulation ---
    df['outcome'] = df.apply(simulate_trade_outcome, axis=1)
    df['pnl'] = df.apply(calculate_pnl, axis=1)
    
    # --- Dashboard Creation ---
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Overall Performance",
            "P&L by Trend Slope",
            "P&L by Market Entropy",
            "Effectiveness of GOAT+ Filters",
            "Performance by Currency Family",
            "Cumulative P&L Over Time"
        ),
        specs=[
            [{"type": "indicator"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "xy"}]
        ]
    )

    # 1. Overall Performance
    total_trades = len(df[df['outcome'].isin(['Win', 'Loss'])])
    win_rate = (df['outcome'] == 'Win').sum() / total_trades if total_trades > 0 else 0
    total_pnl = df['pnl'].sum()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=win_rate * 100,
        title={'text': "Simulated Win Rate (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2E86C1"}},
        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=1)

    # 2. P&L by Trend Slope
    df['slope_bin'] = pd.cut(df['trend_slope'], bins=5)
    pnl_by_slope = df.groupby('slope_bin')['pnl'].sum().reset_index()
    fig.add_trace(go.Bar(x=pnl_by_slope['slope_bin'].astype(str), y=pnl_by_slope['pnl'], name="P&L by Trend"), row=1, col=2)

    # 3. P&L by Entropy
    df['entropy_bin'] = pd.cut(df['entropy'], bins=5)
    pnl_by_entropy = df.groupby('entropy_bin')['pnl'].sum().reset_index()
    fig.add_trace(go.Box(y=df['entropy'], x=df['outcome'], name="Entropy Distribution"), row=2, col=1)

    # 4. Effectiveness of Filters
    df['was_blocked'] = df['label_in'] != df['label_out']
    pnl_blocked = df[df['was_blocked'] == True]['pnl'].sum()
    pnl_not_blocked = df[df['was_blocked'] == False]['pnl'].sum()
    fig.add_trace(go.Bar(
        x=['Money Saved by Blocks', 'P&L from Approved'], 
        y=[-pnl_blocked, pnl_not_blocked], 
        name="Filter Effectiveness"
    ), row=2, col=2)
    
    # 5. Performance by Family
    pnl_by_family = df.groupby('exp_family')['pnl'].sum().reset_index()
    fig.add_trace(go.Bar(x=pnl_by_family['exp_family'], y=pnl_by_family['pnl'], name="P&L by Family"), row=3, col=1)

    # 6. Cumulative P&L
    df['cumulative_pnl'] = df['pnl'].cumsum()
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_pnl'], mode='lines', name='Equity Curve'), row=3, col=2)

    # --- Final Touches ---
    fig.update_layout(
        title_text="<b>The Maestro: Performance Dashboard</b>",
        height=1200,
        showlegend=False,
        template="plotly_dark"
    )

    fig.write_html(OUTPUT_HTML_FILE)
    print(f"\n✅ Maestro has finished. Dashboard saved to '{OUTPUT_HTML_FILE}'.")
    print("   Open this HTML file in your browser to view your bot's performance.")

if __name__ == "__main__":
    analyze_journal()
