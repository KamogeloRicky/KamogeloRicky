"""
GUI Bridge Module - Orchestration Connector (Production Version)

This module ensures that when the GUI requests an analysis, it gets the full
three-tiered orchestration instead of just the GOAT engine.

The bridge captures the output from cli.run_orchestrated_analysis() and returns
it in a format the GUI expects.
"""
import os
import sys
from io import StringIO
from typing import Dict

try:
    from analysis import analyze_pair as fallback_analyze_pair
except ImportError:
    fallback_analyze_pair = None


def analyze_for_gui(symbol: str, timeframe: str) -> Dict:
    """
    Routes GUI analysis requests to the full orchestration.
    
    This function captures the console output from the orchestrator and
    then calls the GOAT engine to get the structured result dict.
    """
    # Set environment variables (GUI should have already set these, but ensure they exist)
    if not os.getenv("ANALYSIS_ACCOUNT_BALANCE"):
        os.environ["ANALYSIS_ACCOUNT_BALANCE"] = "1000"
    if not os.getenv("ANALYSIS_RISK_PCT"):
        os.environ["ANALYSIS_RISK_PCT"] = "1.0"
    
    # Capture orchestrator output for logging
    captured_output = StringIO()
    original_stdout = sys.stdout
    
    try:
        # Run the full orchestration (this will print to console)
        # The orchestrator itself doesn't return a value, so we capture its prints
        sys.stdout = captured_output
        
        try:
            from cli import run_orchestrated_analysis
            
            # Run orchestration (this executes all three tiers)
            run_orchestrated_analysis(
                symbol=symbol,
                timeframe=timeframe,
                risk=float(os.getenv("ANALYSIS_RISK_PCT", "1.0")),
                balance=float(os.getenv("ANALYSIS_ACCOUNT_BALANCE", "1000"))
            )
        except ImportError:
            pass  # Orchestrator not available
        finally:
            sys.stdout = original_stdout
        
        orchestration_log = captured_output.getvalue()
        
        # Now call the GOAT engine directly to get the structured result
        # The orchestration has already set up the environment and run pre-checks
        if fallback_analyze_pair:
            result = fallback_analyze_pair(symbol, timeframe)
            
            # Append orchestration log to the result for transparency
            if isinstance(result, dict):
                original_report = result.get("report", "")
                result["report"] = (
                    "=== ORCHESTRATION LOG ===\n" +
                    orchestration_log +
                    "\n\n=== GOAT ENGINE REPORT ===\n" +
                    str(original_report)
                )
            
            return result
        else:
            return {
                "success": False,
                "error": "No analysis engine available",
                "report": orchestration_log
            }
    
    except Exception as e:
        sys.stdout = original_stdout
        return {
            "success": False,
            "error": f"GUI Bridge error: {str(e)}",
            "report": captured_output.getvalue()
        }
