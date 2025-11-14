"""
CLI - Master Orchestrator v5.0 GOAT EDITION (Fully Autonomous)

Now features fully autonomous position tracking via position_manager.py
No manual updates required for correlation guard or regime sentinel.
"""
import os
import json
import argparse
from typing import Optional, Dict

# --- Core Service & Engine Imports ---
try:
    from data import fetch_candles
except ImportError:
    fetch_candles = None
    print("❌ CRITICAL: `data.py` not found.")

try:
    from market_psychology import calculate_hurst
except ImportError:
    calculate_hurst = None
    print("⚠️ Warning: `market_psychology.py` not found. Adaptive Personality will be disabled.")

try:
    from chimera import find_quantum_entry
except ImportError:
    find_quantum_entry = None
    print("⚠️ Warning: `chimera.py` not found. Tier 1 Analysis will be skipped.")

try:
    from context_engine import contextual_analysis
except ImportError:
    contextual_analysis = None
    print("⚠️ Warning: `context_engine.py` not found. Tier 2 Analysis will be skipped.")

try:
    from analysis import analyze_pair
except ImportError:
    analyze_pair = None
    print("⚠️ Warning: `analysis.py` not found. Tier 3 Analysis will be skipped.")

# --- Quantum Pre‑Execution Stack Imports ---
try:
    from prex_guardian import run_prex, fetch_ohlcv, fetch_symbol_info  # type: ignore
except ImportError:
    run_prex = None         # type: ignore
    fetch_ohlcv = None      # type: ignore
    fetch_symbol_info = None  # type: ignore
    print("⚠️ Warning: `prex_guardian.py` not found. Quantum Pre‑Execution Guardian disabled.")

try:
    from temporal_oracle import run_teo, TEOResult  # type: ignore
except ImportError:
    run_teo = None          # type: ignore
    TEOResult = None        # type: ignore
    print("⚠️ Warning: `temporal_oracle.py` not found. Temporal Execution Oracle disabled.")

try:
    from quantum_execution import run_qef  # type: ignore
except ImportError:
    run_qef = None           # type: ignore
    print("⚠️ Warning: `quantum_execution.py` not found. Quantum Execution Field disabled.")


def load_personas(filepath: str = "adaptive_personas.json") -> Dict:
    """Loads the persona definitions from the JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ CRITICAL: Persona file not found at `{filepath}`. Adaptive core cannot function.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ CRITICAL: Invalid JSON in `{filepath}`. Personas could not be loaded.")
        return {}

def select_persona(hurst_exponent: Optional[float], personas: Dict) -> Optional[Dict]:
    """Selects a trading persona based on the Hurst Exponent."""
    if hurst_exponent is None:
        print("  -> Psychology: Hurst calculation failed. Defaulting to Predator.")
        return personas.get("Predator")

    if hurst_exponent > 0.55:
        print(f"  -> Psychology: Hurst = {hurst_exponent:.3f}. Market is TRENDING. Adopting [PREDATOR] Persona.")
        return personas.get("Predator")
    elif hurst_exponent < 0.45:
        print(f"  -> Psychology: Hurst = {hurst_exponent:.3f}. Market is MEAN-REVERTING. Adopting [VIPER] Persona.")
        return personas.get("Viper")
    else:
        print(f"  -> Psychology: Hurst = {hurst_exponent:.3f}. Market is RANDOM-WALK. Adopting [OBSERVER] Persona.")
        return personas.get("Observer")

def run_orchestrated_analysis(symbol: str, timeframe: str, risk: float, balance: float):
    """Executes the full, persona-aware, five-tiered analysis protocol with post-processing."""
    print(f"\n\n{'='*30}\n🔬 INITIATING MASTER ORCHESTRATION v5.0 GOAT | {symbol} on {timeframe}\n{'='*30}")

    os.environ["ANALYSIS_ACCOUNT_BALANCE"] = str(balance)
    os.environ["ANALYSIS_RISK_PCT"] = str(risk)

    # --- Quantum Pre‑Execution Stack: PREX + TEO (and later QEF) ---
    prex_verdict = None
    teo_result = None

    if run_prex and fetch_ohlcv and fetch_symbol_info:
        try:
            print("\n--- [PreX] Running Pre‑Execution Guardian ---")
            prex_verdict = run_prex(symbol, timeframe, balance)
            print(f"  -> PREX Regime: {prex_verdict.regime} | Allow: {prex_verdict.allow} | Reason: {prex_verdict.reason}")
        except Exception as e:
            print(f"⚠️ PREX failed with exception: {e}. Continuing without pre‑execution guard.")
            prex_verdict = None
    else:
        print("\n--- [PreX] Guardian not available. Skipping PREX layer.")

    # Fetch candles once for TEO (and potentially QEF current price)
    df_for_teo = None
    if fetch_candles:
        try:
            df_for_teo = fetch_candles(symbol, timeframe, n=500)
        except Exception as e:
            print(f"⚠️ Failed to fetch candles for TEO/QEF: {e}")
            df_for_teo = None

    if run_teo and df_for_teo is not None and not df_for_teo.empty:
        try:
            print("\n--- [TEO] Running Temporal Execution Oracle ---")
            # Prepare OHLCV dict for TEO
            ohlcv_dict = {
                "open": df_for_teo["open"].tolist(),
                "high": df_for_teo["high"].tolist(),
                "low": df_for_teo["low"].tolist(),
                "close": df_for_teo["close"].tolist(),
                "volume": df_for_teo.get("volume", df_for_teo["close"] * 0).tolist(),
            }
            prex_regime = prex_verdict.regime if prex_verdict else "NORMAL"
            teo_result = run_teo(
                symbol=symbol,
                timeframe=timeframe,
                prex_regime=prex_regime,
                ohlcv=ohlcv_dict,
                engines_bias=None  # direction from engines comes later
            )
            print(f"  -> TEO Bias: {teo_result.direction_bias} | Wait ~{teo_result.wait_candles} candles "
                  f"(~{teo_result.estimated_wait_minutes} minutes)")
            print(f"     Condition: {teo_result.wait_condition}")
            print(f"     Invalid if: {teo_result.invalid_if}")
        except Exception as e:
            print(f"⚠️ TEO failed with exception: {e}. Continuing without temporal guidance.")
            teo_result = None
    else:
        print("\n--- [TEO] Temporal Oracle not available or no data. Skipping TEO layer.")

    # If PREX explicitly blocks, we do NOT run heavy engines;
    # instead we return a guided HOLD, including TEO guidance if available.
    if prex_verdict is not None and not prex_verdict.allow:
        print("\n--- [PreX Decision] Environment not safe for immediate execution ---")
        print(f"  -> Reason: {prex_verdict.reason}")
        if teo_result:
            print(f"  -> Directional Bias: {teo_result.direction_bias} (strength {teo_result.bias_strength:.2f})")
            print(f"  -> Recommended Wait: ~{teo_result.wait_candles} candles "
                  f"(~{teo_result.estimated_wait_minutes} minutes)")
            print(f"  -> Wait Condition: {teo_result.wait_condition}")
            print(f"  -> Invalidation: {teo_result.invalid_if}")
        print("\n" + "="*30 + "\n✅ ORCHESTRATION COMPLETE\n" + "="*30)
        print("👤 Active Persona: **Standard (PREX HOLD)**")
        print("🚫 Verdict: GUIDED HOLD. The environment is hostile for execution right now.")
        print("   The system has provided bias, wait time, and structural conditions instead of forcing a trade.\n")
        return

    # --- Step Zero: Market Psychology Analysis ---
    print("\n--- [Step 0] Assessing Market Psychology ---")
    active_persona = None
    personas = load_personas()
    
    if calculate_hurst and personas and fetch_candles:
        df_for_hurst = fetch_candles(symbol, '1h', n=500)
        if df_for_hurst is not None and not df_for_hurst.empty:
            hurst = calculate_hurst(df_for_hurst['close'])
            active_persona = select_persona(hurst, personas)
        else:
            print("  -> Psychology: Could not fetch data for Hurst calculation. Defaulting.")
            active_persona = personas.get("Predator")
    else:
        print("  -> Psychology: Adaptive Personality Core disabled. Running in standard mode.")

    if active_persona and active_persona.get("description", "").startswith("For unpredictable"):
        print("\n" + "="*30 + "\n✅ ORCHESTRATION COMPLETE\n" + "="*30)
        print("🏆 Active Persona: **Observer**")
        print("\n🚫 Verdict: STAND DOWN. The market is unpredictable (random-walk).")
        print("   The highest form of skill is knowing when not to trade. Preserving capital.")
        return

    final_trade_plan: Optional[Dict] = None
    source_engine: str = "None"
    
    # --- Tier 1: Hunt with Chimera ---
    if final_trade_plan is None and find_quantum_entry:
        print("\n--- [Tier 1] Engaging Quantum Engine (`chimera.py`) ---")
        try:
            chimera_result = find_quantum_entry(symbol, htf=timeframe)
            if chimera_result and chimera_result.get("status") == "Success":
                print("✅ SUCCESS: Quantum Entry found. Prioritizing signal.")
                final_trade_plan = chimera_result
                source_engine = "Chimera"
            else:
                print("🛡️ INFO: No Quantum Entry found. Proceeding to next tier.")
        except Exception as e:
            print(f"❌ ERROR: Quantum Engine failed with an exception: {e}.")

    # --- Tier 1.5: Liquidity Hunter ---
    if final_trade_plan is None:
        try:
            from liquidity_hunter import hunt_liquidity_sweep
            print("\n--- [Tier 1.5] Engaging Liquidity Hunter ---")
            lh_result = hunt_liquidity_sweep(symbol, timeframe='1m')
            if lh_result and lh_result.get("status") == "Success":
                print("✅ SUCCESS: Liquidity Sweep Entry found. Prioritizing signal.")
                final_trade_plan = lh_result
                source_engine = "Liquidity Hunter"
            else:
                print("🛡️ INFO: No sweep opportunities detected. Proceeding to next tier.")
        except ImportError:
            print("⚠️ Liquidity Hunter module not available. Skipping Tier 1.5.")
        except Exception as e:
            print(f"❌ ERROR: Liquidity Hunter failed with an exception: {e}. Proceeding to next tier.")

    # --- Tier 2: Analyze with Context Engine ---
    if final_trade_plan is None and contextual_analysis and fetch_candles:
        print("\n--- [Tier 2] Engaging Context Engine (`context_engine.py`) ---")
        try:
            df_main = fetch_candles(symbol, timeframe, n=500)
            if df_main is not None and not df_main.empty:
                context_result = contextual_analysis(df_main, pair=symbol, timeframe=timeframe, fetcher=fetch_candles)
                entry_plan = context_result.get("entry_plan", {})
                if entry_plan.get("direction") in ["Buy", "Sell"]:
                    print("✅ SUCCESS: Context Engine provided a valid trade plan.")
                    final_trade_plan = {
                        "final_suggestion": entry_plan["direction"],
                        "entry_point": entry_plan.get("entry_zone"),
                        "sl": entry_plan.get("sl"), 
                        "tp": entry_plan.get("tp1"),
                        "confidence": context_result.get("confidence"),
                        "precision_reason": f"ContextEngine/{context_result.get('signal', 'N/A')}",
                        "lot_size_recommendation": entry_plan.get("size_suggestion")
                    }
                    source_engine = "Context Engine"
                else:
                    print("🛡️ INFO: Context Engine verdict is neutral. Proceeding to next tier.")
            else:
                print("❌ ERROR: Could not fetch data for Context Engine.")
        except Exception as e:
            print(f"❌ ERROR: Context Engine failed with an exception: {e}.")

    # --- Tier 3: Fallback to GOAT Engine ---
    if final_trade_plan is None and analyze_pair:
        print("\n--- [Tier 3] Engaging GOAT Engine (`analysis.py`) ---")
        try:
            goat_result = analyze_pair(symbol, timeframe)
            if goat_result and goat_result.get("final_suggestion") in ["Buy", "Sell"]:
                print("✅ SUCCESS: GOAT Engine found a valid trade signal.")
                final_trade_plan = goat_result
                source_engine = "GOAT Engine"
            elif goat_result:
                print(f"🛡️ INFO: GOAT Engine verdict is neutral ({goat_result.get('final_suggestion')}).")
        except Exception as e:
            print(f"❌ ERROR: GOAT Engine failed with an exception: {e}")

    # --- Quantum Execution Field (QEF) refinement of final trade plan ---
    if final_trade_plan and final_trade_plan.get("final_suggestion") in ["Buy", "Sell"] and run_qef and prex_verdict:
        try:
            print("\n--- [QEF] Refining execution via Quantum Execution Field ---")
            direction = final_trade_plan["final_suggestion"]
            entry = final_trade_plan.get("entry_point")
            sl = final_trade_plan.get("sl")
            tp = final_trade_plan.get("tp")

            if entry is not None and sl is not None and tp is not None and df_for_teo is not None and not df_for_teo.empty:
                current_price = float(df_for_teo["close"].iloc[-1])
                base_entry = {"entry": float(entry), "sl": float(sl), "tp": float(tp)}

                qef_result = run_qef(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    current_price=current_price,
                    base_entry=base_entry,
                    micro=prex_verdict.microstructure,
                    teo=teo_result if teo_result else TEOResult(
                        direction_bias=direction,
                        bias_strength=final_trade_plan.get("confidence", 0.7),
                        wait_candles=0,
                        estimated_wait_minutes=0,
                        wait_condition="Immediate execution.",
                        invalid_if="Opposite breakout cancels setup.",
                        pattern="Normal"
                    ),
                    recent_trend_slope=0.0,
                )

                if qef_result.best_entry:
                    be = qef_result.best_entry
                    print(f"  -> QEF chose optimal entry @ {be.price} (edge {be.edge:.3f})")
                    final_trade_plan["entry_point"] = be.price
                    final_trade_plan["sl"] = be.sl
                    final_trade_plan["tp"] = be.tp
                    final_trade_plan["qef_best_edge"] = be.edge
                else:
                    print("  -> QEF did not override entry (no better candidate).")

                final_trade_plan["qef"] = qef_result.to_dict()
            else:
                print("  -> QEF skipped (missing entry/SL/TP or price data).")
        except Exception as e:
            print(f"⚠️ QEF refinement failed: {e}")

    # --- Post-Processing: Divergence Check & Correlation Guard (NOW AUTONOMOUS) ---
    if final_trade_plan and final_trade_plan.get("final_suggestion") in ["Buy", "Sell"]:
        print("\n--- [Post-Processing] Applying Enhanced Risk Filters ---")
        
        # Divergence Detector
        try:
            from divergence_detector import check_for_negative_divergence
            
            print("  -> Running Divergence Detector...")
            if fetch_candles:
                df_for_div = fetch_candles(symbol, timeframe, n=200)
            else:
                df_for_div = None

            if df_for_div is not None and not df_for_div.empty:
                div_check = check_for_negative_divergence(df_for_div, final_trade_plan["final_suggestion"])
                if div_check["has_negative_divergence"]:
                    print(f"     ⚠️ Divergence Warning: {div_check['reason']}")
                    print(f"     Confidence penalty applied: {div_check['confidence_penalty']}")
                    current_conf = final_trade_plan.get("confidence", 0.7)
                    final_trade_plan["confidence"] = max(0.3, current_conf + div_check["confidence_penalty"])
                    final_trade_plan["divergence_warning"] = div_check["reason"]
                else:
                    print("     ✅ No negative divergence detected.")
            else:
                print("     ℹ️ No data for divergence detector. Skipping.")
        except ImportError:
            print("  ℹ️ Divergence Detector not available. Skipping check.")
        except Exception as e:
            print(f"  ⚠️ Divergence check failed: {e}")
        
        # Correlation Guard (NOW FULLY AUTONOMOUS - no manual positions list needed)
        try:
            from correlation_guard import check_correlation_risk
            
            print("  -> Running Autonomous Correlation Guard...")
            
            # No manual positions needed - it loads from position_manager automatically
            corr_check = check_correlation_risk(
                symbol, 
                final_trade_plan["final_suggestion"], 
                threshold=0.75
            )
            
            if not corr_check["ok"]:
                if corr_check["action"] == "block":
                    print(f"     🚫 CORRELATION BLOCK: {corr_check['recommendation']}")
                    print("     Trade vetoed by Correlation Guard. No trade will be placed.")
                    final_trade_plan = None
                    source_engine = "Blocked by Correlation Guard"
                elif corr_check["action"] == "scale_down":
                    print(f"     ⚠️ CORRELATION SCALE: {corr_check['recommendation']}")
                    if "lot_size_recommendation" in final_trade_plan:
                        original_lot = final_trade_plan["lot_size_recommendation"]
                        final_trade_plan["lot_size_recommendation"] = round(max(0.01, original_lot * 0.5), 2)
                        print(f"     Lot size scaled from {original_lot} to {final_trade_plan['lot_size_recommendation']}")
                    final_trade_plan["correlation_warning"] = corr_check["recommendation"]
            else:
                print("     ✅ No correlation risk detected.")
        except ImportError:
            print("  ℹ️ Correlation Guard not available. Skipping check.")
        except Exception as e:
            print(f"  ⚠️ Correlation guard check failed: {e}")

    # --- Final Verdict and Report ---
    print("\n" + "="*30 + "\n✅ ORCHESTRATION COMPLETE\n" + "="*30)
    persona_name = "Standard"
    if active_persona:
        for name, details in personas.items():
            if details == active_persona:
                persona_name = name
                break

    print(f"👤 Active Persona: **{persona_name}**")

    if final_trade_plan:
        print(f"🏆 Winning Signal from: **{source_engine}**\n")
        print("--- FINAL TRADE PLAN ---")
        print(f"  Symbol:    {symbol} on {timeframe}")
        print(f"  Verdict:   {final_trade_plan.get('final_suggestion')}")
        print(f"  Reason:    {final_trade_plan.get('precision_reason', 'N/A')}")
        print(f"  Entry:     {final_trade_plan.get('entry_point')}")
        print(f"  Stop Loss: {final_trade_plan.get('sl')}")
        print(f"  Take Profit: {final_trade_plan.get('tp')}")
        print(f"  Confidence: {final_trade_plan.get('confidence')}")
        lot_reco = final_trade_plan.get('lot_size_recommendation_goat') or final_trade_plan.get('lot_size_recommendation')
        if lot_reco: 
            print(f"  Lot Reco:  {lot_reco}")
        
        if final_trade_plan.get('divergence_warning'):
            print(f"  ⚠️ Warning: {final_trade_plan.get('divergence_warning')}")
        if final_trade_plan.get('correlation_warning'):
            print(f"  ⚠️ Warning: {final_trade_plan.get('correlation_warning')}")
        
        print("------------------------\n")
    else:
        print(f"🚫 No actionable trade signal found for {symbol} across all analysis tiers.")
        print("   The system advises to STAND DOWN and preserve capital.\n")


def main():
    """Parses arguments and runs the full orchestration."""
    parser = argparse.ArgumentParser(
        description="RickyFX Master Orchestrator CLI v5.0 GOAT EDITION (Fully Autonomous)", 
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("symbol", type=str, help="The symbol to analyze (e.g., 'EURUSD').")
    parser.add_argument("--timeframe", type=str, default="15m", help="Primary analysis timeframe.")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk percentage per trade.")
    parser.add_argument("--balance", type=float, default=1000.0, help="Simulated account balance.")
    args = parser.parse_args()
    run_orchestrated_analysis(args.symbol, args.timeframe, args.risk, args.balance)

if __name__ == "__main__":
    main()
