from analysis_bridge import analyze_with_context

if __name__ == "__main__":
    pair = "EURUSDm"
    timeframe = "1h"
    result = analyze_with_context(pair, timeframe)
    # Print a concise summary
    base = result.get("base", {})
    ctx = result.get("context", {}) or {}
    print("Base suggestion:", base.get("final_suggestion"), "price:", base.get("price"))
    print("Context signal:", ctx.get("signal"), "score:", ctx.get("score"), "conf:", ctx.get("confidence"))
    print("Entry plan:", result.get("entry_plan"))
