"""Per-ticker report: technical signal + recent news, with reasoning."""
from daytrade.data import get_ohlcv
from daytrade.indicators import build_signal
from daytrade.news import get_headlines, score_sentiment


def analyze(symbol: str) -> dict:
    df = get_ohlcv(symbol, period="6mo", interval="1d")
    tech = build_signal(df)
    headlines = get_headlines(symbol)
    sentiment = score_sentiment(headlines)
    return {"symbol": symbol.upper(), "technical": tech, "headlines": headlines, "sentiment": sentiment}


def print_report(report: dict) -> None:
    t = report["technical"]
    print(f"\n=== {report['symbol']} ===")
    print(f"Price: {t['price']:.2f}   Signal: {t['signal']}  (score {t['score']})")
    print("\nWhy:")
    for reason in t["reasons"]:
        print(f"  - {reason}")

    print(f"\nNews sentiment: {report['sentiment']['label']} "
          f"(+{report['sentiment']['positive_hits']} / -{report['sentiment']['negative_hits']} keyword hits)")
    print("Recent headlines:")
    if report["headlines"]:
        for h in report["headlines"]:
            print(f"  - {h}")
    else:
        print("  (none found)")

    print("\nNote: this is a rule-based heuristic + keyword scan, not financial advice.")
