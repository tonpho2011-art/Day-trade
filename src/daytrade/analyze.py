"""Per-ticker report: company profile, financials, technical signal, and
news -- with a synthesis tying the news to the price action, not just a
raw headline dump."""
from daytrade.company import format_profile_block, get_financials, get_profile
from daytrade.data import get_ohlcv
from daytrade.indicators import build_signal
from daytrade.news import get_headlines, score_sentiment, summarize_news_impact


def analyze(symbol: str) -> dict:
    df = get_ohlcv(symbol, period="6mo", interval="1d")
    tech = build_signal(df)
    headlines = get_headlines(symbol)
    sentiment = score_sentiment(headlines)
    news_summary = summarize_news_impact(headlines, sentiment, tech)

    try:
        profile = get_profile(symbol)
        financials = get_financials(symbol)
    except Exception:
        profile, financials = None, {}

    return {
        "symbol": symbol.upper(),
        "technical": tech,
        "headlines": headlines,
        "sentiment": sentiment,
        "news_summary": news_summary,
        "profile": profile,
        "financials": financials,
    }


def print_report(report: dict) -> None:
    t = report["technical"]
    print(f"\n=== {report['symbol']} ===")

    if report.get("profile"):
        print(format_profile_block(report["profile"], report["financials"]))

    print(f"\nPrice: {t['price']:.2f}   Signal: {t['signal']}  (score {t['score']})")
    print("\nWhy:")
    for reason in t["reasons"]:
        print(f"  - {reason}")

    print(f"\nNews sentiment: {report['sentiment']['label']} "
          f"(+{report['sentiment']['positive_hits']} / -{report['sentiment']['negative_hits']} keyword hits)")
    print(f"\nWhat's driving it: {report['news_summary']}")

    print("\nRecent headlines:")
    if report["headlines"]:
        for h in report["headlines"]:
            print(f"  - {h}")
    else:
        print("  (none found)")

    print("\nNote: this is a rule-based heuristic + keyword scan, not financial advice.")
