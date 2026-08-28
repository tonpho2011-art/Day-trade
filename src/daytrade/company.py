"""Company profile and headline fundamentals via yfinance's `info` /
annual financials -- gives an analyze() report actual substance instead
of just a raw headline dump."""
import yfinance as yf


def _fmt_money(value) -> str:
    if value is None:
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def get_profile(symbol: str) -> dict:
    """Business summary + sector/industry + key valuation stats."""
    info = yf.Ticker(symbol).info
    summary = info.get("longBusinessSummary") or ""
    if summary:
        sentences = summary.split(". ")
        summary = ". ".join(sentences[:2]).rstrip(".") + "."

    return {
        "name": info.get("longName") or symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "employees": info.get("fullTimeEmployees"),
        "summary": summary,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
    }


def get_financials(symbol: str) -> dict:
    """Most recent annual revenue/net income plus YoY growth, from the
    annual income statement (last two fiscal years)."""
    fin = yf.Ticker(symbol).financials
    if fin is None or fin.empty or fin.shape[1] < 1:
        return {}

    def row(name):
        return fin.loc[name] if name in fin.index else None

    revenue = row("Total Revenue")
    net_income = row("Net Income")

    result = {}
    if revenue is not None and len(revenue) >= 1:
        result["revenue_latest"] = revenue.iloc[0]
        result["revenue_period"] = str(fin.columns[0])[:10]
        if len(revenue) >= 2 and revenue.iloc[1]:
            result["revenue_yoy"] = (revenue.iloc[0] - revenue.iloc[1]) / abs(revenue.iloc[1])

    if net_income is not None and len(net_income) >= 1:
        result["net_income_latest"] = net_income.iloc[0]
        if len(net_income) >= 2 and net_income.iloc[1]:
            result["net_income_yoy"] = (net_income.iloc[0] - net_income.iloc[1]) / abs(net_income.iloc[1])

    return result


def format_profile_block(profile: dict, financials: dict) -> str:
    lines = []
    header = profile["name"]
    if profile.get("sector") or profile.get("industry"):
        header += f" ({profile.get('sector', '?')} / {profile.get('industry', '?')})"
    lines.append(header)

    if profile.get("summary"):
        lines.append(profile["summary"])

    stats = []
    if profile.get("market_cap"):
        stats.append(f"Market cap: {_fmt_money(profile['market_cap'])}")
    if profile.get("employees"):
        stats.append(f"Employees: {profile['employees']:,}")
    if profile.get("pe_ratio"):
        stats.append(f"P/E: {profile['pe_ratio']:.1f}")
    if stats:
        lines.append("  ".join(stats))

    fin_line = []
    if financials.get("revenue_latest") is not None:
        piece = f"Revenue ({financials.get('revenue_period', 'latest FY')}): {_fmt_money(financials['revenue_latest'])}"
        if financials.get("revenue_yoy") is not None:
            piece += f" ({financials['revenue_yoy'] * 100:+.1f}% YoY)"
        fin_line.append(piece)
    if financials.get("net_income_latest") is not None:
        piece = f"Net income: {_fmt_money(financials['net_income_latest'])}"
        if financials.get("net_income_yoy") is not None:
            piece += f" ({financials['net_income_yoy'] * 100:+.1f}% YoY)"
        fin_line.append(piece)
    if profile.get("profit_margin") is not None:
        fin_line.append(f"Profit margin: {_fmt_pct(profile['profit_margin'])}")
    if fin_line:
        lines.append("  |  ".join(fin_line))

    return "\n".join(lines)
