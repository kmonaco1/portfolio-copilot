import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import io
from edgar import Company, set_identity

# SEC EDGAR identity initialization
set_identity("ResearchUser research@portfolio-copilot.internal")

st.set_page_config(page_title="Portfolio Copilot", layout="wide")

# Metric CSS to prevent text clipping and ellipsis wrapping
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.40rem !important;
    white-space: nowrap !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.80rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Portfolio Intelligence & Risk Copilot")
st.caption("Quantitative factor diagnostics, dynamic transaction replay, and SEC 10-K intelligence")

# ================= Helper: Positions CSV Parser =================
def parse_brokerage_csv(file_bytes):
    try:
        content = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        content = str(file_bytes)
    
    lines = content.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        line_clean = line.replace('"', '').replace("'", "")
        tokens = [t.strip().lower() for t in line_clean.split(",")]
        if any(tok in ["symbol", "ticker"] for tok in tokens):
            header_idx = idx
            break
            
    if header_idx is None:
        raise ValueError("Could not find a valid table header with 'Symbol' or 'Ticker'.")
        
    csv_stream = "\n".join(lines[header_idx:])
    raw_df = pd.read_csv(io.StringIO(csv_stream))
    
    cash_found = 0.0
    sym_col = [c for c in raw_df.columns if any(k in c.lower() for k in ['symbol', 'ticker'])][0]
    
    cash_mask = raw_df[sym_col].astype(str).str.contains('cash', case=False, na=False)
    if cash_mask.any():
        for val_col in ['Mkt Val (Market Value)', 'Current Value', 'Value', 'Market Value']:
            if val_col in raw_df.columns:
                try:
                    c_str = str(raw_df.loc[cash_mask, val_col].values[0]).replace('$', '').replace(',', '').strip()
                    cash_found = float(c_str)
                    break
                except Exception:
                    pass

    clean_df = raw_df[~raw_df[sym_col].astype(str).str.contains('total|cash|no number', case=False, na=False)].copy()
    qty_col = [c for c in clean_df.columns if any(k in c.lower() for k in ['qty', 'quantity', 'shares'])][0]
    
    clean_df['Ticker'] = clean_df[sym_col].astype(str).str.strip().str.upper()
    clean_df = clean_df[clean_df['Ticker'].str.match(r'^[A-Z0-9\.\-]+$', na=False)]
    
    clean_df['Shares'] = pd.to_numeric(clean_df[qty_col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    clean_df = clean_df.dropna(subset=['Shares'])
    clean_df = clean_df[clean_df['Shares'] > 0]

    asset_type_cols = [c for c in clean_df.columns if 'asset type' in c.lower()]
    if asset_type_cols:
        col = asset_type_cols[0]
        clean_df['Asset Class'] = clean_df[col].apply(lambda x: 'ETF' if 'etf' in str(x).lower() else 'Stock')
    else:
        clean_df['Asset Class'] = 'Stock'
    
    return clean_df[['Ticker', 'Shares', 'Asset Class']], cash_found

# ================= Helper: Schwab Transactions Ledger Parser =================
def parse_schwab_transactions(file_bytes):
    try:
        content = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        content = str(file_bytes)
        
    lines = content.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        line_clean = line.replace('"', '').replace("'", "")
        tokens = [t.strip().lower() for t in line_clean.split(",")]
        if any(tok in ["date", "activity"] for tok in tokens) and any(tok in ["action", "description"] for tok in tokens):
            header_idx = idx
            break
            
    if header_idx is None:
        raise ValueError("Could not find header row with 'Date' and 'Action' in transaction export.")
        
    csv_stream = "\n".join(lines[header_idx:])
    tx_df = pd.read_csv(io.StringIO(csv_stream))
    
    date_col = [c for c in tx_df.columns if 'date' in c.lower()][0]
    action_col = [c for c in tx_df.columns if 'action' in c.lower() or 'type' in c.lower()][0]
    sym_col = [c for c in tx_df.columns if 'symbol' in c.lower() or 'ticker' in c.lower()][0]
    qty_col = [c for c in tx_df.columns if 'quantity' in c.lower() or 'qty' in c.lower() or 'shares' in c.lower()][0]
    amt_col = [c for c in tx_df.columns if 'amount' in c.lower() or 'total' in c.lower()][0]
    
    raw_dates = tx_df[date_col].astype(str).str.split(" as of").str[0].str.strip()
    tx_df['ParsedDate'] = pd.to_datetime(raw_dates, errors='coerce').dt.tz_localize(None).dt.normalize()
    tx_df = tx_df.dropna(subset=['ParsedDate']).sort_values(by='ParsedDate', ascending=True)
    
    tx_df['CleanSymbol'] = tx_df[sym_col].astype(str).str.strip().str.upper()
    tx_df = tx_df[tx_df['CleanSymbol'].str.match(r'^[A-Z]{1,5}$', na=False) | (tx_df['CleanSymbol'] == '')].copy()
    
    tx_df['CleanQty'] = pd.to_numeric(tx_df[qty_col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
    tx_df['CleanAmount'] = pd.to_numeric(tx_df[amt_col].astype(str).str.replace('$', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
    tx_df['CleanAction'] = tx_df[action_col].astype(str).str.upper()
    
    net_cash_flow = float(tx_df['CleanAmount'].sum())
    calculated_cash = max(0.0, round(net_cash_flow, 2))
    
    holdings_agg = {}
    for _, r in tx_df.iterrows():
        act = r['CleanAction']
        sym = r['CleanSymbol']
        q = r['CleanQty']
        if sym and sym not in ['--', 'NAN', '']:
            if 'BUY' in act or 'REINVEST' in act:
                holdings_agg[sym] = holdings_agg.get(sym, 0.0) + abs(q)
            elif 'SELL' in act:
                holdings_agg[sym] = max(0.0, holdings_agg.get(sym, 0.0) - abs(q))

    reconstructed_df = pd.DataFrame([
        {
            "Ticker": k, 
            "Shares": round(v, 4), 
            "Asset Class": "ETF" if k in ["VOO", "QQQ", "IUSG", "IUSV", "SCHD", "NUKZ", "VEA", "VTWO", "VWO"] else "Stock"
        } 
        for k, v in holdings_agg.items() if v > 0.001
    ])
    
    return reconstructed_df, calculated_cash, tx_df

# ================= Session State Initialization =================
if "cached_holdings_df" not in st.session_state:
    st.session_state.cached_holdings_df = None
if "cached_tx_df" not in st.session_state:
    st.session_state.cached_tx_df = None
if "cached_cash" not in st.session_state:
    st.session_state.cached_cash = 3.47
if "cached_source" not in st.session_state:
    st.session_state.cached_source = "None"

# ================= Sidebar: Ingestion =================
st.sidebar.header("📥 Portfolio Ingestion")

input_mode = st.sidebar.radio(
    "Choose Input Method:",
    [
        "Upload Schwab Transaction History CSV (Accurate Equity Curve)",
        "Upload Schwab Positions CSV", 
        "Manual Holdings Editor"
    ]
)

initial_holdings = pd.DataFrame([
    {"Ticker": "VOO",  "Shares": 5.39, "Asset Class": "ETF"},
    {"Ticker": "QQQ",  "Shares": 2.14, "Asset Class": "ETF"},
    {"Ticker": "NVDA", "Shares": 7.40, "Asset Class": "Stock"},
    {"Ticker": "AMZN", "Shares": 5.26, "Asset Class": "Stock"},
    {"Ticker": "AAPL", "Shares": 2.01, "Asset Class": "Stock"},
    {"Ticker": "TSLA", "Shares": 2.53, "Asset Class": "Stock"},
    {"Ticker": "MU",   "Shares": 1.19, "Asset Class": "Stock"},
    {"Ticker": "SCHD", "Shares": 16.72, "Asset Class": "ETF"},
])

has_tx_ledger = False
tx_dataframe = None

if input_mode == "Upload Schwab Transaction History CSV (Accurate Equity Curve)":
    uploaded_tx = st.sidebar.file_uploader("Upload Transactions CSV (from Accounts > History):", type=["csv"], key="tx_uploader")
    if uploaded_tx is not None:
        try:
            parsed_holdings, auto_cash, parsed_tx = parse_schwab_transactions(uploaded_tx.getvalue())
            st.session_state.cached_holdings_df = parsed_holdings
            st.session_state.cached_tx_df = parsed_tx
            st.session_state.cached_cash = auto_cash
            st.session_state.cached_source = "Transaction History"
            st.sidebar.success(f"✅ Ingested {len(parsed_tx)} transactions! Cached in session.")
        except Exception as e:
            st.sidebar.error(f"Error parsing transaction file: {str(e)}")

    if st.session_state.cached_source == "Transaction History" and st.session_state.cached_holdings_df is not None:
        raw_df = st.session_state.cached_holdings_df
        tx_dataframe = st.session_state.cached_tx_df
        has_tx_ledger = True
        detected_cash = st.session_state.cached_cash
        st.sidebar.caption(f"💾 Active cached ledger: {len(tx_dataframe)} records ({len(raw_df)} holdings)")
    else:
        st.sidebar.info("Upload your Charles Schwab transaction export from Accounts > History.")
        raw_df = initial_holdings
        detected_cash = 3.47

elif input_mode == "Upload Schwab Positions CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Positions CSV:", type=["csv"], key="pos_uploader")
    if uploaded_file is not None:
        try:
            parsed_holdings, auto_cash = parse_brokerage_csv(uploaded_file.getvalue())
            st.session_state.cached_holdings_df = parsed_holdings
            st.session_state.cached_tx_df = None
            st.session_state.cached_cash = auto_cash
            st.session_state.cached_source = "Positions Export"
            st.sidebar.success(f"✅ Loaded {len(parsed_holdings)} positions from CSV!")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {str(e)}")

    if st.session_state.cached_source == "Positions Export" and st.session_state.cached_holdings_df is not None:
        raw_df = st.session_state.cached_holdings_df
        detected_cash = st.session_state.cached_cash
        st.sidebar.caption(f"💾 Active cached positions: {len(raw_df)} holdings")
    else:
        raw_df = initial_holdings
        detected_cash = 3.47

else:
    st.sidebar.markdown("### Enter Holdings & Shares")
    edited_df = st.sidebar.data_editor(
        initial_holdings,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor"
    )
    raw_df = edited_df.dropna()
    detected_cash = 3.47

# Clear Cache Option
if st.session_state.cached_holdings_df is not None:
    if st.sidebar.button("🗑️ Clear Cached Ledger Data"):
        st.session_state.cached_holdings_df = None
        st.session_state.cached_tx_df = None
        st.session_state.cached_cash = 3.47
        st.session_state.cached_source = "None"
        st.rerun()

cash_balance = st.sidebar.number_input(
    "Uninvested Cash Balance ($):",
    min_value=0.0,
    value=float(detected_cash),
    step=50.0
)

raw_df["Ticker"] = raw_df["Ticker"].astype(str).str.strip().str.upper()
raw_df = raw_df[raw_df["Shares"] > 0]
holdings_dict = dict(zip(raw_df["Ticker"], raw_df["Shares"]))

if 'Asset Class' in raw_df.columns:
    asset_class_dict = dict(zip(raw_df["Ticker"], raw_df["Asset Class"]))
else:
    asset_class_dict = {t: 'Stock' for t in raw_df["Ticker"]}

raw_tickers = list(holdings_dict.keys())
if not raw_tickers:
    st.warning("Please enter at least one holding with shares > 0.")
    st.stop()

benchmark = 'SPY'
all_symbols = list(set(raw_tickers + [benchmark]))

# Timeframe Selector
timeframe = st.radio(
    "Select Performance Window:",
    options=["1M", "3M", "6M", "YTD", "1Y", "2Y"],
    index=4,
    horizontal=True
)

tf_map = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "2Y": "2y"
}

# ================= Engine Calculation =================
with st.spinner("Downloading market quotes and computing true equity curve..."):
    data_full = yf.download(all_symbols, period="2y", auto_adjust=True)['Close']
    
    if hasattr(data_full.index, 'tz') and data_full.index.tz is not None:
        data_full.index = data_full.index.tz_convert(None).normalize()
    else:
        data_full.index = pd.to_datetime(data_full.index).normalize()

    valid_market_tickers = [
        t for t in raw_tickers 
        if t in data_full.columns and not data_full[t].dropna().empty
    ]
    
    holding_details = []
    equity_value = 0.0
    sector_weights = {}

    for ticker, shares in holdings_dict.items():
        ticker_obj = yf.Ticker(ticker)
        try:
            info = ticker_obj.info
            sector = info.get('sector') or info.get('category') or 'Diversified / Index'
            name = info.get('shortName', ticker)
            pe_ratio = info.get('forwardPE', 'N/A')
            quote_type = info.get('quoteType', '')
            detected_type = asset_class_dict.get(ticker, 'ETF' if quote_type == 'ETF' else 'Stock')
        except Exception:
            sector = 'Diversified / Index'
            name = ticker
            pe_ratio = 'N/A'
            detected_type = asset_class_dict.get(ticker, 'Stock')

        if ticker in valid_market_tickers:
            series = data_full[ticker].dropna()
            latest_price = float(series.iloc[-1])
            prev_price = float(series.iloc[-2]) if len(series) >= 2 else latest_price
            stock_day_change = ((latest_price - prev_price) / prev_price) * 100
        else:
            latest_price = 0.0
            stock_day_change = 0.0

        position_value = latest_price * shares
        equity_value += position_value

        holding_details.append({
            'Ticker': ticker,
            'Company': name,
            'Asset Class': detected_type,
            'Price (raw)': latest_price,
            'Price': f"${latest_price:,.2f}",
            'Day Change (%)': stock_day_change,
            'Day Change': f"{stock_day_change:+.2f}%",
            'Shares': shares,
            'Position Value (raw)': position_value,
            'Position Value': f"${position_value:,.2f}",
            'Sector': sector,
            'Forward P/E': pe_ratio
        })

    total_portfolio_value = equity_value + cash_balance
    cash_weight = (cash_balance / total_portfolio_value) * 100.0 if total_portfolio_value > 0 else 0.0

    for h in holding_details:
        raw_val = h['Position Value (raw)']
        weight_val = (raw_val / total_portfolio_value) * 100.0 if total_portfolio_value > 0 else 0.0
        h['Weight (raw)'] = weight_val
        h['Weight (%)'] = f"{weight_val:.1f}%"
        sec = h['Sector']
        sector_weights[sec] = sector_weights.get(sec, 0.0) + weight_val

    if cash_balance > 0:
        sector_weights['Cash'] = cash_weight

    df_holdings = pd.DataFrame(holding_details)

    # Reconstruct Historical Timeline
    price_matrix = data_full[valid_market_tickers].ffill().bfill()
    
    if has_tx_ledger and tx_dataframe is not None and not price_matrix.empty:
        dates = price_matrix.index
        daily_vals = []
        
        sorted_tx = tx_dataframe.sort_values(by='ParsedDate', ascending=True)
        cumulative_shares = {sym: 0.0 for sym in valid_market_tickers}
        tx_idx = 0
        total_tx = len(sorted_tx)
        
        for d in dates:
            while tx_idx < total_tx and sorted_tx.iloc[tx_idx]['ParsedDate'] <= d:
                r = sorted_tx.iloc[tx_idx]
                act = r['CleanAction']
                sym = r['CleanSymbol']
                q = r['CleanQty']
                
                if sym in cumulative_shares:
                    if 'BUY' in act or 'REINVEST' in act:
                        cumulative_shares[sym] += abs(q)
                    elif 'SELL' in act:
                        cumulative_shares[sym] = max(0.0, cumulative_shares[sym] - abs(q))
                tx_idx += 1
            
            day_equity = sum(cumulative_shares[sym] * price_matrix.loc[d, sym] for sym in valid_market_tickers)
            daily_vals.append(day_equity + cash_balance)
            
        full_equity_series = pd.Series(daily_vals, index=dates)
    else:
        shares_vector = pd.Series({t: holdings_dict[t] for t in valid_market_tickers})
        full_equity_series = (price_matrix * shares_vector).sum(axis=1) + cash_balance

    now_date = full_equity_series.index[-1]
    if timeframe == "1M":
        start_cut = now_date - pd.DateOffset(months=1)
    elif timeframe == "3M":
        start_cut = now_date - pd.DateOffset(months=3)
    elif timeframe == "6M":
        start_cut = now_date - pd.DateOffset(months=6)
    elif timeframe == "YTD":
        start_cut = pd.Timestamp(year=now_date.year, month=1, day=1)
    elif timeframe == "1Y":
        start_cut = now_date - pd.DateOffset(years=1)
    else:
        start_cut = now_date - pd.DateOffset(years=2)

    daily_equity_series = full_equity_series.loc[full_equity_series.index >= start_cut]
    if daily_equity_series.empty:
        daily_equity_series = full_equity_series

    df_chart = pd.DataFrame({'Date': daily_equity_series.index, 'Total Value': daily_equity_series.values})

    # Header Metrics
    current_val = total_portfolio_value
    prev_val = daily_equity_series.iloc[-2] if len(daily_equity_series) >= 2 else current_val
    day_dollar_change = current_val - prev_val
    day_pct_change = (day_dollar_change / prev_val) * 100.0 if prev_val > 0 else 0.0

    start_val = daily_equity_series.iloc[0] if not daily_equity_series.empty else current_val
    total_dollar_change = current_val - start_val
    total_pct_change = (total_dollar_change / start_val) * 100.0 if start_val > 0 else 0.0

    # Risk Calculations (1Y Window)
    cutoff_1y = now_date - pd.DateOffset(years=1)
    data_1y = data_full.loc[data_full.index >= cutoff_1y]
    valid_1y_tickers = [t for t in valid_market_tickers if t in data_1y.columns and not data_1y[t].dropna().empty]
    
    if valid_1y_tickers and benchmark in data_1y.columns:
        returns_1y = data_1y[valid_1y_tickers].pct_change().dropna()
        bench_returns_1y = data_1y[benchmark].pct_change().dropna()

        sub_weights_raw = [holdings_dict[t] * float(data_1y[t].dropna().iloc[-1]) for t in valid_1y_tickers]
        sub_weights = np.array(sub_weights_raw)
        
        if sub_weights.sum() > 0 and not returns_1y.empty:
            sub_weights = sub_weights / sub_weights.sum()
            port_returns_1y = returns_1y.dot(sub_weights)
            aligned_bench = bench_returns_1y.loc[port_returns_1y.index]
            cov_val = np.cov(port_returns_1y, aligned_bench)[0][1]
            portfolio_beta = cov_val / np.var(aligned_bench)
            ann_return = port_returns_1y.mean() * 252
            ann_volatility = port_returns_1y.std() * np.sqrt(252)
            risk_free_rate = 0.045
            sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0.0
            corr_matrix = returns_1y.corr()
        else:
            portfolio_beta, ann_return, ann_volatility, sharpe_ratio = 1.0, 0.0, 0.0, 0.0
            corr_matrix = pd.DataFrame()
    else:
        portfolio_beta, ann_return, ann_volatility, sharpe_ratio = 1.0, 0.0, 0.0, 0.0
        corr_matrix = pd.DataFrame()

    all_weights = [h['Weight (raw)'] for h in holding_details] + ([cash_weight] if cash_balance > 0 else [])
    hhi = sum(w ** 2 for w in all_weights)

# ================= Look-Through Overlap Engine =================
KNOWN_ETF_HOLDINGS = {
    "VOO": {"AAPL": 0.069, "NVDA": 0.065, "MSFT": 0.062, "AMZN": 0.038, "META": 0.026, "GOOGL": 0.020, "TSLA": 0.016, "CAT": 0.007, "KO": 0.005, "WMT": 0.007},
    "QQQ": {"AAPL": 0.088, "NVDA": 0.084, "MSFT": 0.079, "AMZN": 0.052, "META": 0.046, "TSLA": 0.029, "MU": 0.008, "COST": 0.025, "AMD": 0.018},
    "IUSG": {"AAPL": 0.114, "NVDA": 0.108, "MSFT": 0.098, "AMZN": 0.062, "META": 0.044, "TSLA": 0.028, "AMD": 0.015},
    "IUSV": {"JNJ": 0.024, "JPM": 0.032, "WMT": 0.018, "CAT": 0.015, "KO": 0.012, "FE": 0.004},
    "SCHD": {"KO": 0.041, "HD": 0.042, "CSCO": 0.038, "CAT": 0.012},
    "VTWO": {"RR": 0.001, "ASTS": 0.003},
    "VEA": {},
    "VWO": {}
}

etf_list = df_holdings[df_holdings['Asset Class'] == 'ETF']['Ticker'].tolist()
etf_holdings_map = {}

for etf in etf_list:
    if etf in KNOWN_ETF_HOLDINGS:
        etf_holdings_map[etf] = KNOWN_ETF_HOLDINGS[etf].copy()
    try:
        t = yf.Ticker(etf)
        funds_obj = getattr(t, 'funds_data', None)
        if funds_obj is not None:
            top = funds_obj.top_holdings
            if top is not None and not top.empty:
                pct_col = [c for c in top.columns if 'percent' in c.lower() or 'holding' in c.lower()][0]
                live_dict = dict(zip(top.index.str.upper(), top[pct_col].astype(float)))
                etf_holdings_map[etf] = {**etf_holdings_map.get(etf, {}), **live_dict}
    except Exception:
        pass

overlap_records = []
individual_stocks = df_holdings[df_holdings['Asset Class'] == 'Stock']

for _, stock in individual_stocks.iterrows():
    sym = stock['Ticker']
    company_name = stock['Company']
    direct_dollars = stock['Position Value (raw)']
    direct_weight = stock['Weight (raw)']
    
    indirect_dollars = 0.0
    overlapping_funds = []
    
    for _, etf_row in df_holdings[df_holdings['Asset Class'] == 'ETF'].iterrows():
        etf_ticker = etf_row['Ticker']
        etf_dollars = etf_row['Position Value (raw)']
        
        if etf_ticker in etf_holdings_map and sym in etf_holdings_map[etf_ticker]:
            fraction = etf_holdings_map[etf_ticker][sym]
            fund_contrib = etf_dollars * fraction
            if fund_contrib > 0.5:
                indirect_dollars += fund_contrib
                overlapping_funds.append(etf_ticker)

    effective_dollars = direct_dollars + indirect_dollars
    effective_weight = (effective_dollars / total_portfolio_value) * 100.0 if total_portfolio_value > 0 else 0.0
    understated_pct = effective_weight - direct_weight

    overlap_records.append({
        'Ticker': sym,
        'Company': company_name,
        'Sector': stock['Sector'],
        'Direct Value ($)': direct_dollars,
        'Indirect ETF Value ($)': indirect_dollars,
        'Effective True Value ($)': effective_dollars,
        'Direct Weight (%)': direct_weight,
        'Effective True Weight (%)': effective_weight,
        'Hidden Boost (+%)': understated_pct,
        'Overlapping ETFs': ", ".join(overlapping_funds) if overlapping_funds else "None"
    })

df_overlap = pd.DataFrame(overlap_records).sort_values(by='Effective True Weight (%)', ascending=False)

# ================= Header Performance Card =================
st.subheader("Total Value")
k1, k2, k3, k4 = st.columns(4)
k1.metric(label="Total Account Value", value=f"${total_portfolio_value:,.2f}")
k2.metric(label="Day Change", value=f"{day_dollar_change:+,.2f} ({day_pct_change:+.2f}%)", delta=f"{day_pct_change:+.2f}%")
k3.metric(label=f"{timeframe} Change", value=f"{total_dollar_change:+,.2f} ({total_pct_change:+.2f}%)", delta=f"{total_pct_change:+.2f}%")
k4.metric(label="Cash Reserve", value=f"${cash_balance:,.2f}", delta=f"{cash_weight:.1f}% Allocation", delta_color="off")

fig_perf = px.area(
    df_chart, 
    x='Date', 
    y='Total Value', 
    labels={'Total Value': 'Portfolio Value ($)'}
)
fig_perf.update_traces(line_color="#007aff", fillcolor="rgba(0, 122, 255, 0.1)")
fig_perf.update_layout(
    xaxis_title="",
    yaxis_title="",
    margin=dict(t=15, l=10, r=10, b=10),
    hovermode="x unified",
    yaxis=dict(tickprefix="$", tickformat=",.0f")
)
st.plotly_chart(fig_perf, use_container_width=True)

# ================= Holdings Breakdown Table Section =================
with st.expander(f"📋 View Holdings Breakdown ({len(df_holdings)} Positions)", expanded=True):
    col_filter, col_stats = st.columns([1, 2])
    with col_filter:
        view_filter = st.selectbox(
            "Filter Holdings View:",
            ["All Holdings (Categorized: Stocks & ETFs)", "All Holdings (Single Unified Table)", "Individual Stocks Only", "ETFs Only"]
        )

    def render_ranked_table(sub_df, title_label=None):
        if sub_df.empty:
            st.info("No holdings found for this category.")
            return
        
        if title_label:
            st.markdown(f"**{title_label}** ({len(sub_df)} positions • ${sub_df['Position Value (raw)'].sum():,.2f} total)")
            
        display_df = sub_df.sort_values(by='Position Value (raw)', ascending=False).copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df = display_df.set_index('Rank')
        
        cols_to_show = ['Ticker', 'Company', 'Price', 'Day Change', 'Shares', 'Position Value', 'Weight (%)', 'Sector']
        table_height = (len(display_df) + 1) * 36 + 5
        
        st.dataframe(
            display_df[cols_to_show],
            use_container_width=True,
            height=table_height
        )

    if view_filter == "All Holdings (Categorized: Stocks & ETFs)":
        stocks_df = df_holdings[df_holdings['Asset Class'] == 'Stock']
        etfs_df = df_holdings[df_holdings['Asset Class'] == 'ETF']
        render_ranked_table(stocks_df, "🏛️ Individual Stocks")
        st.write("")
        render_ranked_table(etfs_df, "🧺 Exchange-Traded Funds (ETFs)")
        
    elif view_filter == "All Holdings (Single Unified Table)":
        render_ranked_table(df_holdings)
        
    elif view_filter == "Individual Stocks Only":
        stocks_df = df_holdings[df_holdings['Asset Class'] == 'Stock']
        render_ranked_table(stocks_df, "🏛️ Individual Stocks")
        
    elif view_filter == "ETFs Only":
        etfs_df = df_holdings[df_holdings['Asset Class'] == 'ETF']
        render_ranked_table(etfs_df, "🧺 Exchange-Traded Funds (ETFs)")

st.divider()

# ================= 5-Tab Analytics Architecture =================
tab_overlap, tab_charts, tab_memo, tab_sec, tab_sim = st.tabs([
    "🪞 ETF Overlap & Hidden Exposure",
    "📊 Factor Visuals & Correlation", 
    "🧠 AI Executive Memo", 
    "📑 SEC 10-K Filings",
    "🧪 What-If Simulator"
])

# ----- TAB 1: ETF Overlap & Hidden Exposure -----
with tab_overlap:
    st.subheader("🪞 ETF Look-Through & Hidden Exposure Analysis")
    st.caption("Dissects ETF constituents to reveal how much of your single stocks you also hold through your funds.")

    top_overlap = df_overlap.iloc[0] if not df_overlap.empty else None
    if top_overlap is not None and top_overlap['Hidden Boost (+%)'] > 0:
        st.warning(
            f"⚠️ **Hidden Concentration Detected:** **{top_overlap['Ticker']}** is held directly ({top_overlap['Direct Weight (%)']:.1f}%) "
            f"and also held by **{top_overlap['Overlapping ETFs']}**. Due to hidden overlap, your true effective concentration in "
            f"**{top_overlap['Ticker']} is {top_overlap['Effective True Weight (%)']:.1f}%** (+{top_overlap['Hidden Boost (+%)']:.1f}% unhedged boost)."
        )

    st.markdown("### ⚠️ Effective True-Weight Concentration Diagnostics")
    
    if not df_overlap.empty:
        top_effective_holding = df_overlap.iloc[0]
        top_3_effective_weight = df_overlap.head(3)['Effective True Weight (%)'].sum()
        double_dipped_positions = df_overlap[df_overlap['Hidden Boost (+%)'] > 0.5]
        
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric(
            "Top Effective Single Asset",
            f"{top_effective_holding['Ticker']} ({top_effective_holding['Effective True Weight (%)']:.1f}%)",
            f"+{top_effective_holding['Hidden Boost (+%)']:.1f}% via ETFs"
        )
        col_d2.metric(
            "Top 3 True Holdings Weight",
            f"{top_3_effective_weight:.1f}%",
            f"{', '.join(df_overlap.head(3)['Ticker'].tolist())}"
        )
        col_d3.metric(
            "Overlapping Single Stocks",
            f"{len(double_dipped_positions)} Holdings",
            "Compound Exposure"
        )

        if top_3_effective_weight > 25.0:
            st.error(
                f"**High Headroom Risk:** Your top 3 positions ({', '.join(df_overlap.head(3)['Ticker'].tolist())}) control **{top_3_effective_weight:.1f}%** of your total wealth. "
                "Because broad-market ETFs (VOO/QQQ) are market-cap weighted, rallies or drawdowns in these three names dictate total portfolio variance."
            )
        else:
            st.success("**Concentration Profile:** Top 3 effective assets control under 25% of total capital.")

        fig_overlap_bar = px.bar(
            df_overlap.head(10),
            x='Ticker',
            y=['Direct Value ($)', 'Indirect ETF Value ($)'],
            title="Direct Stock Holdings vs. Nested ETF Value (Top 10 Holdings)",
            barmode='stack',
            color_discrete_map={'Direct Value ($)': '#007aff', 'Indirect ETF Value ($)': '#ff9500'},
            labels={'value': 'Total Exposure ($)', 'variable': 'Ownership Source'}
        )
        fig_overlap_bar.update_layout(hovermode="x unified", margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_overlap_bar, use_container_width=True)

        st.markdown("#### Detailed Look-Through Overlap Breakdown")
        df_overlap_display = df_overlap.copy()
        df_overlap_display['Rank'] = range(1, len(df_overlap_display) + 1)
        df_overlap_display = df_overlap_display.set_index('Rank')
        
        df_overlap_display['Direct Value ($)'] = df_overlap_display['Direct Value ($)'].apply(lambda x: f"${x:,.2f}")
        df_overlap_display['Indirect ETF Value ($)'] = df_overlap_display['Indirect ETF Value ($)'].apply(lambda x: f"${x:,.2f}")
        df_overlap_display['Effective True Value ($)'] = df_overlap_display['Effective True Value ($)'].apply(lambda x: f"${x:,.2f}")
        df_overlap_display['Direct Weight (%)'] = df_overlap_display['Direct Weight (%)'].apply(lambda x: f"{x:.1f}%")
        df_overlap_display['Effective True Weight (%)'] = df_overlap_display['Effective True Weight (%)'].apply(lambda x: f"{x:.1f}%")
        df_overlap_display['Hidden Boost (+%)'] = df_overlap_display['Hidden Boost (+%)'].apply(lambda x: f"+{x:.1f}%" if x > 0 else "0.0%")

        overlap_height = (len(df_overlap_display) + 1) * 36 + 5
        st.dataframe(
            df_overlap_display[['Ticker', 'Company', 'Direct Value ($)', 'Indirect ETF Value ($)', 'Effective True Value ($)', 'Direct Weight (%)', 'Effective True Weight (%)', 'Hidden Boost (+%)', 'Overlapping ETFs']],
            use_container_width=True,
            height=overlap_height
        )

# ----- TAB 2: Factor Visuals & Correlation -----
with tab_charts:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{ann_return * 100:.1f}%")
    col2.metric("Annualized Volatility", f"{ann_volatility * 100:.1f}%")
    col3.metric("Portfolio Beta (SPY)", f"{portfolio_beta:.2f}")
    col4.metric("Sharpe Ratio (4.5%)", f"{sharpe_ratio:.2f}")

    st.subheader("🗺️ Allocation Treemap (Sector & Holdings)")
    treemap_df = df_holdings.copy()
    if cash_balance > 0:
        cash_row = pd.DataFrame([{
            'Ticker': 'CASH',
            'Company': 'Cash Reserves',
            'Weight (raw)': cash_weight,
            'Sector': 'Cash'
        }])
        treemap_df = pd.concat([treemap_df, cash_row], ignore_index=True)

    fig_treemap = px.treemap(
        treemap_df,
        path=['Sector', 'Ticker'],
        values='Weight (raw)',
        color='Sector'
    )
    fig_treemap.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_treemap, use_container_width=True)

    if not corr_matrix.empty:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("🔥 Holding Correlation Heatmap")
            st.caption("Values near 1.0 (hot red) mean stocks move together in lockstep, eliminating true diversification.")
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_h2:
            st.subheader("⚖️ Concentration Diagnostic")
            st.metric("Concentration Score (HHI)", f"{hhi:.0f} / 10,000")
            if hhi > 2500:
                st.warning("⚠️ High Concentration: Heavily weighted in top assets.")
            elif hhi > 1500:
                st.info("ℹ️ Moderate Concentration: Monitor top position sizing.")
            else:
                st.success("✅ Balanced Allocation: Diversification risk is low.")

# ----- TAB 3: AI Executive Memo -----
with tab_memo:
    st.subheader("📑 Portfolio Executive Research Memo")
    top_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ("None", 0.0)
    
    st.markdown("### 1. Factor Vulnerability Assessment")
    if top_sector[1] > 40.0 and top_sector[0] != "Cash":
        st.error(f"**Structural Skew:** The portfolio has a **{top_sector[1]:.1f}%** concentration in **{top_sector[0]}**.")
    else:
        st.success(f"**Sector Balance:** Top exposure is **{top_sector[0]} ({top_sector[1]:.1f}%)**.")

    if cash_weight > 20.0:
        st.warning(f"**Cash Drag Alert:** You are holding **{cash_weight:.1f}%** in cash.")

    if portfolio_beta > 1.2:
        st.warning(f"**High Systematic Risk (Beta = {portfolio_beta:.2f}):** Amplifies market drawdowns.")
    else:
        st.info(f"**Moderate Beta Profile ({portfolio_beta:.2f}):** Volatility aligned with S&P 500.")

    st.markdown("### 2. Holdings Research & Catalyst Feed")
    top_holdings = df_holdings.sort_values(by='Position Value (raw)', ascending=False).head(8)
    for _, row in top_holdings.iterrows():
        t = row['Ticker']
        with st.expander(f"🔍 {t} — {row['Company']} ({row['Sector']}) | Position: {row['Position Value']}"):
            st.write(f"**Current Price:** {row['Price']} ({row['Day Change']}) | **Forward P/E:** {row['Forward P/E']}")
            try:
                news_items = yf.Ticker(t).news[:2]
                if news_items:
                    for n in news_items:
                        st.markdown(f"- *{n.get('title', 'Corporate Update')}* ({n.get('publisher', 'Financial Wire')})")
                else:
                    st.write("No major 8-K filings flagged in the last 24 hours.")
            except Exception:
                st.write("Catalyst feed updating.")

# ----- TAB 4: SEC 10-K Filings -----
with tab_sec:
    st.subheader("🔍 SEC EDGAR 10-K Diagnostic Feed")
    selected_ticker = st.selectbox("Select a holding to inspect SEC Risk Factors:", raw_tickers)

    if st.button(f"📥 Pull Latest 10-K for {selected_ticker}"):
        with st.spinner(f"Querying SEC EDGAR database for {selected_ticker}..."):
            try:
                c = Company(selected_ticker)
                filing = c.get_filings(form="10-K")[0]
                tenk = filing.obj()
                st.success(f"Successfully retrieved Form 10-K (Filed: {filing.filing_date}) for {selected_ticker}")
                risks = tenk.risk_factors
                if risks:
                    st.markdown("#### Item 1A: Disclosed Corporate Risk Factors (Excerpt)")
                    st.info(risks[:1200] + "\n\n*(Full text available in SEC report)*")
                else:
                    st.warning("Risk Factors section not found under standard Item 1A tag.")
            except Exception as e:
                st.error(f"Could not load SEC filing for {selected_ticker}: {str(e)}")

# ----- TAB 5: What-If Simulator -----
with tab_sim:
    st.subheader("🧪 Portfolio Stress Test & Rebalance Simulator")
    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        new_ticker = st.text_input("Candidate Ticker to Add:", value="COST").strip().upper()
        sim_alloc = st.slider(f"Target Weight for {new_ticker} (%):", min_value=5, max_value=40, value=15, step=5)
        run_sim = st.button("🚀 Run Portfolio Simulation")

    with sim_col2:
        if run_sim:
            with st.spinner(f"Re-running simulation with {new_ticker}..."):
                try:
                    sim_data = yf.download(new_ticker, period="1y", auto_adjust=True)['Close']
                    if sim_data.empty:
                        st.error(f"No market data found for ticker '{new_ticker}'.")
                    else:
                        sim_returns = sim_data.pct_change().dropna()
                        combined_returns = returns_1y.copy()
                        combined_returns[new_ticker] = sim_returns
                        combined_returns = combined_returns.dropna()

                        scale_factor = (100.0 - sim_alloc) / 100.0
                        sim_weights = {t: (w * scale_factor) for t, w in zip(valid_1y_tickers, sub_weights)}
                        sim_weights[new_ticker] = sim_alloc / 100.0

                        sim_tickers = list(sim_weights.keys())
                        sim_weight_vec = np.array([sim_weights[t] for t in sim_tickers])
                        sim_port_returns = combined_returns[sim_tickers].dot(sim_weight_vec)
                        sim_bench_returns = data_1y[benchmark].pct_change().dropna().loc[sim_port_returns.index]

                        sim_cov = np.cov(sim_port_returns, sim_bench_returns)[0][1]
                        sim_beta = sim_cov / np.var(sim_bench_returns)

                        sim_ann_ret = sim_port_returns.mean() * 252
                        sim_ann_vol = sim_port_returns.std() * np.sqrt(252)
                        sim_sharpe = (sim_ann_ret - risk_free_rate) / sim_ann_vol

                        st.markdown(f"#### 📊 Simulated Impact of Adding {sim_alloc}% {new_ticker}")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Simulated Return", f"{sim_ann_ret * 100:.1f}%", f"{(sim_ann_ret - ann_return) * 100:+.1f}%")
                        m2.metric("Simulated Volatility", f"{sim_ann_vol * 100:.1f}%", f"{(sim_ann_vol - ann_volatility) * 100:+.1f}%", delta_color="inverse")
                        m3.metric("Simulated Beta", f"{sim_beta:.2f}", f"{(sim_beta - portfolio_beta):+.2f}", delta_color="inverse")
                        m4.metric("Simulated Sharpe", f"{sim_sharpe:.2f}", f"{(sim_sharpe - sharpe_ratio):+.2f}")
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
        else:
            st.info("👈 Enter a candidate ticker on the left to simulate adding it to your live positions.")