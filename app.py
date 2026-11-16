import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Konfigurasi Halaman
st.set_page_config(page_title="SMC Pro Multi-Timeframe Dashboard", layout="wide")
st.title("Smart Money Concepts (SMC) — Analisa & Multi-Timeframe Backtest")
st.caption("Deteksi FVG Signifikan + Filter Tren EMA 200 + Eksekusi Pending Limit Order (Retest)")

# Sidebar Parameter
st.sidebar.header("Pengaturan Aset & Strategi")
ticker = st.sidebar.text_input("Simbol Ticker (Yahoo Finance)", value="GC=F", help="Contoh: BBCA.JK, EURUSD=X, GC=F, BTC-USD").strip()
period = st.sidebar.selectbox("Rentang Data Historis (Chart Utama)", options=["6mo", "1y", "2y"], index=0)
interval = st.sidebar.selectbox("Timeframe Chart Utama", options=["1h", "4h", "1d"], index=0)

st.sidebar.subheader("Filter Kualitas SMC")
use_ema_filter = st.sidebar.checkbox("Aktifkan Filter Tren (EMA 200)", value=True)
min_fvg_atr = st.sidebar.slider("Lebar Minimum FVG (x ATR)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
order_expiry = st.sidebar.slider("Batas Waktu Tunggu Retest (Candle)", min_value=3, max_value=20, value=8, step=1)

st.sidebar.subheader("Manajemen Risiko & Modal")
initial_capital = st.sidebar.number_input("Modal Awal ($ / Rp)", min_value=100.0, value=10000.0, step=500.0)
risk_per_trade = st.sidebar.slider("Risiko per Transaksi (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
rr_ratio = st.sidebar.slider("Risk-to-Reward Ratio (1 : R)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
fee_pct = st.sidebar.number_input("Biaya Transaksi / Slippage (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

if st.sidebar.button("Refresh / Hapus Cache Data"):
    st.cache_data.clear()
    st.rerun()

# Fungsi Ambil Data dengan Caching & Resample 4h Otomatis
@st.cache_data(ttl=600, show_spinner=False)
def load_data(symbol, per, inter):
    try:
        # Jika 4h diminta, unduh data 1h lalu resample ke 4h agar tidak ditolak Yahoo Finance
        fetch_interval = "1h" if inter == "4h" else inter
        df = yf.download(symbol, period=per, interval=fetch_interval, progress=False, threads=False)
        
        if df is None or df.empty:
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        if inter == "4h" and not df.empty:
            df = df.resample("4h").agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
        return df
    except Exception:
        return pd.DataFrame()

# Fungsi Logika SMC (FVG + EMA 200 + Retest Limit Order)
def run_smc_strategy(df, capital, risk_pct, rr, fee, ema_filter, min_gap_mult, expiry_bars):
    data = df.copy()
    
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()

    bull_gap = data['Low'] - data['High'].shift(2)
    bear_gap = data['Low'].shift(2) - data['High']

    raw_bull_fvg = (bull_gap > (data['ATR'] * min_gap_mult)) & (data['Close'].shift(1) > data['Open'].shift(1))
    raw_bear_fvg = (bear_gap > (data['ATR'] * min_gap_mult)) & (data['Close'].shift(1) < data['Open'].shift(1))

    if ema_filter:
        data['Bull_FVG'] = raw_bull_fvg & (data['Close'] > data['EMA_200'])
        data['Bear_FVG'] = raw_bear_fvg & (data['Close'] < data['EMA_200'])
    else:
        data['Bull_FVG'] = raw_bull_fvg
        data['Bear_FVG'] = raw_bear_fvg

    trades = []
    equity = capital
    equity_curve = [capital] * len(data)
    
    active_trade = None
    pending_order = None
    latest_setup = None

    start_idx = 200 if ema_filter else 15

    for i in range(start_idx, len(data)):
        row = data.iloc[i]
        idx = data.index[i]

        # 1. Cek Posisi Aktif (SL / TP)
        if active_trade is not None:
            if active_trade['type'] == 'LONG':
                if row['Low'] <= active_trade['sl']:
                    pnl = -active_trade['risk_amount'] - active_trade['fee_cost']
                    equity += pnl
                    trades.append({**active_trade, 'exit_date': idx, 'exit_price': active_trade['sl'], 'result': 'LOSS', 'pnl': pnl})
                    active_trade = None
                elif row['High'] >= active_trade['tp']:
                    pnl = (active_trade['risk_amount'] * rr) - active_trade['fee_cost']
                    equity += pnl
                    trades.append({**active_trade, 'exit_date': idx, 'exit_price': active_trade['tp'], 'result': 'WIN', 'pnl': pnl})
                    active_trade = None
            elif active_trade['type'] == 'SHORT':
                if row['High'] >= active_trade['sl']:
                    pnl = -active_trade['risk_amount'] - active_trade['fee_cost']
                    equity += pnl
                    trades.append({**active_trade, 'exit_date': idx, 'exit_price': active_trade['sl'], 'result': 'LOSS', 'pnl': pnl})
                    active_trade = None
                elif row['Low'] <= active_trade['tp']:
                    pnl = (active_trade['risk_amount'] * rr) - active_trade['fee_cost']
                    equity += pnl
                    trades.append({**active_trade, 'exit_date': idx, 'exit_price': active_trade['tp'], 'result': 'WIN', 'pnl': pnl})
                    active_trade = None

        # 2. Cek Pending Limit Order (Menunggu Harga Retest Area FVG)
        if active_trade is None and pending_order is not None:
            bars_waited = i - pending_order['created_bar']
            
            if bars_waited > expiry_bars:
                pending_order = None
            elif pending_order['type'] == 'LONG':
                if row['Low'] <= pending_order['sl']:
                    pending_order = None
                elif row['Low'] <= pending_order['entry']:
                    active_trade = pending_order.copy()
                    active_trade['entry_date'] = idx
                    pending_order = None
            elif pending_order['type'] == 'SHORT':
                if row['High'] >= pending_order['sl']:
                    pending_order = None
                elif row['High'] >= pending_order['entry']:
                    active_trade = pending_order.copy()
                    active_trade['entry_date'] = idx
                    pending_order = None

        # 3. Buat Pending Order Baru jika Muncul FVG Valid
        if active_trade is None and pending_order is None and not np.isnan(row['ATR']):
            if row['Bull_FVG']:
                entry_price = row['Low']
                sl_price = data['Low'].iloc[i-2] - (0.2 * row['ATR'])
                risk_dist = entry_price - sl_price
                
                if risk_dist > 0:
                    tp_price = entry_price + (risk_dist * rr)
                    risk_amt = equity * (risk_pct / 100.0)
                    pos_size = risk_amt / risk_dist
                    fee_cost = (pos_size * entry_price) * (fee / 100.0) * 2
                    
                    pending_order = {
                        'type': 'LONG', 'created_bar': i, 'entry_date': idx,
                        'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
                        'risk_amount': risk_amt, 'fee_cost': fee_cost
                    }
                    latest_setup = pending_order.copy()

            elif row['Bear_FVG']:
                entry_price = row['High']
                sl_price = data['High'].iloc[i-2] + (0.2 * row['ATR'])
                risk_dist = sl_price - entry_price
                
                if risk_dist > 0:
                    tp_price = entry_price - (risk_dist * rr)
                    risk_amt = equity * (risk_pct / 100.0)
                    pos_size = risk_amt / risk_dist
                    fee_cost = (pos_size * entry_price) * (fee / 100.0) * 2
                    
                    pending_order = {
                        'type': 'SHORT', 'created_bar': i, 'entry_date': idx,
                        'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
                        'risk_amount': risk_amt, 'fee_cost': fee_cost
                    }
                    latest_setup = pending_order.copy()

        equity_curve[i] = equity

    data['Equity'] = equity_curve
    return data, pd.DataFrame(trades), latest_setup

def compute_metrics(df_res, trades_df, init_cap):
    final_eq = df_res['Equity'].iloc[-1]
    net_ret = ((final_eq - init_cap) / init_cap) * 100
    tot_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['result'] == 'WIN']) / tot_trades * 100) if tot_trades > 0 else 0.0
    
    rolling_max = df_res['Equity'].cummax()
    max_dd = ((df_res['Equity'] - rolling_max) / rolling_max * 100).min()
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if not trades_df.empty else 0.0
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if not trades_df.empty else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    return {
        "final_equity": final_eq,
        "net_return": net_ret,
        "total_trades": tot_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_dd": max_dd
    }

# Eksekusi Utama dengan Spinner Status
with st.spinner(f"Mengambil data {ticker} dari Yahoo Finance..."):
    df_raw = load_data(ticker, period, interval)

if df_raw.empty:
    st.warning(f"Data untuk simbol **{ticker}** tidak tersedia atau Yahoo Finance sedang membatasi koneksi sementara. Coba ganti ticker (misal: `BTC-USD`, `EURUSD=X`, `BBCA.JK`) atau klik **Refresh / Hapus Cache Data** di sidebar kiri.")
elif len(df_raw) < 205 and use_ema_filter:
    st.warning(f"Jumlah candle ({len(df_raw)} bar) kurang dari 205 untuk menghitung EMA 200. Perbesar **Rentang Data Historis** menjadi `1y` atau `2y`, atau matikan centang **Filter Tren (EMA 200)**.")
else:
    df_res, trades_df, last_setup = run_smc_strategy(
        df_raw, initial_capital, risk_per_trade, rr_ratio, fee_pct,
        use_ema_filter, min_fvg_atr, order_expiry
    )
    metrics = compute_metrics(df_res, trades_df, initial_capital)

    # Kartu Metrik Backtest Utama
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ekuitas Akhir", f"{metrics['final_equity']:,.2f}", f"{metrics['net_return']:+.2f}%")
    col2.metric("Total Eksekusi", f"{metrics['total_trades']}x")
    col3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    col4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    col5.metric("Max Drawdown", f"{metrics['max_dd']:.2f}%")

    # Multi-Timeframe Comparison
    st.markdown("---")
    st.subheader(f"Perbandingan Performa Multi-Timeframe ({ticker})")

    mtf_config = [
        {"tf": "1h", "per": "6mo", "label": "1 Jam (6 Bulan)"},
        {"tf": "4h", "per": "1y",  "label": "4 Jam (1 Tahun)"},
        {"tf": "1d", "per": "2y",  "label": "1 Hari / Daily (2 Tahun)"}
    ]

    mtf_rows = []
    with st.spinner("Menghitung simulasi Multi-Timeframe (1h, 4h, 1d)..."):
        for cfg in mtf_config:
            df_mtf = load_data(ticker, cfg["per"], cfg["tf"])
            min_bars = 205 if use_ema_filter else 20
            if not df_mtf.empty and len(df_mtf) >= min_bars:
                res_mtf, tr_mtf, _ = run_smc_strategy(
                    df_mtf, initial_capital, risk_per_trade, rr_ratio, fee_pct,
                    use_ema_filter, min_fvg_atr, order_expiry
                )
                m = compute_metrics(res_mtf, tr_mtf, initial_capital)
                mtf_rows.append({
                    "Timeframe": cfg["label"],
                    "Total Eksekusi": f"{m['total_trades']}x",
                    "Win Rate (%)": round(m['win_rate'], 1),
                    "Profit Factor": round(m['profit_factor'], 2),
                    "Net Return (%)": round(m['net_return'], 2),
                    "Max Drawdown (%)": round(m['max_dd'], 2),
                    "Ekuitas Akhir": round(m['final_equity'], 2)
                })
            else:
                mtf_rows.append({
                    "Timeframe": cfg["label"],
                    "Total Eksekusi": "Data Kurang",
                    "Win Rate (%)": 0.0,
                    "Profit Factor": 0.0,
                    "Net Return (%)": 0.0,
                    "Max Drawdown (%)": 0.0,
                    "Ekuitas Akhir": initial_capital
                })

    mtf_df = pd.DataFrame(mtf_rows)

    col_mtf_tbl, col_mtf_bar = st.columns([1.4, 1])
    with col_mtf_tbl:
        st.dataframe(
            mtf_df.style.highlight_max(subset=["Win Rate (%)", "Profit Factor", "Net Return (%)"], color="#1b5e20"),
            use_container_width=True,
            hide_index=True
        )
    with col_mtf_bar:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=[r["Timeframe"].split(" ")[0] + r["Timeframe"].split(" ")[1] for r in mtf_rows],
                y=[r["Win Rate (%)"] for r in mtf_rows],
                text=[f"{r['Win Rate (%)']}%" for r in mtf_rows],
                textposition='auto',
                marker_color=['#2979FF', '#00E676', '#FFA726']
            )
        ])
        fig_bar.update_layout(
            title="Perbandingan Win Rate per Timeframe",
            yaxis_title="Win Rate (%)",
            height=220,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Grafik Candlestick + EMA 200 + Setup SMC Terakhir
    st.subheader(f"Grafik Harga, EMA 200 & Setup SMC Terakhir ({ticker} - {interval})")
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_res.index, open=df_res['Open'], high=df_res['High'],
        low=df_res['Low'], close=df_res['Close'], name="Harga"
    ))

    if use_ema_filter:
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['EMA_200'], mode='lines',
            line=dict(color='#FFA726', width=1.8), name='EMA 200 (Tren)'
        ))

    bull_signals = df_res[df_res['Bull_FVG']]
    bear_signals = df_res[df_res['Bear_FVG']]
    fig.add_trace(go.Scatter(
        x=bull_signals.index, y=bull_signals['Low'] * 0.998, mode='markers',
        marker=dict(symbol='triangle-up', size=9, color='#00C853'), name='Bullish FVG Valid'
    ))
    fig.add_trace(go.Scatter(
        x=bear_signals.index, y=bear_signals['High'] * 1.002, mode='markers',
        marker=dict(symbol='triangle-down', size=9, color='#FF1744'), name='Bearish FVG Valid'
    ))

    if last_setup is not None:
        fig.add_hline(y=last_setup['entry'], line_dash="solid", line_color="#2979FF",
                      annotation_text=f"LIMIT ENTRY ({last_setup['type']}): {last_setup['entry']:,.2f}", annotation_position="top left")
        fig.add_hline(y=last_setup['sl'], line_dash="dash", line_color="#FF1744",
                      annotation_text=f"STOP LOSS: {last_setup['sl']:,.2f}", annotation_position="bottom left")
        fig.add_hline(y=last_setup['tp'], line_dash="dash", line_color="#00C853",
                      annotation_text=f"TAKE PROFIT: {last_setup['tp']:,.2f}", annotation_position="top left")

    fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Grafik Kurva Ekuitas & Tabel Riwayat
    col_eq, col_tbl = st.columns([1.2, 1])
    with col_eq:
        st.subheader(f"Kurva Pertumbuhan Modal ({interval})")
        fig_eq = go.Figure(go.Scatter(x=df_res.index, y=df_res['Equity'], mode='lines', line=dict(color='#00E676', width=2), name='Ekuitas'))
        fig_eq.update_layout(height=320, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_tbl:
        st.subheader(f"Riwayat Transaksi Backtest ({interval})")
        if not trades_df.empty:
            display_df = trades_df[['type', 'entry_date', 'entry', 'sl', 'tp', 'result', 'pnl']].copy()
            st.dataframe(display_df.tail(15), use_container_width=True, height=320)
        else:
            st.info("Belum ada transaksi yang tereksekusi pada rentang waktu dan filter ini.")
