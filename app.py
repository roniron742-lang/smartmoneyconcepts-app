import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Konfigurasi Halaman
st.set_page_config(page_title="SMC Trading & Backtest Dashboard", layout="wide")
st.title("Smart Money Concepts (SMC) — Analisa & Backtest")
st.caption("Deteksi Fair Value Gap (FVG), Level Entry/SL/TP Otomatis, dan Simulasi Ekuitas")

# Sidebar Parameter
st.sidebar.header("Pengaturan Aset & Strategi")
ticker = st.sidebar.text_input("Simbol Ticker (Yahoo Finance)", value="BTC-USD", help="Contoh: BBCA.JK, EURUSD=X, GC=F, BTC-USD")
period = st.sidebar.selectbox("Rentang Data Historis", options=["3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Timeframe", options=["1h", "4h", "1d"], index=2)

st.sidebar.subheader("Manajemen Risiko & Modal")
initial_capital = st.sidebar.number_input("Modal Awal ($ / Rp)", min_value=100.0, value=10000.0, step=500.0)
risk_per_trade = st.sidebar.slider("Risiko per Transaksi (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
rr_ratio = st.sidebar.slider("Risk-to-Reward Ratio (1 : R)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
fee_pct = st.sidebar.number_input("Biaya Transaksi / Slippage (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Fungsi Ambil Data dengan Caching
@st.cache_data(ttl=300)
def load_data(symbol, per, inter):
    df = yf.download(symbol, period=per, interval=inter, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# Fungsi Logika SMC (FVG & Sinyal Entry/SL/TP)
def run_smc_strategy(df, capital, risk_pct, rr, fee):
    data = df.copy()
    
    # Identifikasi Bullish & Bearish Fair Value Gap (FVG)
    data['Bull_FVG'] = (data['Low'] > data['High'].shift(2)) & (data['Close'].shift(1) > data['Open'].shift(1))
    data['Bear_FVG'] = (data['High'] < data['Low'].shift(2)) & (data['Close'].shift(1) < data['Open'].shift(1))
    
    # ATR untuk penyangga Stop Loss
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()

    trades = []
    equity = capital
    equity_curve = [capital] * len(data)
    active_trade = None
    latest_setup = None

    for i in range(15, len(data)):
        row = data.iloc[i]
        idx = data.index[i]

        # Cek apakah posisi aktif menyentuh SL atau TP
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

        # Buka sinyal baru jika tidak ada posisi aktif
        if active_trade is None and not np.isnan(row['ATR']):
            if row['Bull_FVG']:
                entry = row['Close']
                sl = data['Low'].iloc[i-2:i+1].min() - (0.2 * row['ATR'])
                risk_dist = entry - sl
                if risk_dist > 0:
                    tp = entry + (risk_dist * rr)
                    risk_amt = equity * (risk_pct / 100.0)
                    pos_size = risk_amt / risk_dist
                    fee_cost = (pos_size * entry) * (fee / 100.0) * 2
                    active_trade = {
                        'type': 'LONG', 'entry_date': idx, 'entry': entry,
                        'sl': sl, 'tp': tp, 'risk_amount': risk_amt, 'fee_cost': fee_cost
                    }
                    latest_setup = active_trade.copy()

            elif row['Bear_FVG']:
                entry = row['Close']
                sl = data['High'].iloc[i-2:i+1].max() + (0.2 * row['ATR'])
                risk_dist = sl - entry
                if risk_dist > 0:
                    tp = entry - (risk_dist * rr)
                    risk_amt = equity * (risk_pct / 100.0)
                    pos_size = risk_amt / risk_dist
                    fee_cost = (pos_size * entry) * (fee / 100.0) * 2
                    active_trade = {
                        'type': 'SHORT', 'entry_date': idx, 'entry': entry,
                        'sl': sl, 'tp': tp, 'risk_amount': risk_amt, 'fee_cost': fee_cost
                    }
                    latest_setup = active_trade.copy()

        equity_curve[i] = equity

    data['Equity'] = equity_curve
    return data, pd.DataFrame(trades), latest_setup

# Eksekusi Utama
df_raw = load_data(ticker, period, interval)

if df_raw.empty:
    st.error("Data tidak ditemukan. Periksa kembali simbol ticker Anda (contoh: BBCA.JK atau BTC-USD).")
else:
    df_res, trades_df, last_setup = run_smc_strategy(df_raw, initial_capital, risk_per_trade, rr_ratio, fee_pct)

    # Kartu Metrik Backtest
    final_equity = df_res['Equity'].iloc[-1]
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    total_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['result'] == 'WIN']) / total_trades * 100) if total_trades > 0 else 0.0
    
    # Hitung Maximum Drawdown
    rolling_max = df_res['Equity'].cummax()
    drawdown = (df_res['Equity'] - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ekuitas Akhir", f"{final_equity:,.2f}", f"{net_return_pct:+.2f}%")
    col2.metric("Total Transaksi", f"{total_trades}x")
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Max Drawdown", f"{max_dd:.2f}%")

    # Grafik Candlestick + Garis Entry, SL, TP Terakhir
    st.subheader(f"Grafik Harga & Setup SMC Terakhir ({ticker})")
    fig = go.Figure(data=[go.Candlestick(
        x=df_res.index,
        open=df_res['Open'],
        high=df_res['High'],
        low=df_res['Low'],
        close=df_res['Close'],
        name="Harga"
    )])

    # Tandai Sinyal FVG
    bull_signals = df_res[df_res['Bull_FVG']]
    bear_signals = df_res[df_res['Bear_FVG']]
    fig.add_trace(go.Scatter(x=bull_signals.index, y=bull_signals['Low'] * 0.995, mode='markers',
                             marker=dict(symbol='triangle-up', size=9, color='#00C853'), name='Bullish FVG'))
    fig.add_trace(go.Scatter(x=bear_signals.index, y=bear_signals['High'] * 1.005, mode='markers',
                             marker=dict(symbol='triangle-down', size=9, color='#FF1744'), name='Bearish FVG'))

    # Tambahkan Garis Horizontal Otomatis untuk Entry, SL, dan TP dari Setup Terakhir
    if last_setup is not None:
        fig.add_hline(y=last_setup['entry'], line_dash="solid", line_color="#2979FF",
                      annotation_text=f"ENTRY ({last_setup['type']}): {last_setup['entry']:,.2f}", annotation_position="top left")
        fig.add_hline(y=last_setup['sl'], line_dash="dash", line_color="#FF1744",
                      annotation_text=f"STOP LOSS: {last_setup['sl']:,.2f}", annotation_position="bottom left")
        fig.add_hline(y=last_setup['tp'], line_dash="dash", line_color="#00C853",
                      annotation_text=f"TAKE PROFIT: {last_setup['tp']:,.2f}", annotation_position="top left")

    fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Grafik Kurva Ekuitas & Tabel Riwayat
    col_eq, col_tbl = st.columns([1.2, 1])
    with col_eq:
        st.subheader("Kurva Pertumbuhan Modal (Equity Curve)")
        fig_eq = go.Figure(go.Scatter(x=df_res.index, y=df_res['Equity'], mode='lines', line=dict(color='#00E676', width=2), name='Ekuitas'))
        fig_eq.update_layout(height=320, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_tbl:
        st.subheader("Riwayat Transaksi Backtest")
        if not trades_df.empty:
            display_df = trades_df[['type', 'entry_date', 'entry', 'sl', 'tp', 'result', 'pnl']].copy()
            st.dataframe(display_df.tail(15), use_container_width=True, height=320)
