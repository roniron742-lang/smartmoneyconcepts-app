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
ticker = st.sidebar.text_input("Simbol Ticker (Yahoo Finance)", value="BTC-USD", help="Contoh: BBCA.JK, EURUSD=X, GC=F, BTC-USD")
period = st.sidebar.selectbox("Rentang Data Historis (Chart Utama)", options=["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Timeframe Chart Utama", options=["1h", "4h", "1d"], index=2)

st.sidebar.subheader("Filter Kualitas SMC")
use_ema_filter = st.sidebar.checkbox("Aktifkan Filter Tren (EMA 200)", value=True)
min_fvg_atr = st.sidebar.slider("Lebar Minimum FVG (x ATR)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
order_expiry = st.sidebar.slider("Batas Waktu Tunggu Retest (Candle)", min_value=3, max_value=20, value=8, step=1)

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

# Fungsi Logika SMC (FVG + EMA 200 + Retest Limit Order)
def run_smc_strategy(df, capital, risk_pct, rr, fee, ema_filter, min_gap_mult, expiry_bars):
    data = df.copy()
    
    # Hitung Indikator Pendukung: EMA 200 & ATR 14
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()

    # Hitung Celah FVG (Fair Value Gap)
    bull_gap = data['Low'] - data['High'].shift(2)
    bear_gap = data['Low'].shift(2) - data['High']

    # Kondisi FVG Valid (Struktur 3 Candle + Lebar Minimum ATR)
    raw_bull_fvg = (bull_gap > (data['ATR'] * min_gap_mult)) & (data['Close'].shift(1) > data['Open'].shift(1))
    raw_bear_fvg = (bear_gap > (data['ATR'] * min_gap_mult)) & (data['Close'].shift(1) < data['Open'].shift(1))

    # Terapkan Filter Tren EMA 200 jika diaktifkan
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
                risk_dist = entry_price - sl
