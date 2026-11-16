import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from smartmoneyconcepts import smc

st.set_page_config(page_title="SMC Trading Dashboard", layout="wide")
st.title("📊 Smart Money Concepts (SMC) Strategy Dashboard")

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("Pengaturan Aset & Parameter")
ticker = st.sidebar.text_input("Simbol Aset (Contoh: BTC-USD, GC=F, BBCA.JK)", value="BTC-USD")
period = st.sidebar.selectbox("Rentang Waktu Data", ["1mo", "3mo", "6mo", "1y"], index=1)
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
swing_length = st.sidebar.slider("Swing Length (Sensitivitas Struktur)", min_value=5, max_value=50, value=10)

@st.cache_data(ttl=300)
def load_data(symbol, per, inter):
    df = yf.download(symbol, period=per, interval=inter, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    return df

df = load_data(ticker, period, interval)

if df.empty:
    st.error("Data tidak ditemukan. Periksa kembali simbol aset atau interval waktu.")
else:
    # --- HITUNG INDIKATOR SMC ---
    swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    fvg_data = smc.fvg(df, join_consecutive=True)
    ob_data = smc.ob(df, swing_hl)
    bos_choch = smc.bos_choch(df, swing_hl)

    # --- LOGIKA SINYAL ENTRY SEDERHANA ---
    # Sinyal Buy jika harga penutupan masuk/menyentuh Bullish FVG terakhir
    df['Signal'] = None
    last_close = df['close'].iloc[-1]
    
    bullish_fvg = fvg_data[fvg_data['FVG'] == 1].dropna()
    bearish_fvg = fvg_data[fvg_data['FVG'] == -1].dropna()

    status_sinyal = "Menunggu Konfirmasi Zona (Wait)"
    if not bullish_fvg.empty:
        top_fvg = bullish_fvg['Top'].iloc[-1]
        bot_fvg = bullish_fvg['Bottom'].iloc[-1]
        if bot_fvg <= last_close <= top_fvg:
            status_sinyal = f"POTENSI BUY (Harga di zona Bullish FVG: {bot_fvg:.2f} - {top_fvg:.2f})"
            
    if not bearish_fvg.empty:
        top_fvg = bearish_fvg['Top'].iloc[-1]
        bot_fvg = bearish_fvg['Bottom'].iloc[-1]
        if bot_fvg <= last_close <= top_fvg:
            status_sinyal = f"POTENSI SELL (Harga di zona Bearish FVG: {bot_fvg:.2f} - {top_fvg:.2f})"

    st.info(f"*Status Analisa Saat Ini ({ticker}):* {status_sinyal} | Harga Terakhir: *{last_close:.2f}*")

    # --- VISUALISASI GRAFIK CANDLESTICK + SMC ---
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    )])

    # Gambar kotak Fair Value Gap (FVG) terbaru (maksimal 15 terakhir agar ringan)
    recent_fvg = fvg_data.dropna(subset=['FVG']).tail(15)
    for idx, row in recent_fvg.iterrows():
        color = "rgba(0, 255, 0, 0.2)" if row['FVG'] == 1 else "rgba(255, 0, 0, 0.2)"
        fig.add_shape(
            type="rect",
            x0=idx, y0=row['Bottom'],
            x1=df.index[-1], y1=row['Top'],
            fillcolor=color, line=dict(width=0),
            layer="below"
        )

    # Gambar kotak Order Block (OB) terbaru (maksimal 10 terakhir)
    recent_ob = ob_data.dropna(subset=['OB']).tail(10)
    for idx, row in recent_ob.iterrows():
        color = "rgba(0, 128, 255, 0.25)" if row['OB'] == 1 else "rgba(255, 165, 0, 0.25)"
        fig.add_shape(
            type="rect",
            x0=idx, y0=row['Bottom'],
            x1=df.index[-1], y1=row['Top'],
            fillcolor=color, line=dict(color="gray", width=1, dash="dot"),
            layer="below"
        )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=f"Peta Struktur SMC: Hijau/Merah = FVG | Biru/Oranye = Order Block"
    )
    st.plotly_chart(fig, use_container_width=True)
