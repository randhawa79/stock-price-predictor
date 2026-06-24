import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress visual disruptions
warnings.filterwarnings('ignore')

# Set layout structure
st.set_page_config(
    page_title="Quantitative TA Forecast Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Quantitative Technical Analysis & Next-Day Forecast Engine")
st.write("This app uses pure statistical volatility boundaries and multi-indicator momentum scoring rules to project tomorrow's expected trading range.")

# ==========================================
# SIDEBAR CONFIGURATION LAYER
# ==========================================
st.sidebar.header("Data & Parameter Input")
user_stock = st.sidebar.text_input("NSE/BSE Ticker Symbol", value="20MICRONS.NS")
lookback_period = st.sidebar.selectbox("Historical Lookback Window", ["6mo", "1y", "2y"], index=1)

st.sidebar.markdown("""
**Ticker Formatting Tips:**
- For NSE (National Stock Exchange): Append `.NS` (e.g., `RELIANCE.NS`)
- For BSE (Bombay Stock Exchange): Append `.BO` (e.g., `500325.BO`)
""")

if st.sidebar.button("Run Quantitative Analysis"):
    with st.spinner(f"Ingesting structured market data matrices for {user_stock.upper()}..."):
        # Fetch high-fidelity daily data
        raw_data = yf.download(user_stock, period=lookback_period, interval="1d")
        
        if raw_data.empty:
            st.error("Data ingestion failed. Please verify that the ticker symbol format is correct.")
        else:
            # Flatten yfinance multi-index columns if present
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [col[0] for col in raw_data.columns]
                
            df = raw_data.copy()
            
            # ==========================================
            # MATHEMATICAL INDICATOR CALCULATIONS
            # ==========================================
            # 1. Moving Averages
            df["MA_5"] = df["Close"].rolling(5).mean()
            df["MA_20"] = df["Close"].rolling(20).mean()
            df["MA_50"] = df["Close"].rolling(50).mean()
            
            # 2. Bollinger Bands (20-day, 2 Standard Deviations)
            df["BB_Middle"] = df["MA_20"]
            df["BB_Std"] = df["Close"].rolling(20).std()
            df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
            df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)
            
            # 3. Relative Strength Index (RSI - 14 Days)
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
            rs = gain / (loss.replace(0.0, 0.00001)) # Avoid division by zero
            df["RSI"] = 100 - (100 / (1 + rs))
            
            # 4. MACD (Moving Average Convergence Divergence)
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            
            # 5. Average True Range (ATR - 14 Days for Volatility Bounds)
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df["ATR"] = true_range.rolling(14).mean()
            
            df.dropna(inplace=True)
            
            # ==========================================
            # EXTRACT LATEST COMPLETE DATA VECTORS
            # ==========================================
            latest = df.iloc[-1]
            cmp = latest["Close"]
            prev_high = latest["High"]
            prev_low = latest["Low"]
            atr = latest["ATR"]
            
            # ==========================================
            # PIVOT POINT MATRIX FORMULATION (TOMORROW)
            # ==========================================
            # Standard Floor Pivot Calculations
            pivot = (prev_high + prev_low + cmp) / 3
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            
            # ==========================================
            # MULTI-FACTOR DIRECTIONAL SCORING ENGINE
            # ==========================================
            score = 0
            max_score = 6
            
            # Trend Rules
            if cmp > latest["MA_20"]: score += 1
            else: score -= 1
            if cmp > latest["MA_50"]: score += 1
            else: score -= 1
                
            # Momentum Rules
            if latest["MACD"] > latest["MACD_Signal"]: score += 2
            else: score -= 2
                
            # RSI Rules
            if 50 <= latest["RSI"] <= 70: score += 2  # Bullish expansion
            elif latest["RSI"] > 70: score += 0       # Stretched / Overbought consolidation risk
            elif 30 <= latest["RSI"] < 50: score -= 2 # Bearish contraction
            elif latest["RSI"] < 30: score -= 1      # Heavily Oversold / Watch for reversal bounce
            
            # Normalize Directional Bias Score
            bias_percentage = (score / max_score) * 100
            if score >= 3: outlook = "BULLISH ACCELERATION"
            elif score <= -3: outlook = "BEARISH CONTRACTION"
            else: outlook = "NEUTRAL / MEAN REVERSION"
                
            # ==========================================
            # VOLATILITY-ADJUSTED NEXT-DAY PRICE TARGET
            # ==========================================
            # Expected price trajectory derived from standard deviation daily volatility translation
            expected_change = (score / max_score) * atr
            predicted_target = cmp + expected_change
            
            # Statistical floor and ceiling boundaries (Max expected daily variance envelope)
            upper_statistical_bound = cmp + atr
            lower_statistical_bound = cmp - atr
            
            # ==========================================
            # RENDER INTERACTIVE DASHBOARD VIEW
            # ==========================================
            st.success("Technical Analysis Matrix Calculated Successfully!")
            
            # Main Structural Banner
            st.markdown(f"""
            <div style="background-color:#111625; padding:25px; border-radius:12px; border: 1px solid #2e374d; text-align:center; margin-bottom: 25px;">
                <h4 style="margin:0; color:#8892b0; text-transform: uppercase; letter-spacing: 1.5px;">Next Trading Day Quantitative Forecast Target</h4>
                <h1 style="margin:10px 0; font-size:46px; color:#ffffff;">₹ {predicted_target:.2f}</h1>
                <p style="margin:0; font-size:16px; color:#2ECC71; font-weight:bold;">Outlook Context: {outlook} (Bias Score: {bias_percentage:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Row 1: Key Boundary Parameters
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Market Price (CMP)", f"₹{cmp:.2f}")
            c2.metric("Average True Range (ATR)", f"₹{atr:.2f}", help="Average dynamic daily price movement range.")
            c3.metric("Expected Volatility Ceiling", f"₹{upper_statistical_bound:.2f}", help="Maximum statistical upper boundary for tomorrow.")
            c4.metric("Expected Volatility Floor", f"₹{lower_statistical_bound:.2f}", help="Maximum statistical lower boundary for tomorrow.")
            
            st.markdown("---")
            
            # Secondary Tabbed Panel Interface Layout
            tab_trader, tab_indicators, tab_charts = st.tabs(["🎯 Next-Day Trading Cheat Sheet", "📋 Indicator Signals Engine", "📊 Trend & Momentum Charts"])
            
            with tab_trader:
                st.subheader("Floor Pivot Target Matrices (Next Session)")
                st.write("These levels are generated using the price action of the most recent trading session to define support and resistance thresholds.")
                
                pivot_col1, pivot_col2 = st.columns(2)
                with pivot_col1:
                    st.markdown("##### Resistance Targets (Upside)")
                    st.table(pd.DataFrame({
                        "Target Zone": ["Resistance 2 (R2 - Max Extension)", "Resistance 1 (R1 - Breakout Node)", "Central Pivot Point (P)"],
                        "Price Target": [f"₹{r2:.2f}", f"₹{r1:.2f}", f"₹{pivot:.2f}"]
                    }))
                with pivot_col2:
                    st.markdown("##### Support Targets (Downside)")
                    st.table(pd.DataFrame({
                        "Target Zone": ["Central Pivot Point (P)", "Support 1 (S1 - Accumulation Node)", "Support 2 (S2 - Capitulation Floor)"],
                        "Price Target": [f"₹{pivot:.2f}", f"₹{s1:.2f}", f"₹{s2:.2f}"]
                    }))
            
            with tab_indicators:
                st.subheader("Current Technical Parameters Ingestion")
                
                ind_col1, ind_col2 = st.columns(2)
                with ind_col1:
                    st.markdown("**Momentum Vectors**")
                    st.write(f"- **Relative Strength Index (RSI):** {latest['RSI']:.2f} " + ("(Overbought 🔥)" if latest['RSI'] > 70 else ("(Oversold ❄️)" if latest['RSI'] < 30 else "(Neutral Momentum)")))
                    st.write(f"- **MACD Line:** {latest['MACD']:.4f}")
                    st.write(f"- **MACD Signal Line:** {latest['MACD_Signal']:.4f}")
                    st.write(f"- **MACD Histogram Alignment:** {latest['MACD'] - latest['MACD_Signal']:.4f}")
                
                with ind_col2:
                    st.markdown("**Trend Following Alignments**")
                    st.write(f"- **5-Day Short-Term EMA:** ₹{latest['MA_5']:.2f} (" + ("Price Above" if cmp > latest['MA_5'] else "Price Below") + ")")
                    st.write(f"- **20-Day Intermediate Horizon:** ₹{latest['MA_20']:.2f}")
                    st.write(f"- **50-Day Structural Trendline Floor:** ₹{latest['MA_50']:.2f}")
            
            with tab_charts:
                st.subheader("Visual Technical Analysis Mapping Canvas")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
                plot_dates = df.index[-60:] # Filter out the last 60 trading sessions for visual clarity
                
                # Plot 1: Price Tracking with Overlaid Volatility Envelopes
                ax1.plot(plot_dates, df["Close"].astype(float)[-60:], label="Closing Price", color="#1F77B4", linewidth=2.5)
                ax1.plot(plot_dates, df["MA_20"].astype(float)[-60:], label="20 MA Baseline", color="#E67E22", linestyle="--")
                ax1.fill_between(plot_dates, df["BB_Upper"].astype(float)[-60:], df["BB_Lower"].astype(float)[-60:], color="#1F77B4", alpha=0.1, label="Bollinger Bands Variance Range")
                ax1.set_title("Price Tracking Matrix & Underlying Structural Volatility Boundaries", fontsize=12, fontweight="bold")
                ax1.set_ylabel("Price (INR)")
                ax1.grid(True, linestyle=":", alpha=0.5)
                ax1.legend(loc="upper left")
                
                # Plot 2: Relative Strength Index (RSI Indicator Tracking)
                ax2.plot(plot_dates, df["RSI"].astype(float)[-60:], label="RSI (14)", color="#9B59B6", linewidth=2)
                ax2.axhline(70, color="#E74C3C", linestyle=":", linewidth=1.5)
                ax2.axhline(30, color="#2ECC71", linestyle=":", linewidth=1.5)
                ax2.axhline(50, color="gray", linestyle="-.", linewidth=1.0, alpha=0.5)
                ax2.fill_between(plot_dates, 70, 30, color="#9B59B6", alpha=0.05)
                ax2.set_title("Relative Strength Index (RSI Horizon)", fontsize=11, fontweight="bold")
                ax2.set_ylabel("Oscillator Range")
                ax2.set_ylim(10, 90)
                ax2.grid(True, linestyle=":", alpha=0.5)
                ax2.legend(loc="upper left")
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)