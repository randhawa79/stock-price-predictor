import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Stock Price Predictor Engine",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Stock Price Prediction Engine using Machine Learning")
st.write("This app pulls real-time market data, engineers technical indicators, trains a Random Forest model, and estimates the next day's closing price.")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("Model Configuration")
user_stock = st.sidebar.text_input("NSE/BSE Ticker Symbol", value="20MICRONS.NS")
historical_period = st.sidebar.selectbox("Data Lookback Period", ["2y", "5y", "10y"], index=1)
n_trees = st.sidebar.slider("Number of Trees (n_estimators)", min_value=50, max_value=500, value=200, step=50)

st.sidebar.markdown("""
**Quick Formatting Tip:**
- For NSE stocks append `.NS` (e.g., `TCS.NS`, `RELIANCE.NS`)
- For BSE stocks append `.BO` (e.g., `500325.BO`)
""")

# Execute Pipeline on Button Click
if st.sidebar.button("Train Model & Predict"):
    with st.spinner(f"Ingesting live historical rows for {user_stock}..."):
        # 1. Fetching data
        raw_data = yf.download(user_stock, period=historical_period, interval="1d")
        
        if raw_data.empty:
            st.error("Error fetching data. Please verify the ticker format.")
        else:
            # Flatten multi-index columns if returned by yfinance
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [col[0] for col in raw_data.columns]
                
            data = raw_data.copy()
            
            # 2. Feature Engineering
            data["Return"] = data["Close"].pct_change()
            data["MA_5"] = data["Close"].rolling(5).mean()
            data["MA_20"] = data["Close"].rolling(20).mean()
            data["MA_50"] = data["Close"].rolling(50).mean()
            data["Volatility"] = data["Return"].rolling(20).std()
            data["Daily_Range"] = data["High"] - data["Low"]
            
            # 3. Create Target (Tomorrow's Price)
            data["Target"] = data["Close"].shift(-1)
            data.dropna(inplace=True)
            
            # 4. Select Features
            features = ["Open", "High", "Low", "Close", "Volume", "Return", "MA_5", "MA_20", "MA_50", "Volatility", "Daily_Range"]
            X = data[features]
            Y = data["Target"]
            
            # 5. Chronological Train-Test Split (Fixes Time-Series Data Leakage)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
            
            # Save the index timestamps for plotting
            test_dates = X_test.index
            
            # 6. Model Training
            model = RandomForestRegressor(n_estimators=n_trees, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, Y_train)
            
            # 7. Predictions & Evaluation
            prediction = model.predict(X_test)
            
            mae = mean_absolute_error(Y_test, prediction)
            rmse = np.sqrt(mean_squared_error(Y_test, prediction))
            r2 = r2_score(Y_test, prediction)
            
            # Next Day Projection
            latest_data = X.tail(1)
            next_day_price = model.predict(latest_data)[0]
            
            # ==========================================
            # RENDER DASHBOARD INTERFACE
            # ==========================================
            st.success("Machine Learning Model Restructured & Trained Successfully!")
            
            # Target Metric Banner
            st.markdown(f"""
            <div style="background-color:#0E1117; padding:20px; border-radius:10px; border: 1px solid #464b55; text-align:center; margin-bottom: 25px;">
                <h3 style="margin:0; color:#2ECC71;">Projected Next Trading Day Close</h3>
                <h1 style="margin:10px 0 0 0; font-size:42px; color:white;">₹ {next_day_price:.2f}</h1>
                <p style="margin:5px 0 0 0; color:#888;">Ticker Code Context: {user_stock.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics Layout Block
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error (MAE)", f"₹{mae:.2f}", help="Average prediction variance error profile.")
            col2.metric("Root Mean Squared Error (RMSE)", f"₹{rmse:.2f}", help="Penalizes larger outliers in errors.")
            col3.metric("R² Score (Goodness of Fit)", f"{r2:.4f}", help="Proportion of variance explained by model parameters.")
            
            st.markdown("---")
            
            # Visualization Section
            col_chart1, col_chart2 = st.columns([2, 1])
            
            with col_chart1:
                st.subheader("Backtest Performance: Actual vs. Model Prediction")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(test_dates, Y_test.values, label="Actual Out-of-Sample Price", color="#1f77b4", linewidth=1.5)
                ax.plot(test_dates, prediction, label="Random Forest Prediction", color="#ff7f0e", linestyle="--", linewidth=1.5)
                ax.set_ylabel("Price (INR)")
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            with col_chart2:
                st.subheader("Feature Importance Distribution")
                importance = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=True)
                
                fig2, ax2 = plt.subplots(figsize=(5, 6.7))
                ax2.barh(importance["Feature"], importance["Importance"], color="#2ECC71", alpha=0.8)
                ax2.set_xlabel("Relative Weight Calculation Factor")
                ax2.grid(axis='x', linestyle=":", alpha=0.5)
                st.pyplot(fig2)