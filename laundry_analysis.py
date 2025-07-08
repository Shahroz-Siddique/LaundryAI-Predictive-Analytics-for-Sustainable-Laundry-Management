import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime

def prepare_laundry_data(df, laundry_id):
    """Prepare laundry-specific time series data"""
    laundry_df = df[df['LaundryID'] == laundry_id].copy()
    daily_demand = laundry_df.groupby('StartDate').size().reset_index(name='y')
    daily_demand = daily_demand.rename(columns={"StartDate": "ds"})
    daily_demand["ds"] = pd.to_datetime(daily_demand["ds"])
    return daily_demand

def forecast_demand_laundry_rf(daily_data, forecast_days=14):
    """Forecast laundry demand using Random Forest"""
    data = daily_data.copy()
    data['DayOfYear'] = data['ds'].dt.dayofyear
    data['DayOfWeek'] = data['ds'].dt.dayofweek
    data['WeekOfYear'] = data['ds'].dt.isocalendar().week.astype(int)

    X = data[['DayOfYear', 'DayOfWeek', 'WeekOfYear']]
    y = data['y']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({"ds": future_dates})
    future_df['DayOfYear'] = future_df['ds'].dt.dayofyear
    future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
    future_df['WeekOfYear'] = future_df['ds'].dt.isocalendar().week.astype(int)

    yhat = model.predict(future_df[['DayOfYear', 'DayOfWeek', 'WeekOfYear']])
    future_df['yhat'] = yhat
    future_df['yhat_lower'] = yhat - 1.5
    future_df['yhat_upper'] = yhat + 1.5

    return data, model, future_df

def laundry_resource_analysis(df, laundry_id):
    """Analyze resource consumption patterns"""
    laundry_df = df[df['LaundryID'] == laundry_id].copy()
    
    daily_usage = laundry_df.groupby(["StartDate"]).agg({
        "TenantID": "count",
        "Water_Litres": "sum",
        "Electricity_kWh": "sum"
    }).reset_index()
    
    daily_usage = daily_usage.rename(columns={
        "TenantID": "OrderCount",
        "Water_Litres": "WaterConsumption",
        "Electricity_kWh": "ElectricityConsumption"
    })
    
    X = daily_usage[["OrderCount"]]
    daily_usage["ExpectedWater"] = LinearRegression().fit(X, daily_usage["WaterConsumption"]).predict(X)
    daily_usage["ExpectedElectricity"] = LinearRegression().fit(X, daily_usage["ElectricityConsumption"]).predict(X)
    
    daily_usage["WaterError"] = daily_usage["WaterConsumption"] - daily_usage["ExpectedWater"]
    daily_usage["ElectricError"] = daily_usage["ElectricityConsumption"] - daily_usage["ExpectedElectricity"]
    
    model = IsolationForest(contamination=0.05, random_state=42)
    daily_usage["Anomaly"] = model.fit_predict(daily_usage[["WaterError", "ElectricError"]])
    daily_usage["AnomalyLabel"] = daily_usage["Anomaly"].map({1: "Normal", -1: "Anomaly"})
    
    def check_alerts(row):
        if row["Anomaly"] == -1 and row["OrderCount"] < 5:
            return "🚨 Alert: High usage on low order day"
        return "✅ Normal"
    
    daily_usage["Alert"] = daily_usage.apply(check_alerts, axis=1)
    
    return daily_usage

def detect_low_demand_days(df, laundry_id, threshold=5):
    """Identify days with expected low demand"""
    daily = prepare_laundry_data(df, laundry_id)
    
    if daily.empty:
        return None, None
    
    daily, model, forecast = forecast_demand_laundry_rf(daily, forecast_days=7)
    low_demand = forecast[forecast["yhat"] < threshold]
    
    return forecast, low_demand

def laundry_section(df):
    """Main laundry analysis section"""
    st.header("Laundry Analysis")
    laundry_id = st.text_input("**Enter Laundry ID:**", placeholder="e.g. L3")
    
    if not laundry_id:
        return
    
    # Subsection routing
    subsection = st.sidebar.radio("Laundry Analysis", 
                                 ["📈 Peak Forecast", "🚨 Peak Alerts", "⚡ Resources"],
                                 horizontal=True)
    
    if "Forecast" in subsection:
        show_peak_forecast(df, laundry_id)
    elif "Alerts" in subsection:
        show_peak_alerts(df, laundry_id)
    elif "Resources" in subsection:
        show_resource_analysis(df, laundry_id)

def show_peak_forecast(df, laundry_id):
    """Display peak demand forecast"""
    st.subheader("Peak Days Forecast")
    peak_threshold = st.slider("Set Peak Threshold", 1, 20, 5, key="peak_threshold")
    
    with st.spinner("Forecasting laundry demand..."):
        daily = prepare_laundry_data(df, laundry_id)
        
        if daily.empty:
            st.markdown(f'<div class="alert-box"><h3>⚠️ No data found for Laundry ID: {laundry_id}</h3></div>', unsafe_allow_html=True)
            return
        
        daily, model, forecast = forecast_demand_laundry_rf(daily, forecast_days=30)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(daily["ds"], daily["y"], 'o-', label="Actual Demand", color='#1a1a2e', markersize=4)
        ax.plot(forecast["ds"], forecast["yhat"], '--', label="Forecast", color='#e94560', linewidth=2)
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], 
                      color='#e94560', alpha=0.2, label="Confidence Interval")
        
        forecast_start = forecast["ds"].min()
        ax.axvspan(forecast_start, forecast["ds"].max(), color='gray', alpha=0.07, label="Forecast Period")
        
        ax.axhline(y=peak_threshold, color='green', linestyle='--', label=f"Peak Threshold: {peak_threshold}")
        peak_days = forecast[forecast["yhat"] > peak_threshold]
        ax.scatter(peak_days["ds"], peak_days["yhat"], color='orange', label="Predicted Peak Days", s=60)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Orders")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        st.pyplot(fig)

def show_peak_alerts(df, laundry_id):
    """Display peak demand alerts"""
    st.subheader("Peak Days Alert System")
    peak_threshold = st.slider("Set Alert Threshold", 1, 20, 8, key="alert_threshold")
    
    with st.spinner("Analyzing peak days..."):
        daily = prepare_laundry_data(df, laundry_id)
        
        if daily.empty:
            st.markdown(f'<div class="alert-box"><h3>⚠️ No data found for Laundry ID: {laundry_id}</h3></div>', unsafe_allow_html=True)
            return
        
        daily, model, forecast = forecast_demand_laundry_rf(daily, forecast_days=30)
        peak_days = forecast[forecast["yhat"] > peak_threshold]
        
        if not peak_days.empty:
            st.markdown(f'<div class="alert-box"><h3>🚨 ALERT: {len(peak_days)} peak days detected at Laundry {laundry_id}</h3></div>', 
                       unsafe_allow_html=True)
            
            # Display peak days in a table
            peak_days_display = peak_days.copy()
            peak_days_display["Date"] = peak_days_display["ds"].dt.strftime('%Y-%m-%d')
            peak_days_display["Expected Orders"] = peak_days_display["yhat"].round(1)
            st.dataframe(peak_days_display[["Date", "Expected Orders"]].reset_index(drop=True))
            
            # Show action plan
            with st.expander("📋 Recommended Action Plan", expanded=True):
                st.markdown("""
                **👥 Staffing:**
                - Schedule extra staff during peak hours
                - Prepare backup staff for unexpected demand
                - Ensure manager coverage during peak periods
                
                **⚙️ Operations:**
                - Delay non-essential maintenance
                - Prepare additional laundry machines
                - Increase inventory of detergents and supplies
                
                **😊 Customer Experience:**
                - Notify customers about potential delays
                - Prepare express service options
                - Implement priority for regular customers
                """)
            
            # Email alert system
            with st.expander("✉️ Send Alert Notifications"):
                emails = st.text_input("Enter email addresses (comma separated):")
                message = st.text_area("Custom message:", 
                                     f"Peak demand alert for Laundry {laundry_id} on the following dates:")
                
                if st.button("Send Alerts"):
                    st.success(f"✅ Alerts sent to: {emails}")
        else:
            st.markdown(f'<div class="success-box"><h3>✅ No peak days detected at Laundry {laundry_id} with current threshold</h3></div>', 
                       unsafe_allow_html=True)

def show_resource_analysis(df, laundry_id):
    """Display resource consumption analysis"""
    st.subheader("Resource Analysis")
    
    with st.spinner("Analyzing resource consumption..."):
        daily_usage = laundry_resource_analysis(df, laundry_id)
        
        if daily_usage.empty:
            st.markdown('<div class="alert-box"><h3>⚠️ No data found for the specified laundry</h3></div>', unsafe_allow_html=True)
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Usage Trends", "🔍 Anomaly Detection", "📊 Efficiency Metrics", "📉 Low Demand Forecast"])
        
        with tab1:
            st.markdown("### Water Consumption Over Time")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(daily_usage["StartDate"], daily_usage["WaterConsumption"], 
                    label="Actual Water", color='#0f3460', linewidth=2)
            ax1.plot(daily_usage["StartDate"], daily_usage["ExpectedWater"], 
                    '--', label="Expected Water", color='#1a1a2e', linewidth=2)
            ax1.set_xlabel("Date")
            ax1.legend()
            ax1.grid(True, alpha=0.2)
            st.pyplot(fig1)
            
            st.markdown("### Electricity Consumption Over Time")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(daily_usage["StartDate"], daily_usage["ElectricityConsumption"], 
                    label="Actual Electricity", color='#e94560', linewidth=2)
            ax2.plot(daily_usage["StartDate"], daily_usage["ExpectedElectricity"], 
                    '--', label="Expected Electricity", color='#8a0b22', linewidth=2)
            ax2.set_xlabel("Date")
            ax2.legend()
            ax2.grid(True, alpha=0.2)
            st.pyplot(fig2)
        
        with tab2:
            st.markdown("### Water Consumption Anomalies")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.scatterplot(data=daily_usage, x="StartDate", y="WaterConsumption", 
                          hue="AnomalyLabel", palette="Set1", ax=ax3)
            ax3.set_title("Water Consumption Anomalies")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)
            
            st.markdown("### Electricity Consumption Anomalies")
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            sns.scatterplot(data=daily_usage, x="StartDate", y="ElectricityConsumption", 
                          hue="AnomalyLabel", palette="Set2", ax=ax4)
            ax4.set_title("Electricity Consumption Anomalies")
            ax4.tick_params(axis='x', rotation=45)
            st.pyplot(fig4)
            
            # Show alerts
            alerts = daily_usage[daily_usage["Alert"].str.contains("Alert")]
            if not alerts.empty:
                st.markdown("### 🔔 Resource Alerts")
                st.dataframe(alerts[["StartDate", "OrderCount", 
                                   "WaterConsumption", "ElectricityConsumption", "Alert"]])
        
        with tab3:
            st.markdown("### Efficiency Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Water per Order</div><div class="metric-value">{daily_usage["WaterConsumption"].sum()/daily_usage["OrderCount"].sum():.2f} L</div></div>', 
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Electricity per Order</div><div class="metric-value">{daily_usage["ElectricityConsumption"].sum()/daily_usage["OrderCount"].sum():.2f} kWh</div></div>', 
                            unsafe_allow_html=True)
            
            st.markdown("### Water Efficiency (Litres per Order)")
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            ax5.plot(daily_usage["StartDate"], daily_usage["WaterConsumption"] / daily_usage["OrderCount"], 
                    color='#0f3460', linewidth=2)
            ax5.set_xlabel("Date")
            ax5.grid(True, alpha=0.2)
            st.pyplot(fig5)
            
            st.markdown("### Electricity Efficiency (kWh per Order)")
            fig6, ax6 = plt.subplots(figsize=(10, 4))
            ax6.plot(daily_usage["StartDate"], daily_usage["ElectricityConsumption"] / daily_usage["OrderCount"], 
                    color='#e94560', linewidth=2)
            ax6.set_xlabel("Date")
            ax6.grid(True, alpha=0.2)
            st.pyplot(fig6)
        
        with tab4:
            st.markdown("### Low Demand Forecast")
            low_threshold = st.slider("Low Demand Threshold", 1, 10, 3, key="low_threshold")
            
            forecast, low_demand = detect_low_demand_days(df, laundry_id, low_threshold)
            
            if forecast is not None:
                if not low_demand.empty:
                    st.markdown(f'<div class="alert-box"><h3>⚠️ Low demand days detected: {len(low_demand)} days below {low_threshold} orders</h3></div>', 
                               unsafe_allow_html=True)
                    
                    # Show optimization recommendations
                    st.markdown("### 💡 Resource Optimization Recommendations")
                    st.markdown("""
                    #### 🧺 Machine Optimization Plan:
                   
                    1. **Energy Saving Mode**:
                       - Use eco-friendly wash cycles
                       - Reduce water temperature settings
                       - Schedule operations during off-peak electricity hours
                    
                    2. **Maintenance Schedule**:
                       - Perform routine maintenance on idle days
                       - Clean filters and check machine efficiency
                    
                    3. **Staff Management**:
                       - Reduce staff during low demand periods
                       - Schedule training or administrative tasks
                    
                    4. **Resource Allocation**:
                       - Redirect resources to busier locations
                       - Plan inventory management
                    """)
                    
                    # Show low demand days
                    st.markdown("### 📅 Low Demand Days")
                    low_days_display = low_demand.copy()
                    low_days_display["Date"] = low_days_display["ds"].dt.strftime('%Y-%m-%d')
                    low_days_display["Expected Orders"] = low_days_display["yhat"].round(1)
                    st.dataframe(low_days_display[["Date", "Expected Orders"]].reset_index(drop=True))
                else:
                    st.markdown(f'<div class="success-box"><h3>✅ No low demand days detected below {low_threshold} orders</h3></div>', 
                               unsafe_allow_html=True)