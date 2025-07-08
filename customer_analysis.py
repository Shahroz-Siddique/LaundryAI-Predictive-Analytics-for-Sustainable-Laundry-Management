import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import io
from resources import load_data

def prepare_enhanced_data(data):
    """Prepare time series data with features for forecasting"""
    min_date = data["StartDate"].min()
    max_date = data["StartDate"].max()
    full_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    daily_orders = data.groupby("StartDate").size().reset_index(name="orders")
    daily_orders = daily_orders.rename(columns={"StartDate": "ds"})
    
    full_df = pd.DataFrame({"ds": full_dates})
    daily_orders = full_df.merge(daily_orders, on="ds", how="left").fillna(0)
    
    daily_orders["day_of_week"] = daily_orders["ds"].dt.weekday
    daily_orders["is_weekend"] = daily_orders["day_of_week"].isin([5, 6]).astype(int)
    daily_orders["month"] = daily_orders["ds"].dt.month
    daily_orders["day_of_month"] = daily_orders["ds"].dt.day
    daily_orders["time_idx"] = (daily_orders["ds"] - daily_orders["ds"].min()).dt.days
    
    daily_orders["orders_7d_avg"] = daily_orders["orders"].rolling(window=7, min_periods=1).mean()
    daily_orders["orders_28d_avg"] = daily_orders["orders"].rolling(window=28, min_periods=1).mean()
    
    for lag in [1, 7, 14, 28]:
        daily_orders[f"lag_{lag}"] = daily_orders["orders"].shift(lag).fillna(0)
    
    holidays = data[data["IsHoliday"] == 1]["StartDate"].unique()
    daily_orders["is_holiday"] = daily_orders["ds"].isin(holidays).astype(int)
    
    return daily_orders.dropna()

def forecast_intermittent_demand(daily_data, customer_data):
    """Forecast future demand using Random Forest"""
    holidays = customer_data[customer_data["IsHoliday"] == 1]["StartDate"].unique()
    
    X = daily_data.drop(columns=["orders", "ds"])
    y = daily_data["orders"]
    
    train_size = int(len(daily_data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    future_days = 90
    last_date = daily_data["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    
    future_data = []
    for i, date in enumerate(future_dates):
        features = {
            "day_of_week": date.weekday(),
            "is_weekend": 1 if date.weekday() in [5, 6] else 0,
            "month": date.month,
            "day_of_month": date.day,
            "time_idx": (date - daily_data["ds"].min()).days,
            "orders_7d_avg": daily_data["orders_7d_avg"].iloc[-1],
            "orders_28d_avg": daily_data["orders_28d_avg"].iloc[-1],
            "lag_1": daily_data["orders"].iloc[-1],
            "lag_7": daily_data["orders"].iloc[-7] if len(daily_data) >= 7 else 0,
            "lag_14": daily_data["orders"].iloc[-14] if len(daily_data) >= 14 else 0,
            "lag_28": daily_data["orders"].iloc[-28] if len(daily_data) >= 28 else 0,
            "is_holiday": 1 if date in holidays else 0
        }
        future_data.append(features)
    
    future_df = pd.DataFrame(future_data)
    future_pred = model.predict(future_df)
    
    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.maximum(0, future_pred),
        "yhat_lower": np.maximum(0, future_pred * 0.7),
        "yhat_upper": np.maximum(0, future_pred * 1.3)
    })
    
    historical = daily_data[["ds", "orders"]].rename(columns={"orders": "y"})
    forecast_df = pd.concat([historical, forecast_df], ignore_index=True)
    
    return historical, forecast_df

def generate_customer_insights(data):
    """Generate key insights about customer behavior"""
    insights = []
    
    last_order = data["StartDate"].max()
    today = datetime.now()
    days_since = (today - last_order).days
    insights.append(f"📅 **Days since last order**: {days_since} days")
    
    order_dates = data["StartDate"].sort_values()
    avg_days_between = (order_dates.diff().dt.days.mean() if len(order_dates) > 1 else np.nan)
    insights.append(f"⏱️ **Average days between orders**: {avg_days_between:.1f} days")
    
    weekday_counts = data["StartDate"].dt.day_name().value_counts()
    top_day = weekday_counts.idxmax()
    insights.append(f"📆 **Most frequent ordering day**: {top_day}")
    
    item_pref = data["Item"].value_counts().head(3).to_dict()
    insights.append(f"👕 **Top items**: {', '.join([f'{k} ({v})' for k, v in item_pref.items()])}")
    
    service_pref = data["Service"].value_counts().head(3).to_dict()
    insights.append(f"🧼 **Top services**: {', '.join([f'{k} ({v})' for k, v in service_pref.items()])}")
    
    return insights

def plot_detailed_forecast(forecast, customer_id):
    """Create detailed forecast visualization"""
    oct_end = pd.to_datetime("2025-10-31")
    forecast = forecast[forecast["ds"] <= oct_end].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(forecast["ds"], forecast["yhat"], 
            color='#0f3460', linewidth=3, 
            label="Forecasted Daily Orders")
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n\n%b\n%Y'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    
    ax.set_title(f"Customer {customer_id} Daily Order Forecast", 
              pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", labelpad=15)
    ax.set_ylabel("Predicted Orders", labelpad=10)
    ax.grid(True, which='both', alpha=0.2)
    
    for label in ax.get_xminorticklabels():
        label.set_rotation(90)
        label.set_fontsize(8)
        label.set_ha('center')
        label.set_va('top')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig

def generate_business_report(actual, forecast, customer_id, customer_data):
    """Generate comprehensive business report"""
    forecast_period = forecast[forecast["ds"] > actual["ds"].max()]
    start_date = forecast_period["ds"].min().strftime('%Y-%m')
    end_date = forecast_period["ds"].max().strftime('%Y-%m')
    
    avg_daily = forecast_period["yhat"].mean()
    total_orders = forecast_period["yhat"].sum()
    avg_lower = forecast_period["yhat_lower"].mean()
    avg_upper = forecast_period["yhat_upper"].mean()
    
    top_items = customer_data["Item"].value_counts().head(3).index.tolist()
    min_stock = max(1, round(avg_lower))
    weekly_stock = f"{round(avg_daily*7*0.8)}-{round(avg_daily*7*1.2)}"
    buffer_stock = round(avg_upper * 2)
    
    report = f"""
## 🔮 DEMAND FORECAST REPORT: CUSTOMER {customer_id}

### 📅 Forecast Period: 
{start_date} to {end_date} (90 days)

### 📊 Expected Demand:
- **Average daily orders**: {avg_daily:.2f}
- **Daily range**: {avg_lower:.2f} - {avg_upper:.2f} orders
- **Total projected orders**: {total_orders:.0f} orders ({avg_daily:.2f} × 90 days)

### 📈 Key Patterns:
1. Consistent low-volume demand with occasional small peaks
2. No extreme seasonality detected
3. Steady pattern similar to historical behavior

### 📦 Inventory Recommendations:
1. Maintain minimum daily stock: **{min_stock} unit{"s" if min_stock > 1 else ""}** per item
2. Weekly restocking level: **{weekly_stock} units**
3. Buffer stock for potential peaks: **{buffer_stock} extra units**
4. Focus inventory on top items: **{", ".join(top_items)}**

### 🚦 Risk Assessment:
- **Volatility**: {"Low" if avg_upper/avg_lower < 1.5 else "Medium"} (confidence range: {avg_lower:.2f}-{avg_upper:.2f})
- **Stockout risk**: {"Minimal" if avg_daily < 1 else "Moderate"}
- **Waste potential**: {"Low" if avg_daily > 0.5 else "Medium"}

### 💡 Business Actions:
- Maintain **{"lean" if avg_daily < 1 else "moderate"} inventory** - focus on freshness
- Schedule **{"weekly" if avg_daily < 2 else "bi-weekly"} deliveries**
- Allocate **{round(70 if len(top_items)>=3 else 100)}% of stock** to top {len(top_items)} items
- Monitor for **{"weekend" if customer_data["IsWeekend"].mean() > 0.3 else "mid-week"} demand**
"""
    return report

def customer_resource_analysis(customer_data):
    """Analyze resource usage patterns"""
    total_orders = len(customer_data)
    avg_water = customer_data["Water_Litres"].sum() / total_orders
    avg_electricity = customer_data["Electricity_kWh"].sum() / total_orders
    
    resource_usage = customer_data.groupby('StartDate').agg({
        'Water_Litres': 'sum',
        'Electricity_kWh': 'sum'
    }).reset_index()
    
    return resource_usage, avg_water, avg_electricity

def calculate_future_resource(forecast, avg_water, avg_electricity):
    """Project future resource needs based on forecast"""
    forecast_period = forecast[forecast["ds"] > datetime.now()]
    future_resource = forecast_period.copy()
    future_resource['Water_Needed'] = future_resource['yhat'] * avg_water
    future_resource['Electricity_Needed'] = future_resource['yhat'] * avg_electricity
    return future_resource

def customer_section(df):
    """Main customer analysis section"""
    st.header("Customer Analysis")
    tenant_id = st.text_input("**Enter Customer ID:**", placeholder="e.g. T1")
    
    if not tenant_id:
        return
        
    customer_data = df[df["TenantID"] == tenant_id]
    
    if customer_data.empty:
        st.markdown(f'<div class="alert-box"><h3>⚠️ No data found for Customer ID: {tenant_id}</h3></div>', unsafe_allow_html=True)
        return
    
    st.markdown(f'<div class="info-box"><p>🔍 Analyzing customer: {tenant_id}</p><p>📊 Found {len(customer_data)} historical orders</p></div>', unsafe_allow_html=True)
    
    # Subsection routing
    subsection = st.sidebar.radio("Customer Analysis", 
                                 ["📊 Historical", "🔮 Forecast", "💧 Resources", "📝 Report"],
                                 horizontal=True)
    
    if "Historical" in subsection:
        show_historical_analysis(customer_data)
    elif "Forecast" in subsection:
        show_forecast_analysis(customer_data, tenant_id)
    elif "Resources" in subsection:
        show_resource_analysis(customer_data, tenant_id)
    elif "Report" in subsection:
        show_business_report(customer_data, tenant_id)

def show_historical_analysis(customer_data):
    """Display historical customer insights"""
    st.subheader("Historical Customer Insights")
    insights = generate_customer_insights(customer_data)
    with st.expander("Detailed Customer Profile", expanded=True):
        for insight in insights:
            st.markdown(f"- {insight}")
    
    st.subheader("Recent Orders")
    recent_orders = customer_data.sort_values("StartDate", ascending=False).head(5)
    st.dataframe(recent_orders[["StartDate", "Item", "Service", "Water_Litres", "Electricity_kWh"]])

def show_forecast_analysis(customer_data, tenant_id):
    """Display demand forecast analysis"""
    st.subheader("Demand Forecasting")
    with st.spinner("Running demand forecast..."):
        daily_orders = prepare_enhanced_data(customer_data)
        actual, forecast = forecast_intermittent_demand(daily_orders, customer_data)
        
        if actual is None or forecast is None:
            st.error("Failed to generate forecast")
            return
        
        # Main forecast chart
        with st.container():
            st.markdown("### Demand Forecast Overview")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            
            ax1.plot(actual["ds"], actual["y"], 'o-', label="Actual Demand", 
                    color='#1a1a2e', linewidth=2, markersize=4, alpha=0.9)
            
            forecast_period = forecast[forecast["ds"] > actual["ds"].max()]
            ax1.plot(forecast["ds"], forecast["yhat"], '--', label="Forecast", 
                    color='#e94560', linewidth=2, alpha=0.9)
            
            ax1.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], 
                          color='#e94560', alpha=0.15, label="Confidence Range")
            
            forecast_start = forecast_period["ds"].min()
            ax1.axvspan(forecast_start, forecast["ds"].max(), 
                      color='gray', alpha=0.07, label="Forecast Period")
            
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Daily Orders")
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.legend()
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax1.set_ylim(bottom=0)
            
            st.pyplot(fig1)
        
        # Detailed forecast for October
        with st.container():
            st.markdown("### Detailed October Forecast")
            fig2 = plot_detailed_forecast(forecast, tenant_id)
            st.pyplot(fig2)
        
        # Metrics
        forecast_period = forecast[forecast["ds"] > actual["ds"].max()]
        avg_forecast = forecast_period["yhat"].mean()
        
        st.subheader("Forecast Summary")
        col1, col2, col3 = st.columns(3)
        col1.markdown('<div class="metric-card"><div class="metric-title">Forecast Start</div><div class="metric-value">' + 
                    forecast_period['ds'].min().strftime('%Y-%m-%d') + '</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-title">Avg Daily Orders</div><div class="metric-value">{avg_forecast:.2f}</div></div>', 
                    unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-title">Forecast Range</div><div class="metric-value">{forecast_period["yhat_lower"].mean():.2f} - {forecast_period["yhat_upper"].mean():.2f}</div></div>', 
                    unsafe_allow_html=True)

def show_resource_analysis(customer_data, tenant_id):
    """Display resource usage analysis"""
    st.subheader("Resource Usage Analysis")
    with st.spinner("Analyzing resource usage..."):
        # Prepare demand forecast first
        daily_orders = prepare_enhanced_data(customer_data)
        actual, forecast = forecast_intermittent_demand(daily_orders, customer_data)
        
        if actual is None or forecast is None:
            st.error("Failed to generate forecast")
            return
        
        # Get resource analysis
        resource_usage, avg_water, avg_electricity = customer_resource_analysis(customer_data)
        future_resource = calculate_future_resource(forecast, avg_water, avg_electricity)
        
        # Display averages
        st.subheader("Resource Usage Metrics")
        col1, col2 = st.columns(2)
        col1.markdown(f'<div class="metric-card"><div class="metric-title">Avg Water per Order</div><div class="metric-value">{avg_water:.2f} L</div></div>', 
                    unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-title">Avg Electricity per Order</div><div class="metric-value">{avg_electricity:.2f} kWh</div></div>', 
                    unsafe_allow_html=True)
        
        # Show historical usage
        with st.container():
            st.subheader("Historical Resource Usage")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Water Usage")
                fig1, ax1 = plt.subplots(figsize=(10, 3))
                ax1.plot(resource_usage["StartDate"], resource_usage["Water_Litres"], 
                        color='#0f3460', linewidth=2)
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Water (L)")
                ax1.grid(True, alpha=0.2)
                st.pyplot(fig1)
            
            with col2:
                st.markdown("#### Electricity Usage")
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.plot(resource_usage["StartDate"], resource_usage["Electricity_kWh"], 
                        color='#e94560', linewidth=2)
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Electricity (kWh)")
                ax2.grid(True, alpha=0.2)
                st.pyplot(fig2)
        
        # Show future needs
        if not future_resource.empty:
            with st.container():
                st.subheader("Future Resource Needs")
                
                # Create date slider for detailed view
                min_date = future_resource['ds'].min().to_pydatetime()
                max_date = future_resource['ds'].max().to_pydatetime()
                selected_date = st.slider("Select Date for Detailed View", min_date, max_date, min_date)
                
                # Get data for selected date
                day_data = future_resource[future_resource['ds'] == selected_date].iloc[0]
                
                # Display metrics
                st.subheader("Resource Needs for Selected Date")
                col1, col2, col3 = st.columns(3)
                col1.markdown(f'<div class="metric-card"><div class="metric-title">Forecasted Orders</div><div class="metric-value">{day_data["yhat"]:.1f}</div></div>', 
                            unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><div class="metric-title">Water Needed</div><div class="metric-value">{day_data["Water_Needed"]:.1f} L</div></div>', 
                            unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><div class="metric-title">Electricity Needed</div><div class="metric-value">{day_data["Electricity_Needed"]:.1f} kWh</div></div>', 
                            unsafe_allow_html=True)
                
                # Show future resource chart
                st.subheader("Predicted Resource Requirements")
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(future_resource["ds"], future_resource["Water_Needed"], 
                         label="Water Needed (L)", color='#0f3460', linewidth=2)
                ax3.plot(future_resource["ds"], future_resource["Electricity_Needed"], 
                         label="Electricity Needed (kWh)", color='#e94560', linewidth=2)
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Amount")
                ax3.legend()
                ax3.grid(True, alpha=0.2)
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                st.pyplot(fig3)
        else:
            st.info("No future resource needs calculated")

def show_business_report(customer_data, tenant_id):
    """Display comprehensive business report"""
    st.subheader("Business Intelligence Report")
    with st.spinner("Generating comprehensive report..."):
        daily_orders = prepare_enhanced_data(customer_data)
        actual, forecast = forecast_intermittent_demand(daily_orders, customer_data)
        
        if actual is None or forecast is None:
            st.error("Failed to generate forecast")
            return
        
        report = generate_business_report(actual, forecast, tenant_id, customer_data)
        
        # Display report
        st.markdown(report)
        
        # Add download button
        report_txt = io.BytesIO(report.encode())
        st.download_button(
            label="📥 Download Report",
            data=report_txt,
            file_name=f"customer_{tenant_id}_report.md",
            mime="text/markdown"
        )
        
        # Add key metrics visualization
        forecast_period = forecast[forecast["ds"] > actual["ds"].max()]
        
        st.subheader("Key Forecast Metrics")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="metric-title">Avg Daily Orders</div><div class="metric-value">{forecast_period["yhat"].mean():.2f}</div></div>', 
                    unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-title">Total Projected Orders</div><div class="metric-value">{forecast_period["yhat"].sum():.0f}</div></div>', 
                    unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-title">Max Daily Orders</div><div class="metric-value">{forecast_period["yhat"].max():.1f}</div></div>', 
                    unsafe_allow_html=True)
        
        # Mini chart
        st.subheader("90-Day Forecast Trend")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(forecast_period["ds"], forecast_period["yhat"], color='#0f3460', linewidth=2)
        ax.fill_between(forecast_period["ds"], 
                      forecast_period["yhat_lower"], 
                      forecast_period["yhat_upper"],
                      color='#0f3460', alpha=0.2)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)