import streamlit as st
from datetime import datetime
from customer_analysis import customer_section
from laundry_analysis import laundry_section
from resources import load_data
import styles

def main():
    st.set_page_config(
        page_title="Laundry Analytics Dashboard", 
        layout="wide", 
        page_icon="🧺"
    )
    styles.apply_custom_styles()
    
    st.title("🧺 Laundry Service Analytics Dashboard")
    df = load_data()
    
    # Initialize session state
    if 'main_section' not in st.session_state:
        st.session_state.main_section = None
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Laundry Dashboard")
        st.divider()
        
        col1, col2 = st.columns(2)
        if col1.button("👤 Customer", use_container_width=True):
            st.session_state.main_section = "customer"
            st.subheader("Customer Analysis")
            if st.button("📊 Historical", use_container_width=True):
                st.session_state.customer_subsection = "historical"
            if st.button("🔮 Forecast", use_container_width=True):
                st.session_state.customer_subsection = "demand"
            if st.button("💧 Resources", use_container_width=True):
                st.session_state.customer_subsection = "resource"
            if st.button("📝 Report", use_container_width=True):
                st.session_state.customer_subsection = "report"
        
        if col2.button("🏭 Laundry", use_container_width=True):
            st.session_state.main_section = "laundry"
            if st.button("📈 Peak Forecast", use_container_width=True):
                st.session_state.laundry_subsection = "forecast"
            if st.button("🚨 Peak Alerts", use_container_width=True):
                st.session_state.laundry_subsection = "alert"
            if st.button("⚡ Resources", use_container_width=True):
                st.session_state.laundry_subsection = "consumption"
        
        st.divider()
        st.markdown("**System Status:**")
        st.success("Operational")
        st.markdown(f"**Data Records:** {len(df)}")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Main content routing
    if st.session_state.main_section == "customer":
        customer_section(df)
    elif st.session_state.main_section == "laundry":
        laundry_section(df)
    else:
        show_welcome_screen(df)

def show_welcome_screen(df):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3081/3081519.png", width=200)
    with col2:
        st.header("Welcome to Laundry Analytics")
        st.markdown("""
        <div class="info-box">
            <p>This dashboard helps you analyze customer behavior and laundry operations with:</p>
            <ul>
                <li>📊 Customer demand forecasting</li>
                <li>🔍 Resource usage analysis</li>
                <li>🚨 Peak day alerts</li>
                <li>📈 Operational efficiency metrics</li>
            </ul>
            <p>Get started by selecting an analysis type in the sidebar 👉</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👤 Customer Insights</h3>
            <p>Analyze historical patterns, forecast demand, and optimize inventory</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏭 Laundry Operations</h3>
            <p>Identify peak days, optimize resources, and detect anomalies</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Business Intelligence</h3>
            <p>Generate comprehensive reports with actionable recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Sample Data Overview")
    st.dataframe(df.head(10))

if __name__ == "__main__":
    main()