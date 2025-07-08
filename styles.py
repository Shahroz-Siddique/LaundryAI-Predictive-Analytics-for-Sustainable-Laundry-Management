import streamlit as st
CUSTOM_CSS = """
<style>
        :root {
            --primary: #1a1a2e;
            --secondary: #0f3460;
            --accent: #e94560;
            --light: #f5f7fa;
            --gray: #6c757d;
            --dark: #121212;
            --bg: #ffffff;
        }

        .stApp {
            background-color: var(--light);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--dark);
        }

        /* Main containers */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Headings */
        h1 {
            color: var(--primary);
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 0.3rem;
            margin-bottom: 1.2rem;
        }

        h2, h3 {
            color: var(--primary);
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 0.2rem;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--primary);
            color: white;
        }

        [data-testid="stSidebar"] .st-bd {
            padding: 1rem;
        }

        [data-testid="stSidebar"] .stMarkdown h2 {
            color: #f0f0f0;
            border-bottom: 1px solid var(--accent);
        }

        [data-testid="stSidebar"] .stButton > button {
            background-color: var(--secondary);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.6rem;
            margin: 0.5rem 0;
            transition: all 0.2s ease-in-out;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #073b7a;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }

        /* Input elements */
        .stTextInput input, .stNumberInput input {
            border-radius: 6px;
            padding: 0.5rem;
            border: 1px solid #ccc;
        }

        .stSlider .thumb {
            background-color: var(--accent) !important;
        }

        /* Tabs */
        .stTabs [role="tablist"] {
            gap: 10px;
            margin-bottom: 1rem;
        }

        .stTabs [role="tab"] {
            background: #f0f0f0;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            color: var(--primary);
            border: none;
        }

        .stTabs [aria-selected="true"] {
            background: var(--secondary) !important;
            color: white !important;
        }

        /* Cards & Metrics */
        .metric-card {
            background: var(--bg);
            border-left: 4px solid var(--secondary);
            border-radius: 8px;
            padding: 1.2rem 1rem;
            box-shadow: 0 3px 6px rgba(0,0,0,0.05);
            text-align: center;
        }

        .metric-title {
            font-size: 0.95rem;
            color: var(--gray);
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--primary);
        }

        /* Alerts */
        .alert-box {
            background: #fff5f5;
            border-left: 4px solid var(--accent);
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }

        .success-box {
            background: #e6f4ea;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }

        .info-box {
            background: #e7f3fe;
            border-left: 4px solid var(--secondary);
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }

        /* Tables/DataFrames */
        .stDataFrame {
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
    </style>
"""

def apply_custom_styles():
    """Apply custom CSS styles to the app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def get_color_palette():
    """Return consistent color palette for visualizations"""
    return {
        'primary': '#1a1a2e',
        'secondary': '#0f3460',
        'accent': '#e94560',
        'light': '#f5f7fa',
        'dark': '#121212'
    }