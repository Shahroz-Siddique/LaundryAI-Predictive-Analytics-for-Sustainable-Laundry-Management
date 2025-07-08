import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    return pd.read_csv("data/better_laundry_service_dataset.csv", parse_dates=["StartDate"])

def save_report(content, filename, directory="reports"):
    """Save generated reports to files"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath

def load_report(filename, directory="reports"):
    """Load saved reports"""
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def calculate_water_savings(current_usage, optimized_usage):
    """Calculate potential water savings"""
    return current_usage - optimized_usage

def calculate_energy_savings(current_usage, optimized_usage):
    """Calculate potential energy savings"""
    return current_usage - optimized_usage