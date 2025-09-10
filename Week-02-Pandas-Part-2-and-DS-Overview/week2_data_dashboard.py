#!/usr/bin/env python3
"""
Week 2 Data Science Dashboard

A comprehensive Streamlit dashboard showcasing data cleaning and analysis techniques
using the datasets from Week 2: Food Choices, Salary Survey, and Airbnb Listings.

This dashboard demonstrates:
- Data cleaning techniques
- Handling missing values
- Data type conversions
- Exploratory data analysis
- Interactive visualizations

Author: Data Science Student
Date: 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import re
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Week 2 Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_food_data():
    """Load and preprocess the food choices data."""
    try:
        df = pd.read_csv('data/food_coded.csv')
        
        # Clean and process the data
        # Convert weight to numeric, handling non-numeric values
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        
        # Note: Food dataset doesn't have age column, so we skip age groups for this dataset
        # Age analysis is available in the Salary Survey dataset instead
        
        # Clean gender data (1=Female, 2=Male based on the data)
        df['gender_clean'] = df['Gender'].map({1: 'Female', 2: 'Male'})
        
        # Process comfort food reasons
        df['comfort_food_clean'] = df['comfort_food_reasons'].fillna('No response')
        
        return df
    except FileNotFoundError:
        st.error("Food data file not found. Please ensure food_coded.csv is in the data folder.")
        return None

@st.cache_data
def load_salary_data():
    """Load and preprocess the salary survey data."""
    try:
        # Load the TSV file
        df = pd.read_csv('data/ask_a_manager_salary_survey_2021_responses.tsv', sep='\t')
        
        # Rename columns for easier handling
        df = df.rename(columns={
            "Timestamp": "timestamp",
            "How old are you?": "age",
            "What industry do you work in?": "industry",
            "Job title": "title",
            "If your job title needs additional context, please clarify here:": "title_context",
            "What is your annual salary? (You'll indicate the currency in a later question. If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.)": "salary",
            "How much additional monetary compensation do you get, if any (for example, bonuses or overtime in an average year)? Please only include monetary compensation here, not the value of benefits.": "additional_compensation",
            "Please indicate the currency": "currency",
            'If "Other," please indicate the currency here:': "other_currency",
            "If your income needs additional context, please provide it here:": "salary_context",
            "What country do you work in?": "country",
            "If you're in the U.S., what state do you work in?": "state",
            "What city do you work in?": "city",
            "How many years of professional work experience do you have overall?": "total_yoe",
            "How many years of professional work experience do you have in your field?": "field_yoe",
            "What is your highest level of education completed?": "highest_education_completed",
            "What is your gender?": "gender",
            "What is your race? (Choose all that apply.)": "race"
        })
        
        # Convert salary to numeric
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Clean country data - standardize US variations
        df['country'] = df['country'].replace([
            'United States', 'USA', 'U.S.', 'United States of America', 
            'United State Of America', 'United State', 'Unite States', 'US'
        ], 'US')
        
        # Filter for US data and USD currency for cleaner analysis
        df_usd = df[(df['country'] == 'US') & (df['currency'] == 'USD')].copy()
        
        return df, df_usd
    except FileNotFoundError:
        st.error("Salary data file not found. Please ensure the TSV file is in the data folder.")
        return None, None

@st.cache_data
def load_airbnb_data():
    """Load and preprocess the Airbnb listings data."""
    try:
        df = pd.read_csv('data/listings.csv')
        
        # Price is already numeric in this dataset, no conversion needed
        # df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Convert last_review to datetime
        df['last_review'] = pd.to_datetime(df['last_review'])
        
        # Create price categories
        df['price_category'] = pd.cut(df['price'], 
                                    bins=[0, 100, 200, 300, 500, float('inf')], 
                                    labels=['$0-100', '$100-200', '$200-300', '$300-500', '$500+'])
        
        # Create review score categories
        df['review_category'] = pd.cut(df['number_of_reviews'], 
                                     bins=[0, 10, 50, 100, float('inf')], 
                                     labels=['0-10', '10-50', '50-100', '100+'])
        
        return df
    except FileNotFoundError:
        st.error("Airbnb data file not found. Please ensure listings.csv is in the data folder.")
        return None

@st.cache_data
def load_cities_data():
    """Load the cities JSON data."""
    try:
        with open('data/cities.json', 'r') as f:
            cities_data = json.load(f)
        return pd.DataFrame(cities_data)
    except FileNotFoundError:
        st.warning("Cities data file not found. Some features may not be available.")
        return None

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    st.title("📊 Week 2 Data Science Dashboard")
    st.markdown("**Demonstrating Data Cleaning and Analysis Techniques**")
    st.markdown("---")
    
    # Load all datasets
    food_df = load_food_data()
    salary_df, salary_usd_df = load_salary_data()
    airbnb_df = load_airbnb_data()
    cities_df = load_cities_data()
    
    # Check if data loaded successfully
    if food_df is None and salary_df is None and airbnb_df is None:
        st.error("No data files found. Please ensure the data files are in the correct location.")
        return
    
    # Sidebar for dataset selection
    st.sidebar.header("🔍 Dataset Selection")
    available_datasets = []
    if food_df is not None:
        available_datasets.append("Food Choices")
    if salary_df is not None:
        available_datasets.append("Salary Survey")
    if airbnb_df is not None:
        available_datasets.append("Airbnb Listings")
    if cities_df is not None:
        available_datasets.append("Cities Data")
    
    selected_dataset = st.sidebar.selectbox("Choose Dataset", available_datasets)
    
    # Display dataset info
    st.sidebar.markdown("---")
    if selected_dataset == "Food Choices" and food_df is not None:
        st.sidebar.metric("Total Records", f"{len(food_df):,}")
        st.sidebar.metric("Columns", f"{len(food_df.columns)}")
        st.sidebar.metric("Missing Values", f"{food_df.isnull().sum().sum():,}")
    elif selected_dataset == "Salary Survey" and salary_df is not None:
        st.sidebar.metric("Total Records", f"{len(salary_df):,}")
        st.sidebar.metric("US USD Records", f"{len(salary_usd_df):,}")
        st.sidebar.metric("Columns", f"{len(salary_df.columns)}")
    elif selected_dataset == "Airbnb Listings" and airbnb_df is not None:
        st.sidebar.metric("Total Listings", f"{len(airbnb_df):,}")
        st.sidebar.metric("Neighborhoods", f"{airbnb_df['neighbourhood'].nunique()}")
        st.sidebar.metric("Avg Price", f"${airbnb_df['price'].mean():.0f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview", "🧹 Data Cleaning", "📈 Analysis", "🔍 Insights"
    ])
    
    with tab1:
        st.header("📋 Data Overview")
        
        if selected_dataset == "Food Choices" and food_df is not None:
            st.subheader("Food Choices Dataset")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Info:**")
                st.write(f"- Shape: {food_df.shape}")
                st.write(f"- Memory usage: {food_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.write(f"- Missing values: {food_df.isnull().sum().sum()}")
            
            with col2:
                st.write("**Data Types:**")
                dtype_counts = food_df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} columns")
            
            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(food_df.head(10), use_container_width=True)
            
            # Show missing values heatmap
            st.subheader("Missing Values Pattern")
            missing_data = food_df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.values, 
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Missing Count', 'y': 'Column'}
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        elif selected_dataset == "Salary Survey" and salary_df is not None:
            st.subheader("Salary Survey Dataset")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Full Dataset Info:**")
                st.write(f"- Shape: {salary_df.shape}")
                st.write(f"- Date range: {salary_df['timestamp'].min().date()} to {salary_df['timestamp'].max().date()}")
                st.write(f"- Countries: {salary_df['country'].nunique()}")
            
            with col2:
                st.write("**US USD Subset Info:**")
                st.write(f"- Shape: {salary_usd_df.shape}")
                st.write(f"- Salary range: ${salary_usd_df['salary'].min():,.0f} - ${salary_usd_df['salary'].max():,.0f}")
                st.write(f"- States: {salary_usd_df['state'].nunique()}")
            
            # Show sample data
            st.subheader("Sample Data (US USD subset)")
            display_cols = ['timestamp', 'age', 'industry', 'title', 'salary', 'state', 'gender']
            st.dataframe(salary_usd_df[display_cols].head(10), use_container_width=True)
        
        elif selected_dataset == "Airbnb Listings" and airbnb_df is not None:
            st.subheader("Airbnb Listings Dataset")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Info:**")
                st.write(f"- Shape: {airbnb_df.shape}")
                st.write(f"- Price range: ${airbnb_df['price'].min():,.0f} - ${airbnb_df['price'].max():,.0f}")
                st.write(f"- Room types: {airbnb_df['room_type'].nunique()}")
            
            with col2:
                st.write("**Geographic Coverage:**")
                st.write(f"- Boroughs: {airbnb_df['neighbourhood_group'].nunique()}")
                st.write(f"- Neighborhoods: {airbnb_df['neighbourhood'].nunique()}")
                st.write(f"- Hosts: {airbnb_df['host_id'].nunique()}")
            
            # Show sample data
            st.subheader("Sample Data")
            display_cols = ['name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price', 'number_of_reviews']
            st.dataframe(airbnb_df[display_cols].head(10), use_container_width=True)
    
    with tab2:
        st.header("🧹 Data Cleaning Techniques")
        
        if selected_dataset == "Food Choices" and food_df is not None:
            st.subheader("Food Data Cleaning Examples")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Cleaning - Weight Column:**")
                st.write("Original data types and sample values:")
                weight_sample = food_df[['weight', 'Gender']].head(10)
                st.dataframe(weight_sample)
                
                st.write("**Issues Found:**")
                st.write("- Non-numeric values in weight column")
                st.write("- Missing values (NaN)")
                st.write("- Mixed data types")
            
            with col2:
                st.write("**After Cleaning - Weight Column:**")
                # Clean the weight data
                food_clean = food_df.copy()
                food_clean['weight'] = pd.to_numeric(food_clean['weight'], errors='coerce')
                food_clean['gender_clean'] = food_clean['Gender'].map({1: 'Female', 2: 'Male'})
                
                cleaned_sample = food_clean[['weight', 'gender_clean']].head(10)
                st.dataframe(cleaned_sample)
                
                st.write("**Cleaning Applied:**")
                st.write("- Converted to numeric with error handling")
                st.write("- Created clean gender labels")
                st.write("- Handled missing values appropriately")
            
            # Show data quality metrics
            st.subheader("Data Quality Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Complete Records", f"{food_clean.dropna().shape[0]:,}")
            
            with col2:
                st.metric("Missing Weight Values", f"{food_clean['weight'].isnull().sum():,}")
            
            with col3:
                st.metric("Data Completeness", f"{(1 - food_clean.isnull().sum().sum() / (food_clean.shape[0] * food_clean.shape[1])) * 100:.1f}%")
        
        elif selected_dataset == "Salary Survey" and salary_df is not None:
            st.subheader("Salary Data Cleaning Examples")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Cleaning - Country Column:**")
                country_counts = salary_df['country'].value_counts().head(10)
                st.write("Top 10 country variations:")
                st.dataframe(country_counts.reset_index())
                
                st.write("**Issues Found:**")
                st.write("- Multiple variations of 'United States'")
                st.write("- Inconsistent naming conventions")
                st.write("- Mixed case and abbreviations")
            
            with col2:
                st.write("**After Cleaning - Country Column:**")
                # Clean country data
                salary_clean = salary_df.copy()
                salary_clean['country'] = salary_clean['country'].replace([
                    'United States', 'USA', 'U.S.', 'United States of America', 
                    'United State Of America', 'United State', 'Unite States', 'US'
                ], 'US')
                
                country_clean_counts = salary_clean['country'].value_counts().head(10)
                st.write("Standardized country names:")
                st.dataframe(country_clean_counts.reset_index())
                
                st.write("**Cleaning Applied:**")
                st.write("- Standardized US variations")
                st.write("- Consistent naming convention")
                st.write("- Reduced data fragmentation")
            
            # Show salary cleaning
            st.subheader("Salary Data Processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Salary Conversion:**")
                st.write("Converting string salaries to numeric values...")
                salary_clean['salary'] = pd.to_numeric(salary_clean['salary'], errors='coerce')
                
                salary_stats = salary_clean['salary'].describe()
                st.dataframe(salary_stats)
            
            with col2:
                st.write("**Data Filtering:**")
                usd_data = salary_clean[(salary_clean['country'] == 'US') & (salary_clean['currency'] == 'USD')]
                
                st.write(f"Original records: {len(salary_clean):,}")
                st.write(f"US USD records: {len(usd_data):,}")
                st.write(f"Filtered out: {len(salary_clean) - len(usd_data):,}")
        
        elif selected_dataset == "Airbnb Listings" and airbnb_df is not None:
            st.subheader("Airbnb Data Cleaning Examples")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Cleaning - Price Column:**")
                price_sample = airbnb_df[['price']].head(10)
                st.dataframe(price_sample)
                
                st.write("**Issues Found:**")
                st.write("- Dollar signs ($) in price values")
                st.write("- Comma separators")
                st.write("- String data type instead of numeric")
            
            with col2:
                st.write("**After Cleaning - Price Column:**")
                # Clean price data
                airbnb_clean = airbnb_df.copy()
                airbnb_clean['price'] = airbnb_clean['price'].str.replace('$', '').str.replace(',', '').astype(float)
                
                cleaned_price_sample = airbnb_clean[['price']].head(10)
                st.dataframe(cleaned_price_sample)
                
                st.write("**Cleaning Applied:**")
                st.write("- Removed dollar signs and commas")
                st.write("- Converted to numeric type")
                st.write("- Ready for mathematical operations")
            
            # Show price analysis
            st.subheader("Price Analysis After Cleaning")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Price", f"${airbnb_clean['price'].mean():.0f}")
            
            with col2:
                st.metric("Median Price", f"${airbnb_clean['price'].median():.0f}")
            
            with col3:
                st.metric("Price Range", f"${airbnb_clean['price'].min():.0f} - ${airbnb_clean['price'].max():.0f}")
    
    with tab3:
        st.header("📈 Data Analysis")
        
        if selected_dataset == "Food Choices" and food_df is not None:
            st.subheader("Food Preferences Analysis")
            
            # Clean data for analysis
            food_analysis = food_df.copy()
            food_analysis['weight'] = pd.to_numeric(food_analysis['weight'], errors='coerce')
            food_analysis['gender_clean'] = food_analysis['Gender'].map({1: 'Female', 2: 'Male'})
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gender distribution
                gender_counts = food_analysis['gender_clean'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values, 
                    names=gender_counts.index,
                    title="Gender Distribution"
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            
            with col2:
                # Weight distribution by gender
                weight_data = food_analysis.dropna(subset=['weight', 'gender_clean'])
                fig_weight = px.box(
                    weight_data, 
                    x='gender_clean', 
                    y='weight',
                    title="Weight Distribution by Gender"
                )
                st.plotly_chart(fig_weight, use_container_width=True)
            
            # Comfort food analysis
            st.subheader("Comfort Food Analysis")
            comfort_foods = food_analysis['comfort_food'].value_counts().head(10)
            
            fig_comfort = px.bar(
                x=comfort_foods.values, 
                y=comfort_foods.index,
                orientation='h',
                title="Top 10 Comfort Foods",
                labels={'x': 'Count', 'y': 'Comfort Food'}
            )
            st.plotly_chart(fig_comfort, use_container_width=True)
        
        elif selected_dataset == "Salary Survey" and salary_usd_df is not None:
            st.subheader("Salary Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Salary by industry
                industry_salary = salary_usd_df.groupby('industry')['salary'].mean().sort_values(ascending=False).head(10)
                
                fig_industry = px.bar(
                    x=industry_salary.values, 
                    y=industry_salary.index,
                    orientation='h',
                    title="Average Salary by Industry (Top 10)",
                    labels={'x': 'Average Salary ($)', 'y': 'Industry'}
                )
                st.plotly_chart(fig_industry, use_container_width=True)
            
            with col2:
                # Salary by gender
                gender_salary = salary_usd_df.groupby('gender')['salary'].mean()
                
                fig_gender_salary = px.bar(
                    x=gender_salary.index, 
                    y=gender_salary.values,
                    title="Average Salary by Gender",
                    labels={'x': 'Gender', 'y': 'Average Salary ($)'}
                )
                st.plotly_chart(fig_gender_salary, use_container_width=True)
            
            # Salary distribution
            st.subheader("Salary Distribution")
            
            fig_salary_dist = px.histogram(
                salary_usd_df, 
                x='salary',
                nbins=50,
                title="Salary Distribution",
                labels={'salary': 'Salary ($)', 'count': 'Number of People'}
            )
            st.plotly_chart(fig_salary_dist, use_container_width=True)
            
            # Geographic analysis
            st.subheader("Geographic Salary Analysis")
            
            state_salary = salary_usd_df.groupby('state')['salary'].agg(['mean', 'count']).reset_index()
            state_salary = state_salary[state_salary['count'] >= 10]  # Filter for meaningful sample sizes
            state_salary = state_salary.sort_values('mean', ascending=False).head(15)
            
            fig_state = px.bar(
                state_salary, 
                x='mean', 
                y='state',
                orientation='h',
                title="Average Salary by State (Top 15)",
                labels={'mean': 'Average Salary ($)', 'state': 'State'}
            )
            st.plotly_chart(fig_state, use_container_width=True)
        
        elif selected_dataset == "Airbnb Listings" and airbnb_df is not None:
            st.subheader("Airbnb Market Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price by neighborhood group
                borough_price = airbnb_df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
                
                fig_borough = px.bar(
                    x=borough_price.index, 
                    y=borough_price.values,
                    title="Average Price by Borough",
                    labels={'x': 'Borough', 'y': 'Average Price ($)'}
                )
                st.plotly_chart(fig_borough, use_container_width=True)
            
            with col2:
                # Room type distribution
                room_type_counts = airbnb_df['room_type'].value_counts()
                
                fig_room_type = px.pie(
                    values=room_type_counts.values, 
                    names=room_type_counts.index,
                    title="Room Type Distribution"
                )
                st.plotly_chart(fig_room_type, use_container_width=True)
            
            # Price vs Reviews scatter plot
            st.subheader("Price vs Reviews Analysis")
            
            fig_scatter = px.scatter(
                airbnb_df, 
                x='number_of_reviews', 
                y='price',
                color='neighbourhood_group',
                title="Price vs Number of Reviews",
                labels={'number_of_reviews': 'Number of Reviews', 'price': 'Price ($)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Top neighborhoods by price
            st.subheader("Top Neighborhoods by Average Price")
            
            neighborhood_price = airbnb_df.groupby('neighbourhood')['price'].agg(['mean', 'count']).reset_index()
            neighborhood_price = neighborhood_price[neighborhood_price['count'] >= 10]
            neighborhood_price = neighborhood_price.sort_values('mean', ascending=False).head(15)
            
            fig_neighborhood = px.bar(
                neighborhood_price, 
                x='mean', 
                y='neighbourhood',
                orientation='h',
                title="Average Price by Neighborhood (Top 15)",
                labels={'mean': 'Average Price ($)', 'neighbourhood': 'Neighborhood'}
            )
            st.plotly_chart(fig_neighborhood, use_container_width=True)
    
    with tab4:
        st.header("🔍 Key Insights")
        
        if selected_dataset == "Food Choices" and food_df is not None:
            st.subheader("Food Choices Insights")
            
            # Clean data for insights
            food_insights = food_df.copy()
            food_insights['weight'] = pd.to_numeric(food_insights['weight'], errors='coerce')
            food_insights['gender_clean'] = food_insights['Gender'].map({1: 'Female', 2: 'Male'})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Findings:**")
                st.write("• **Gender Distribution**: The dataset shows a balanced representation of genders")
                st.write("• **Weight Patterns**: Clear differences in weight distribution between genders")
                st.write("• **Comfort Foods**: Chocolate and ice cream are the most common comfort foods")
                st.write("• **Data Quality**: Several columns have missing values that need attention")
            
            with col2:
                st.write("**Data Cleaning Impact:**")
                st.write("• **Weight Column**: Converted from mixed types to numeric")
                st.write("• **Gender Column**: Mapped numeric codes to meaningful labels")
                st.write("• **Missing Values**: Identified patterns in missing data")
                st.write("• **Data Types**: Standardized data types for analysis")
        
        elif selected_dataset == "Salary Survey" and salary_usd_df is not None:
            st.subheader("Salary Survey Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Findings:**")
                st.write("• **Industry Differences**: Significant salary variations across industries")
                st.write("• **Gender Gap**: Observable differences in average salaries by gender")
                st.write("• **Geographic Patterns**: State-level salary variations are substantial")
                st.write("• **Data Quality**: Country standardization reduced data fragmentation")
            
            with col2:
                st.write("**Data Cleaning Impact:**")
                st.write("• **Country Standardization**: Unified US variations into single category")
                st.write("• **Currency Filtering**: Focused on USD for consistent analysis")
                st.write("• **Salary Conversion**: Transformed string salaries to numeric values")
                st.write("• **Data Filtering**: Reduced dataset to most relevant subset")
        
        elif selected_dataset == "Airbnb Listings" and airbnb_df is not None:
            st.subheader("Airbnb Market Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Findings:**")
                st.write("• **Borough Differences**: Manhattan has highest average prices")
                st.write("• **Room Types**: Entire home/apt listings are most common")
                st.write("• **Price-Review Correlation**: Complex relationship between price and reviews")
                st.write("• **Market Concentration**: Certain neighborhoods dominate high-price segments")
            
            with col2:
                st.write("**Data Cleaning Impact:**")
                st.write("• **Price Standardization**: Removed currency symbols and separators")
                st.write("• **Data Type Conversion**: Converted prices to numeric for analysis")
                st.write("• **Categorical Creation**: Built price and review categories")
                st.write("• **Date Processing**: Converted timestamps to datetime objects")
        
        # General insights
        st.subheader("📚 Data Science Learning Outcomes")
        
        st.write("""
        **This dashboard demonstrates essential data science skills:**
        
        1. **Data Loading & Exploration**: Understanding dataset structure and content
        2. **Data Cleaning**: Handling missing values, type conversions, and standardization
        3. **Data Transformation**: Creating derived variables and categories
        4. **Exploratory Data Analysis**: Uncovering patterns and relationships
        5. **Data Visualization**: Creating meaningful charts and graphs
        6. **Data Quality Assessment**: Identifying and addressing data issues
        
        **Key Techniques Shown:**
        - `pd.to_numeric()` with error handling
        - String manipulation and cleaning
        - Categorical data mapping
        - Missing value analysis
        - Data filtering and subsetting
        - Statistical aggregation and grouping
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🎓 Week 2 Data Science Dashboard")
    st.markdown("""
    This dashboard showcases the data cleaning and analysis techniques learned in Week 2 of the Data Science course.
    It demonstrates how to handle real-world data challenges and transform messy data into actionable insights.
    """)

if __name__ == "__main__":
    main()
