# dashboard_app.py - Enhanced User Interface

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date # Import date for default date inputs
from supplier_recommendation import SupplierRecommendationEngine

# Configuration
DASHBOARD_DATA_PATH = 'dashboard_data/'
SOURCE_DATA_PATH = 'data/'
FORECASTS_FILE = 'fact_demand_forecasts.csv'
METRICS_FILE = 'metrics_forecast_accuracy.csv'
PRODUCTS_FILE = 'dim_products.csv'
LOCATIONS_FILE = 'dim_locations.csv'
INVENTORY_INSIGHTS_FILE = 'inventory_insights.csv'
SUPPLIERS_FILE = 'dim_suppliers.csv'
PURCHASES_FILE = 'fact_purchases.csv'
DEMAND_FILE = 'demand.csv'

# Set Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Supply Chain Optimization Dashboard", # More descriptive title
    page_icon="📈"
)

# Function to load data 
@st.cache_data # Cache data to avoid reloading on every rerun
def load_data(file_path, parse_dates=None, id_columns=None):
    """
    Loads a CSV file into a DataFrame, optionally parsing dates and converting
    specified ID columns to string type for consistent merging.
    """
    if not os.path.exists(file_path):
        st.error(f"Data file not found: `{file_path}`. Please ensure all preceding data generation and analysis scripts have run successfully.")
        st.stop() # Stop the app if crucial data is missing
    
    try:
        df = pd.read_csv(file_path, parse_dates=parse_dates)
        
        # Convert specified ID columns to string type for consistent merges
        if id_columns:
            for col in id_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading `{os.path.basename(file_path)}`: {e}. Please check the file content and format.")
        st.stop()


# Load all datasets
# Use st.spinner for better user feedback during loading
with st.spinner("Loading data... This might take a moment."):
    forecasts_df = load_data(os.path.join(DASHBOARD_DATA_PATH, FORECASTS_FILE), 
                             parse_dates=['forecast_date'], 
                             id_columns=['product_id', 'location_id'])
    forecasts_df.rename(columns={'forecast_date': 'ds', 'forecasted_units': 'yhat'}, inplace=True)

    metrics_df = load_data(os.path.join(DASHBOARD_DATA_PATH, METRICS_FILE), 
                           id_columns=['product_id', 'location_id'])

    historical_demand_df = load_data(os.path.join(SOURCE_DATA_PATH, DEMAND_FILE), 
                                     parse_dates=['date'], 
                                     id_columns=['product_id', 'location_id'])
    historical_demand_df.rename(columns={'date': 'ds', 'units_sold': 'y'}, inplace=True)

    products_df = load_data(os.path.join(SOURCE_DATA_PATH, PRODUCTS_FILE), 
                            id_columns=['product_id'])
    locations_df = load_data(os.path.join(SOURCE_DATA_PATH, LOCATIONS_FILE), 
                             id_columns=['location_id'])
    suppliers_df = load_data(os.path.join(SOURCE_DATA_PATH, SUPPLIERS_FILE), 
                             id_columns=['supplier_id'])

    inventory_insights_df = load_data(os.path.join(DASHBOARD_DATA_PATH, INVENTORY_INSIGHTS_FILE), 
                                      id_columns=['product_id', 'location_id'])
    fact_purchases_df = load_data(os.path.join(SOURCE_DATA_PATH, PURCHASES_FILE), 
                                  parse_dates=['order_date', 'delivery_date'], 
                                  id_columns=['purchase_id', 'product_id', 'location_id', 'supplier_id'])


# Data Preprocessing for Dashboard
# Merge product name into forecasts_df for display
if not products_df.empty and not forecasts_df.empty:
    products_info = products_df[['product_id', 'product_name']].copy()
    forecasts_df = pd.merge(forecasts_df, products_info, on='product_id', how='left')
    forecasts_df['product_name'].fillna('Unknown Product', inplace=True)
else:
    st.warning("Product dimension data (`dim_products.csv`) not found or is empty. Product names will not be available in forecasts.")
    if not forecasts_df.empty:
        forecasts_df['product_name'] = 'N/A'

# Prepare historical demand with product name
if not products_df.empty and not historical_demand_df.empty:
    historical_demand_df_processed = pd.merge(historical_demand_df, products_info, on='product_id', how='left')
    historical_demand_df_processed['product_name'].fillna('Unknown Product', inplace=True)
else:
    st.warning("Historical demand data or product dimension data is empty. Some historical plots may be incomplete.")
    historical_demand_df_processed = historical_demand_df.copy()

# Merge product name into inventory insights for display
if not products_df.empty and not inventory_insights_df.empty:
    inventory_insights_df = pd.merge(inventory_insights_df, products_info, on='product_id', how='left')
    inventory_insights_df['product_name'].fillna('Unknown Product', inplace=True)

# Merge supplier name into purchases for display
if not suppliers_df.empty and not fact_purchases_df.empty:
    suppliers_info = suppliers_df[['supplier_id', 'supplier_name']].copy()
    fact_purchases_df = pd.merge(fact_purchases_df, suppliers_info, on='supplier_id', how='left')
    fact_purchases_df['supplier_name'].fillna('Unknown Supplier', inplace=True)

# Get unique product, location, and supplier IDs for sidebar filters
# Ensure these lists are populated even if dataframes are empty to prevent errors
unique_products = sorted(products_df['product_id'].unique().tolist()) if not products_df.empty else []
unique_locations = sorted(locations_df['location_id'].unique().tolist()) if not locations_df.empty else []
unique_suppliers = sorted(suppliers_df['supplier_id'].unique().tolist()) if not suppliers_df.empty else []


# --- Dashboard Header ---
st.title("📊 Supply Chain Optimization Dashboard") # Updated title for main content
st.markdown("A comprehensive tool for **Demand Forecasting**, **Inventory Management**, and **Supplier Analytics**.")

# --- Sidebar Filters ---
st.sidebar.header("⚙️ Dashboard Filters")

# Product and Location Filters
st.sidebar.subheader("Product & Location Selection")
selected_product = st.sidebar.selectbox(
    "Select Product ID",
    options=unique_products if unique_products else ["No Products Available"],
    disabled=not unique_products,
    help="Choose a specific product to analyze its demand, inventory, and supplier interactions."
)

selected_location = st.sidebar.selectbox(
    "Select Location ID",
    options=unique_locations if unique_locations else ["No Locations Available"],
    disabled=not unique_locations,
    help="Choose a specific operational location (e.g., warehouse, factory) for analysis."
)

# Date Range Filters
st.sidebar.subheader("📅 Date Range Selection")

# Determine min/max dates from relevant dataframes for default values
# Use .dt.date to get Python date objects for st.date_input
min_date_hist = historical_demand_df['ds'].min().date() if not historical_demand_df.empty else date(2022, 1, 1)
max_date_hist = historical_demand_df['ds'].max().date() if not historical_demand_df.empty else date(2023, 12, 31)

min_date_forecast = forecasts_df['ds'].min().date() if not forecasts_df.empty else date(2024, 1, 1)
max_date_forecast = forecasts_df['ds'].max().date() if not forecasts_df.empty else date(2025, 12, 31)

min_overall_date = min(min_date_hist, min_date_forecast)
max_overall_date = max(max_date_hist, max_date_forecast)

# Ensure min_overall_date is before max_overall_date for default
if min_overall_date > max_overall_date:
    min_overall_date, max_overall_date = max_overall_date, min_overall_date

start_date = st.sidebar.date_input(
    "Start Date",
    value=min_overall_date,
    min_value=min_overall_date,
    max_value=max_overall_date,
    help="Select the beginning of the period for analysis."
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_overall_date,
    min_value=min_overall_date,
    max_value=max_overall_date,
    help="Select the end of the period for analysis."
)

if start_date > end_date:
    st.sidebar.error("⚠️ Error: End date must be after start date. Please adjust your selection.")
    st.stop() # Stop execution if dates are invalid

# Supplier Filter (for Purchases tab)
st.sidebar.subheader("📦 Supplier Filter (Purchases Tab)")
selected_supplier = st.sidebar.selectbox(
    "Select Supplier ID",
    options=['All'] + unique_suppliers,
    index=0,
    help="Filter purchase orders to view performance of specific suppliers."
)


# Main Content Area - Using Tabs for Clear Navigation
tab_forecast, tab_inventory, tab_supplier, tab_recommendation = st.tabs(["📈 Demand Forecast", "📦 Inventory Management", "🚚 Supplier Analytics", "🏆 Supplier Recommendation"])

with tab_forecast:
    st.header(f"Demand Forecast for Product: **{selected_product}** at Location: **{selected_location}**")
    
    current_product_name_forecast = forecasts_df[
        (forecasts_df['product_id'] == selected_product) &
        (forecasts_df['location_id'] == selected_location)
    ]['product_name'].iloc[0] if not forecasts_df[(forecasts_df['product_id'] == selected_product) & (forecasts_df['location_id'] == selected_location)].empty else 'N/A'
    st.markdown(f"**Product Name:** `{current_product_name_forecast}`")

    # Filter data based on selections and date range
    filtered_forecasts = forecasts_df[
        (forecasts_df['product_id'] == selected_product) &
        (forecasts_df['location_id'] == selected_location) &
        (forecasts_df['ds'].dt.date >= start_date) &
        (forecasts_df['ds'].dt.date <= end_date)
    ].sort_values('ds')

    filtered_metrics = metrics_df[
        (metrics_df['product_id'] == selected_product) &
        (metrics_df['location_id'] == selected_location)
    ]

    filtered_historical = historical_demand_df_processed[
        (historical_demand_df_processed['product_id'] == selected_product) &
        (historical_demand_df_processed['location_id'] == selected_location) &
        (historical_demand_df_processed['ds'].dt.date >= start_date) &
        (historical_demand_df_processed['ds'].dt.date <= end_date)
    ].sort_values('ds')

    st.subheader("📊 Forecast Accuracy Metrics")
    if not filtered_metrics.empty:
        col1, col2, col3 = st.columns(3)
        mae = filtered_metrics[filtered_metrics['Metric'] == 'MAE']['Value'].iloc[0] if 'MAE' in filtered_metrics['Metric'].values else 'N/A'
        rmse = filtered_metrics[filtered_metrics['Metric'] == 'RMSE']['Value'].iloc[0] if 'RMSE' in filtered_metrics['Metric'].values else 'N/A'
        mape = filtered_metrics[filtered_metrics['Metric'] == 'MAPE']['Value'].iloc[0] if 'MAPE' in filtered_metrics['Metric'].values else 'N/A'

        with col1:
            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}" if isinstance(mae, (int, float)) else mae, help="Average absolute difference between actual and forecasted values.")
        with col2:
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}" if isinstance(rmse, (int, float)) else rmse, help="Measures the magnitude of the errors, penalizing larger errors more.")
        with col3:
            st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%" if isinstance(mape, (int, float)) else mape, help="Average percentage difference between actual and forecasted values. Easier for business interpretation.")
    else:
        st.info("ℹ️ No accuracy metrics available for the selected combination. This might be due to insufficient historical data for validation.")

    st.subheader("📈 Demand Trend & Forecast")

    if not filtered_forecasts.empty or not filtered_historical.empty:
        plot_df_historical = filtered_historical.rename(columns={'ds': 'Date', 'y': 'Units'})
        plot_df_forecast = filtered_forecasts.rename(columns={'ds': 'Date', 'yhat': 'Units'})

        fig = go.Figure()

        if not plot_df_historical.empty:
            fig.add_trace(go.Scatter(
                x=plot_df_historical['Date'],
                y=plot_df_historical['Units'],
                mode='lines',
                name='Historical Demand',
                line=dict(color='blue')
            ))
        
        if not plot_df_forecast.empty:
            fig.add_trace(go.Scatter(
                x=plot_df_forecast['Date'],
                y=plot_df_forecast['Units'],
                mode='lines',
                name='Forecasted Demand',
                line=dict(color='red', dash='dot')
            ))

            if 'yhat_lower' in plot_df_forecast.columns and 'yhat_upper' in plot_df_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([plot_df_forecast['Date'], plot_df_forecast['Date'].iloc[::-1]]),
                    y=pd.concat([plot_df_forecast['yhat_upper'], plot_df_forecast['yhat_lower'].iloc[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))

        fig.update_layout(
            title='Historical and Forecasted Units Sold Over Time',
            xaxis_title='Date',
            yaxis_title='Units Sold',
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No forecast or historical data available for the selected combination and date range. Please adjust filters.")

    # Raw Forecast Data Table (inside an expander for cleanliness)
    with st.expander("📋 View Raw Forecast Data"):
        if not filtered_forecasts.empty:
            display_columns = ['product_name', 'product_id', 'location_id', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']
            display_df = filtered_forecasts[display_columns].rename(columns={'ds': 'Forecast Date', 'yhat': 'Forecasted Units'})
            st.dataframe(display_df.set_index('Forecast Date'))
        else:
            st.info("ℹ️ No raw forecast data to display for the selected date range.")

with tab_inventory:
    st.header(f"Inventory Management for Product: **{selected_product}** at Location: **{selected_location}**")

    # Overall Inventory Status Metrics
    if not inventory_insights_df.empty:
        total_sku_locations = inventory_insights_df.shape[0]
        reorder_needed = inventory_insights_df[inventory_insights_df['inventory_status'] == 'Reorder Needed'].shape[0]
        critical_stock = inventory_insights_df[inventory_insights_df['inventory_status'] == 'Critical (Below Safety Stock)'].shape[0]
        out_of_stock = inventory_insights_df[inventory_insights_df['inventory_status'] == 'Out of Stock'].shape[0]
        overstock = inventory_insights_df[inventory_insights_df['inventory_status'] == 'Potential Overstock'].shape[0]
        
        st.subheader("📊 Overall Inventory Status Summary (All SKU-Locations)")
        col_inv1, col_inv2, col_inv3, col_inv4, col_inv5 = st.columns(5)
        with col_inv1: st.metric("Total SKU-Locations", total_sku_locations, help="Total unique product-location combinations analyzed.")
        with col_inv2: st.metric("Reorder Needed", reorder_needed, help="Number of SKU-locations where current stock is below the Reorder Point.")
        with col_inv3: st.metric("Critical Stock", critical_stock, help="Number of SKU-locations where current stock is below the Safety Stock level.")
        with col_inv4: st.metric("Out of Stock", out_of_stock, help="Number of SKU-locations with zero current stock.")
        with col_inv5: st.metric("Overstock", overstock, help="Number of SKU-locations with potentially excessive stock levels.")
        
        # Display specific inventory insights for selected product/location
        st.subheader(f"Detailed Inventory Metrics for `{selected_product}` at `{selected_location}`")
        filtered_inventory_insights = inventory_insights_df[
            (inventory_insights_df['product_id'] == selected_product) &
            (inventory_insights_df['location_id'] == selected_location)
        ]

        if not filtered_inventory_insights.empty:
            inv_row = filtered_inventory_insights.iloc[0]
            st.markdown(f"**Product Name:** `{inv_row['product_name']}`")
            
            col_detail1, col_detail2, col_detail3 = st.columns(3)
            with col_detail1: st.metric("Current Stock", inv_row['current_stock'])
            with col_detail2: st.metric("Avg Daily Demand", f"{inv_row['avg_daily_demand']:.2f}")
            with col_detail3: st.metric("Avg Lead Time", f"{inv_row['avg_lead_time_days']} days")
            
            col_detail4, col_detail5, col_detail6 = st.columns(3)
            with col_detail4: st.metric("Calculated Safety Stock", inv_row['safety_stock'], help="Buffer stock to prevent stockouts during lead time variations.")
            with col_detail5: st.metric("Calculated Reorder Point", inv_row['reorder_point'], help="The inventory level at which a new order should be placed.")
            with col_detail6: st.metric("Recommended Reorder Quantity", inv_row['reorder_quantity'], help="The suggested quantity to order to replenish stock.")
            
            # Highlight status with emojis
            status_text = inv_row['inventory_status']
            if status_text == "Out of Stock":
                st.error(f"**Inventory Status:** {status_text} 🚨 (Immediate action required!)")
            elif status_text == "Critical (Below Safety Stock)":
                st.warning(f"**Inventory Status:** {status_text} 🟠 (Risk of stockout, consider expediting orders.)")
            elif status_text == "Reorder Needed":
                st.info(f"**Inventory Status:** {status_text} 🟡 (Time to place a new order.)")
            elif status_text == "Potential Overstock":
                st.info(f"**Inventory Status:** {status_text} 🔵 (Monitor for excess costs, consider promotions.)")
            else: # Optimal
                st.success(f"**Inventory Status:** {status_text} 🟢 (Inventory levels are healthy.)")

            st.subheader("📈 Current Stock vs. Reorder/Safety Levels")
            chart_data = pd.DataFrame({
                'Metric': ['Current Stock', 'Safety Stock', 'Reorder Point'],
                'Value': [inv_row['current_stock'], inv_row['safety_stock'], inv_row['reorder_point']]
            })
            fig_inv = px.bar(chart_data, x='Metric', y='Value', 
                             title='Comparison of Inventory Levels',
                             color='Metric',
                             color_discrete_map={
                                 'Current Stock': 'darkgreen' if inv_row['current_stock'] > inv_row['reorder_point'] else 'darkorange',
                                 'Safety Stock': 'darkred',
                                 'Reorder Point': 'darkblue'
                             })
            fig_inv.update_layout(yaxis_title="Units")
            st.plotly_chart(fig_inv, use_container_width=True)

            with st.expander("📋 View All Inventory Insights (Filtered)"):
                st.dataframe(filtered_inventory_insights[[
                    'product_name', 'location_id', 'current_stock', 'avg_daily_demand',
                    'avg_lead_time_days', 'safety_stock', 'reorder_point', 'reorder_quantity',
                    'inventory_status'
                ]].set_index('product_name'))
        else:
            st.info("ℹ️ No inventory insights available for the selected product and location. Please ensure the product-location combination exists in `inventory_insights.csv`.")
    else:
        st.info("ℹ️ Inventory insights data not found. Please run `inventory_optimizer.py` first to generate `inventory_insights.csv`.")


with tab_supplier:
    st.header("Supplier Analytics")

    # Apply date filter for purchases
    filtered_purchases = fact_purchases_df[
        (fact_purchases_df['order_date'].dt.date >= start_date) &
        (fact_purchases_df['order_date'].dt.date <= end_date)
    ].copy()

    if selected_supplier != 'All':
        filtered_purchases = filtered_purchases[filtered_purchases['supplier_id'] == selected_supplier]

    if not filtered_purchases.empty:
        st.subheader("📊 Key Supplier Performance Metrics")
        
        on_time_delivery_rate = (filtered_purchases['on_time_delivery'].sum() / len(filtered_purchases)) * 100
        
        merged_purchases_for_lead_time = pd.merge(
            filtered_purchases,
            suppliers_df[['supplier_id', 'lead_time_days']],
            on='supplier_id',
            how='left'
        )
        merged_purchases_for_lead_time['actual_lead_time_days'] = (merged_purchases_for_lead_time['delivery_date'] - merged_purchases_for_lead_time['order_date']).dt.days
        
        avg_actual_lead_time = merged_purchases_for_lead_time['actual_lead_time_days'].mean()
        
        supplier_name_for_metric = "All Suppliers"
        if selected_supplier != 'All' and not suppliers_df[suppliers_df['supplier_id'] == selected_supplier].empty:
            supplier_name_for_metric = suppliers_df[suppliers_df['supplier_id'] == selected_supplier]['supplier_name'].iloc[0]


        col_sup1, col_sup2, col_sup3 = st.columns(3)
        with col_sup1: st.metric("Total Purchase Orders", len(filtered_purchases), help="Total number of purchase orders within the selected date range and supplier filter.")
        with col_sup2: st.metric(f"On-Time Delivery Rate ({supplier_name_for_metric})", f"{on_time_delivery_rate:.2f}%", help="Percentage of purchase orders delivered on or before the planned delivery date.")
        with col_sup3: st.metric(f"Avg Actual Lead Time ({supplier_name_for_metric})", f"{avg_actual_lead_time:.1f} days", help="Average number of days from order placement to actual delivery.")


        st.subheader("📋 Recent Purchase Orders (Filtered)")
        filtered_purchases_display = pd.merge(
            filtered_purchases,
            products_df[['product_id', 'product_name']],
            on='product_id',
            how='left'
        ).sort_values(by='order_date', ascending=False)

        display_cols_purchases = [
            'order_date', 'delivery_date', 'supplier_name', 'product_name', 
            'location_id', 'ordered_quantity', 'unit_price', 'on_time_delivery'
        ]
        st.dataframe(filtered_purchases_display[display_cols_purchases].head(20))
    else:
        st.info("ℹ️ No purchase order data available for the selected criteria and date range. Adjust supplier/date filters or ensure `fact_purchases.csv` is generated.")


with tab_recommendation:
    st.header("🏆 Supplier Recommendation Engine")
    st.markdown(f"**Product:** `{selected_product}` | **Location:** `{selected_location}`")
    
    try:
        # Initialize recommendation engine
        @st.cache_resource
        def get_recommendation_engine():
            return SupplierRecommendationEngine(
                demand_forecasts_path=os.path.join(DASHBOARD_DATA_PATH, FORECASTS_FILE),
                inventory_path=os.path.join(SOURCE_DATA_PATH, INVENTORY_FILE) 
                    if 'INVENTORY_FILE' in globals() else os.path.join(SOURCE_DATA_PATH, 'inventory.csv'),
                suppliers_path=os.path.join(SOURCE_DATA_PATH, SUPPLIERS_FILE),
                products_path=os.path.join(SOURCE_DATA_PATH, PRODUCTS_FILE),
                purchases_path=os.path.join(SOURCE_DATA_PATH, PURCHASES_FILE)
            )
        
        engine = get_recommendation_engine()
        
        # Get supply gap analysis
        gap_analysis = engine.get_supply_gap_analysis(selected_product, selected_location)
        
        # Display supply gap metrics
        st.subheader("📊 Supply Gap Analysis")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Stock", f"{gap_analysis['current_stock']:.0f} units")
        with col2:
            st.metric("7-Day Demand", f"{gap_analysis['demand_7d']:.0f} units")
        with col3:
            st.metric("30-Day Demand", f"{gap_analysis['demand_30d']:.0f} units")
        with col4:
            st.metric("Supply Gap (30d)", f"{gap_analysis['gap_30d']:.0f} units")
        
        # Display urgency status with color coding
        st.markdown("### Urgency Status")
        urgency_text = gap_analysis['urgency']
        
        if gap_analysis['urgency_level'] == 4:
            st.error(urgency_text)
        elif gap_analysis['urgency_level'] == 3:
            st.warning(urgency_text)
        elif gap_analysis['urgency_level'] == 2:
            st.info(urgency_text)
        else:
            st.success(urgency_text)
        
        # Get supplier recommendations
        recommendations_df, context_info = engine.recommend_suppliers(
            selected_product, selected_location, forecast_days=30
        )
        
        if not recommendations_df.empty:
            st.subheader("🏆 Top Supplier Recommendations")
            
            # Create tabs for different view types
            view_all, view_best = st.tabs(["📋 All Suppliers Ranked", "✅ Best Supplier Summary"])
            
            with view_all:
                # Display all suppliers ranked by score
                display_df = recommendations_df[[
                    'supplier_name', 'standard_lead_time', 'estimated_delivery_date',
                    'on_time_rate', 'avg_unit_price', 'estimated_order_cost',
                    'reliability_score', 'speed_score', 'cost_score', 'overall_score'
                ]].copy()
                
                display_df.columns = [
                    'Supplier Name', 'Lead Time (days)', 'Est. Delivery Date',
                    'On-Time Rate (%)', 'Avg Unit Price ($)', 'Est. Order Cost ($)',
                    'Reliability Score', 'Speed Score', 'Cost Score', 'Overall Score'
                ]
                
                # Format numeric columns
                display_df['Est. Delivery Date'] = display_df['Est. Delivery Date'].dt.strftime('%Y-%m-%d')
                display_df['On-Time Rate (%)'] = display_df['On-Time Rate (%)'].apply(lambda x: f"{x:.1f}%")
                display_df['Avg Unit Price ($)'] = display_df['Avg Unit Price ($)'].apply(lambda x: f"${x:.2f}")
                display_df['Est. Order Cost ($)'] = display_df['Est. Order Cost ($)'].apply(lambda x: f"${x:.2f}")
                
                for col in ['Reliability Score', 'Speed Score', 'Cost Score', 'Overall Score']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}/100")
                
                st.dataframe(display_df, use_container_width=True)
            
            with view_best:
                # Show best supplier in detail
                best_supplier = recommendations_df.iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### ✅ Recommended Supplier")
                    st.markdown(f"**{best_supplier['supplier_name']}**")
                    st.markdown(f"ID: `{best_supplier['supplier_id']}`")
                    contact_col = 'contact_person' if 'contact_person' in best_supplier.index else 'contact'
                    if contact_col in best_supplier.index:
                        st.markdown(f"Contact: {best_supplier[contact_col]}")
                
                with col2:
                    st.markdown(f"### � Order & Pricing Details")
                    st.markdown(f"**Required Quantity:** {context_info['required_stock']:.0f} units")
                    st.markdown(f"**Unit Price:** ${best_supplier['avg_unit_price']:.2f}")
                    st.markdown(f"**Total Estimated Cost:** ${best_supplier['estimated_order_cost']:.2f}")
                                
                # Delivery Timeline
                st.markdown("### 📅 Delivery Timeline")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lead Time", f"{best_supplier['standard_lead_time']:.0f} days")
                with col2:
                    st.metric("Est. Delivery", 
                             best_supplier['estimated_delivery_date'].strftime('%Y-%m-%d'))
                with col3:
                    days_until = (best_supplier['estimated_delivery_date'] - pd.Timestamp.now()).days
                    st.metric("Days Until Delivery", f"{max(0, days_until)} days")
                
                # Performance Metrics
                st.markdown("### 📊 Supplier Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("On-Time Delivery Rate", 
                             f"{best_supplier['on_time_rate']:.1f}%")
                with col2:
                    st.metric("Avg Actual Lead Time", 
                             f"{best_supplier['avg_actual_lead_time']:.1f} days")
                with col3:
                    st.metric("Total Orders Placed", 
                             f"{best_supplier['total_orders']:.0f}")
                
                # Score Breakdown
                st.markdown("### 🎯 Score Breakdown")
                scores_data = {
                    'Metric': ['Reliability', 'Speed', 'Cost', 'Overall'],
                    'Score': [
                        best_supplier['reliability_score'],
                        best_supplier['speed_score'],
                        best_supplier['cost_score'],
                        best_supplier['overall_score']
                    ]
                }
                scores_df = pd.DataFrame(scores_data)
                
                fig_scores = go.Figure(data=[
                    go.Bar(x=scores_df['Metric'], y=scores_df['Score'],
                           text=scores_df['Score'].apply(lambda x: f"{x:.1f}"),
                           textposition='outside',
                           marker=dict(color=['#00AA00', '#0066FF', '#FF6600', '#FF0000']))
                ])
                fig_scores.update_layout(
                    title='Supplier Score Breakdown',
                    yaxis_title='Score (0-100)',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_scores, use_container_width=True)
                
                # Comparison with other top suppliers
                if len(recommendations_df) > 1:
                    st.markdown("### 🔄 Comparison with Other Top Suppliers")
                    
                    comparison_df = recommendations_df.head(3)[[
                        'supplier_name', 'overall_score', 'on_time_rate', 
                        'standard_lead_time', 'avg_unit_price'
                    ]].copy()
                    comparison_df.columns = ['Supplier', 'Overall Score', 'On-Time %', 'Lead Time (days)', 'Unit Price ($)']
                    
                    fig_compare = go.Figure()
                    for idx, row in comparison_df.iterrows():
                        fig_compare.add_trace(go.Scatterpolar(
                            r=[
                                row['Overall Score'],
                                row['On-Time %'],
                                100 - row['Lead Time (days)'],
                                100 - row['Unit Price ($)'] if row['Unit Price ($)'] < 100 else 10
                            ],
                            theta=['Overall Score', 'On-Time Rate', 'Speed', 'Cost'],
                            fill='toself',
                            name=row['Supplier']
                        ))
                    
                    fig_compare.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        height=500
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.warning("⚠️ No supplier data available for this product. Please check if suppliers data exists.")
    
    except Exception as e:
        st.error(f"Error in Supplier Recommendation Engine: {str(e)}")
        st.info("Please ensure all required data files are generated:\n- `fact_demand_forecasts.csv`\n- `inventory.csv`\n- `fact_purchases.csv`")


st.markdown("---")
st.markdown(f"Developed by Sreenu | Data last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
