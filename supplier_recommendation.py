"""
Supplier Recommendation Engine
Recommends the best supplier based on:
- Forecasted demand
- Current inventory levels
- Lead time vs urgency
- Cost comparison
- Supplier reliability (on-time delivery rate)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys

# Set encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class SupplierRecommendationEngine:
    def __init__(self, demand_forecasts_path, inventory_path, 
                 suppliers_path, products_path, purchases_path):
        """
        Initialize the recommendation engine with data paths
        
        Parameters:
        -----------
        demand_forecasts_path : str
            Path to demand forecasts CSV (from demand_forecaster.py)
        inventory_path : str
            Path to current inventory CSV
        suppliers_path : str
            Path to suppliers master data CSV
        products_path : str
            Path to products master data CSV
        purchases_path : str
            Path to historical purchases CSV
        """
        self.forecasts_df = pd.read_csv(demand_forecasts_path)
        self.inventory_df = pd.read_csv(inventory_path)
        self.suppliers_df = pd.read_csv(suppliers_path)
        self.products_df = pd.read_csv(products_path)
        self.purchases_df = pd.read_csv(purchases_path)
        
        # Convert date columns
        self.forecasts_df['forecast_date'] = pd.to_datetime(self.forecasts_df['forecast_date'])
        self.inventory_df['inventory_date'] = pd.to_datetime(self.inventory_df['inventory_date'])
        self.purchases_df['order_date'] = pd.to_datetime(self.purchases_df['order_date'])
        self.purchases_df['delivery_date'] = pd.to_datetime(self.purchases_df['delivery_date'])
        
    def calculate_supplier_metrics(self):
        """
        Calculate reliability metrics for each supplier:
        - On-time delivery rate
        - Average actual lead time
        - Cost competitiveness
        """
        supplier_metrics = []
        
        for supplier_id in self.suppliers_df['supplier_id'].unique():
            supplier_purchases = self.purchases_df[
                self.purchases_df['supplier_id'] == supplier_id
            ]
            
            if supplier_purchases.empty:
                continue
                
            # On-time delivery rate
            on_time_deliveries = supplier_purchases['on_time_delivery'].sum()
            total_deliveries = len(supplier_purchases)
            on_time_rate = (on_time_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0
            
            # Average actual lead time
            supplier_purchases['actual_lead_time'] = (
                supplier_purchases['delivery_date'] - supplier_purchases['order_date']
            ).dt.days
            avg_actual_lead_time = supplier_purchases['actual_lead_time'].mean()
            
            # Average unit price
            avg_unit_price = supplier_purchases['unit_price'].mean()
            
            # Standard lead time from suppliers_df
            standard_lead_time = self.suppliers_df[
                self.suppliers_df['supplier_id'] == supplier_id
            ]['lead_time_days'].iloc[0]
            
            supplier_metrics.append({
                'supplier_id': supplier_id,
                'on_time_rate': on_time_rate,
                'avg_actual_lead_time': avg_actual_lead_time,
                'standard_lead_time': standard_lead_time,
                'avg_unit_price': avg_unit_price,
                'total_orders': total_deliveries
            })
        
        return pd.DataFrame(supplier_metrics)
    
    def get_demand_over_period(self, product_id, location_id, days=30):
        """
        Get total forecasted demand over next N days
        
        Parameters:
        -----------
        product_id : str
        location_id : str
        days : int
            Number of days to forecast (default 30)
            
        Returns:
        --------
        float : Total forecasted units over period
        """
        today = datetime.now()
        future_date = today + timedelta(days=days)
        
        future_demand = self.forecasts_df[
            (self.forecasts_df['product_id'] == product_id) &
            (self.forecasts_df['location_id'] == location_id) &
            (self.forecasts_df['forecast_date'] >= today) &
            (self.forecasts_df['forecast_date'] <= future_date)
        ]['forecasted_units'].sum()
        
        return future_demand
    
    def get_current_stock(self, product_id, location_id):
        """Get current inventory for a product-location"""
        current_stock = self.inventory_df[
            (self.inventory_df['product_id'] == product_id) &
            (self.inventory_df['location_id'] == location_id)
        ]
        
        if current_stock.empty:
            return 0
        
        return current_stock['quantity_on_hand'].iloc[0]
    
    def recommend_suppliers(self, product_id, location_id, 
                           forecast_days=30, urgency_threshold=7):
        """
        Recommend suppliers for a specific product-location combination
        
        Parameters:
        -----------
        product_id : str
        location_id : str
        forecast_days : int
            Days to forecast demand over (default 30)
        urgency_threshold : int
            Lead time threshold to flag as urgent (default 7 days)
            
        Returns:
        --------
        DataFrame : Ranked supplier recommendations with detailed metrics
        """
        
        # Get metrics for all suppliers
        supplier_metrics = self.calculate_supplier_metrics()
        
        # Get demand forecast
        demand_over_period = self.get_demand_over_period(
            product_id, location_id, forecast_days
        )
        
        # Get current stock
        current_stock = self.get_current_stock(product_id, location_id)
        
        # Calculate required stock
        required_stock = max(0, demand_over_period - current_stock)
        
        # Get product information
        product_info = self.products_df[
            self.products_df['product_id'] == product_id
        ]
        
        if product_info.empty:
            product_name = 'Unknown Product'
        else:
            product_name = product_info['product_name'].iloc[0]
        
        # Merge with supplier info
        supplier_cols = ['supplier_id', 'supplier_name']
        if 'contact_person' in self.suppliers_df.columns:
            supplier_cols.append('contact_person')
        elif 'contact' in self.suppliers_df.columns:
            supplier_cols.append('contact')
        
        recommendations = supplier_metrics.merge(
            self.suppliers_df[supplier_cols],
            on='supplier_id',
            how='left'
        )
        
        # Calculate scores
        recommendations['urgency'] = recommendations['standard_lead_time'] > urgency_threshold
        
        # Reliability score (0-100): On-time delivery rate
        recommendations['reliability_score'] = recommendations['on_time_rate']
        
        # Cost score (0-100): Inverse of unit price (cheaper = higher score)
        min_price = recommendations['avg_unit_price'].min()
        max_price = recommendations['avg_unit_price'].max()
        recommendations['cost_score'] = 100 * (
            (max_price - recommendations['avg_unit_price']) / 
            (max_price - min_price + 1)  # +1 to avoid division by zero
        )
        
        # Speed score (0-100): Inverse of lead time (faster = higher score)
        min_lead = recommendations['standard_lead_time'].min()
        max_lead = recommendations['standard_lead_time'].max()
        recommendations['speed_score'] = 100 * (
            (max_lead - recommendations['standard_lead_time']) / 
            (max_lead - min_lead + 1)
        )
        
        # Overall composite score (weighted average)
        # Weights: Reliability 40%, Speed 35%, Cost 25%
        recommendations['overall_score'] = (
            0.40 * recommendations['reliability_score'] +
            0.35 * recommendations['speed_score'] +
            0.25 * recommendations['cost_score']
        )
        
        # Calculate delivery date estimate
        today = datetime.now()
        recommendations['estimated_delivery_date'] = (
            today + pd.to_timedelta(recommendations['standard_lead_time'], unit='D')
        )
        
        # Total cost to fulfill order
        recommendations['estimated_order_cost'] = (
            required_stock * recommendations['avg_unit_price']
        )
        
        # Add context information
        recommendations['product_id'] = product_id
        recommendations['product_name'] = product_name
        recommendations['location_id'] = location_id
        recommendations['current_stock'] = current_stock
        recommendations['forecasted_demand_30d'] = demand_over_period
        recommendations['required_stock'] = required_stock
        recommendations['analysis_date'] = today
        
        # Sort by overall score (highest first)
        recommendations = recommendations.sort_values('overall_score', ascending=False)
        
        # Select relevant columns for output
        output_columns = [
            'supplier_id', 'supplier_name',
            'standard_lead_time', 'estimated_delivery_date',
            'on_time_rate', 'avg_unit_price', 'avg_actual_lead_time', 'total_orders',
            'reliability_score', 'speed_score', 'cost_score', 'overall_score',
            'estimated_order_cost', 'urgency'
        ]
        
        # Add contact column if it exists
        if 'contact_person' in recommendations.columns:
            output_columns.insert(2, 'contact_person')
        
        return recommendations[output_columns].reset_index(drop=True), {
            'product_id': product_id,
            'product_name': product_name,
            'location_id': location_id,
            'current_stock': current_stock,
            'forecasted_demand_30d': demand_over_period,
            'required_stock': required_stock,
            'analysis_date': today
        }
    
    def get_supply_gap_analysis(self, product_id, location_id):
        """
        Detailed analysis of supply gap and urgency
        
        Returns:
        --------
        dict : Detailed supply gap metrics
        """
        current_stock = self.get_current_stock(product_id, location_id)
        demand_30d = self.get_demand_over_period(product_id, location_id, days=30)
        demand_7d = self.get_demand_over_period(product_id, location_id, days=7)
        
        days_of_stock = current_stock / (demand_30d / 30) if demand_30d > 0 else float('inf')
        
        gap_30d = max(0, demand_30d - current_stock)
        gap_7d = max(0, demand_7d - current_stock)
        
        # Determine urgency level
        if current_stock == 0:
            urgency = "🔴 CRITICAL - Out of Stock"
            urgency_level = 4
        elif days_of_stock < 7:
            urgency = "🔴 CRITICAL - Less than 7 days of stock"
            urgency_level = 4
        elif days_of_stock < 14:
            urgency = "🟠 HIGH - Less than 14 days of stock"
            urgency_level = 3
        elif days_of_stock < 30:
            urgency = "🟡 MEDIUM - Less than 30 days of stock"
            urgency_level = 2
        else:
            urgency = "🟢 LOW - Adequate stock"
            urgency_level = 1
        
        return {
            'current_stock': current_stock,
            'demand_7d': demand_7d,
            'demand_30d': demand_30d,
            'gap_7d': gap_7d,
            'gap_30d': gap_30d,
            'days_of_stock': days_of_stock,
            'urgency': urgency,
            'urgency_level': urgency_level
        }


def save_recommendations_to_csv(recommendations_df, context_info, output_path='dashboard_data/'):
    """Save supplier recommendations to CSV for dashboard"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save recommendations
    output_file = os.path.join(
        output_path, 
        f"supplier_recommendations_{context_info['product_id']}_"
        f"{context_info['location_id']}.csv"
    )
    recommendations_df.to_csv(output_file, index=False)
    
    return output_file


if __name__ == "__main__":
    """
    Example usage of the Supplier Recommendation Engine
    """
    
    # Initialize engine
    engine = SupplierRecommendationEngine(
        demand_forecasts_path='dashboard_data/fact_demand_forecasts.csv',
        inventory_path='data/inventory.csv',
        suppliers_path='data/dim_suppliers.csv',
        products_path='data/dim_products.csv',
        purchases_path='data/fact_purchases.csv'
    )
    
    # Example: Get recommendations for Product P001 at Location L01
    product_id = 'P001'
    location_id = 'L01'
    
    print(f"\n{'='*80}")
    print(f"SUPPLIER RECOMMENDATION ANALYSIS")
    print(f"Product: {product_id} | Location: {location_id}")
    print(f"{'='*80}\n")
    
    # Get supply gap analysis
    gap_analysis = engine.get_supply_gap_analysis(product_id, location_id)
    print("📊 SUPPLY GAP ANALYSIS")
    print(f"Current Stock: {gap_analysis['current_stock']:.0f} units")
    print(f"7-Day Forecasted Demand: {gap_analysis['demand_7d']:.0f} units")
    print(f"30-Day Forecasted Demand: {gap_analysis['demand_30d']:.0f} units")
    print(f"Supply Gap (30-day): {gap_analysis['gap_30d']:.0f} units")
    print(f"Days of Stock: {gap_analysis['days_of_stock']:.1f} days")
    print(f"Urgency: {gap_analysis['urgency']}\n")
    
    # Get supplier recommendations
    recommendations_df, context_info = engine.recommend_suppliers(
        product_id, location_id, forecast_days=30, urgency_threshold=7
    )
    
    print("🏆 TOP SUPPLIER RECOMMENDATIONS (Ranked by Overall Score)")
    print(f"\n{recommendations_df.to_string(index=False)}\n")
    
    # Show best supplier
    if not recommendations_df.empty:
        best_supplier = recommendations_df.iloc[0]
        print(f"\n{'='*80}")
        print(f"✅ RECOMMENDED SUPPLIER: {best_supplier['supplier_name']}")
        print(f"{'='*80}")
        print(f"Supplier ID: {best_supplier['supplier_id']}")
        print(f"Lead Time: {best_supplier['standard_lead_time']:.0f} days")
        print(f"Est. Delivery Date: {best_supplier['estimated_delivery_date'].strftime('%Y-%m-%d')}")
        print(f"On-Time Delivery Rate: {best_supplier['on_time_rate']:.1f}%")
        print(f"Unit Price: ${best_supplier['avg_unit_price']:.2f}")
        print(f"Order Cost for {context_info['required_stock']:.0f} units: ${best_supplier['estimated_order_cost']:.2f}")
        print(f"Overall Score: {best_supplier['overall_score']:.1f}/100")
        print(f"{'='*80}\n")
