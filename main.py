# Import necessary modules
import numpy as np
from demand_forecasting import DemandForecastingLSTM
from inventory_management import InventoryManagementRL
from route_optimization import RouteOptimizationGNN

def main():
    # Step 1: Demand Forecasting
    print("Step 1: Demand Forecasting using LSTM")
    
    # Initialize and load the LSTM model for demand forecasting
    lstm_model = DemandForecastingLSTM()
    
    # Load the sales data (time-series data)
    sales_data_path = 'data/sales_data.csv'
    predicted_demand = lstm_model.predict_demand(sales_data_path)
    
    print(f"Predicted Demand: {predicted_demand}")

    # Step 2: Inventory Management with Reinforcement Learning
    print("Step 2: Inventory Management using Reinforcement Learning")
    
    # Initialize and load the reinforcement learning model for inventory management
    inventory_model = InventoryManagementRL()
    
    # Use the predicted demand from LSTM as input to the RL model
    optimal_inventory = inventory_model.optimize_inventory(predicted_demand)
    
    print(f"Optimal Inventory: {optimal_inventory}")

    # Step 3: Route Optimization using GNN
    print("Step 3: Route Optimization using GNN")
    
    # Initialize and load the GNN model for route optimization
    gnn_model = RouteOptimizationGNN()
    
    # Generate graph data for delivery routes (using synthetic or real road data)
    graph_data_path = 'data/route_graph_data.csv'
    optimized_routes = gnn_model.optimize_routes(graph_data_path, optimal_inventory)
    
    print(f"Optimized Delivery Routes: {optimized_routes}")

if __name__ == "__main__":
    main()
