Given : vessels info, cargoes info, port distances, bunker prices, 

freight_calculator.py

# Setup Guide
- Create a virtual environment (using conda/venv)
- pip install -r requirements.txt
# 1. Calculate Time                                                                                                                                                                                 
  ballast_days = distance / (speed * 24)  # hours in a day                                                                                                                                            
  laden_days = distance / (speed * 24)                                                                                                                                                                
  port_days = (quantity/load_rate + load_tt) + (quantity/discharge_rate + discharge_tt)                                                                                                               
  total_days = ballast_days + laden_days + port_days                                                                                                                                                  
                                                                                                                                                                                                      
  # 2. Calculate Fuel Consumption                                                                                                                                                                     
  vlsf_at_sea = (ballast_days * vessel_ballast_consumption) + (laden_days * vessel_laden_consumption)                                                                                                 
  vlsf_in_port = port_days * vessel_port_consumption                                                                                                                                                  
  total_vlsf = vlsf_at_sea + vlsf_in_port                                                                                                                                                             
                                                                                                                                                                                                      
  # 3. Calculate Costs                                                                                                                                                                                
  hire_cost = vessel_hire_rate * total_days                                                                                                                                                           
  bunker_cost = (total_vlsf * vlsf_price) + (total_mgo * mgo_price)                                                                                                                                   
  port_cost = cargo['port_cost']                                                                                                                                                                      
  total_costs = hire_cost + bunker_cost + port_cost                                                                                                                                                   
                                                                                                                                                                                                      
  # 4. Calculate Revenue                                                                                                                                                                              
  freight_revenue = quantity * freight_rate                                                                                                                                                           
  commission = freight_revenue * commission_rate                                                                                                                                                      
  net_revenue = freight_revenue - commission                                                                                                                                                          
                                                                                                                                                                                                      
  # 5. Calculate Profit & TCE                                                                                                                                                                         
  voyage_profit = net_revenue - total_costs                                                                                                                                                           
  TCE = (net_revenue - bunker_cost - port_cost) / total_days                                                                                                                                          
                                                                                                                                                                                                      
  TCE (Time Charter Equivalent):                                                                                                                                                                      
  - Industry-standard metric for comparing voyage profitability                                                                                                                                       
  - Formula: (Net Revenue - Bunker - Port Costs) / Total Days                                                                                                                                         
  - Higher TCE = more profitable per day                                                                                                                                                              
  - Think of it as "daily earnings after variable costs" 

main_analysis.py

1. Port Normalization = estimate unknown port distances using nearby ports
    Classify ports to regions, look up distance between regsions, return inter-region distance
2. calculate all combinations of vessels and cargos
3. find optimal allocation using greedy by TCE. take top 3 and save the rest in csv

datathon_submission.ipynb

ML Models Used (5 models compared):                                                                                                                                                                 
                                                                                                                                                                                                      
  1. Random Forest Regressor                                                                                                                                             
    - Ensemble of 100 decision trees                                                                                                                                                                  
    - max_depth=10, min_samples_split=5                                                                                                                                                               
    - Good at capturing non-linear relationships                                                                                                                                                      
    - Handles feature interactions well                                                                                                                                                               
  2. Gradient Boosting Regressor                                                                                                                                                                      
    - Sequential boosting (learns from errors)                                                                                                                                                        
    - n_estimators=100, max_depth=5, learning_rate=0.1                                                                                                                                                
    - Often more accurate but slower                                                                                                                                                                  
  3. Linear Regression                                                                                                                                                                                
    - Simple baseline model                                                                                                                                                                           
    - Assumes linear relationships                                                                                                                                                                    
    - Easy to interpret                                                                                                                                                                               
  4. Ridge Regression                                                                                                                                                                                 
    - Linear regression with L2 regularization                                                                                                                                                        
    - Prevents overfitting                                                                                                                                                                            
    - alpha=1.0                                                                                                                                                                                       
  5. Decision Tree Regressor                                                                                                                                                                          
    - Single tree (simpler than Random Forest)                                                                                                                                                        
    - max_depth=8                                                                                                                                                                                     
    - Easy to visualize                                                                                                                                                                               
                                                                                                                                                                                                      
  What the ML predicts: Risk-adjusted TCE   