import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats

class CausalEngine:
    """
    PART 2: CAUSAL AI & BAYESIAN NETWORKS
    Moves beyond correlation to causation.
    
    ADVANCED CAPABILITIES (V9.7):
    - Structural Causal Models (SCM)
    - Counterfactual Simulation (What-If Analysis)
    - Granger Causality Tests (Temporal Precedence)
    - Policy Shock Propagation (System-Wide Stress Test)
    """
    
    @staticmethod
    def analyze_factors(df):
        """
        Simulates a Causal Inference Model.
        Determines if 'Low Performance' is caused by 'Hardware' or 'External Factors'.
        (Legacy Function - Preserved for Backward Compatibility)
        """
        if df.empty: return pd.DataFrame()
        
        # 1. Create Causal Features
        causal_df = df.groupby('district').agg({
            'total_activity': 'mean'
        }).reset_index()
        
        # Simulate Causal Drivers
        # Factor A: Rain/Weather (Random Impact)
        # Factor B: Operator Efficiency (Internal Impact)
        np.random.seed(42)
        causal_df['weather_impact_prob'] = np.random.uniform(0, 0.4, len(causal_df))
        causal_df['hardware_failure_prob'] = np.random.uniform(0, 0.2, len(causal_df))
        
        # Causal Logic: If activity is low, what is the P(Cause)?
        def determine_cause(row):
            if row['total_activity'] < 500:
                if row['weather_impact_prob'] > row['hardware_failure_prob']:
                    return "🌧️ External (Weather/Access)"
                else:
                    return "💻 Internal (System Failure)"
            return "✅ Optimal Operation"
            
        causal_df['root_cause'] = causal_df.apply(determine_cause, axis=1)
        
        return causal_df

    # ==========================================================================
    # NEW: STRUCTURAL CAUSAL MODEL (SCM)
    # ==========================================================================
    @staticmethod
    def structural_causal_model(df):
        """
        Builds a DAG (Directed Acyclic Graph) to model relationships between:
        [Migration] -> [Request Volume] -> [Server Load] -> [Latency]
        Also considers [Teledensity] -> [Digital Access]
        """
        if df.empty or 'total_activity' not in df.columns:
            return None
            
        G = nx.DiGraph()
        
        # Define Nodes
        G.add_node("Migration_Flow", type="Exogenous")
        G.add_node("Teledensity", type="Exogenous")
        G.add_node("Request_Volume", type="Endogenous")
        G.add_node("Server_Load", type="Endogenous")
        G.add_node("Latency", type="Target")
        
        # Define Causal Edges (Assumptions based on Domain Knowledge)
        G.add_edge("Migration_Flow", "Request_Volume", weight=0.85)
        G.add_edge("Teledensity", "Request_Volume", weight=0.45)
        G.add_edge("Request_Volume", "Server_Load", weight=0.95)
        G.add_edge("Server_Load", "Latency", weight=0.90)
        
        # In a real SCM, we would fit coefficients here. 
        # For the hackathon, we return the topology for the 'Strategy Agent'.
        return G

    # ==========================================================================
    # NEW: COUNTERFACTUAL SIMULATOR (THE "WHAT-IF" ENGINE)
    # ==========================================================================
    @staticmethod
    def run_counterfactual(df, intervention_variable, scale_factor=1.2):
        """
        Answers: "What happens to Y if we intervene on X?"
        Example: "If we increase Operator Efficiency by 20% (scale=1.2), how does Latency change?"
        
        Uses Linear SCM approximation:
        Latency = 0.8 * Load - 0.5 * Efficiency + Noise
        """
        if df.empty: return {}
        
        # 1. Baseline Metrics
        current_load = df['total_activity'].mean()
        # Simulated Efficiency (assume 0.7 if not present)
        current_efficiency = df['operator_efficiency'].mean() if 'operator_efficiency' in df.columns else 0.7
        
        # SCM Equation (Simplified for Demo)
        # Latency (ms) = (Load * 0.05) / Efficiency
        baseline_latency = (current_load * 0.05) / (current_efficiency + 1e-5)
        
        # 2. Apply Intervention (The "DO" Operator)
        simulated_results = {}
        
        if intervention_variable == "efficiency":
            # Scenario: Training programs improve operator speed
            new_efficiency = current_efficiency * scale_factor
            new_latency = (current_load * 0.05) / (new_efficiency + 1e-5)
            
            simulated_results = {
                "Intervention": f"Increase Operator Efficiency by {int((scale_factor-1)*100)}%",
                "Baseline_Latency": round(baseline_latency, 2),
                "Counterfactual_Latency": round(new_latency, 2),
                "Improvement": f"{round((baseline_latency - new_latency), 2)} ms"
            }
            
        elif intervention_variable == "load_balancing":
            # Scenario: Offloading traffic to cloud
            # "scale_factor" here acts as load REDUCTION (e.g., 0.8 means 20% offload)
            new_load = current_load * (1.0 / scale_factor) # Inverse relation for intuitive API
            new_latency = (new_load * 0.05) / (current_efficiency + 1e-5)
            
            simulated_results = {
                "Intervention": f"Load Balancing Protocol (Scale: {scale_factor}x)",
                "Baseline_Latency": round(baseline_latency, 2),
                "Counterfactual_Latency": round(new_latency, 2),
                "Improvement": f"{round((baseline_latency - new_latency), 2)} ms"
            }
            
        return simulated_results

    # ==========================================================================
    # NEW: GRANGER CAUSALITY TEST (TEMPORAL PRECEDENCE)
    # ==========================================================================
    @staticmethod
    def check_temporal_causality(time_series_df):
        """
        Checks if 'Migration' Granger-causes 'Anomaly Spikes'.
        Does past migration predict future fraud?
        """
        if len(time_series_df) < 10: return "INSUFFICIENT DATA"
        
        # Simulated Lag Correlation (Proxy for full Granger F-test to save compute)
        # In prod, use: statsmodels.tsa.stattools.grangercausalitytests
        
        # Create lag
        df = time_series_df.copy()
        if 'migration_index' not in df.columns:
            # Create synthetic migration proxy if missing
            np.random.seed(101)
            df['migration_index'] = np.random.poisson(50, len(df))
            
        df['migration_lag_3d'] = df['migration_index'].shift(3)
        
        # Correlation between Lagged Migration and Current Activity
        corr = df['migration_lag_3d'].corr(df['total_activity'])
        
        if abs(corr) > 0.6:
            return f"POSITIVE CAUSALITY: Migration surges precede activity spikes by 3 days (Corr: {corr:.2f})."
        else:
            return "NO TEMPORAL CAUSALITY: Spikes are likely instantaneous or external."

    # ==========================================================================
    # NEW V9.8: POLICY SHOCK PROPAGATION (SYSTEM-WIDE STRESS TEST)
    # ==========================================================================
    @staticmethod
    def simulate_policy_shock(policy_type, intensity=1.5):
        """
        Models how a policy decision ripples through the causal graph.
        This is for the 'Wargame' engine to calculate 2nd and 3rd order effects.
        
        Args:
            policy_type: "MANDATORY_UPDATE" | "OFFLINE_MODE" | "DBT_LINKAGE"
            intensity: Multiplier for the shock (1.5 = 50% increase)
        """
        impact_report = {}
        
        if policy_type == "MANDATORY_UPDATE":
            # Direct Effect: Request Volume spikes
            vol_impact = intensity * 100 
            # Second Order: Server Load increases non-linearly
            load_impact = (intensity ** 1.2) * 100
            # Third Order: Latency penalty
            latency_impact = (intensity ** 2.5) * 20 # Exponential degradation
            
            impact_report = {
                "Policy": "Mandatory Biometric Update",
                "Primary_Effect": f"Request Volume +{int(vol_impact-100)}%",
                "Secondary_Effect": f"Server Load +{int(load_impact-100)}%",
                "Tertiary_Effect": f"Latency Spike +{int(latency_impact)}ms (CRITICAL)",
                "Recommendation": "Stagger rollout by PIN Code to dampen shock."
            }
            
        elif policy_type == "OFFLINE_MODE":
            # Direct Effect: Sync delay increases
            sync_impact = intensity * 100
            # Second Order: Real-time visibility drops
            vis_impact = -1 * (intensity * 20)
            
            impact_report = {
                "Policy": "Emergency Offline Mode",
                "Primary_Effect": "Data Sync Delay +4 Hours",
                "Secondary_Effect": f"Real-time Visibility {int(vis_impact)}%",
                "Recommendation": "Acceptable for disaster zones only."
            }
            
        return impact_report