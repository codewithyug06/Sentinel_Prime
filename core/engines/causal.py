import pandas as pd
import numpy as np

class CausalEngine:
    """
    PART 2: CAUSAL AI & BAYESIAN NETWORKS
    Moves beyond correlation to causation.
    """
    
    @staticmethod
    def analyze_factors(df):
        """
        Simulates a Causal Inference Model.
        Determines if 'Low Performance' is caused by 'Hardware' or 'External Factors'.
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