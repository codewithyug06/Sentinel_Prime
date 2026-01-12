import pandas as pd
import numpy as np
import math
from sklearn.ensemble import IsolationForest

class ForensicEngine:
    """
    ADVANCED FORENSIC SUITE v5.0 (God Mode)
    Includes: Whipple, Benford, Digit Fingerprint, and ISOLATION FOREST.
    """
    
    @staticmethod
    def calculate_whipple(df):
        """
        Detects Age Heaping (rounding to 0 or 5).
        """
        if 'total_activity' not in df.columns: return pd.DataFrame()
        stats = df.groupby(['state', 'district'])['total_activity'].sum().reset_index()
        # Proxy Logic: Check if the total activity count ends in 0 or 5
        stats['is_suspicious'] = stats['total_activity'].apply(lambda x: 1 if x % 5 == 0 else 0)
        return stats.sort_values('total_activity', ascending=False)

    @staticmethod
    def calculate_benfords_law(df):
        """
        Detects data fabrication.
        Returns standardized columns: ['Digit', 'Expected', 'Observed', 'Deviation']
        """
        if 'total_activity' not in df.columns: return pd.DataFrame(), False
        
        def get_leading_digit(x):
            try:
                s = str(int(x))
                return int(s[0]) if s[0] != '0' else None
            except: return None
            
        digits = df['total_activity'].apply(get_leading_digit).dropna()
        if len(digits) < 50: return pd.DataFrame(), False
        
        observed = digits.value_counts(normalize=True).sort_index()
        expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        
        analysis = pd.DataFrame({
            'Digit': range(1, 10),
            'Expected': [expected[d] for d in range(1, 10)],
            'Observed': [observed.get(d, 0) for d in range(1, 10)]
        })
        analysis['Deviation'] = abs(analysis['Expected'] - analysis['Observed'])
        
        return analysis, analysis['Deviation'].mean() > 0.05

    @staticmethod
    def calculate_digit_fingerprint(df):
        """
        Detects manual data entry bias (Last Digit Analysis).
        """
        if 'total_activity' not in df.columns: return 0.0
        
        def get_last_digit(x): 
            try: return int(str(int(x))[-1])
            except: return -1
        
        last_digits = df['total_activity'].apply(get_last_digit)
        last_digits = last_digits[last_digits != -1]
        
        if len(last_digits) == 0: return 0.0
        
        counts = last_digits.value_counts(normalize=True).sort_index()
        fingerprint_score = sum([abs(counts.get(d, 0) - 0.1) for d in range(10)])
        
        return fingerprint_score

    # ==========================================================================
    # NEW GOD-LEVEL FEATURE: MULTIVARIATE ANOMALY DETECTION (UNSUPERVISED AI)
    # ==========================================================================
    @staticmethod
    def detect_high_dimensional_fraud(df):
        """
        Uses Isolation Forest to detect anomalies based on multi-dimensional features:
        (Activity Volume, Latitude, Longitude).
        
        Finds 'Spatial Outliers' - centers that have high activity in low-density zones.
        """
        if len(df) < 50: return pd.DataFrame()
        
        # Prepare Features
        # Using Lat/Lon acts as a proxy for "Location Context"
        # In a real scenario, we would add 'Time of Day', 'Operator ID', etc.
        features = df[['total_activity', 'lat', 'lon']].fillna(0)
        
        # Model: Isolation Forest
        # Designed to detect anomalies that are 'few and different'
        clf = IsolationForest(contamination=0.02, random_state=42)
        df = df.copy() 
        df['anomaly_score'] = clf.fit_predict(features)
        
        # -1 indicates anomaly, 1 indicates normal.
        anomalies = df[df['anomaly_score'] == -1].copy()
        
        # Calculate Severity (How far from the mean activity)
        mean_activity = df['total_activity'].mean()
        anomalies['severity'] = anomalies['total_activity'] / (mean_activity + 1e-5)
        
        return anomalies.sort_values('severity', ascending=False)