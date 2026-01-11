import pandas as pd
import numpy as np
import math

class ForensicEngine:
    """
    ADVANCED FORENSIC SUITE v3.5 (Ensemble)
    Includes: Whipple Index, Benford's Law, and Digit Frequency Fingerprinting.
    """
    
    @staticmethod
    def calculate_whipple(df):
        """
        Detects Age Heaping (rounding to 0 or 5).
        """
        if 'total_activity' not in df.columns: return pd.DataFrame()
        stats = df.groupby(['state', 'district'])['total_activity'].sum().reset_index()
        # Proxy: Check if count ends in 0 or 5
        stats['is_suspicious'] = stats['total_activity'].apply(lambda x: 1 if x % 5 == 0 else 0)
        return stats.sort_values('total_activity', ascending=False)

    @staticmethod
    def calculate_benfords_law(df):
        """
        Detects data fabrication by checking leading digit distribution.
        """
        if 'total_activity' not in df.columns: return pd.DataFrame(), False
        
        def get_leading_digit(x):
            s = str(int(x))
            return int(s[0]) if s[0] != '0' else None
            
        digits = df['total_activity'].apply(get_leading_digit).dropna()
        if len(digits) < 50: return pd.DataFrame(), False # Need decent sample size
        
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
        NEW FEATURE: Digit-Frequency Fingerprinting.
        Analyzes the LAST digit of counts. In natural large datasets, 
        last digits (0-9) should be uniformly distributed (approx 10% each).
        Human-entered data often biases towards 0, 5, or even numbers.
        """
        if 'total_activity' not in df.columns: return 0.0
        
        def get_last_digit(x): 
            try:
                return int(str(int(x))[-1])
            except:
                return -1
        
        last_digits = df['total_activity'].apply(get_last_digit)
        last_digits = last_digits[last_digits != -1]
        
        if len(last_digits) == 0: return 0.0
        
        counts = last_digits.value_counts(normalize=True).sort_index()
        
        # Calculate Sum of Errors from Uniform Distribution (0.1 for each digit 0-9)
        # Higher score = Less Random (More likely fabricated/manual)
        fingerprint_score = sum([abs(counts.get(d, 0) - 0.1) for d in range(10)])
        
        return fingerprint_score