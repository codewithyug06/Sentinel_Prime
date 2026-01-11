import pandas as pd
import numpy as np
import math

class ForensicEngine:
    """
    ADVANCED FORENSIC SUITE v4.5 (Robust & Safe)
    Includes: Whipple Index, Benford's Law, and Digit Frequency Fingerprinting.
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
        Detects data fabrication by checking leading digit distribution.
        Returns:
            1. analysis_df: DataFrame with 'Digit', 'Expected', 'Observed', 'Deviation'
            2. is_anomalous: Boolean flag (True if deviation is high)
        """
        if 'total_activity' not in df.columns: return pd.DataFrame(), False
        
        # Helper to get first digit safely
        def get_leading_digit(x):
            try:
                s = str(int(x))
                # Skip 0 as leading digit
                return int(s[0]) if s[0] != '0' else None
            except:
                return None
            
        digits = df['total_activity'].apply(get_leading_digit).dropna()
        
        # Need decent sample size for statistical significance
        if len(digits) < 50: 
            return pd.DataFrame(), False 
        
        # Calculate Observed Frequencies
        observed = digits.value_counts(normalize=True).sort_index()
        
        # Calculate Expected Frequencies (Benford's Law: P(d) = log10(1 + 1/d))
        expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        
        # Create Analysis DataFrame
        analysis = pd.DataFrame({
            'Digit': range(1, 10),
            'Expected': [expected[d] for d in range(1, 10)],
            'Observed': [observed.get(d, 0) for d in range(1, 10)]
        })
        
        # Calculate Deviation
        analysis['Deviation'] = abs(analysis['Expected'] - analysis['Observed'])
        
        # Flag if mean deviation is high (Threshold 0.05 is standard for this volume)
        is_anomalous = analysis['Deviation'].mean() > 0.05
        
        return analysis, is_anomalous

    @staticmethod
    def calculate_digit_fingerprint(df):
        """
        NEW FEATURE: Digit-Frequency Fingerprinting.
        Analyzes the LAST digit of counts. 
        
        Theory: In natural large datasets, last digits (0-9) are uniformly distributed (~10% each).
        Fraud: Human-entered data often biases towards 0, 5, or even numbers.
        
        Returns:
            fingerprint_score (float): Sum of errors from uniform distribution.
            Higher score = Higher probability of manual manipulation.
        """
        if 'total_activity' not in df.columns: return 0.0
        
        def get_last_digit(x): 
            try:
                # Convert to int, string, take last char, convert back to int
                return int(str(int(x))[-1])
            except:
                return -1
        
        # Extract last digits and filter invalid ones
        last_digits = df['total_activity'].apply(get_last_digit)
        last_digits = last_digits[last_digits != -1]
        
        if len(last_digits) == 0: return 0.0
        
        # Calculate actual distribution
        counts = last_digits.value_counts(normalize=True).sort_index()
        
        # Calculate Sum of Errors from Uniform Distribution (0.1 for each digit 0-9)
        # We sum the absolute difference between Observed Frequency and 0.1
        fingerprint_score = sum([abs(counts.get(d, 0) - 0.1) for d in range(10)])
        
        return fingerprint_score