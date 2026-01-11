import os
from pathlib import Path

class Config:
    """
    CENTRAL CONFIGURATION MATRIX
    Controls Paths, Themes, AI Hyperparameters, and Security Protocols.
    """
    # ==========================================================================
    # 1. SYSTEM PATHS
    # ==========================================================================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    # ==========================================================================
    # 2. SOVEREIGN VISUAL IDENTITY (Cyberpunk / Command Center Theme)
    # ==========================================================================
    THEME_BG = "#050505"       # Void Black
    THEME_PRIMARY = "#00FF9D"  # Neon Mint
    THEME_SECONDARY = "#1A1A1A"
    THEME_ALERT = "#FF2A2A"    # Cyber Red for Fraud
    
    # ==========================================================================
    # 3. AI HYPERPARAMETERS & THRESHOLDS
    # ==========================================================================
    LSTM_HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    ANOMALY_THRESHOLD = 0.01
    FORECAST_HORIZON = 30
    
    # ==========================================================================
    # 4. SECURITY & RBAC PROTOCOLS (Zero-Trust)
    # ==========================================================================
    RBAC_ROLES = ["Director General", "State Secretary", "District Magistrate", "Auditor"]
    DEFAULT_ROLE = "Director General"
    
    # ==========================================================================
    # 5. EXTERNAL API KEYS
    # ==========================================================================
    # CRITICAL FIX: Added default fallback to empty string to prevent AttributeError
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")

config = Config()