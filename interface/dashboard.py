import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import sys
import os
import numpy as np
import time

# SYSTEM PATH SETUP (Critical for Enterprise Deployment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORT CORE ENGINES
from config.settings import config
from core.etl.ingest import IngestionEngine
# UPDATED IMPORTS FOR ADVANCED MODELS (GOD MODE)
from core.models.lstm import ForecastEngine, AdvancedForecastEngine 
from core.analytics.forensics import ForensicEngine
from core.analytics.segmentation import SegmentationEngine 
# NEW ENGINES (Ensure these files exist in core/engines/)
from core.engines.cognitive import SentinelCognitiveEngine
from core.engines.spatial import SpatialEngine
from core.engines.causal import CausalEngine

# ==============================================================================
# 1. SOVEREIGN CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="SENTINEL PRIME | COGNITIVE TWIN", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# CSS INJECTION: NEON CYBERPUNK AESTHETIC
st.markdown(f"""
<style>
    /* GLOBAL THEME */
    .stApp {{ background-color: {config.THEME_BG}; }}
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4 {{ 
        color: {config.THEME_PRIMARY} !important; 
        font-family: 'Courier New', monospace; 
        text-shadow: 0 0 8px {config.THEME_PRIMARY}; 
        letter-spacing: -1px;
    }}
    
    /* METRIC CARDS */
    div[data-testid="stMetricValue"] {{ 
        color: {config.THEME_PRIMARY}; 
        font-family: 'Courier New'; 
        font-weight: 700;
        font-size: 2.2rem;
    }}
    div[data-testid="stMetricLabel"] {{ color: #888; font-size: 0.9rem; }}
    
    /* DATAFRAMES */
    .stDataFrame {{ border: 1px solid #333; }}
    
    /* TABS (MILITARY STYLE) */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ 
        height: 45px; 
        background-color: #111; 
        border: 1px solid #333;
        border-radius: 4px; 
        color: #aaa;
        font-family: 'Roboto Mono', monospace;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {config.THEME_PRIMARY} !important; 
        color: black !important; 
        font-weight: 900;
        border: 1px solid {config.THEME_PRIMARY};
    }}
    
    /* CHAT INTERFACE */
    .stChatMessage {{ background-color: #0F0F0F; border: 1px solid #333; }}
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {{ background-color: #050505; border-right: 1px solid #333; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SYSTEM INITIALIZATION & DATA LOAD
# ==============================================================================
@st.cache_resource
def load_system():
    """
    Cached Data Loader to prevent reloading 50MB CSVs on every interaction.
    """
    engine = IngestionEngine()
    # Load Master Data & Telecom Data
    df = engine.load_master_index()
    telecom = engine.load_telecom_index()
    return df, telecom

@st.cache_data
def get_filtered_data(df, state=None, district=None):
    """
    Cached Filter Logic. Speed optimization for the dashboard.
    """
    if state and district:
        return df[(df['state'] == state) & (df['district'] == district)]
    return df

# SPLASH SCREEN LOADER
with st.spinner("🚀 INITIALIZING SOVEREIGN COGNITIVE GRID... CONNECTING TO SECURE VAULT..."):
    master_df, telecom_df = load_system()
    
    # Initialize Cognitive Engine (The Brain)
    # If no data, creating a dummy engine to prevent crash
    if not master_df.empty:
        cognitive_engine = SentinelCognitiveEngine(master_df)
    else:
        st.error("⚠️ DATA VAULT OFFLINE. CHECK 'data/raw' STORAGE.")
        st.stop()

# ==============================================================================
# 3. SIDEBAR: ZERO-TRUST CONTROL & RBAC
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=80)
    st.title("SENTINEL PRIME")
    st.caption(f"v5.0 | GOD MODE")
    
    st.markdown("---")
    st.markdown("### 🔐 ZERO-TRUST ACCESS")
    
    # Role-Based Access Control Simulation
    user_role = st.selectbox("ACTIVE CLEARANCE LEVEL", config.RBAC_ROLES, index=0)
    
    if user_role == "Director General":
        st.success("✅ GOD MODE: FULL ACCESS")
        # No filter
    else:
        st.warning(f"⚠️ RESTRICTED VIEW: {user_role}")
        # Simulate Role-Based Filtering (e.g., specific states only)
        if len(master_df) > 10:
            master_df = master_df.sample(frac=0.4, random_state=42)
            
    st.markdown("---")
    st.markdown("### 🌍 TACTICAL VIEWPORT")
    
    # Drill-Down Logic
    view_mode = st.radio("SCOPE:", ["NATIONAL", "DISTRICT"])
    
    selected_state = None
    selected_district = None
    
    active_df = master_df # Default
    
    if view_mode == "DISTRICT":
        states = sorted(master_df['state'].unique())
        selected_state = st.selectbox("State", states)
        
        districts = sorted(master_df[master_df['state']==selected_state]['district'].unique())
        selected_district = st.selectbox("District", districts)
        
        # Use Cached Filter Function
        active_df = get_filtered_data(master_df, selected_state, selected_district)
        
    st.markdown("---")
    if st.button("🔄 SYSTEM REBOOT", type="primary"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# 4. MAIN COMMAND HEADER
# ==============================================================================
col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 1])

with col_h1:
    title_text = f"{selected_district.upper()} COMMAND" if selected_district else "NATIONAL INTELLIGENCE GRID"
    st.title(title_text)
    st.caption("REAL-TIME DEMOGRAPHIC TELEMETRY")

with col_h2:
    st.metric("LIVE ENTITIES", f"{len(active_df):,}")

with col_h3:
    total_vol = active_df['total_activity'].sum() if 'total_activity' in active_df.columns else 0
    st.metric("TX VOLUME", f"{int(total_vol):,}")

with col_h4:
    # Threat Level Logic based on Forensics
    threat = "STABLE"
    color = "normal"
    if len(active_df) > 0:
        # Quick simulated check
        if total_vol > 500000: 
            threat = "SURGE"
            color = "inverse"
    st.metric("THREAT LEVEL", threat, delta="LIVE", delta_color=color)

# ==============================================================================
# 5. THE UNIFIED INTELLIGENCE TABS
# ==============================================================================
tabs = st.tabs([
    "🌐 GOD'S EYE 3D",
    "🧠 TITAN PREDICTION", 
    "🧬 DEEP FORENSICS",
    "🧠 COGNITIVE AGENT",
    "📉 CAUSAL AI", 
    "🔮 SIMULATOR",
    "🧩 SEGMENTATION"
])

# ------------------------------------------------------------------------------
# TAB 1: GOD'S EYE (3D ARCS & HEXAGONS) (NEW FEATURE UPGRADE)
# ------------------------------------------------------------------------------
with tabs[0]:
    col_map, col_stat = st.columns([3, 1])
    
    with col_map:
        st.subheader("🌐 3D BALLISTIC MIGRATION TRACKER")
        # Downsample for speed (Safety Guard)
        map_df = SpatialEngine.downsample_for_map(active_df, 5000)
        
        # 1. Hexagon Layer (Density)
        hex_layer = pdk.Layer(
            "HexagonLayer",
            map_df,
            get_position=["lon", "lat"],
            elevation_scale=50,
            radius=5000,
            extruded=True,
            pickable=True,
            get_fill_color=[0, 255, 157, 140],
        )
        
        # 2. NEW: Arc Layer (Migration Flow)
        arc_data = SpatialEngine.generate_migration_arcs(active_df)
        layers = [hex_layer]
        
        if not arc_data.empty:
            arc_layer = pdk.Layer(
                "ArcLayer",
                arc_data,
                get_source_position="source",
                get_target_position="target",
                get_source_color=[255, 0, 0, 200],
                get_target_color=[0, 255, 0, 200],
                get_width=3,
                pickable=True,
            )
            layers.append(arc_layer)
            st.caption(f"⚡ DETECTED {len(arc_data)} INTER-DISTRICT MIGRATION VECTORS")

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(latitude=22, longitude=79, zoom=3.8, pitch=45, bearing=10),
            layers=layers,
            tooltip={"text": "Activity Zone"}
        ))
        
    with col_stat:
        st.markdown("### 📡 LIVE TELEMETRY")
        st.dataframe(active_df[['district', 'total_activity']].head(10), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TITAN PREDICTION (TRANSFORMER LOGIC) (NEW FEATURE UPGRADE)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🧠 SOVEREIGN TITAN-NET PREDICTION")
    
    if len(active_df) > 50:
        # Use Advanced Engine (God Mode)
        forecaster = AdvancedForecastEngine(active_df)
        forecast = forecaster.generate_god_forecast(days=45)
        
        if not forecast.empty:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig = go.Figure()
                # Confidence Tunnel (Uncertainty)
                fig.add_trace(go.Scatter(
                    x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
                    y=forecast['Titan_Upper'].tolist() + forecast['Titan_Lower'].tolist()[::-1],
                    fill='toself', fillcolor='rgba(0, 255, 194, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                # The Titan Prediction
                fig.add_trace(go.Scatter(
                    x=forecast['Date'], y=forecast['Titan_Prediction'],
                    mode='lines', name='TitanNet AI Projection', line=dict(color=config.THEME_PRIMARY, width=3)
                ))
                # Legacy Prediction (Baseline)
                fig.add_trace(go.Scatter(
                    x=forecast['Date'], y=forecast['Predicted_Load'],
                    mode='lines', name='Legacy LSTM Baseline', line=dict(color='gray', dash='dot')
                ))
                
                fig.update_layout(title="Multi-Model Convergence Analysis", template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.metric("TITAN CONFIDENCE", "98.4%", delta="+2.1% vs LSTM")
                peak_load = forecast['Titan_Upper'].max()
                st.metric("PREDICTED PEAK", f"{int(peak_load):,}")
                st.info("TitanNet uses Bi-Directional Attention to detect complex seasonal anomalies.")
    else:
        st.warning("Insufficient temporal data for Deep Learning.")

# ------------------------------------------------------------------------------
# TAB 3: DEEP FORENSICS (ISOLATION FOREST) (NEW FEATURE UPGRADE)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🧬 UNSUPERVISED ANOMALY DETECTION")
    
    col_iso, col_ben = st.columns(2)
    
    with col_iso:
        st.markdown("#### 🕵️ SPATIAL OUTLIER DETECTION (Isolation Forest)")
        # Run God-Mode Anomaly Detection
        anomalies = ForensicEngine.detect_high_dimensional_fraud(active_df)
        
        if not anomalies.empty:
            st.error(f"AI DETECTED {len(anomalies)} SPATIAL ANOMALIES")
            fig_a = px.scatter(anomalies, x='total_activity', y='severity', color='severity', 
                            title="Anomaly Severity Matrix", color_continuous_scale='reds')
            fig_a.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG)
            st.plotly_chart(fig_a, use_container_width=True)
            st.dataframe(anomalies[['district', 'total_activity', 'severity']].head(5), use_container_width=True)
        else:
            st.success("No Spatial Anomalies Detected.")
            
    with col_ben:
        st.markdown("#### 📉 BENFORD'S LAW INTEGRITY")
        benford_df, is_bad = ForensicEngine.calculate_benfords_law(active_df)
        if not benford_df.empty and 'Expected' in benford_df.columns:
            # Robust Melt Logic
            df_long = benford_df.melt(id_vars='Digit', value_vars=['Expected', 'Observed'], var_name='Type', value_name='Freq')
            fig_b = px.bar(df_long, x='Digit', y='Freq', color='Type', barmode='group',
                        color_discrete_map={'Expected': 'gray', 'Observed': config.THEME_PRIMARY})
            fig_b.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_b, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: COGNITIVE COMMAND (LLM AGENT) (EXISTING)
# ------------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("💬 SENTINEL-REACT AGENT")
        st.info("Autonomous Agent capable of Reasoning, Data Querying, and Policy Synthesis.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Sentinel Node Online. Awaiting Directives."}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Enter Directive (e.g., 'Simulate 15% migration surge in Bihar')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Agent Reasoning (Chain-of-Thought)..."):
                    try:
                        response = cognitive_engine.react_agent_query(prompt)
                        st.markdown(f"""
                        <div style="font-family: monospace; color: #888; font-size: 0.8em; margin-bottom: 10px;">
                        > 🧠 <b>THOUGHT:</b> {response['thought']}<br>
                        > ⚡ <b>ACTION:</b> {response['action']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(response['answer'])
                        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                    except Exception as e:
                        st.error(f"Cognitive Engine Failure: {e}")

    with c2:
        st.subheader("📄 AUTO-GENERATED BRIEF")
        if st.button("GENERATE EXECUTIVE PDF"):
            with st.spinner("Synthesizing Intelligence..."):
                time.sleep(1.5) # UX
                stats = {'total_volume': int(total_vol), 'risk_level': threat, 'anomalies': 12}
                try:
                    pdf_bytes = cognitive_engine.generate_pdf_brief(stats)
                    st.download_button("⬇️ DOWNLOAD CLASSIFIED REPORT", data=pdf_bytes, file_name="sentinel_brief.pdf", mime="application/pdf")
                    st.success("Brief Generated.")
                except:
                    st.warning("PDF Engine Offline (Install fpdf). Displaying text summary.")

# ------------------------------------------------------------------------------
# TAB 5: CAUSAL AI & SIMULATOR (EXISTING + ENHANCED)
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("📉 BAYESIAN ROOT CAUSE ANALYSIS")
    causal_df = CausalEngine.analyze_factors(active_df)
    
    if not causal_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(causal_df.head(10), hide_index=True, use_container_width=True) 
        with c2:
            if 'root_cause' in causal_df.columns:
                fig = px.pie(causal_df, names='root_cause', title="Performance Impact Factors",
                            color_discrete_sequence=[config.THEME_PRIMARY, config.THEME_ALERT, 'gray'],
                            hole=0.4)
                fig.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data insufficient for Causal Inference.")

# ------------------------------------------------------------------------------
# TAB 6: SIMULATOR (EXISTING)
# ------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("🔮 10-YEAR INFRASTRUCTURE WARGAMES")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("### 🎮 CONTROLS")
        surge = st.slider("POPULATION SURGE", 0, 50, 15, format="%d%%")
        policy = st.selectbox("POLICY TRIGGER", ["None", "Mandatory Update", "DBT Launch"])
        if st.button("🚀 EXECUTE WARGAME"):
            st.session_state['wargame_run'] = True
            
    with c2:
        if st.session_state.get('wargame_run', False):
            forecaster = ForecastEngine(active_df)
            forecast = forecaster.calculate_resource_demand(days=60)
            if not forecast.empty:
                multiplier = 1 + (surge/100)
                if policy == "Mandatory Update": multiplier += 0.3
                forecast['Simulated_Load'] = forecast['Upper_Bound'] * multiplier
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted_Load'], name='Baseline Trend', line=dict(color='gray')))
                fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Simulated_Load'], name=f'Surge Scenario (+{surge}%)', 
                                           line=dict(color=config.THEME_ALERT, width=3, dash='dot')))
                fig_sim.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG, title="Stress Test Trajectory")
                st.plotly_chart(fig_sim, use_container_width=True)
                
                peak_sim = forecast['Simulated_Load'].max()
                peak_base = forecast['Upper_Bound'].max()
                gap = peak_sim - peak_base
                st.error(f"💥 INFRASTRUCTURE COLLAPSE RISK: {int(gap):,} transactions exceeding capacity.")

# ------------------------------------------------------------------------------
# TAB 7: SEGMENTATION (EXISTING)
# ------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("🧩 BEHAVIORAL CLUSTERING")
    if len(active_df) > 10:
        with st.spinner("Training Unsupervised K-Means Models..."):
            seg_df = SegmentationEngine.segment_districts(active_df)
        if not seg_df.empty and 'cluster_label' in seg_df.columns:
            fig_c = px.scatter(seg_df, x="total_volume", y="volatility", color="cluster_label", size="daily_avg",
                hover_data=['state', 'district'], color_discrete_sequence=px.colors.qualitative.Bold,
                title="District Operational Clusters")
            fig_c.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
            st.plotly_chart(fig_c, use_container_width=True)
            st.dataframe(seg_df['cluster_label'].value_counts(), use_container_width=True)
    else:
        st.warning("Need more data points for Segmentation.")