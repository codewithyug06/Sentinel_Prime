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
from core.models.lstm import ForecastEngine
from core.analytics.forensics import ForensicEngine
from core.analytics.segmentation import SegmentationEngine 
# NEW ENGINES (Ensure these files exist in core/engines/ as per previous instruction)
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
    st.caption(f"v4.0 | SOVEREIGN TWIN")
    
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
    "🧠 COGNITIVE AGENT",
    "🌐 4D WAR ROOM",
    "📉 CAUSAL AI",
    "🧬 FORENSIC LAB",
    "🔮 SIMULATOR",
    "🧩 SEGMENTATION"
])

# ------------------------------------------------------------------------------
# TAB 1: COGNITIVE COMMAND (LLM AGENT)
# ------------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("💬 SENTINEL-REACT AGENT")
        st.info("Autonomous Agent capable of Reasoning, Data Querying, and Policy Synthesis.")
        
        # Session State for Chat
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Sentinel Node Online. Awaiting Directives."}]

        # Display History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Enter Directive (e.g., 'Simulate 15% migration surge in Bihar')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Agent Reasoning (Chain-of-Thought)..."):
                    # Call the Cognitive Engine
                    try:
                        response = cognitive_engine.react_agent_query(prompt)
                        
                        # Show the "Brain" working (The ReAct Magic)
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
                stats = {
                    'total_volume': int(total_vol),
                    'risk_level': threat,
                    'anomalies': 12 # Simulated
                }
                # PDF Generation (Mocked for Display if FPDF missing, else Real)
                try:
                    pdf_bytes = cognitive_engine.generate_pdf_brief(stats)
                    st.download_button(
                        "⬇️ DOWNLOAD CLASSIFIED REPORT",
                        data=pdf_bytes,
                        file_name=f"sentinel_brief_{selected_district if selected_district else 'NATIONAL'}.pdf",
                        mime="application/pdf"
                    )
                    st.success("Brief Generated.")
                except:
                    st.warning("PDF Engine Offline (Install fpdf). Displaying text summary.")
                    st.text(f"SUMMARY REPORT\nTarget: {selected_district}\nVolume: {total_vol}\nStatus: {threat}")

# ------------------------------------------------------------------------------
# TAB 2: 4D WAR ROOM (H3 SPATIAL TWIN)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🌐 H3 FLUID DENSITY DIGITAL TWIN")
    
    col_vis, col_info = st.columns([3, 1])
    with col_vis:
        # PERFORMANCE FIX: Downsample data if too large (>500MB potential)
        map_df = SpatialEngine.downsample_for_map(active_df, max_points=5000)
        
        # Generate H3 Hexagons (Spatial Engine)
        hex_data = SpatialEngine.generate_h3_hexagons(map_df)
        
        if not hex_data.empty:
            # 3D Hexagon Map
            layer = pdk.Layer(
                "HexagonLayer",
                map_df, # Using downsampled points prevents MessageSizeError
                get_position=["lon", "lat"],
                elevation_scale=50,
                radius=10000 if view_mode == "NATIONAL" else 2000,
                extruded=True,
                pickable=True,
                get_fill_color=[0, 255, 157, 200],
                auto_highlight=True
            )
            
            # Camera State
            view_state = pdk.ViewState(
                latitude=map_df['lat'].mean(), 
                longitude=map_df['lon'].mean(), 
                zoom=4 if view_mode == "NATIONAL" else 9, 
                pitch=60,
                bearing=30
            )
            
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"html": "<b>Zone Density:</b> High Activity"}
            ))
        else:
            st.warning("Insufficient Geodata for H3 Tiling.")
            
    with col_info:
        st.markdown("### 🕸️ MIGRATION GRAPH")
        # Graph Viz
        try:
            G, centrality = SpatialEngine.build_migration_graph(active_df)
            if G:
                st.metric("ACTIVE NODES", G.number_of_nodes())
                st.metric("FLOW VECTORS", G.number_of_edges())
                st.info("Simulating migration corridors using Graph Centrality (NetworkX).")
                
                # Simple list of hubs
                st.write("**Top Transit Hubs:**")
                sorted_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                for node, score in sorted_hubs:
                    st.text(f"• {node} (Score: {score:.2f})")
        except:
            st.warning("Graph Engine Offline.")

# ------------------------------------------------------------------------------
# TAB 3: CAUSAL AI (WHY IT HAPPENED)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("📉 BAYESIAN ROOT CAUSE ANALYSIS")
    st.caption("Determines probability of performance drops (Weather vs. Hardware vs. Human).")
    
    causal_df = CausalEngine.analyze_factors(active_df)
    
    if not causal_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            # FIX: Removed width=None to prevent StreamlitInvalidWidthError
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
# TAB 4: ENSEMBLE FORENSICS
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🧬 NEURAL FRAUD FINGERPRINTING")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 1. WHIPPLE INDEX (Age Heaping)")
        forensics = ForensicEngine.calculate_whipple(active_df)
        if not forensics.empty:
            anomalies = forensics[forensics['is_suspicious'] == 1]
            st.metric("ANOMALOUS DISTRICTS", len(anomalies), delta="RISK")
            # FIX: Deprecation Warning
            st.dataframe(
                anomalies.style.map(lambda x: "background-color: #330000; color: #ff4b4b", subset=['total_activity']),
                height=200,
                use_container_width=True
            )
    
    with c2:
        st.markdown("#### 2. BENFORD'S LAW (Fabrication)")
        benford_df, is_bad = ForensicEngine.calculate_benfords_law(active_df)
        
        if not benford_df.empty:
            # FIX: CRITICAL CRASH FIX - Check columns before melting
            if 'Expected' in benford_df.columns and 'Observed' in benford_df.columns:
                benford_long = benford_df.melt(id_vars='Digit', value_vars=['Expected', 'Observed'], 
                                             var_name='Type', value_name='Frequency')
                
                fig_b = px.bar(benford_long, x='Digit', y='Frequency', color='Type', barmode='group',
                              color_discrete_map={'Expected': 'gray', 'Observed': config.THEME_PRIMARY})
                
                fig_b.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_b, use_container_width=True)
                
                if is_bad: st.error("⚠️ DATA FABRICATION DETECTED")
                else: st.success("✅ DATA INTEGRITY VERIFIED")
            else:
                st.info("Benford Analysis pending data sufficiency.")

    st.divider()
    
    # Digit Fingerprint
    st.markdown("#### 3. DIGIT FREQUENCY FINGERPRINT")
    
    try:
        digit_score = ForensicEngine.calculate_digit_fingerprint(active_df)
        
        col_d1, col_d2 = st.columns([1, 3])
        col_d1.metric("FINGERPRINT SCORE", f"{digit_score:.4f}", delta="Lower is Better", delta_color="inverse")
        col_d2.info("A high score (> 0.05) indicates the operator is manually typing numbers instead of real counting.")
    except AttributeError:
        st.error("⚠️ FORENSIC ENGINE UPDATE REQUIRED. 'calculate_digit_fingerprint' missing.")
    except Exception as e:
        st.error(f"Computation Failed: {e}")

# ------------------------------------------------------------------------------
# TAB 5: STRATEGIC SIMULATOR
# ------------------------------------------------------------------------------
with tabs[4]:
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
            # Run Forecasting
            forecaster = ForecastEngine(active_df)
            forecast = forecaster.calculate_resource_demand(days=60)
            
            if not forecast.empty:
                # Apply surge multiplier
                multiplier = 1 + (surge/100)
                if policy == "Mandatory Update": multiplier += 0.3
                
                forecast['Simulated_Load'] = forecast['Upper_Bound'] * multiplier
                
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted_Load'], name='Baseline Trend', line=dict(color='gray')))
                fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Simulated_Load'], name=f'Surge Scenario (+{surge}%)', 
                                           line=dict(color=config.THEME_ALERT, width=3, dash='dot')))
                
                fig_sim.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG,
                                     title="Stress Test Trajectory")
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # Gap Analysis
                peak_sim = forecast['Simulated_Load'].max()
                peak_base = forecast['Upper_Bound'].max()
                gap = peak_sim - peak_base
                
                st.error(f"💥 INFRASTRUCTURE COLLAPSE RISK: {int(gap):,} transactions exceeding capacity.")

# ------------------------------------------------------------------------------
# TAB 6: SEGMENTATION (CLUSTERING)
# ------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("🧩 BEHAVIORAL CLUSTERING")
    
    if len(active_df) > 10:
        with st.spinner("Training Unsupervised K-Means Models..."):
            seg_df = SegmentationEngine.segment_districts(active_df)
            
        if not seg_df.empty and 'cluster_label' in seg_df.columns:
            fig_c = px.scatter(
                seg_df, 
                x="total_volume", 
                y="volatility", 
                color="cluster_label",
                size="daily_avg",
                hover_data=['state', 'district'],
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="District Operational Clusters"
            )
            fig_c.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
            st.plotly_chart(fig_c, use_container_width=True)
            
            # Removed invalid width=None
            st.dataframe(seg_df['cluster_label'].value_counts(), use_container_width=True)
    else:
        st.warning("Need more data points for Segmentation.")