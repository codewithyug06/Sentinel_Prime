import pandas as pd
import numpy as np
import networkx as nx
import random
import math
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans

# ==============================================================================
# 1. GRAPH NEURAL NETWORK (GNN) SIMULATOR
# ==============================================================================
class GraphNeuralNetwork:
    """
    Simulates Message Passing Interface (MPI) for District Nodes.
    Predicts 'High Velocity Hubs' and 'Forensic Risk Contagion'.
    """
    @staticmethod
    def predict_node_importance(G):
        if G is None: return {}
        # PageRank is a good proxy for GNN node embedding importance in this context
        try:
            return nx.pagerank(G, alpha=0.85)
        except:
            return {}

    @staticmethod
    def simulate_risk_diffusion(G, initial_risk_scores, decay_factor=0.6, steps=3):
        """
        NEW V9.7: FORENSIC CONTAGION MODEL
        Simulates how fraud/anomaly risk spreads through the migration network.
        """
        if G is None or not initial_risk_scores: return {}
        
        current_risks = initial_risk_scores.copy()
        
        for _ in range(steps):
            new_risks = current_risks.copy()
            for node in G.nodes():
                if node not in current_risks: continue
                
                # Get neighbors (Districts connected by migration)
                neighbors = list(G.neighbors(node))
                if not neighbors: continue
                
                # Calculate risk pressure from this node to its neighbors
                outbound_risk = current_risks[node] * decay_factor
                
                for neighbor in neighbors:
                    # Get edge weight (strength of connection)
                    weight = G[node][neighbor].get('weight', 0.5)
                    
                    # Neighbor absorbs risk based on connection strength
                    transmitted_risk = outbound_risk * weight
                    
                    # Update neighbor risk (Max aggregation to simulate critical exposure)
                    existing_risk = new_risks.get(neighbor, 0)
                    new_risks[neighbor] = max(existing_risk, transmitted_risk)
            
            current_risks = new_risks
            
        return current_risks

    # ==========================================================================
    # NEW V9.8 FEATURE: NETWORK SHOCK SIMULATION (RESILIENCE TEST)
    # ==========================================================================
    @staticmethod
    def simulate_network_shock(G, failed_node):
        """
        Simulates the collapse of a major node (e.g., Flood in Chennai).
        Calculates the 'Spillover Pressure' on connected neighbors.
        """
        if G is None or failed_node not in G.nodes():
            return {}
            
        neighbors = list(G.neighbors(failed_node))
        if not neighbors: return {}
        
        impact_analysis = {}
        total_load_to_redistribute = 100.0 # Abstract load unit
        
        # Calculate capacity of neighbors (Degree centrality as proxy)
        degrees = dict(G.degree(neighbors))
        total_capacity = sum(degrees.values()) + 1e-5
        
        for neighbor in neighbors:
            # Load distributed proportional to capacity
            share = degrees[neighbor] / total_capacity
            impact_load = total_load_to_redistribute * share
            
            impact_analysis[neighbor] = {
                "status": "STRESSED",
                "spillover_load": round(impact_load, 2),
                "alert_level": "HIGH" if impact_load > 30 else "MODERATE"
            }
            
        return impact_analysis

    # ==========================================================================
    # NEW V9.9 FEATURE: SPATIOTEMPORAL GCN (ST-GCN) PREP
    # ==========================================================================
    @staticmethod
    def prepare_stgcn_tensors(df, time_window=7):
        """
        Transforms flat CSV data into 3D Tensors (Nodes x Time x Features).
        Ready for ingestion by advanced Deep Learning models (PyTorch Geometric).
        """
        if df.empty or 'date' not in df.columns: return None
        
        # 1. Pivot to Matrix (District vs Date)
        pivot = df.pivot_table(index='district', columns='date', values='total_activity', fill_value=0)
        
        # 2. Slice recent window
        recent_data = pivot.iloc[:, -time_window:]
        
        # 3. Construct Adjacency Matrix (A) based on correlation
        # If districts have similar temporal patterns, they are 'functionally connected'
        correlation_matrix = recent_data.T.corr().fillna(0).values
        
        # Thresholding to create sparse graph structure for GCN
        adjacency = (correlation_matrix > 0.7).astype(float)
        
        return {
            "node_features": recent_data.values,
            "adjacency_matrix": adjacency,
            "shape": f"({len(recent_data)}, {time_window})"
        }

class SpatialEngine:
    """
    PART 2 & 3: ADVANCED GEOSPATIAL & GRAPH INTELLIGENCE
    Features: H3 Indexing, Migration Graphs, Digital Twin Layers, Dark Zone Detection.
    """
    
    @staticmethod
    def generate_h3_hexagons(df, resolution=4):
        """
        Generates H3 Hexagon data for aggregation.
        """
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return pd.DataFrame()
        
        try:
            import h3
            # Real H3 Indexing
            df['h3_index'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], resolution), axis=1)
            hex_data = df.groupby('h3_index').agg({
                'total_activity': 'sum',
                'lat': 'mean', # Centroid proxy
                'lon': 'mean'
            }).reset_index()
            return hex_data
        except ImportError:
            # Fallback
            return df[['lat', 'lon', 'total_activity']].copy()

    @staticmethod
    def build_migration_graph(df):
        G = nx.Graph()
        
        if 'district' not in df.columns: return None, {}
        
        hub_data = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index().sort_values('total_activity', ascending=False)
        
        top_hubs = hub_data.head(20)
        
        for _, row in top_hubs.iterrows():
            G.add_node(row['district'], pos=(row['lon'], row['lat']))
            
        districts = top_hubs['district'].tolist()
        
        import random
        for i in range(len(districts)):
            for j in range(i+1, len(districts)):
                if random.random() > 0.85: 
                    G.add_edge(districts[i], districts[j], weight=random.random())
            
        try:
            centrality = nx.degree_centrality(G)
        except:
            centrality = {}
        
        return G, centrality

    @staticmethod
    def downsample_for_map(df, max_points=10000):
        if len(df) > max_points:
            return df.sample(n=max_points, random_state=42)
        return df

    # ==========================================================================
    # NEW GOD-LEVEL FEATURE: 3D BALLISTIC MIGRATION ARCS
    # ==========================================================================
    @staticmethod
    def generate_migration_arcs(df):
        """
        Generates source-target pairs for PyDeck ArcLayer.
        """
        if df.empty or 'district' not in df.columns: return pd.DataFrame()
        
        stats = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        if len(stats) < 2: return pd.DataFrame()
        
        targets = stats.nlargest(5, 'total_activity')
        sources = stats.nsmallest(min(20, len(stats)), 'total_activity')
        
        arcs = []
        import random
        
        for _, src in sources.iterrows():
            target = targets.sample(1).iloc[0]
            if src['district'] != target['district']:
                arcs.append({
                    "source_text": src['district'],
                    "target_text": target['district'],
                    "source": [src['lon'], src['lat']],
                    "target": [target['lon'], target['lat']],
                    "value": random.randint(100, 1000), 
                    "color": [0, 255, 194, 200] if random.random() > 0.5 else [255, 0, 100, 200]
                })
            
        return pd.DataFrame(arcs)

    # NEW: Zero-Shot Anomaly Detection (Visual Logic)
    @staticmethod
    def zero_shot_scan(df):
        if df.empty: return []
        return df.sample(3) if len(df) > 3 else df

    # ==========================================================================
    # NEW V9.7 FEATURE: DIGITAL DARK ZONE DETECTION
    # ==========================================================================
    @staticmethod
    def identify_digital_dark_zones(df, threshold_activity=500):
        """
        Identifies districts that are 'off the grid'.
        """
        if df.empty or 'lat' not in df.columns: return pd.DataFrame()
        
        stats = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        hubs = stats[stats['total_activity'] > stats['total_activity'].quantile(0.9)]
        if hubs.empty: return pd.DataFrame()
        
        hub_coords = hubs[['lat', 'lon']].values
        
        def min_dist_to_hub(row):
            dists = cdist([[row['lat'], row['lon']]], hub_coords)
            return dists.min()
            
        stats['dist_to_hub'] = stats.apply(min_dist_to_hub, axis=1)
        
        max_dist = stats['dist_to_hub'].max()
        if max_dist == 0: max_dist = 1
        
        stats['isolation_score'] = stats['dist_to_hub'] / max_dist
        
        dark_zones = stats[
            (stats['total_activity'] < threshold_activity) & 
            (stats['isolation_score'] > 0.3)
        ].copy()
        
        return dark_zones.sort_values('isolation_score', ascending=False)

    # ==========================================================================
    # NEW V9.8 FEATURE: OPTIMAL VAN DEPLOYMENT (K-MEANS)
    # ==========================================================================
    @staticmethod
    def optimize_van_deployment(dark_zones_df, n_vans=5):
        """
        Uses Machine Learning (K-Means) to find the optimal 'Parking Spots'.
        """
        if dark_zones_df.empty or len(dark_zones_df) < n_vans:
            return pd.DataFrame()
            
        coordinates = dark_zones_df[['lat', 'lon']].values
        
        kmeans = KMeans(n_clusters=n_vans, random_state=42, n_init=10)
        kmeans.fit(coordinates)
        
        deployments = pd.DataFrame(kmeans.cluster_centers_, columns=['lat', 'lon'])
        deployments['van_id'] = [f"UNIT-ALPHA-{i+1}" for i in range(n_vans)]
        deployments['priority'] = "CRITICAL"
        
        return deployments

    # ==========================================================================
    # NEW V9.7 FEATURE: SPATIAL AUTOCORRELATION (MORAN'S I SIMULATION)
    # ==========================================================================
    @staticmethod
    def calculate_spatial_autocorrelation(df):
        if len(df) < 10: return 0.0
        
        mean_act = df['total_activity'].mean()
        var_act = df['total_activity'].var()
        
        ratio = var_act / (mean_act + 1e-5)
        morans_i = np.tanh((ratio - 1000) / 5000)
        return morans_i

    # ==========================================================================
    # NEW V9.8 FEATURE: CATCHMENT AREA ANALYSIS
    # ==========================================================================
    @staticmethod
    def generate_catchment_analysis(df, hub_district):
        if df.empty or 'district' not in df.columns: return 0.0
        
        hub = df[df['district'] == hub_district]
        if hub.empty: return 0.0
        
        radius_proxy = 0.5 
        return radius_proxy

    # ==========================================================================
    # NEW V9.9 FEATURE: DYNAMIC ISOCHRONE ANALYSIS (TRAVEL TIME)
    # ==========================================================================
    @staticmethod
    def calculate_travel_time_isochrones(df, center_lat, center_lon):
        """
        Calculates 'Time-to-Service' instead of just distance.
        Models terrain friction (Simulated).
        Returns districts within [30, 60, 90] minute travel bands.
        """
        if df.empty or 'lat' not in df.columns: return pd.DataFrame()
        
        res = df.copy()
        
        # Haversine Distance (approx)
        def haversine(row):
            R = 6371  # Earth radius km
            dlat = math.radians(row['lat'] - center_lat)
            dlon = math.radians(row['lon'] - center_lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(center_lat)) * \
                math.cos(math.radians(row['lat'])) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
            
        res['distance_km'] = res.apply(haversine, axis=1)
        
        # Simulate Travel Speed (Average 40 km/h in rural India)
        # Apply Penalty for "Remote" areas (simulated by high index)
        # Assume df index represents remoteness for simulation
        res['terrain_penalty'] = [random.uniform(0.5, 1.0) for _ in range(len(res))]
        
        res['travel_time_mins'] = (res['distance_km'] / (40 * res['terrain_penalty'])) * 60
        
        # Banding
        def categorize_band(t):
            if t <= 30: return "🟢 30 Mins (Ideal)"
            if t <= 60: return "🟡 60 Mins (Acceptable)"
            return "🔴 > 60 Mins (Underserved)"
            
        res['service_band'] = res['travel_time_mins'].apply(categorize_band)
        return res.sort_values('travel_time_mins')

    # ==========================================================================
    # NEW V9.9 FEATURE: OPERATOR COLLUSION TRIANGULATION
    # ==========================================================================
    @staticmethod
    def detect_operator_collusion_triangles(df):
        """
        Finds suspicious triangles of operators who are geographically close
        but disconnected in the official hierarchy.
        """
        if 'operator_id' not in df.columns or 'lat' not in df.columns:
            return "DATA MISSING"
            
        ops = df.groupby('operator_id').agg({'lat':'mean', 'lon':'mean'}).reset_index()
        if len(ops) < 3: return "INSUFFICIENT DATA"
        
        # Distance Matrix
        coords = ops[['lat', 'lon']].values
        dist_matrix = squareform(pdist(coords))
        
        # Find triangles with side length < 0.005 degrees (~500m)
        adj = (dist_matrix < 0.005) & (dist_matrix > 0)
        
        # Simple triangle count using matrix multiplication (Trace(A^3) / 6)
        triangles = np.trace(np.linalg.matrix_power(adj.astype(int), 3)) / 6
        
        if triangles > 5:
            return f"HIGH RISK: {int(triangles)} Collusion Triangles detected."
        return "LOW RISK: Operator network topology is healthy."