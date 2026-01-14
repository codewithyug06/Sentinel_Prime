# ==============================================================================
# SENTINEL PRIME V9.9 | SOVEREIGN CONTAINER PROTOCOL
# ENTERPRISE-GRADE DOCKERFILE FOR HIGH-SECURITY DEPLOYMENT
# COMPLIANCE: UIDAI Data Security Guidelines 2026 | ISO 27001
# CAPABILITIES: 
#   - Physics-Informed Neural Networks (PINNs)
#   - Spatiotemporal Graph Convolution (ST-GCN)
#   - Cross-Lingual Voice Processing (Whisper/FFmpeg)
#   - Vector Search (ChromaDB/Legal-RAG)
#   - Distributed Computing (Ray/Dask Clusters)
# ==============================================================================

# 1. BASE IMAGE SELECTION
# Using Python 3.9 Slim for a balance of size and scientific library compatibility (Wheels)
FROM python:3.9-slim as base

# 2. METADATA LABELS (GOVERNANCE & AUDIT TRAIL)
LABEL maintainer="Sentinel Prime Team <admin@sentinel-prime.gov>"
LABEL version="9.9.0-AEGIS-GOD-MODE"
LABEL description="Sovereign Digital Twin Analytics Engine for Aadhaar Hackathon 2026"
LABEL security.compliance="Zero-Trust/Local-Compute/Non-Root"
LABEL module.audio="Voice Uplink Enabled (FFmpeg/LibSndFile)"
LABEL module.compute="Ray/Dask Distributed Ready"
LABEL module.viz="Graphviz/OpenGL Enabled"

# 3. ENVIRONMENT CONFIGURATION (PERFORMANCE TUNING)
# Prevent Python from writing pyc files to disc (Security & Size)
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure stdout/stderr are flushed immediately (Real-time Logging)
ENV PYTHONUNBUFFERED=1
# Set Locale settings to UTF-8 for Multi-lingual NLP (Hindi/Tamil)
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Hardcode Timezone to IST (Indian Standard Time) for UIDAI Audit Accuracy
ENV TZ=Asia/Kolkata

# Sentinel Config Defaults (Override these in docker run)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV OPENAI_API_KEY=""
ENV MAPBOX_TOKEN=""

# High-Performance Math Acceleration (Intel MKL / OpenMP)
# Critical for Bayesian Neural Networks & PINN Differential Equations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Ray/Dask Distributed Config
ENV RAY_DEDUP_LOGS=0
ENV DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=30s

# 4. SYSTEM DEPENDENCY LAYER (THE HEAVY LIFTING)
# Install build essentials for compiling PyTorch/SciPy/NumPy extensions
# Install GDAL/GEOS for Advanced Geospatial Engine (Spatial.py)
# Install Fonts for FPDF Executive Brief generation (Cognitive.py) with Indian Language Support
# Install FFmpeg & libsndfile for Voice Interface (librosa/audio)
# Install Graphviz for Causal Graph Visualization
# Install BLAS/LAPACK for Matrix Acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    curl \
    git \
    wget \
    libgdal-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgeos-dev \
    graphviz \
    libgl1-mesa-glx \
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    fontconfig \
    ffmpeg \
    libsndfile1 \
    tzdata \
    iputils-ping \
    netcat-openbsd \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && fc-cache -f -v \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. WORKSPACE SETUP
WORKDIR /app

# 6. PYTHON DEPENDENCY LAYER (OPTIMIZED CACHING)
# Copy requirements first. Docker will cache this layer if requirements.txt doesn't change.
COPY requirements.txt .

# Upgrade pip and install dependencies with extended timeout for large ML wheels
# Installs: torch, pandas, numpy, scikit-learn, networkx, pydeck, streamlit, fpdf, h3, shapely
# NEW V9.9: Installs librosa, chromadb, dask, distributed, ray, scipy, graphviz
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Install production server utilities & distributed computing agents explicitly
RUN pip install --no-cache-dir streamlit gunicorn uvicorn dask[complete] distributed ray[default]

# 7. SOURCE CODE INGESTION
# Copy the entire project structure into the container
COPY . .

# 8. SECURITY HARDENING: NON-ROOT USER
# Create a specialized user 'sentinel' to run the application
# This prevents potential privilege escalation attacks (Zero-Trust)
RUN groupadd -r sentinel && useradd -r -g sentinel sentinel

# Create specific directories for Vector DB, Logs, and Reports with correct permissions
# Crucial for Legal-RAG (ChromaDB) to write without root privileges
# Creates /app/data/chromadb_store for persistent vector storage
# Creates /app/reports for generated PDFs
RUN mkdir -p /app/data/chromadb_store \
             /app/data/processed \
             /app/logs \
             /app/outputs \
             /app/reports \
             /app/models && \
    chown -R sentinel:sentinel /app

# Switch context to non-root user
USER sentinel

# 9. HEALTHCHECK PROTOCOL
# Periodically pings the Streamlit health endpoint to ensure the Aegis Command is active
# Fails if the dashboard hangs during heavy GNN computation
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 10. NETWORK EXPOSURE
# Expose Streamlit Dashboard Port
EXPOSE 8501
# Expose API Port (Reserved for future microservices/FastAPI)
EXPOSE 8000
# NEW V9.9: Expose Dask Scheduler & Dashboard Ports
EXPOSE 8786
EXPOSE 8787 
# NEW V9.9: Expose Ray Dashboard Port for Cluster Visualization
EXPOSE 8265

# 11. EXECUTION ENTRYPOINT
# By default, runs the Main Analysis Pipeline (God Mode Output)
# To run the Dashboard, override this CMD with: ["streamlit", "run", "interface/dashboard.py"]
# Configures Streamlit to be headless and listen on all interfaces
CMD ["python", "main.py"]