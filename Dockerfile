# syntax=docker/dockerfile:1
#
# Fraud risk operations console (Streamlit) for the IEEE-CIS fraud model.
#
# The image contains the app code, src/, the trained bundle (app/model.pkl), the
# result files in outputs/ and the inference artifacts in artifacts/ (form model,
# category lists, card history), all read-only. The Kaggle-derived feature table is NOT
# baked in. Mount data/processed/ read-only at runtime to enable live SHAP
# explanations on the Transaction investigation page.
#
# Build:  docker build -t validax-console .
# Run:    see README.md, "Deployment"

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4 \
    MALLOC_ARENA_MAX=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8501

# libgomp1: OpenMP runtime required by the xgboost wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/fraud-console

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY outputs/ outputs/
COPY artifacts/ artifacts/
COPY scripts/run_app.sh scripts/run_app.sh
COPY .streamlit/config.toml .streamlit/config.toml

# Normalise line endings (the script may be checked out on Windows), create the
# mount point for the optional feature table, and make the artifacts read-only
# for the unprivileged runtime user.
RUN sed -i 's/\r$//' scripts/run_app.sh \
    && chmod 0555 scripts/run_app.sh \
    && mkdir -p data/processed \
    && useradd --create-home --uid 10001 appuser \
    && chmod -R a-w app outputs artifacts src

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["./scripts/run_app.sh"]
