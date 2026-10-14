# vplan Jobs image (spec §4 "Jobs container", docs/plan.md W3.4 R8): CPU only, exactly the uv.lock dependencies,
# headless PyBullet. Build from the repo root:  docker build --file docker/Dockerfile --tag vplan:local .
# linux/amd64 only: pybullet 3.2.7 has a CPython 3.10 Linux wheel for x86_64 only, so no source build is needed.
#
# Base: python:3.10.22-slim-trixie, which python:3.10-slim pointed to on 2026-10-15. The digest is the multi-arch
# index from Docker Hub; its linux/amd64 manifest is sha256:7b6bd969dda3ecfca0655524e338b57a8b8cd692a9362e3d55b5a3ef49b07256.
FROM python:3.10.22-slim-trixie@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV PYTHONUTF8=1 \
    MANIPULAPY_QUIET=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependencies before sources, so a source edit reuses this layer. docker/requirements.txt is uv.lock exported with
# hashes (its header has the command; tests/test_docker.py keeps the two equal). --require-hashes accepts only the
# locked files, --no-deps installs nothing the lock does not list, and --only-binary refuses source builds.
COPY docker/requirements.txt docker/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check \
        --no-deps --require-hashes --only-binary=:all: -r docker/requirements.txt

# What the package and its tests read: tests import scripts/spikes, tests/test_docker.py reads .dockerignore,
# uv.lock and docker/, and LICENSE carries the AGPL text with the image. config/ must sit next to src/, because
# vplan.config resolves it from the source tree (so the package runs from /app/src rather than site-packages).
COPY .dockerignore LICENSE pyproject.toml uv.lock ./
COPY config/ config/
COPY docker/ docker/
COPY scripts/ scripts/
COPY src/ src/
COPY tests/ tests/

# Root owns the tree and the runtime user can only read it, so a run cannot change the sources its manifest hashes.
# The explicit modes also undo the 0777 that a checkout under /mnt/c reports.
RUN find /app -type d -exec chmod 0755 {} + \
    && find /app -type f -exec chmod 0644 {} + \
    && groupadd --gid 10001 vplan \
    && useradd --no-log-init --create-home --uid 10001 --gid vplan vplan
USER vplan

# Ties the image ID that run manifests record back to a commit (run-local.sh passes it).
ARG VPLAN_GIT_SHA=unknown
LABEL org.opencontainers.image.revision=$VPLAN_GIT_SHA

ENTRYPOINT ["python", "-m", "vplan"]
