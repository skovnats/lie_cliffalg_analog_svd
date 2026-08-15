# Build context for this Dockerfile MUST be the repository root (the main
# crate's own directory, one level up from `compare/`), since the compare
# crate reaches the main crate via a `path = ".."` dependency:
#
#   docker build -f compare/Dockerfile -t lie_svd_reference_compare .
#
# This mirrors, deliberately, the isolation `compare/Cargo.toml` already
# sets up: the main crate's own `Dockerfile` (repository root) never reads
# this file or anything under `compare/`, and vice versa -- two completely
# independent images from two completely independent dependency trees,
# sharing only the main crate's source as a read-only reference.
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    m4 \
    build-essential \
    diffutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml ./Cargo.toml
COPY src ./src
COPY compare/Cargo.toml ./compare/Cargo.toml
COPY compare/src ./compare/src

ENV RUSTFLAGS="-C target-cpu=native"

WORKDIR /app/compare
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/compare/target/release/lie_svd_reference_compare /usr/local/bin/lie_svd_reference_compare

CMD ["lie_svd_reference_compare"]
