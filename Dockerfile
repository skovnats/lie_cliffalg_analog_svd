FROM rust:1-bookworm AS builder

WORKDIR /app

COPY Cargo.toml ./
COPY Cargo.lock ./
COPY src ./src

ENV RUSTFLAGS="-C target-cpu=native"

RUN cargo test --release --lib --locked
RUN cargo build --release --bin stress_cpu --locked

FROM debian:bookworm-slim

WORKDIR /app
COPY --from=builder /app/target/release/stress_cpu /usr/local/bin/stress_cpu

CMD ["stress_cpu", "64", "--analog"]
