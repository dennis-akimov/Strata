# syntax=docker/dockerfile:1
#
# Strata's chat server (the Rust server, server/) in a container.  A Linux container on a Mac has no Metal, so the
# engine stays on the host: the server starts it through strata-relay (make relay runs the host side; README below).
#
#   docker compose up -d --build        (from server/, with server/.env)   or: make up
#
# Build context: the repository root (the web app's files are in serve/web); Dockerfile.dockerignore keeps it small.

FROM rust:1.90-slim-bookworm AS build
WORKDIR /src/server
COPY server/Cargo.toml server/Cargo.lock ./
COPY server/src ./src
# the registry and the target folder are cached between builds; the binaries are copied out of the cache mount
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/server/target \
    cargo build --release --locked --bin strata-server --bin strata-relay && \
    cp target/release/strata-server target/release/strata-relay /usr/local/bin/

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1000 --create-home strata
COPY --from=build /usr/local/bin/strata-server /usr/local/bin/strata-relay /usr/local/bin/
COPY serve/web /strata/serve/web
COPY serve/chat_template.jinja /strata/serve/chat_template.jinja
ENV STRATA_ROOT=/strata
USER strata
EXPOSE 8090
# the port answers once the server listens (it listens before the model has loaded; /health says when it is ready)
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8090"]
ENTRYPOINT ["strata-server"]
