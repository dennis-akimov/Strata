# Strata for an x86_64 Linux PC, built on another machine (a Mac: Docker's amd64 containers run under Rosetta):
# Ubuntu 24.04, the GTK / WebKitGTK / tray libraries the app links, and Rust.  make linux-package uses it.
FROM --platform=linux/amd64 ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config libssl-dev curl git ca-certificates python3 \
      libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev libxdo-dev librsvg2-dev \
      libvulkan-dev glslc glslang-tools spirv-headers \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
