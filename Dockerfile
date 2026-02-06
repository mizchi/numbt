# Dockerfile for debugging Linux LAPACK issues
# Use x86_64 platform since MoonBit doesn't support Linux aarch64
FROM --platform=linux/amd64 ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Install MoonBit
RUN curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash
ENV PATH="/root/.moon/bin:${PATH}"

WORKDIR /app

# Copy source
COPY . .

# Configure link flags for Linux
RUN cat > src/moon.pkg << 'EOF'
import {
  "mizchi/blas" as @blas,
  "moonbitlang/core/math" as @math,
}
options(
  link: { "native": { "cc-link-flags": "-lopenblas -llapack -lm" } },
  "native-stub": [ "numbt_stub.c" ],
  "supported-targets": [ "native" ],
)
EOF

# Update dependencies
RUN moon update

CMD ["bash"]
