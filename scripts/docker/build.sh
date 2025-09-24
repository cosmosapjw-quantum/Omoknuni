#!/bin/bash
# Docker build script for AlphaZero Engine
# Supports multi-stage builds with caching and optimization

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-alphazero}"
TAG="${TAG:-latest}"
TARGET="${TARGET:-runtime}"
CACHE="${CACHE:-true}"
PUSH="${PUSH:-false}"
PLATFORM="${PLATFORM:-linux/amd64}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Docker Build Script for AlphaZero Engine

Usage: $0 [OPTIONS]

Options:
    -t, --target TARGET     Build target (runtime, development, training) [default: runtime]
    -r, --registry REGISTRY Docker registry prefix [default: alphazero]
    -g, --tag TAG           Docker image tag [default: latest]
    -p, --push              Push image to registry after build
    -n, --no-cache          Disable build cache
    -h, --help              Show this help message

Environment Variables:
    REGISTRY                Docker registry prefix
    TAG                     Docker image tag
    TARGET                  Build target
    CACHE                   Enable/disable cache (true/false)
    PUSH                    Push to registry (true/false)
    PLATFORM                Target platform (linux/amd64, linux/arm64)

Examples:
    # Build runtime image
    $0 -t runtime

    # Build and push development image
    $0 -t development -p

    # Build with custom registry and tag
    $0 -r myregistry.com/alphazero -g v1.0.0

    # Build all targets
    $0 --target all
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -g|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -n|--no-cache)
            CACHE=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate Docker installation
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check for NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_warning "NVIDIA Docker support not available - GPU features will be disabled"
fi

# Build cache options
CACHE_FROM_ARGS=""
CACHE_TO_ARGS=""
if [[ "$CACHE" == "true" ]]; then
    CACHE_FROM_ARGS="--cache-from=type=local,src=/tmp/.buildx-cache"
    CACHE_TO_ARGS="--cache-to=type=local,dest=/tmp/.buildx-cache-new,mode=max"
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Function to build a specific target
build_target() {
    local target=$1
    local image_name="${REGISTRY}/alphazero-engine:${target}-${TAG}"

    log_info "Building target: $target"
    log_info "Image name: $image_name"

    # Build command
    docker build \
        --target "$target" \
        --tag "$image_name" \
        --platform "$PLATFORM" \
        $CACHE_FROM_ARGS \
        $CACHE_TO_ARGS \
        --progress=plain \
        .

    # Tag as latest for the target
    docker tag "$image_name" "${REGISTRY}/alphazero-engine:${target}-latest"

    log_success "Built $image_name successfully"

    # Push if requested
    if [[ "$PUSH" == "true" ]]; then
        log_info "Pushing $image_name to registry..."
        docker push "$image_name"
        docker push "${REGISTRY}/alphazero-engine:${target}-latest"
        log_success "Pushed $image_name successfully"
    fi
}

# Main build logic
main() {
    log_info "Starting Docker build process..."
    log_info "Configuration:"
    log_info "  Registry: $REGISTRY"
    log_info "  Tag: $TAG"
    log_info "  Target: $TARGET"
    log_info "  Platform: $PLATFORM"
    log_info "  Cache: $CACHE"
    log_info "  Push: $PUSH"

    # Clean up old cache if using new cache
    if [[ "$CACHE" == "true" ]] && [[ -d "/tmp/.buildx-cache-new" ]]; then
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
    fi

    case "$TARGET" in
        "all")
            log_info "Building all targets..."
            build_target "runtime"
            build_target "development"
            build_target "training"
            ;;
        "runtime"|"development"|"training")
            build_target "$TARGET"
            ;;
        *)
            log_error "Invalid target: $TARGET"
            log_error "Valid targets: runtime, development, training, all"
            exit 1
            ;;
    esac

    log_success "Docker build process completed successfully!"

    # Show final images
    log_info "Built images:"
    docker images --filter "reference=${REGISTRY}/alphazero-engine" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

# Run main function
main