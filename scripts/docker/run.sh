#!/bin/bash
# Docker run script for AlphaZero Engine
# Provides convenient commands for running different services

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-alphazero}"
TAG="${TAG:-latest}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

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
Docker Run Script for AlphaZero Engine

Usage: $0 COMMAND [OPTIONS]

Commands:
    dev                     Start development environment with Jupyter
    training               Start training service
    runtime                Start production runtime
    benchmark              Run performance benchmarks
    tensorboard            Start TensorBoard monitoring
    shell                  Open shell in running container
    logs                   Show logs for a service
    stop                   Stop all services
    clean                  Clean up containers and volumes

Options:
    -d, --detach           Run in detached mode
    -f, --follow           Follow log output
    -h, --help             Show this help message

Examples:
    # Start development environment
    $0 dev

    # Start training in background
    $0 training -d

    # View training logs
    $0 logs training

    # Run benchmarks
    $0 benchmark

    # Clean up everything
    $0 clean

    # Open shell in development container
    $0 shell dev
EOF
}

# Check prerequisites
check_prerequisites() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
}

# Function to start development environment
start_dev() {
    local detach_flag=""
    if [[ "${1:-}" == "-d" ]] || [[ "${1:-}" == "--detach" ]]; then
        detach_flag="-d"
    fi

    log_info "Starting development environment..."
    docker-compose -f "$COMPOSE_FILE" up $detach_flag dev

    if [[ -z "$detach_flag" ]]; then
        log_info "Development environment started. Jupyter Lab available at http://localhost:8888"
    else
        log_success "Development environment started in background"
        log_info "Access Jupyter Lab at http://localhost:8888"
        log_info "View logs with: $0 logs dev"
    fi
}

# Function to start training
start_training() {
    local detach_flag=""
    if [[ "${1:-}" == "-d" ]] || [[ "${1:-}" == "--detach" ]]; then
        detach_flag="-d"
    fi

    log_info "Starting training service..."
    docker-compose -f "$COMPOSE_FILE" up $detach_flag training

    if [[ -n "$detach_flag" ]]; then
        log_success "Training service started in background"
        log_info "Monitor progress with: $0 logs training"
        log_info "TensorBoard available at http://localhost:6007"
    fi
}

# Function to start runtime
start_runtime() {
    local detach_flag=""
    if [[ "${1:-}" == "-d" ]] || [[ "${1:-}" == "--detach" ]]; then
        detach_flag="-d"
    fi

    log_info "Starting runtime service..."
    docker-compose -f "$COMPOSE_FILE" up $detach_flag runtime

    if [[ -n "$detach_flag" ]]; then
        log_success "Runtime service started in background"
    fi
}

# Function to run benchmarks
run_benchmark() {
    log_info "Running performance benchmarks..."
    docker-compose -f "$COMPOSE_FILE" run --rm benchmark

    log_success "Benchmarks completed"
}

# Function to start TensorBoard
start_tensorboard() {
    local detach_flag=""
    if [[ "${1:-}" == "-d" ]] || [[ "${1:-}" == "--detach" ]]; then
        detach_flag="-d"
    fi

    log_info "Starting TensorBoard..."
    docker-compose -f "$COMPOSE_FILE" up $detach_flag tensorboard

    if [[ -n "$detach_flag" ]]; then
        log_success "TensorBoard started in background"
        log_info "Access TensorBoard at http://localhost:6008"
    fi
}

# Function to open shell in container
open_shell() {
    local service="${1:-dev}"

    log_info "Opening shell in $service container..."

    # Check if container is running
    if ! docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
        log_info "Starting $service container..."
        docker-compose -f "$COMPOSE_FILE" up -d "$service"
        sleep 5
    fi

    docker-compose -f "$COMPOSE_FILE" exec "$service" /bin/bash
}

# Function to show logs
show_logs() {
    local service="${1:-}"
    local follow_flag=""

    if [[ "${2:-}" == "-f" ]] || [[ "${2:-}" == "--follow" ]]; then
        follow_flag="-f"
    fi

    if [[ -z "$service" ]]; then
        log_info "Showing logs for all services..."
        docker-compose -f "$COMPOSE_FILE" logs $follow_flag
    else
        log_info "Showing logs for $service..."
        docker-compose -f "$COMPOSE_FILE" logs $follow_flag "$service"
    fi
}

# Function to stop services
stop_services() {
    log_info "Stopping all services..."
    docker-compose -f "$COMPOSE_FILE" down
    log_success "All services stopped"
}

# Function to clean up
clean_up() {
    log_warning "This will remove all containers, images, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up containers and volumes..."
        docker-compose -f "$COMPOSE_FILE" down -v --rmi local

        # Remove unused Docker objects
        docker system prune -f

        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Main function
main() {
    check_prerequisites

    local command="${1:-help}"
    shift || true

    case "$command" in
        "dev")
            start_dev "$@"
            ;;
        "training")
            start_training "$@"
            ;;
        "runtime")
            start_runtime "$@"
            ;;
        "benchmark")
            run_benchmark "$@"
            ;;
        "tensorboard")
            start_tensorboard "$@"
            ;;
        "shell")
            open_shell "$@"
            ;;
        "logs")
            show_logs "$@"
            ;;
        "stop")
            stop_services "$@"
            ;;
        "clean")
            clean_up "$@"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"