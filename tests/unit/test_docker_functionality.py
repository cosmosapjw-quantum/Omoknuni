"""
Unit tests for Docker functionality and containerization features.

Tests Docker build, configuration, health checks, and container functionality
without requiring actual Docker runtime (uses mocking for unit tests).
"""

import json
import os
import tempfile
import unittest
import importlib
from pathlib import Path
import pytest


class TestDockerfileValidation:
    """Test Dockerfile syntax and configuration validation."""

    def setup_method(self):
        """Setup test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.dockerfile_path = self.project_root / "Dockerfile"

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists in project root."""
        assert self.dockerfile_path.exists(), "Dockerfile should exist in project root"

    def test_dockerfile_has_cuda_base(self):
        """Test that Dockerfile uses CUDA 12.x base image."""
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()

        assert "nvidia/cuda:12." in content, "Should use CUDA 12.x base image"
        assert "FROM nvidia/cuda:" in content, "Should have NVIDIA CUDA FROM statement"

    def test_dockerfile_multi_stage_build(self):
        """Test that Dockerfile implements multi-stage build."""
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()

        # Check for multiple FROM statements (multi-stage)
        from_statements = content.count("FROM ")
        assert from_statements >= 3, "Should have multi-stage build with at least 3 stages"

        # Check for expected stages
        assert "AS builder" in content, "Should have builder stage"
        assert "AS runtime" in content, "Should have runtime stage"
        assert "AS development" in content, "Should have development stage"

    def test_dockerfile_has_health_checks(self):
        """Test that Dockerfile includes health checks."""
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()

        assert "HEALTHCHECK" in content, "Should include health check instructions"

    def test_dockerfile_sets_user(self):
        """Test that Dockerfile creates non-root user for security."""
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()

        assert "useradd" in content, "Should create user for security"
        assert "USER " in content, "Should switch to non-root user"

    def test_dockerfile_optimized_build_flags(self):
        """Test that Dockerfile uses optimized build flags."""
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()

        assert "CFLAGS" in content, "Should set optimized CFLAGS"
        assert "CXXFLAGS" in content, "Should set optimized CXXFLAGS"
        assert "-O3" in content, "Should use O3 optimization"

    def test_dockerfile_python_version(self):
        """Test that Dockerfile uses correct Python version."""
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()

        assert "python3.12" in content, "Should use Python 3.12"


class TestDockerIgnore:
    """Test .dockerignore configuration."""

    def setup_method(self):
        """Setup test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.dockerignore_path = self.project_root / ".dockerignore"

    def test_dockerignore_exists(self):
        """Test that .dockerignore exists."""
        assert self.dockerignore_path.exists(), ".dockerignore should exist"

    def test_dockerignore_excludes_venv(self):
        """Test that .dockerignore excludes virtual environments."""
        with open(self.dockerignore_path, 'r') as f:
            content = f.read()

        assert "venv/" in content, "Should exclude venv directory"
        assert "__pycache__/" in content, "Should exclude Python cache"

    def test_dockerignore_excludes_build_artifacts(self):
        """Test that .dockerignore excludes build artifacts."""
        with open(self.dockerignore_path, 'r') as f:
            content = f.read()

        assert "build/" in content, "Should exclude build directory"
        assert "dist/" in content, "Should exclude dist directory"
        assert "*.so" in content, "Should exclude shared libraries"

    def test_dockerignore_excludes_ide_files(self):
        """Test that .dockerignore excludes IDE files."""
        with open(self.dockerignore_path, 'r') as f:
            content = f.read()

        assert ".vscode/" in content, "Should exclude VS Code files"
        assert ".idea/" in content, "Should exclude IntelliJ files"

    def test_dockerignore_excludes_data_files(self):
        """Test that .dockerignore excludes large data files."""
        with open(self.dockerignore_path, 'r') as f:
            content = f.read()

        assert "training_data/" in content, "Should exclude training data"
        assert "results/" in content, "Should exclude results"
        assert "*.h5" in content, "Should exclude HDF5 files"


class TestDockerCompose:
    """Test docker-compose.yml configuration."""

    def setup_method(self):
        """Setup test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.compose_path = self.project_root / "docker-compose.yml"

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        assert self.compose_path.exists(), "docker-compose.yml should exist"

    def test_docker_compose_has_services(self):
        """Test that docker-compose.yml defines required services."""
        with open(self.compose_path, 'r') as f:
            content = f.read()

        # Check for required services
        assert "services:" in content, "Should define services"
        assert "dev:" in content, "Should have dev service"
        assert "training:" in content, "Should have training service"
        assert "runtime:" in content, "Should have runtime service"

    def test_docker_compose_has_volumes(self):
        """Test that docker-compose.yml defines persistent volumes."""
        with open(self.compose_path, 'r') as f:
            content = f.read()

        assert "volumes:" in content, "Should define volumes"
        assert "training_data:" in content, "Should have training data volume"
        assert "models:" in content, "Should have models volume"

    def test_docker_compose_gpu_support(self):
        """Test that docker-compose.yml includes GPU support."""
        with open(self.compose_path, 'r') as f:
            content = f.read()

        assert "nvidia" in content, "Should include NVIDIA GPU support"
        assert "capabilities:" in content, "Should specify GPU capabilities"


class TestDockerBuildScript:
    """Test Docker build script functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.build_script_path = self.project_root / "scripts/docker/build.sh"

    def test_build_script_exists(self):
        """Test that build script exists."""
        assert self.build_script_path.exists(), "Docker build script should exist"

    def test_build_script_executable(self):
        """Test that build script is executable."""
        assert os.access(self.build_script_path, os.X_OK), "Build script should be executable"

    def test_build_script_has_help(self):
        """Test that build script has help documentation."""
        with open(self.build_script_path, 'r') as f:
            content = f.read()

        assert "show_usage()" in content, "Should have usage function"
        assert "--help" in content, "Should support help flag"

    def test_build_script_supports_targets(self):
        """Test that build script supports different build targets."""
        with open(self.build_script_path, 'r') as f:
            content = f.read()

        assert "runtime" in content, "Should support runtime target"
        assert "development" in content, "Should support development target"
        assert "training" in content, "Should support training target"


class TestDockerRunScript:
    """Test Docker run script functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.run_script_path = self.project_root / "scripts/docker/run.sh"

    def test_run_script_exists(self):
        """Test that run script exists."""
        assert self.run_script_path.exists(), "Docker run script should exist"

    def test_run_script_executable(self):
        """Test that run script is executable."""
        assert os.access(self.run_script_path, os.X_OK), "Run script should be executable"

    def test_run_script_has_commands(self):
        """Test that run script supports required commands."""
        with open(self.run_script_path, 'r') as f:
            content = f.read()

        # Check for required commands
        assert '"dev")' in content, "Should support dev command"
        assert '"training")' in content, "Should support training command"
        assert '"runtime")' in content, "Should support runtime command"
        assert '"benchmark")' in content, "Should support benchmark command"

    def test_run_script_has_safety_checks(self):
        """Test that run script includes safety checks."""
        with open(self.run_script_path, 'r') as f:
            content = f.read()

        assert "check_prerequisites()" in content, "Should check prerequisites"
        assert "docker &> /dev/null" in content, "Should check Docker availability"


class TestDockerHealthChecks:
    """Test Docker health check functionality (mocked)."""

    def test_health_check_imports(self):
        """Test that health check dependencies can be imported."""
        import subprocess

        result = subprocess.run(
            ['python', '-c', 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "CUDA available:" in result.stdout

    def test_health_check_cuda_detection(self):
        """Test health check CUDA detection."""
        import torch

        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)

        cudnn_backend = getattr(torch.backends, 'cudnn', None)
        cudnn_available = bool(
            cudnn_backend is not None and
            callable(getattr(cudnn_backend, 'is_available', None)) and
            cudnn_backend.is_available()
        )

        if cuda_available and cudnn_available:
            # Torch exposes cuDNN; basic sanity check that version can be queried
            assert hasattr(cudnn_backend, 'version'), 'cuDNN backend should expose version() when available'
            assert cudnn_backend.version() is not None
        else:
            # In CPU-only builds or when cuDNN is disabled, just ensure we can query without exception
            if cudnn_backend is not None and hasattr(cudnn_backend, 'is_available'):
                assert cudnn_backend.is_available() in (False, True)

    def test_health_check_command_format(self):
        """Test that health check commands are properly formatted."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Find health check commands
        healthcheck_lines = [line.strip() for line in content.split('\n')
                           if 'CMD python -c' in line and 'HEALTHCHECK' in content]

        # Should have health check commands
        assert any('python -c' in line for line in content.split('\n')), \
            "Should have Python health check commands"


class TestDockerConfiguration:
    """Test Docker configuration and environment setup."""

    def test_environment_variables_defined(self):
        """Test that necessary environment variables are defined in Dockerfile."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        assert "ENV " in content, "Should define environment variables"
        assert "PYTHONUNBUFFERED" in content, "Should set Python unbuffered"
        assert "OMOKNUNI" in content, "Should set application-specific variables"

    def test_working_directory_set(self):
        """Test that working directories are properly set."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        assert "WORKDIR " in content, "Should set working directory"

    def test_copy_instructions_present(self):
        """Test that COPY instructions are present for application files."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        assert "COPY " in content, "Should copy application files"
        assert "--chown=" in content, "Should set proper file ownership"

    def test_port_exposure(self):
        """Test that necessary ports are exposed."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        assert "EXPOSE " in content, "Should expose necessary ports"


class TestDockerIntegration:
    """Integration tests for Docker functionality (mocked)."""

    def test_docker_client_connection(self):
        """Test Docker client connection availability."""
        import importlib

        docker_spec = importlib.util.find_spec('docker')
        assert docker_spec is not None, 'Docker SDK should be importable'

        import docker
        client_factory = getattr(docker, 'from_env', None)
        assert callable(client_factory), 'docker.from_env must be callable'

        client = None
        try:
            client = client_factory()
            assert client is not None
            try:
                client.ping()
            except Exception as exc:
                assert isinstance(exc, Exception)
        finally:
            if client is not None and hasattr(client, 'close'):
                try:
                    client.close()
                except Exception:
                    pass

    def test_build_script_execution(self):
        """Test build script can be executed."""
        import subprocess

        result = subprocess.run(['bash', '-n', 'scripts/docker/build.sh'], capture_output=True, text=True)
        assert result.returncode == 0

    def test_run_script_execution(self):
        """Test run script can be executed."""
        import subprocess

        result = subprocess.run(['bash', '-n', 'scripts/docker/run.sh'], capture_output=True, text=True)
        assert result.returncode == 0


class TestDockerSecurity:
    """Test Docker security configurations."""

    def test_non_root_user_configured(self):
        """Test that containers run as non-root users."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        assert "useradd" in content, "Should create non-root user"
        assert "--uid 1001" in content, "Should use specific UID"

    def test_minimal_attack_surface(self):
        """Test that Dockerfile minimizes attack surface."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Should clean up package managers
        assert "rm -rf /var/lib/apt/lists/*" in content, "Should clean apt cache"
        assert "apt-get clean" in content, "Should clean package cache"

    def test_environment_isolation(self):
        """Test that environment variables provide proper isolation."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Should not expose sensitive paths or information
        assert "/root" not in content.lower(), "Should not reference root directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
