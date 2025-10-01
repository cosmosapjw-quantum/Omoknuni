"""Lightweight Docker SDK compatibility layer for test environments.

This module provides a minimal subset of the python-docker API that the unit
tests rely on. It does not attempt to interact with a real Docker daemon, but
instead raises explicit exceptions to signal the absence of Docker services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import errors


@dataclass
class _DockerEnvironment:
    """Represents a minimal Docker environment configuration."""

    host: str = "unix://var/run/docker.sock"
    tls: bool = False


class DockerClient:
    """Stand-in client that mimics the docker.DockerClient API surface."""

    def __init__(self, environment: Optional[_DockerEnvironment] = None):
        self.environment = environment or _DockerEnvironment()

    def ping(self) -> None:
        """Ping the Docker daemon.

        Raises:
            docker.errors.DockerException: Always, to indicate that the Docker
                daemon is not available in the current test environment.
        """
        raise errors.DockerException(
            "Docker daemon not available in the sandboxed test environment"
        )

    def close(self) -> None:
        """Close the client (no-op for the compatibility layer)."""
        return None


def from_env() -> DockerClient:
    """Create a Docker client using environment configuration.

    Returns:
        DockerClient: Compatibility client that raises informative errors when
            pinged.
    """
    return DockerClient()


__all__ = [
    "DockerClient",
    "from_env",
    "errors",
]

