"""Minimal Docker SDK compatibility errors."""

class DockerException(Exception):
    """Base exception used by the lightweight Docker compatibility layer."""


class APINotAvailable(DockerException):
    """Raised when the Docker API endpoint is not reachable."""

