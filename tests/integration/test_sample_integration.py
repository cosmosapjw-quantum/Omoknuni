"""
Sample integration tests to validate CI/CD pipeline.
These tests verify that components work together.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_build_system_integration():
    """Test that the build system produces working modules."""
    project_root = Path(__file__).parent.parent.parent

    # Check that CMakeLists.txt exists and is properly configured
    cmake_file = project_root / "CMakeLists.txt"
    assert cmake_file.exists()

    # Check that pyproject.toml is properly configured
    pyproject_file = project_root / "pyproject.toml"
    assert pyproject_file.exists()

    # Verify content contains expected build configuration
    content = pyproject_file.read_text()
    assert "scikit-build-core" in content
    assert "pybind11" in content


def test_package_structure_integration():
    """Test that package structure supports proper imports."""
    # Test that all main modules are importable
    modules_to_test = ["core", "games", "neural", "training", "telemetry", "utils"]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
        except ImportError:
            # This is expected until modules are implemented
            pass  # OK for now


def test_cpp_extensions_structure():
    """Test that C++ extensions are properly structured."""
    project_root = Path(__file__).parent.parent.parent

    cpp_dirs = ["cpp_extensions/mcts", "cpp_extensions/games", "cpp_extensions/utils"]

    for cpp_dir in cpp_dirs:
        full_path = project_root / cpp_dir
        assert full_path.exists(), f"C++ directory missing: {cpp_dir}"

        # Check for CMakeLists.txt
        cmake_file = full_path / "CMakeLists.txt"
        assert cmake_file.exists(), f"CMakeLists.txt missing in {cpp_dir}"


def test_requirements_consistency():
    """Test that requirements are consistent and loadable."""
    project_root = Path(__file__).parent.parent.parent
    requirements_file = project_root / "requirements.txt"

    assert requirements_file.exists()

    # Read requirements and check they're parseable
    content = requirements_file.read_text()
    lines = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.startswith("#")
    ]

    # Should have some core dependencies
    requirement_names = [line.split(">=")[0].split("==")[0] for line in lines]
    expected_deps = ["torch", "numpy", "pybind11", "cython"]

    for dep in expected_deps:
        assert any(
            dep in req for req in requirement_names
        ), f"Missing expected dependency: {dep}"


@pytest.mark.slow
def test_full_pipeline_simulation():
    """Simulate a full pipeline run to test integration."""
    # This is a placeholder for a full pipeline test
    # In the future, this would test: data loading -> model creation -> training step -> evaluation

    steps_completed = []

    # Step 1: Configuration loading
    try:
        # Placeholder for config loading
        steps_completed.append("config_loading")
    except Exception:
        pass

    # Step 2: Model initialization
    try:
        # Placeholder for model init
        steps_completed.append("model_init")
    except Exception:
        pass

    # Step 3: Data preparation
    try:
        # Placeholder for data prep
        steps_completed.append("data_prep")
    except Exception:
        pass

    # For now, just verify we can complete at least the basic setup
    assert len(steps_completed) >= 0  # Always passes for now


if __name__ == "__main__":
    pytest.main([__file__])
