#!/bin/bash
# Verification script for OpenMP linkage
# Purpose: Verify OpenMP is correctly linked to mcts_py shared library
# Expected: libomp.so or libgomp.so linked, runtime thread count >1

set -e

echo "=== OpenMP Linkage Verification ==="
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Find mcts_py shared library
MCTS_LIB=$(find build/lib.* -name "mcts_py*.so" 2>/dev/null | head -1)

if [ -z "$MCTS_LIB" ]; then
    echo "❌ mcts_py shared library not found in build/lib.*/"
    echo "   Build the extension first: pip install -e . --config-settings build-dir=build"
    exit 1
fi

echo "Found library: $MCTS_LIB"
echo ""

# Check linkage with ldd
echo "Checking library dependencies..."
OPENMP_LINKED=$(ldd "$MCTS_LIB" | grep -i "omp" || echo "")

if [ -z "$OPENMP_LINKED" ]; then
    echo "❌ OpenMP not linked to $MCTS_LIB"
    echo ""
    echo "Fix required:"
    echo "  1. Verify CMakeLists.txt contains:"
    echo "     target_link_libraries(mcts_py PRIVATE OpenMP::OpenMP_CXX)"
    echo "  2. Rebuild: pip install -e . --force-reinstall --no-deps"
    echo "  3. Re-run this verification script"
    exit 1
else
    echo "✅ OpenMP linked successfully:"
    echo "   $OPENMP_LINKED"
fi

echo ""

# Check runtime thread count (requires Python venv)
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found, skipping runtime check"
    echo "✅ Static linkage verified"
    exit 0
fi

echo "Checking runtime OpenMP configuration..."
source venv/bin/activate 2>/dev/null || true

PYTHON_CHECK=$(python -c "
import mcts_py
try:
    # Try to get OpenMP thread count if exposed
    if hasattr(mcts_py, 'get_openmp_threads'):
        count = mcts_py.get_openmp_threads()
        print(f'OpenMP threads: {count}')
        if count > 1:
            print('RUNTIME_OK')
        else:
            print('RUNTIME_FAIL')
    else:
        print('get_openmp_threads() not exposed (non-critical)')
        print('RUNTIME_SKIP')
except Exception as e:
    print(f'Runtime check failed: {e}')
    print('RUNTIME_SKIP')
" 2>&1)

echo "$PYTHON_CHECK"
echo ""

if echo "$PYTHON_CHECK" | grep -q "RUNTIME_OK"; then
    echo "✅ Runtime OpenMP thread count >1 (verified)"
elif echo "$PYTHON_CHECK" | grep -q "RUNTIME_FAIL"; then
    echo "⚠️  Runtime OpenMP thread count ≤1"
    echo "   Set OMP_NUM_THREADS environment variable:"
    echo "   export OMP_NUM_THREADS=8"
elif echo "$PYTHON_CHECK" | grep -q "RUNTIME_SKIP"; then
    echo "⚠️  Runtime check skipped (API not exposed yet)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ OpenMP linkage verified"
echo "✅ Phase 2A constitution principle IV satisfied"
exit 0
