#!/bin/bash
#
# Run batch tests across multiple models and languages.
#
# Demonstrates parameterized testing with environment variables,
# model comparison, and CI-style assertions.
#
# Usage:
#   ./examples/run-batch-testing.sh
#   ./examples/run-batch-testing.sh --continue-on-error

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/batch-testing.sc"

# Pass through any arguments (like --continue-on-error)
EXTRA_ARGS="$@"

# Models to test (must exist in config.toml)
MODELS=(
    "deepseek/deepseek-chat"
    "deepseek/deepseek-reasoner"
)

# Languages to test
LANGUAGES=(
    "Python"
    "JavaScript"
    "Rust"
)

echo "========================================"
echo "Running batch tests"
echo "Models: ${MODELS[*]}"
echo "Languages: ${LANGUAGES[*]}"
echo "========================================"
echo ""

PASS=0
FAIL=0

for model in "${MODELS[@]}"; do
    for lang in "${LANGUAGES[@]}"; do
        echo "----------------------------------------"
        echo "Testing: $model / $lang"
        echo "----------------------------------------"

        if MODEL="$model" LANGUAGE="$lang" scriptchat --run "$SCRIPT" $EXTRA_ARGS; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
        fi

        echo ""
    done
done

echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo "========================================"

# Exit with error if any tests failed
[ "$FAIL" -eq 0 ]
