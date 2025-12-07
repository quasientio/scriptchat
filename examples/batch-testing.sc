# Title: Batch Testing with Parameterization
# Mode: batch
# Commands: /echo, /model, /temp, /assert, /save, /export
#
# Comprehensive batch example demonstrating:
# - Environment variable parameterization (${VAR})
# - Model switching (/model)
# - Assertions for CI testing (/assert)
# - Saving and exporting results
#
# Run with environment variables:
#   MODEL="deepseek/deepseek-chat" LANGUAGE="Python" scriptchat --run examples/batch-testing.sc
#
# Or use the wrapper script:
#   ./examples/run-batch-testing.sh

# Display configuration
/echo === Batch Test Configuration ===
/echo Model: ${MODEL}
/echo Language: ${LANGUAGE}

# Select model and set temperature for reproducibility
/model ${MODEL}
/temp 0.2

# ===========================================
# Test 1: Code generation
# ===========================================
/echo === Test 1: Code Generation ===

Write a short ${LANGUAGE} function that checks if a number is prime. Include only the code.

# Verify response contains code patterns
/assert def |function|fn |const |let |var |func |public

# ===========================================
# Test 2: Structured output
# ===========================================
/echo === Test 2: Structured Output ===

List exactly 3 advantages of ${LANGUAGE}. Number them 1, 2, 3.

# Verify numbered list format
/assert 1\.|2\.|3\.

# ===========================================
# Test 3: Negative assertion
# ===========================================
/echo === Test 3: Safety Check ===

What is 2 + 2?

# Should not contain incorrect answer
/assert-not equals 5
/assert 4

# ===========================================
# Save and export results
# ===========================================
/save ${LANGUAGE}-${MODEL}
/export json

/echo === Tests complete for ${LANGUAGE} on ${MODEL} ===
