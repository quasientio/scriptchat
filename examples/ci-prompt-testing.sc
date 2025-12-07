# Title: CI Prompt Regression Testing
# Mode: batch
# Commands: /echo, /temp, /prompt, /assert, /assert-not
#
# Automated tests for prompt behavior in CI pipelines.
# Run with: scriptchat --run examples/ci-prompt-testing.sc
# Use --continue-on-error to run all tests even if some fail.

# Set low temperature for consistent outputs
/temp 0.1

# ===========================================
# Test 1: Math explanation
# ===========================================
/echo === Test 1: Math Explanation ===

/prompt You are a math tutor. Give clear, concise explanations.

Explain the Pythagorean theorem in one sentence.

# Verify response mentions key concepts
/assert theorem|triangle|hypotenuse
/assert-not calculus|derivative

# ===========================================
# Test 2: Code generation
# ===========================================
/echo === Test 2: Code Generation ===

/prompt clear
/prompt You are a Python developer. Write only code, no explanations.

Write a Python function to check if a number is prime.

# Verify it's actual Python code
/assert def.*prime
/assert return
# Should not include conversational fluff
/assert-not here is|sure|certainly

# ===========================================
# Test 3: Summarization
# ===========================================
/echo === Test 3: Summarization ===

/prompt You are a technical summarizer. Be concise.

Summarize in one short (< 8 words) sentence: Python is a high-level, interpreted programming language known for its simple syntax and readability.

# Should produce a summary, not a long explanation
/assert python|programming|language
/assert-not chapter|introduction|firstly

# ===========================================
/echo === All tests completed ===
