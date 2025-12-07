# Title: Code Review with File References
# Mode: interactive
# Commands: /file, /files, /note, /save, /export, /stream
#
# Conduct a PR code review by registering files and using @references.
# Document findings with /note and export for team sharing.

# Enable streaming for longer responses
/stream on

# Register the files from your PR
# (example files provided in examples/files/)
/file examples/files/src/api/routes.py
/file examples/files/tests/test_routes.py

# Verify registered files with details
/files --long

# Review the implementation for security issues
"""
Review @routes.py for security vulnerabilities, focusing on:
- Input validation
- Authentication/authorization
- SQL injection or other injection attacks
- Sensitive data exposure
"""

# Capture key security findings (example)
# /note Security: found SQL injection in search_users, weak MD5 hashing

# Check test coverage
Does @test_routes.py adequately cover the endpoints in @routes.py? List any gaps.

# Note coverage findings (example)
# /note Coverage gaps: missing tests for DELETE, search, admin promotion

# Generate a comprehensive PR summary
"""
Summarize this PR review:
- Implementation: @routes.py
- Tests: @test_routes.py

Provide:
1. Overall assessment (approve/request changes)
2. Security concerns
3. Test coverage gaps
4. Suggested improvements
"""

# Save and export for the PR
/save pr-review-routes
/export md

# The markdown export can be copied into PR comments
