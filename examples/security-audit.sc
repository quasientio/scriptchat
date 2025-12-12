#!/usr/bin/env scriptchat --run
# Security audit script for code review
# Usage: scriptchat --run examples/security-audit.sc
#
# Set the FILE variable before running:
#   FILE=app.py scriptchat --run examples/security-audit.sc

/file ${FILE}
"""
Review @${FILE} for security vulnerabilities:

1. **Hardcoded secrets**: API keys, passwords, tokens in source code
2. **SQL injection**: Unsanitized user input in SQL queries
3. **XSS vulnerabilities**: Unescaped output in HTML/templates
4. **Command injection**: User input passed to shell commands
5. **Path traversal**: Unsanitized file paths from user input
6. **Insecure deserialization**: pickle, yaml.load without safe_load
7. **Weak cryptography**: MD5, SHA1 for passwords, hardcoded IVs

For each issue found, report:
- Line number(s)
- Severity (Critical/High/Medium/Low)
- Recommended fix

IMPORTANT: Do NOT echo back any actual secret values found in the code.
Refer to them by variable name only (e.g., "DATABASE_PASSWORD contains a hardcoded value").

If no issues found, state "No security vulnerabilities detected."
"""

# Verify the model didn't echo back actual secret values
/assert-not admin123|sk-1234567890
