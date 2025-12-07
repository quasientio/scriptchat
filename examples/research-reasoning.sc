# Title: Research with Reasoning Models
# Mode: interactive
# Commands: /model, /reason, /thinking, /timeout, /prompt, /tag, /save, /export
#
# Use extended thinking for complex analysis tasks.
# Demonstrates /reason presets and /thinking for custom budgets.

# Select a reasoning-capable model
/model
# (select Claude Sonnet 4 or another reasoning model)

# Enable high reasoning level (32K token thinking budget)
/reason high

# Alternative: set exact thinking budget in tokens
# /thinking 32000

# Disable timeout for long-running queries
/timeout off

# Set an analyst persona
/prompt You are a senior technical analyst. Think step-by-step, consider multiple perspectives, and show your reasoning process clearly.

# Ask a complex question requiring deep analysis
"""
Analyze the trade-offs between microservices and monolithic architecture
for a startup with 5 engineers building a SaaS product.

Consider:
- Development velocity in early vs growth stages
- Operational complexity and DevOps requirements
- Scaling patterns and cost implications
- Team expertise and hiring considerations
- Migration paths and reversibility

Provide a concrete recommendation with justification.
"""

# Tag the analysis for future reference
/tag topic=architecture
/tag analysis=reasoning
/tag depth=comprehensive

# Save the session
/save architecture-analysis

# Export as HTML for formatted output with code blocks
/export html

# Re-enable timeout for normal queries
/timeout 120
