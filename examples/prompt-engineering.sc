# Title: Iterative Prompt Engineering
# Mode: interactive
# Commands: /save, /branch, /prompt, /retry, /tag, /note, /history, /export, /open
#
# Refine a prompt across multiple iterations using conversation branching.
# Compare different approaches by saving, branching, and exporting results.

# Start with a baseline question
What are the key principles of REST API design?

# Save the baseline conversation
/save rest-api-baseline

# Add a note about your initial observations
/note Baseline response - checking for completeness and technical depth

# Branch to try a more specific approach
/branch rest-api-detailed

# Set a system prompt for more focused responses
/prompt You are an expert API architect. Be concise and technical. Focus on practical implementation details.

# Retry with the new system prompt
/retry

# Review user messages in this conversation
/history

# Tag for organization
/tag topic=api-design
/tag variant=detailed

# Add observations about this variant
/note Detailed variant - more technical, better structure

# Export this version
/export md

# Open the original baseline
/open rest-api-baseline

# Tag and export for comparison
/tag variant=baseline
/export md

# Now you have two markdown files to compare side-by-side
