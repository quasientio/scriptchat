# Test script for /search command
# This demonstrates the search functionality in batch mode

# Create a conversation with some content
What is Python?
/sleep 1

Tell me about error handling in Python
/sleep 1

How do I handle exceptions in Python code?
/sleep 1

# Now we can't test the interactive menu in batch mode,
# but we can verify the command runs without error
# In interactive mode, this would show a selection menu with matches
/echo Testing search - in interactive mode, this would show a menu

# Note: /search doesn't work in batch mode (needs UI interaction)
# But we can verify command recognition
/help search
