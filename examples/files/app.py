"""Sample Flask app with intentional security vulnerabilities for demo purposes."""

import os
import pickle
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Hardcoded secrets (vulnerability #1)
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"

def get_db():
    return sqlite3.connect("users.db")

@app.route("/user")
def get_user():
    # SQL injection vulnerability (#2)
    user_id = request.args.get("id")
    db = get_db()
    cursor = db.execute(f"SELECT * FROM users WHERE id = {user_id}")
    user = cursor.fetchone()

    # XSS vulnerability (#3)
    return render_template_string(f"<h1>Welcome, {user[1]}!</h1>")

@app.route("/run")
def run_command():
    # Command injection vulnerability (#4)
    filename = request.args.get("file")
    os.system(f"cat {filename}")
    return "Done"

@app.route("/download")
def download():
    # Path traversal vulnerability (#5)
    filename = request.args.get("name")
    with open(f"./uploads/{filename}", "rb") as f:
        return f.read()

@app.route("/load")
def load_data():
    # Insecure deserialization (#6)
    data = request.args.get("data")
    return pickle.loads(bytes.fromhex(data))

if __name__ == "__main__":
    app.run(debug=True)
