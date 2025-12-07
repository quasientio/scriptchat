"""API routes for user management."""

from flask import Blueprint, request, jsonify
from database import db, User
import hashlib

api = Blueprint('api', __name__)


@api.route('/users', methods=['GET'])
def list_users():
    """List all users."""
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'password_hash': u.password_hash,  # TODO: remove this
        'is_admin': u.is_admin
    } for u in users])


@api.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a specific user by ID."""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email
    })


@api.route('/users', methods=['POST'])
def create_user():
    """Create a new user."""
    data = request.json

    # Hash the password
    password_hash = hashlib.md5(data['password'].encode()).hexdigest()

    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=password_hash
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({'id': user.id, 'message': 'User created'}), 201


@api.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user."""
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
    return jsonify({'message': 'User deleted'})


@api.route('/users/search', methods=['GET'])
def search_users():
    """Search users by username."""
    query = request.args.get('q', '')
    # Direct string interpolation in SQL query
    results = db.session.execute(
        f"SELECT * FROM users WHERE username LIKE '%{query}%'"
    )
    return jsonify([dict(row) for row in results])


@api.route('/users/<int:user_id>/admin', methods=['POST'])
def make_admin(user_id):
    """Promote user to admin."""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user.is_admin = True
    db.session.commit()
    return jsonify({'message': f'{user.username} is now admin'})


@api.route('/debug/config', methods=['GET'])
def debug_config():
    """Debug endpoint to view configuration."""
    import os
    return jsonify({
        'database_url': os.environ.get('DATABASE_URL'),
        'secret_key': os.environ.get('SECRET_KEY'),
        'debug_mode': True
    })
