"""Tests for API routes."""

import pytest
from flask import Flask
from src.api.routes import api
from database import db, User


@pytest.fixture
def app():
    """Create test application."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.register_blueprint(api, url_prefix='/api')
    db.init_app(app)
    with app.app_context():
        db.create_all()
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_user(app):
    """Create a sample user."""
    with app.app_context():
        user = User(username='testuser', email='test@example.com', password_hash='abc123')
        db.session.add(user)
        db.session.commit()
        return user.id


class TestListUsers:
    """Tests for GET /users endpoint."""

    def test_list_users_empty(self, client):
        """Test listing users when database is empty."""
        response = client.get('/api/users')
        assert response.status_code == 200
        assert response.json == []

    def test_list_users_with_data(self, client, sample_user):
        """Test listing users returns user data."""
        response = client.get('/api/users')
        assert response.status_code == 200
        assert len(response.json) == 1
        assert response.json[0]['username'] == 'testuser'


class TestGetUser:
    """Tests for GET /users/<id> endpoint."""

    def test_get_user_exists(self, client, sample_user):
        """Test getting an existing user."""
        response = client.get(f'/api/users/{sample_user}')
        assert response.status_code == 200
        assert response.json['username'] == 'testuser'

    def test_get_user_not_found(self, client):
        """Test getting a non-existent user."""
        response = client.get('/api/users/9999')
        assert response.status_code == 404


class TestCreateUser:
    """Tests for POST /users endpoint."""

    def test_create_user_success(self, client):
        """Test creating a new user."""
        response = client.post('/api/users', json={
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'secret123'
        })
        assert response.status_code == 201
        assert 'id' in response.json


# TODO: Add tests for:
# - DELETE /users/<id>
# - GET /users/search
# - POST /users/<id>/admin
# - GET /debug/config
# - Error handling (missing fields, invalid data)
# - Authentication/authorization
