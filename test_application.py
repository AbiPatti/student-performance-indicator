import pytest
from application import application

@pytest.fixture
def client():
    # This creates a "fake" version of your app for testing
    application.config['TESTING'] = True
    with application.test_client() as client:
        yield client

def test_home_route(client):
    """
    This test checks if the home page ('/') loads correctly.
    It expects a '200 OK' status code.
    """
    response = client.get('/')
    assert response.status_code == 200