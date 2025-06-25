def test_root(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI MVC project!"}