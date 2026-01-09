"""Tests for FastAPI application and OpenAPI documentation."""

from fastapi.testclient import TestClient

from main import API_DESCRIPTION, API_TITLE, API_VERSION, app


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation endpoints."""

    def test_openapi_json_endpoint_returns_200(self) -> None:
        """Test that /openapi.json endpoint returns valid OpenAPI schema."""
        client = TestClient(app)
        response = client.get("/openapi.json")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_openapi_schema_has_correct_metadata(self) -> None:
        """Test that OpenAPI schema contains correct API metadata."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()

        assert schema["info"]["title"] == API_TITLE
        assert schema["info"]["version"] == API_VERSION
        assert API_DESCRIPTION in schema["info"]["description"]

    def test_openapi_schema_has_security_scheme(self) -> None:
        """Test that OpenAPI schema defines Azure Entra ID security scheme."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()

        assert "securitySchemes" in schema["components"]
        assert "AzureEntraID" in schema["components"]["securitySchemes"]

        security_scheme = schema["components"]["securitySchemes"]["AzureEntraID"]
        assert security_scheme["type"] == "http"
        assert security_scheme["scheme"] == "bearer"
        assert security_scheme["bearerFormat"] == "JWT"

    def test_openapi_schema_has_tags(self) -> None:
        """Test that OpenAPI schema defines all expected tags."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()

        tag_names = [tag["name"] for tag in schema["tags"]]
        assert "Health" in tag_names
        assert "Authentication" in tag_names
        assert "Transcription" in tag_names
        assert "Transcription History" in tag_names

    def test_openapi_schema_has_all_endpoints(self) -> None:
        """Test that OpenAPI schema includes all expected endpoints."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()

        paths = schema["paths"]

        # Health endpoints
        assert "/health" in paths
        assert "/ready" in paths

        # Auth endpoints
        assert "/auth/me" in paths

        # Transcription endpoints
        assert "/transcribe" in paths
        assert "/transcribe/diarize" in paths

        # Transcription history endpoints
        assert "/transcriptions" in paths
        assert "/transcriptions/{transcription_id}" in paths

    def test_swagger_ui_endpoint_returns_200(self) -> None:
        """Test that Swagger UI is available at /docs."""
        client = TestClient(app)
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "swagger" in response.text.lower()

    def test_redoc_endpoint_returns_200(self) -> None:
        """Test that ReDoc is available at /redoc."""
        client = TestClient(app)
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "redoc" in response.text.lower()

    def test_openapi_schema_has_contact_info(self) -> None:
        """Test that OpenAPI schema includes contact information."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()

        assert "contact" in schema["info"]
        assert "name" in schema["info"]["contact"]

    def test_openapi_schema_has_license_info(self) -> None:
        """Test that OpenAPI schema includes license information."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()

        assert "license" in schema["info"]
        assert schema["info"]["license"]["name"] == "MIT"


class TestApplicationRouters:
    """Tests for application routers and endpoints."""

    def test_health_endpoint_accessible(self) -> None:
        """Test that health endpoint is accessible without authentication."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_ready_endpoint_accessible(self) -> None:
        """Test that ready endpoint is accessible without authentication."""
        client = TestClient(app)
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_auth_endpoint_requires_authentication(self) -> None:
        """Test that auth endpoint returns 401 without authentication."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/auth/me")

        assert response.status_code == 401

    def test_transcribe_endpoint_requires_authentication(self) -> None:
        """Test that transcribe endpoint returns 401 without authentication."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/transcribe")

        assert response.status_code == 401

    def test_transcriptions_endpoint_requires_authentication(self) -> None:
        """Test that transcriptions endpoint returns 401 without authentication."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/transcriptions")

        assert response.status_code == 401
