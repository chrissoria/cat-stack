"""
Tests for task #40 (PREFLIGHT): _sanitize_google_schema strips schema
keys Google's responseSchema doesn't accept.

The preflight probe sends `additionalProperties: false` (valid JSON
Schema) which makes most Gemini models 400 Bad Request. Same issue
for `oneOf` / `anyOf` / `$schema` / `$ref` etc. when callers pass
richer JSON schemas. The sanitizer recursively removes the
unsupported keys before the payload reaches Google.

Per Google's docs:
  https://ai.google.dev/api/generate-content#schema
responseSchema accepts a subset of OpenAPI 3.0 (no
`additionalProperties`, no `oneOf`/`anyOf`/`allOf`/`not`, no
`$schema`/`$ref`/`definitions`/`patternProperties`).
"""

import pytest

from cat_stack._providers import _sanitize_google_schema


class TestSanitizerStripsUnsupportedKeys:
    def test_strips_additional_properties_at_root(self):
        schema = {
            "type": "object",
            "properties": {"1": {"type": "string"}},
            "required": ["1"],
            "additionalProperties": False,
        }
        result = _sanitize_google_schema(schema)
        assert "additionalProperties" not in result
        assert result["type"] == "object"
        assert result["properties"] == {"1": {"type": "string"}}
        assert result["required"] == ["1"]

    def test_strips_additional_properties_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "additionalProperties": False},
            },
        }
        result = _sanitize_google_schema(schema)
        assert "additionalProperties" not in result["properties"]["tags"]

    def test_strips_oneof_anyof_allof(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"oneOf": [{"type": "string"}]},
                "b": {"anyOf": [{"type": "string"}]},
                "c": {"allOf": [{"type": "string"}]},
                "d": {"not": {"type": "null"}},
            },
        }
        result = _sanitize_google_schema(schema)
        assert "oneOf" not in result["properties"]["a"]
        assert "anyOf" not in result["properties"]["b"]
        assert "allOf" not in result["properties"]["c"]
        assert "not" not in result["properties"]["d"]

    def test_strips_schema_metadata(self):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$ref": "#/definitions/Foo",
            "definitions": {"Foo": {"type": "string"}},
            "type": "string",
        }
        result = _sanitize_google_schema(schema)
        assert "$schema" not in result
        assert "$ref" not in result
        assert "definitions" not in result
        assert result["type"] == "string"

    def test_preserves_supported_keys(self):
        """Keep type/properties/required/items/description/format/enum/nullable."""
        schema = {
            "type": "object",
            "description": "User record",
            "properties": {
                "name": {"type": "string", "format": "email"},
                "role": {"type": "string", "enum": ["admin", "user"]},
                "age": {"type": "integer", "nullable": True},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        }
        result = _sanitize_google_schema(schema)
        # Round-trip — nothing should have been stripped
        assert result == schema


class TestSanitizerHandlesEdgeCases:
    def test_non_dict_non_list_passthrough(self):
        assert _sanitize_google_schema("scalar") == "scalar"
        assert _sanitize_google_schema(42) == 42
        assert _sanitize_google_schema(None) is None
        assert _sanitize_google_schema(True) is True

    def test_empty_dict(self):
        assert _sanitize_google_schema({}) == {}

    def test_empty_list(self):
        assert _sanitize_google_schema([]) == []

    def test_list_of_schemas_sanitized_individually(self):
        """When a list shows up in a schema position (e.g. enum values),
        recursion still strips bad keys from any dict elements."""
        schemas = [
            {"type": "string", "additionalProperties": False},
            {"type": "integer", "$schema": "X"},
        ]
        result = _sanitize_google_schema(schemas)
        assert result == [{"type": "string"}, {"type": "integer"}]


class TestGooglePayloadBuilderUsesSanitizer:
    """The fix is wired in via _build_google_payload — verify the
    sanitized schema is what lands in the payload."""

    def test_payload_strips_additional_properties_from_response_schema(self):
        from cat_stack._providers import UnifiedLLMClient

        client = UnifiedLLMClient(provider="google", api_key="fake", model="gemini-2.5-flash")
        payload = client._build_payload(
            messages=[{"role": "user", "content": "hi"}],
            json_schema={
                "type": "object",
                "properties": {"1": {"type": "string"}},
                "required": ["1"],
                "additionalProperties": False,
            },
        )

        rs = payload["generationConfig"]["responseSchema"]
        assert "additionalProperties" not in rs
        assert rs["type"] == "object"
        assert rs["properties"] == {"1": {"type": "string"}}
