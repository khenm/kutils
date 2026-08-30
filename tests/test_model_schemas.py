"""Tests for kutils.models.schemas: strict spec parsing, metadata and
digesting."""

import pytest

from kutils.models.schemas import ModelSpec

VALID = """
[model]
provider = "stub"
model_id = "stub/vit"
family = "vision_transformer"
modality = "image"
output = "cls_token"
revision = "abc123"

[model.capability]
depth = 12
width = 768
parameters = 86000000
pretraining_samples = 142000000
"""


def write_spec(tmp_path, text, name="model.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parse_valid_spec(tmp_path):
    spec = ModelSpec.from_toml(write_spec(tmp_path, VALID, "my_model.toml"))
    assert spec.name == "my_model"
    assert spec.provider == "stub"
    assert spec.model_id == "stub/vit"
    assert spec.family == "vision_transformer"
    assert spec.capability == {
        "depth": 12,
        "width": 768,
        "parameters": 86_000_000,
        "pretraining_samples": 142_000_000,
    }


def test_unknown_key_raises(tmp_path):
    text = VALID.replace("[model.capability]", 'nope = 1\n\n[model.capability]')
    with pytest.raises(ValueError, match="Unknown model spec key 'nope'"):
        ModelSpec.from_toml(write_spec(tmp_path, text))


def test_unknown_top_level_section_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown top-level section"):
        ModelSpec.from_toml(write_spec(tmp_path, "[other]\nx = 1\n"))


def test_missing_required_key_raises(tmp_path):
    with pytest.raises(ValueError, match="missing required key"):
        ModelSpec.from_toml(write_spec(tmp_path, '[model]\nmodel_id = "x"\n'))


def test_capability_must_be_scalar(tmp_path):
    text = '[model]\nprovider = "s"\nmodel_id = "m"\n[model.capability]\ndepth = [1]\n'
    with pytest.raises(ValueError, match="must be a scalar"):
        ModelSpec.from_toml(write_spec(tmp_path, text))


def test_digest_stable_and_sensitive(tmp_path):
    path = write_spec(tmp_path, VALID)
    a = ModelSpec.from_toml(path)
    b = ModelSpec.from_toml(path)
    assert a.digest() == b.digest()

    changed_text = VALID.replace('revision = "abc123"', 'revision = "def456"')
    changed = ModelSpec.from_toml(write_spec(tmp_path, changed_text, "other.toml"))
    assert a.digest() != changed.digest()


def test_to_dict_roundtrip(tmp_path):
    spec = ModelSpec.from_toml(write_spec(tmp_path, VALID))
    data = spec.to_dict()
    assert data["provider"] == "stub"
    assert data["capability"]["depth"] == 12
