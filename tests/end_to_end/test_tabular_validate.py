# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
End-to-end tests for the top-level validate() function.

These tests exercise the full validation pipeline: workspace preparation,
federated-state retrieval via train(), and validation via validate().
"""

import numpy as np
import pytest

from mostlyai.engine import analyze, encode, split, train, validate
from mostlyai.engine.domain import ModelEncodingType

from .conftest import MockData

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_tgt_data():
    """Minimal flat tabular dataset for fast validate() round-trip tests."""
    mock = MockData(n_samples=500)
    mock.add_numeric_column(
        name="age",
        quantiles={0.0: 18, 0.25: 25, 0.5: 35, 0.75: 50, 1.0: 70},
        dtype="int32",
    )
    mock.add_categorical_column(
        name="category",
        probabilities={"A": 0.4, "B": 0.35, "C": 0.25},
    )
    mock.add_numeric_column(
        name="score",
        quantiles={0.0: 0.0, 0.5: 50.0, 1.0: 100.0},
        dtype="float32",
    )
    return mock.df


@pytest.fixture(scope="module")
def prepared_workspace(small_tgt_data, tmp_path_factory):
    """Workspace that has been split, analysed, and encoded — ready for train/validate."""
    ws_dir = tmp_path_factory.mktemp("validate_ws")
    split(
        tgt_data=small_tgt_data,
        tgt_encoding_types={
            "age": ModelEncodingType.tabular_numeric_auto,
            "category": ModelEncodingType.tabular_categorical,
            "score": ModelEncodingType.tabular_numeric_auto,
        },
        workspace_dir=ws_dir,
    )
    analyze(workspace_dir=ws_dir)
    encode(workspace_dir=ws_dir)
    return ws_dir


@pytest.fixture(scope="module")
def trained_federated_state(prepared_workspace):
    """Federated state (model weights) obtained after 1 federated training epoch."""
    state = train(
        workspace_dir=prepared_workspace,
        model="MOSTLY_AI/Small",
        federated_epochs=1,
        device="cpu",
    )
    assert state is not None, "train() with federated_epochs should return a federated_state dict"
    return state


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestValidateReturnStructure:
    """Verify the shape and types of the validate() return value."""

    def test_returns_dict(self, prepared_workspace, trained_federated_state):
        result = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            device="cpu",
            federated_state=trained_federated_state,
        )
        assert isinstance(result, dict)

    def test_has_model_weights_key(self, prepared_workspace, trained_federated_state):
        result = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            device="cpu",
            federated_state=trained_federated_state,
        )
        assert "model_weights" in result
        assert isinstance(result["model_weights"], dict)
        assert len(result["model_weights"]) > 0

    def test_has_training_metrics_with_val_loss(self, prepared_workspace, trained_federated_state):
        result = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            device="cpu",
            federated_state=trained_federated_state,
        )
        assert "training_metrics" in result
        metrics = result["training_metrics"]
        assert "val_loss" in metrics
        assert isinstance(metrics["val_loss"], float)
        assert metrics["val_loss"] > 0.0

    def test_training_metrics_reflect_validate_only_mode(self, prepared_workspace, trained_federated_state):
        """Validate-only mode should have zero epoch/steps and no trn_loss, but samples reflects the validation set size."""
        result = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            device="cpu",
            federated_state=trained_federated_state,
        )
        metrics = result["training_metrics"]
        assert metrics["trn_loss"] is None
        assert metrics["epoch"] == 0.0
        assert metrics["steps"] == 0
        assert metrics["samples"] > 0


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestValidateCorrectness:
    """Verify that validate() computes the expected values correctly."""

    def test_model_weights_unchanged(self, prepared_workspace, trained_federated_state):
        """validate() must return the coordinator-supplied weights, not modify them."""
        input_weights = {k: v.copy() for k, v in trained_federated_state["model_weights"].items()}

        result = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            device="cpu",
            federated_state=trained_federated_state,
        )

        out_weights = result["model_weights"]
        assert set(input_weights.keys()) == set(out_weights.keys()), "Layer names must be identical"
        for key in input_weights:
            np.testing.assert_array_equal(
                input_weights[key],
                out_weights[key],
                err_msg=f"Weights for '{key}' were modified by validate()",
            )

    def test_val_loss_is_deterministic_on_cpu(self, prepared_workspace, trained_federated_state):
        """Calling validate() twice with the same weights on CPU must produce the same val_loss."""
        common_kwargs = dict(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            batch_size=32,
            device="cpu",
            federated_state=trained_federated_state,
        )
        result1 = validate(**common_kwargs)
        result2 = validate(**common_kwargs)

        assert result1["training_metrics"]["val_loss"] == pytest.approx(
            result2["training_metrics"]["val_loss"], rel=1e-5
        )

    def test_different_weights_produce_different_val_loss(self, prepared_workspace, trained_federated_state):
        """Perturbing model weights should (almost always) change the val_loss."""
        result_original = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            batch_size=32,
            device="cpu",
            federated_state=trained_federated_state,
        )

        # Create a clearly different federated state by zeroing out all weights
        zeroed_weights = {k: np.zeros_like(v) for k, v in trained_federated_state["model_weights"].items()}
        zeroed_state = {**trained_federated_state, "model_weights": zeroed_weights}

        result_zeroed = validate(
            workspace_dir=prepared_workspace,
            model="MOSTLY_AI/Small",
            batch_size=32,
            device="cpu",
            federated_state=zeroed_state,
        )

        original_loss = result_original["training_metrics"]["val_loss"]
        zeroed_loss = result_zeroed["training_metrics"]["val_loss"]
        assert original_loss != pytest.approx(zeroed_loss, rel=1e-3), (
            "Zeroing all model weights should produce a measurably different val_loss"
        )
