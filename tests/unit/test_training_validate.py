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
Unit tests for the top-level validate() dispatcher.

These tests cover the API contract, error handling, and correct forwarding
without requiring a full workspace or actual training.
"""

from unittest.mock import patch

import numpy as np
import pytest

import mostlyai.engine
from mostlyai.engine import validate
from mostlyai.engine.domain import ModelType

# ---------------------------------------------------------------------------
# Export / API surface
# ---------------------------------------------------------------------------


def test_validate_is_exported_in_all():
    assert "validate" in mostlyai.engine.__all__


def test_validate_is_callable():
    assert callable(validate)


# ---------------------------------------------------------------------------
# Input validation (error paths — no workspace needed)
# ---------------------------------------------------------------------------


def test_validate_raises_if_federated_state_is_none(tmp_path):
    with pytest.raises(ValueError, match="federated_state"):
        validate(workspace_dir=tmp_path, federated_state=None)


def test_validate_raises_if_model_weights_key_missing(tmp_path):
    with pytest.raises(ValueError, match="model_weights"):
        validate(workspace_dir=tmp_path, federated_state={})


def test_validate_raises_for_language_workspace(tmp_path):
    """validate() must raise NotImplementedError for language workspaces."""
    federated_state = {"model_weights": {"layer": np.zeros(1)}}
    with patch("mostlyai.engine.training.resolve_model_type", return_value=ModelType.language):
        with pytest.raises(NotImplementedError, match="tabular"):
            validate(workspace_dir=tmp_path, federated_state=federated_state)


# ---------------------------------------------------------------------------
# Dispatch correctness (tabular path, mocked low-level train)
# ---------------------------------------------------------------------------


def test_validate_passes_validate_only_true_to_tabular_train(tmp_path):
    """validate() must forward validate_only=True to the low-level tabular train."""
    fake_weights = {"layer": np.zeros(1)}
    fake_result = {
        "model_weights": fake_weights,
        "training_metrics": {
            "epoch": 0.0,
            "steps": 0,
            "samples": 0,
            "learn_rate": None,
            "trn_loss": None,
            "val_loss": 0.42,
        },
    }
    with patch("mostlyai.engine.training.resolve_model_type", return_value=ModelType.tabular):
        # autospec=True preserves the real function's signature so that
        # inspect.signature() inside validate() gets correct parameter defaults.
        with patch(
            "mostlyai.engine._tabular.training.train",
            autospec=True,
            return_value=fake_result,
        ) as mock_tabular_train:
            result = validate(
                workspace_dir=tmp_path,
                federated_state={"model_weights": fake_weights},
            )

    mock_tabular_train.assert_called_once()
    call_kwargs = mock_tabular_train.call_args.kwargs
    assert call_kwargs["validate_only"] is True
    assert call_kwargs["federated_state"] == {"model_weights": fake_weights}
    assert result == fake_result


def test_validate_does_not_pass_training_only_params(tmp_path):
    """validate() must NOT forward max_epochs, federated_epochs, or model_state_strategy."""
    fake_weights = {"layer": np.zeros(1)}
    fake_result = {
        "model_weights": fake_weights,
        "training_metrics": {
            "epoch": 0.0,
            "steps": 0,
            "samples": 0,
            "learn_rate": None,
            "trn_loss": None,
            "val_loss": 0.1,
        },
    }
    with patch("mostlyai.engine.training.resolve_model_type", return_value=ModelType.tabular):
        with patch(
            "mostlyai.engine._tabular.training.train",
            autospec=True,
            return_value=fake_result,
        ) as mock_tabular_train:
            validate(
                workspace_dir=tmp_path,
                federated_state={"model_weights": fake_weights},
            )

    call_kwargs = mock_tabular_train.call_args.kwargs
    # These training-only parameters should not be forwarded by validate()
    assert "max_epochs" not in call_kwargs
    assert "federated_epochs" not in call_kwargs
    assert "model_state_strategy" not in call_kwargs
    assert "gradient_accumulation_steps" not in call_kwargs
    assert "upload_model_data_callback" not in call_kwargs
    assert "fixed_learning_rate" not in call_kwargs


def test_validate_forwards_optional_params(tmp_path):
    """validate() must correctly forward model, batch_size, device, etc. when provided."""
    import torch

    fake_weights = {"layer": np.zeros(1)}
    fake_result = {
        "model_weights": fake_weights,
        "training_metrics": {
            "epoch": 0.0,
            "steps": 0,
            "samples": 0,
            "learn_rate": None,
            "trn_loss": None,
            "val_loss": 0.3,
        },
    }
    with patch("mostlyai.engine.training.resolve_model_type", return_value=ModelType.tabular):
        with patch(
            "mostlyai.engine._tabular.training.train",
            autospec=True,
            return_value=fake_result,
        ) as mock_tabular_train:
            validate(
                workspace_dir=tmp_path,
                model="MOSTLY_AI/Small",
                batch_size=32,
                device=torch.device("cpu"),
                federated_state={"model_weights": fake_weights},
            )

    call_kwargs = mock_tabular_train.call_args.kwargs
    assert call_kwargs["model"] == "MOSTLY_AI/Small"
    assert call_kwargs["batch_size"] == 32
    assert call_kwargs["device"] == torch.device("cpu")
