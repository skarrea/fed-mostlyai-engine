# Copyright 2025 MOSTLY AI
# Copyright 2026 Clinical Data Science Maastricht and Bendik Skarre Abrahamsen
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

import inspect
from collections.abc import Callable
from pathlib import Path

import torch

from mostlyai.engine._common import ProgressCallback
from mostlyai.engine._workspace import resolve_model_type
from mostlyai.engine.domain import DifferentialPrivacyConfig, ModelStateStrategy, ModelType


def train(
    *,
    model: str | None = None,
    max_training_time: float | None = 14400.0,  # 10 days
    max_epochs: float | None = 100.0,  # 100 epochs
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    enable_flexible_generation: bool = True,
    max_sequence_window: int | None = None,
    differential_privacy: DifferentialPrivacyConfig | dict | None = None,
    model_state_strategy: ModelStateStrategy = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    upload_model_data_callback: Callable | None = None,
    federated_epochs: int | None = None,
    federated_state: dict | None = None,
    fixed_learning_rate: float | None = None,
) -> dict | None:
    """
    Trains a model with optional early stopping and differential privacy.

    Creates the following folder structure within the `workspace_dir`:

    - `ModelStore`: Trained model checkpoints and logs.

    Args:
        model: The identifier of the model to train. If tabular, defaults to MOSTLY_AI/Medium. If language,
            defaults to MOSTLY_AI/LSTMFromScratch-3m. For language models, Hugging Face hub ids are supported;
            verified pretrained checkpoints include HuggingFaceTB/SmolLM2-135M, HuggingFaceTB/SmolLM3-3B,
            Qwen/Qwen3-0.6B, and microsoft/phi-4.
        max_training_time: Maximum training time in minutes. If None, defaults to 10 days.
        max_epochs: Maximum number of training epochs. If None, defaults to 100 epochs.
        batch_size: Per-device batch size for training and validation. If None, determined automatically.
        gradient_accumulation_steps: Number of steps to accumulate gradients. If None, determined automatically.
        enable_flexible_generation: Whether to enable flexible order generation. Defaults to True.
        max_sequence_window: Maximum sequence window for tabular sequential models. Only applicable for tabular models.
        differential_privacy: Configuration for differential privacy training. If None, DP is disabled.
        model_state_strategy: Strategy for handling existing model state (reset/resume/reuse).
        device: Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace. Training outputs are stored in ModelStore subdirectory.
        update_progress: Callback function to report training progress.
        upload_model_data_callback: Callback function to upload model data during training.
        federated_epochs: If specified, train for exactly this number of epochs and return model weights. Overrides max_epochs.
        federated_state: Optional federated state dictionary containing model weights, optimizer state, and DP accountant state for continuing federated training.
        fixed_learning_rate: Fixed learning rate for training. If specified, overrides learning rate schedule.

    Returns:
        dict | None: Comprehensive federated state dictionary if federated_epochs is specified, otherwise None.
    """
    model_type = resolve_model_type(workspace_dir)
    if model_type == ModelType.tabular:
        from mostlyai.engine._tabular.training import train as train_tabular

        args = inspect.signature(train_tabular).parameters
        result = train_tabular(
            model=model if model else args["model"].default,
            workspace_dir=workspace_dir,
            max_training_time=max_training_time if max_training_time else args["max_training_time"].default,
            max_epochs=max_epochs if max_epochs else args["max_epochs"].default,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_flexible_generation=enable_flexible_generation,
            differential_privacy=differential_privacy,
            update_progress=update_progress,
            upload_model_data_callback=upload_model_data_callback,
            model_state_strategy=model_state_strategy,
            device=device,
            max_sequence_window=max_sequence_window if max_sequence_window else args["max_sequence_window"].default,
            federated_epochs=federated_epochs,
            federated_state=federated_state,
            fixed_learning_rate=fixed_learning_rate,
        )
        return result
    else:
        from mostlyai.engine._language.training import train as train_language

        if max_sequence_window is not None:
            raise ValueError("max_sequence_window is not supported for language models")

        args = inspect.signature(train_language).parameters
        result = train_language(
            model=model if model else args["model"].default,
            workspace_dir=workspace_dir,
            max_training_time=max_training_time if max_training_time else args["max_training_time"].default,
            max_epochs=max_epochs if max_epochs else args["max_epochs"].default,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_flexible_generation=enable_flexible_generation,
            differential_privacy=differential_privacy,
            update_progress=update_progress,
            upload_model_data_callback=upload_model_data_callback,
            model_state_strategy=model_state_strategy,
            device=device,
            federated_epochs=federated_epochs,
            federated_state=federated_state,
        )
        return result


def validate(
    *,
    model: str | None = None,
    batch_size: int | None = None,
    max_sequence_window: int | None = None,
    enable_flexible_generation: bool = True,
    differential_privacy: DifferentialPrivacyConfig | dict | None = None,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    federated_state: dict | None = None,
) -> dict:
    """
    Validate a model against the workspace's validation data, using coordinator-supplied weights.

    This is the federated counterpart to `train()`. It loads model weights from
    ``federated_state["model_weights"]``, computes the local validation loss, and returns
    a federated state dict containing both the (unchanged) weights and the new ``val_loss``.

    No training is performed and no checkpoints are written. The workspace must contain
    ``tgt_stats``, ``ctx_stats``, and encoded validation parquet files — nothing else is required.

    Args:
        model: The identifier of the model architecture. Defaults to MOSTLY_AI/Medium.
        batch_size: Per-device batch size for validation. If None, determined automatically.
        max_sequence_window: Maximum sequence window for tabular sequential models.
        enable_flexible_generation: Whether to enable flexible order generation. Defaults to True.
        differential_privacy: DP configuration, used to correctly reconstruct the model architecture.
        device: Device to run validation on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        workspace_dir: Directory containing the prepared workspace (stats + encoded val data).
        update_progress: Callback function to report progress.
        federated_state: Dict containing model weights to evaluate. Must include 'model_weights'.

    Returns:
        dict: Federated state dict with ``model_weights`` (unchanged) and ``training_metrics``
            containing ``val_loss`` reflecting the local validation loss.

    Raises:
        ValueError: If ``federated_state`` is None or missing the ``model_weights`` key.
        NotImplementedError: If the workspace contains a language model.
    """
    if federated_state is None or "model_weights" not in federated_state:
        raise ValueError(
            "validate() requires a federated_state dict containing 'model_weights'. "
            "Obtain model weights from a prior train() call with federated_epochs set."
        )
    model_type = resolve_model_type(workspace_dir)
    if model_type == ModelType.tabular:
        from mostlyai.engine._tabular.training import train as train_tabular

        args = inspect.signature(train_tabular).parameters
        return train_tabular(
            model=model if model is not None else args["model"].default,
            workspace_dir=workspace_dir,
            batch_size=batch_size,
            enable_flexible_generation=enable_flexible_generation,
            differential_privacy=differential_privacy,
            update_progress=update_progress,
            device=device,
            max_sequence_window=max_sequence_window
            if max_sequence_window is not None
            else args["max_sequence_window"].default,
            federated_state=federated_state,
            validate_only=True,
        )
    else:
        raise NotImplementedError("validate() is currently only supported for tabular workspaces")
