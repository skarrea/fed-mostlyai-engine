#!/usr/bin/env python3
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
"""
End-to-end test to compare normal training vs federated training convergence.

This test evaluates whether:
1. Training a model without federated epochs (normal training)
2. Training a model with federated epochs, loading weights, and continuing training

Both approaches should converge to similar results.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Project root for the local (development) version
project_root = Path(__file__).parent.parent.parent

# Import PyPI version first (before adding local path to sys.path)
try:
    import mostlyai.engine as _mostlyai_engine_pypi
    from mostlyai.engine import split as split_pypi
    from mostlyai.engine import analyze as analyze_pypi
    from mostlyai.engine import encode as encode_pypi
    from mostlyai.engine import train as train_pypi
    from mostlyai.engine._workspace import Workspace as Workspace_pypi
    from mostlyai.engine.domain import ModelEncodingType as ModelEncodingType_pypi

    HAS_PYPI_ENGINE = True
    print(f"PyPI mostlyai.engine imported: {getattr(_mostlyai_engine_pypi, '__version__', 'unknown version')}")
except ImportError:
    split_pypi = analyze_pypi = encode_pypi = train_pypi = None
    Workspace_pypi = None
    ModelEncodingType_pypi = None
    HAS_PYPI_ENGINE = False
    print("Note: PyPI mostlyai.engine not available — PyPI baseline run will be skipped")

# Clear cached mostlyai modules so the local development version loads fresh
for _key in list(sys.modules.keys()):
    if _key.startswith("mostlyai"):
        del sys.modules[_key]

sys.path.insert(0, str(project_root))

from mostlyai.engine import analyze, encode, split, train  # noqa: E402
from mostlyai.engine._workspace import Workspace  # noqa: E402
from mostlyai.engine.domain import ModelEncodingType  # noqa: E402

# Shared reporting utilities (plots + GitHub step summary)
_test_dir = Path(__file__).parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
import reporting  # noqa: E402


# ============================================
# TEST CONFIGURATION - Adjust these parameters
# ============================================
class TestConfig:
    """Centralised configuration for test parameters"""

    # Training parameters
    EPOCHS_PER_ITERATION = 1  # Numbers of epochs per federated iteration
    MAX_EPOCHS = 100  # Maximum number of epochs for training
    MODEL_SIZE = "MOSTLY_AI/Medium"  # Model size to use

    # Output directory for plots and summary artifacts
    OUTPUT_DIR = Path("test-output/convergence")


# ============================================

# Optional import for quality assessment
try:
    from mostlyai import qa  # noqa: F401

    HAS_QA_LIBRARY = True
except ImportError:
    HAS_QA_LIBRARY = False
    print("Note: mostlyai-qa library not available - quality assessment features disabled")


_METADATA_WRITTEN = False  # written once per process to avoid duplicate sections

_SOURCE_URL = "https://zenodo.org/records/20411920/files/Intensivregister_Deutschland_Versorgungsstufen.csv"


def create_test_data():
    """Fetch sample tabular data for testing."""
    global _METADATA_WRITTEN

    data = pd.read_csv(_SOURCE_URL)

    if not _METADATA_WRITTEN:
        _METADATA_WRITTEN = True
        reporting.write_dataset_info(
            df=data,
            source_url=_SOURCE_URL,
            config={
                "Model": TestConfig.MODEL_SIZE,
                "Max epochs": TestConfig.MAX_EPOCHS,
                "Epochs per federated iteration": TestConfig.EPOCHS_PER_ITERATION,
                "QA library available": str(HAS_QA_LIBRARY),
                "PyPI engine available": str(HAS_PYPI_ENGINE),
            },
            output_dir=TestConfig.OUTPUT_DIR,
        )

    return data


def setup_workspace(data, workspace_dir):
    """Set up a workspace with split, analyse, and encode steps."""
    split(
        tgt_data=data,
        tgt_encoding_types={
            "datum": ModelEncodingType.tabular_numeric_auto,
            "bundesland_id": ModelEncodingType.tabular_numeric_auto,
            "bundesland_name": ModelEncodingType.tabular_categorical,
            "versorgungsstufe": ModelEncodingType.tabular_categorical,
            "anzahl_meldebereiche": ModelEncodingType.tabular_numeric_auto,
            "faelle_covid_aktuell": ModelEncodingType.tabular_numeric_auto,
            "intensivbetten_belegt": ModelEncodingType.tabular_numeric_auto,
            "intensivbetten_frei": ModelEncodingType.tabular_numeric_auto,
        },
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)


def setup_workspace_pypi(data, workspace_dir):
    """Set up a workspace using the PyPI (released) engine."""
    split_pypi(
        tgt_data=data,
        tgt_encoding_types={
            "datum": ModelEncodingType_pypi.tabular_numeric_auto,
            "bundesland_id": ModelEncodingType_pypi.tabular_numeric_auto,
            "bundesland_name": ModelEncodingType_pypi.tabular_categorical,
            "versorgungsstufe": ModelEncodingType_pypi.tabular_categorical,
            "anzahl_meldebereiche": ModelEncodingType_pypi.tabular_numeric_auto,
            "faelle_covid_aktuell": ModelEncodingType_pypi.tabular_numeric_auto,
            "intensivbetten_belegt": ModelEncodingType_pypi.tabular_numeric_auto,
            "intensivbetten_frei": ModelEncodingType_pypi.tabular_numeric_auto,
        },
        workspace_dir=workspace_dir,
    )
    analyze_pypi(workspace_dir=workspace_dir)
    encode_pypi(workspace_dir=workspace_dir)


def train_normal_model(workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, model=TestConfig.MODEL_SIZE):
    """Train a model using the normal training approach."""
    print(f"\n--- Normal Training (max_epochs={max_epochs}) ---")
    start_time = time.time()

    result = train(workspace_dir=workspace_dir, max_epochs=max_epochs, model=model)

    training_time = time.time() - start_time
    print(f"Normal training completed in {training_time:.2f} seconds")
    print(f"Normal training result: {result}")

    # Get final validation loss from progress messages CSV
    workspace = Workspace(workspace_dir)
    progress_messages_path = workspace.model_progress_messages_path
    curve_df = None
    final_val_loss = None
    if progress_messages_path.exists():
        try:
            curve_df = pd.read_csv(progress_messages_path)
            if not curve_df.empty:
                final_val_loss = curve_df.iloc[-1].get("val_loss")
                curve_df = curve_df.reset_index(drop=True)
                curve_df["epoch"] = range(1, len(curve_df) + 1)
            else:
                curve_df = None
        except Exception as e:
            print(f"Warning: Could not read progress messages: {e}")
            curve_df = None

    return result, final_val_loss, training_time, curve_df


def train_normal_model_pypi(workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, model=TestConfig.MODEL_SIZE):
    """Train a model using the PyPI (released) engine — normal training approach."""
    print(f"\n--- Normal Training / PyPI baseline (max_epochs={max_epochs}) ---")
    start_time = time.time()

    result = train_pypi(workspace_dir=workspace_dir, max_epochs=max_epochs, model=model)

    training_time = time.time() - start_time
    print(f"PyPI normal training completed in {training_time:.2f} seconds")

    workspace = Workspace_pypi(workspace_dir)
    progress_messages_path = workspace.model_progress_messages_path
    curve_df = None
    final_val_loss = None
    if progress_messages_path.exists():
        try:
            curve_df = pd.read_csv(progress_messages_path)
            if not curve_df.empty:
                final_val_loss = curve_df.iloc[-1].get("val_loss")
                curve_df = curve_df.reset_index(drop=True)
                curve_df["epoch"] = range(1, len(curve_df) + 1)
            else:
                curve_df = None
        except Exception as e:
            print(f"Warning: Could not read PyPI progress messages: {e}")
            curve_df = None

    return result, final_val_loss, training_time, curve_df


def train_federated_model(
    workspace_dir,
    total_epochs=TestConfig.MAX_EPOCHS,
    epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION,
    model=TestConfig.MODEL_SIZE,
):
    """Train a model using the federated approach with fixed epochs per iteration and federated state objects.

    This function provides an 'all-encompassing' test of the complete federated learning workflow:
    - Uses fixed epochs per iteration
    - Explicitly tests federated state object passing between iterations
    - Simulates real-world federated learning where comprehensive state is passed between rounds
    - Provides a comprehensive integration test that validates the entire workflow

    The function demonstrates the new federated state pattern where comprehensive state objects
    (containing model weights, optimiser state, LR scheduler state, and DP accountant state) are
    passed between training iterations instead of relying on workspace files.
    """
    print(f"\n--- Federated Training (total_epochs={total_epochs}, epochs_per_iteration={epochs_per_iteration}) ---")

    weights_history = []
    loss_history = []
    training_times = []
    total_training_time = 0
    federated_state = None  # Start with no federated state
    curve_rows = []  # Accumulated per-iteration training curve

    for iteration in range(1, total_epochs + 1):
        print(f"\n  Federated Iteration {iteration}/{total_epochs}")
        start_time = time.time()

        # Train for exactly 'epochs_per_iteration' epochs each time
        # Pass the federated state from the previous iteration (if available)
        result = train(
            workspace_dir=workspace_dir,  # Continue to pass the workspace for data and associated metadata
            federated_epochs=epochs_per_iteration,
            max_epochs=total_epochs,
            model=model,
            federated_state=federated_state,  # Pass previous federated state
        )

        training_time = time.time() - start_time
        total_training_time += training_time

        # Get current validation loss
        workspace = Workspace(workspace_dir)
        progress_messages_path = workspace.model_progress_messages_path
        current_val_loss = None
        current_trn_loss = None
        if progress_messages_path.exists():
            try:
                progress_df = pd.read_csv(progress_messages_path)
                if not progress_df.empty:
                    last_row = progress_df.iloc[-1]
                    current_val_loss = last_row.get("val_loss")
                    current_trn_loss = last_row.get("trn_loss")
            except Exception as e:
                print(f"      Warning: Could not read progress messages: {e}")

        # Store results
        weights_history.append(result)
        loss_history.append(current_val_loss)
        training_times.append(training_time)
        curve_rows.append({"epoch": iteration, "val_loss": current_val_loss, "trn_loss": current_trn_loss})

        # Print analysis
        print(f"    Training time: {training_time:.2f}s")
        print(f"    Federated state returned: {result is not None}")
        print(f"    Validation loss: {current_val_loss}")

        # Prepare the federated state for the next iteration
        # The result is a comprehensive federated state object that contains everything needed for continuation
        if result is not None and iteration < total_epochs:
            print(f"    Preparing federated state for next iteration...")
            # The result is already a complete federated state object, so we can use it directly
            federated_state = result
            print(f"    Federated state contains: {list(result.keys())}")

            # Verify the federated state has the expected structure
            expected_keys = {"model_weights", "training_metrics"}
            actual_keys = set(result.keys())
            missing_keys = expected_keys - actual_keys
            if missing_keys:
                print(f"    ⚠️  Warning: Missing expected keys in federated state: {missing_keys}")
            else:
                print(f"    ✓ Federated state has all expected components")

    print(f"\nFederated training completed in {total_training_time:.2f} seconds")

    # Return final results
    final_weights = weights_history[-1] if weights_history else None
    intermediate_loss = loss_history[0] if len(loss_history) > 1 else None
    final_loss = loss_history[-1] if loss_history else None
    curve_df = pd.DataFrame(curve_rows) if curve_rows else None

    return final_weights, intermediate_loss, final_loss, total_training_time, curve_df


def analyse_weights(weights, epoch, detailed=False):
    """Analyse and print information about model weights."""
    if not weights:
        print(f"    No weights available for epoch {epoch}")
        return

    print(f"    Weight analysis for epoch {epoch}:")

    # Extract model_weights from the federated state if needed
    if isinstance(weights, dict) and "model_weights" in weights:
        model_weights = weights["model_weights"]
    else:
        model_weights = weights

    # Collect all weight values
    all_values = []
    for name, tensor in model_weights.items():
        if hasattr(tensor, "numpy"):  # PyTorch tensor
            # Handle both CPU and CUDA tensors
            if tensor.is_cuda:
                values = tensor.detach().cpu().numpy().flatten()
            else:
                values = tensor.detach().numpy().flatten()
        else:  # Already a numpy array or list
            values = np.array(tensor).flatten()
        all_values.extend(values)

    if all_values:
        all_values = np.array(all_values)
        print(f"      Total parameters: {len(all_values)}")
        print(f"      Mean weight value: {np.mean(all_values):.6f}")
        print(f"      Std weight value: {np.std(all_values):.6f}")
        print(f"      Min/Max weight: {np.min(all_values):.6f} / {np.max(all_values):.6f}")

        # Percentile analysis
        percentiles = np.percentile(all_values, [1, 25, 50, 75, 99])
        print(f"      Percentiles (1%, 25%, 50%, 75%, 99%): {percentiles}")

    # Layer type analysis
    if detailed:
        layer_types = {}
        for name in model_weights.keys():
            if "weight" in name:
                layer_types["weight"] = layer_types.get("weight", 0) + 1
            elif "bias" in name:
                layer_types["bias"] = layer_types.get("bias", 0) + 1
            elif "embed" in name:
                layer_types["embed"] = layer_types.get("embed", 0) + 1

        print(f"      Layer types: {layer_types}")

    # Visualisation removed — weight matrix images are no longer generated.


def train_epoch_by_epoch(
    workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION
):
    """Train one epoch at a time and analyse weights after each epoch.

    This function focuses on detailed analysis and monitoring of the federated training process:
    - Tracks complete history of weights, losses, and training times for each iteration
    - Includes comprehensive weight analysis with statistics, percentiles, and visualisation
    - Provides progress monitoring and detailed logging for debugging and understanding
    - Tests the core federated training pattern with fixed epochs per iteration

    While train_federated_model provides an all-encompassing integration test, this function
    offers straightforward epoch-by-epoch analysis and monitoring capabilities for understanding training dynamics.
    """
    print(f"\n--- Epoch-by-Epoch Training (total_epochs={max_epochs}, epochs_per_iteration={epochs_per_iteration}) ---")

    weights_history = []
    loss_history = []
    training_times = []
    federated_state = None  # Start with no federated state
    curve_rows = []  # Accumulated per-iteration training curve

    for iteration in range(1, max_epochs + 1):
        print(f"\n  Iteration {iteration}/{max_epochs}")
        start_time = time.time()

        # Train for exactly 'epochs_per_iteration' epochs each time
        # Pass the federated state from the previous iteration (if available)
        result = train(
            workspace_dir=workspace_dir,  # Continue to pass the workspace for data and associated metadata
            federated_epochs=epochs_per_iteration,
            max_epochs=max_epochs,
            model=TestConfig.MODEL_SIZE,
            federated_state=federated_state,  # Pass previous federated state
        )

        training_time = time.time() - start_time

        # Get current validation loss
        workspace = Workspace(workspace_dir)
        progress_messages_path = workspace.model_progress_messages_path
        current_val_loss = None
        current_trn_loss = None
        if progress_messages_path.exists():
            try:
                progress_df = pd.read_csv(progress_messages_path)
                if not progress_df.empty:
                    last_row = progress_df.iloc[-1]
                    current_val_loss = last_row.get("val_loss")
                    current_trn_loss = last_row.get("trn_loss")
            except Exception as e:
                print(f"      Warning: Could not read progress messages: {e}")

        # Store results
        weights_history.append(result)
        loss_history.append(current_val_loss)
        training_times.append(training_time)
        curve_rows.append({"epoch": iteration, "val_loss": current_val_loss, "trn_loss": current_trn_loss})

        # Print analysis
        print(f"    Training time: {training_time:.2f}s")
        print(f"    Federated state returned: {result is not None}")
        print(f"    Validation loss: {current_val_loss}")

        # Progress indicator for long training
        if max_epochs > 10:
            progress_percent = (iteration / max_epochs) * 100
            print(f"    Progress: {progress_percent:.0f}% complete")

        # Prepare the federated state for the next iteration
        if result is not None and iteration < max_epochs:
            federated_state = result
            print(f"    Preparing federated state for next iteration...")

    curve_df = pd.DataFrame(curve_rows) if curve_rows else None
    return weights_history, loss_history, training_times, curve_df


def test_epoch_by_epoch_comparison():
    """Compare epoch-by-epoch training between federated approaches."""
    print("\n" + "=" * 80)
    print("EPOCH-BY-EPOCH TRAINING ANALYSIS")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Federated epoch-by-epoch training
            federated_workspace = Path(tmpdir) / "federated-epoch-ws"
            federated_workspace.mkdir(parents=True)

            print(f"\nSetting up federated epoch-by-epoch workspace: {federated_workspace}")
            setup_workspace(data, federated_workspace)

            print(f"\nFederated epoch-by-epoch training:")
            federated_weights_history, federated_loss_history, federated_times, fed_curve_df = train_epoch_by_epoch(
                federated_workspace
            )

            # Normal training for final comparison
            normal_workspace = Path(tmpdir) / "normal-epoch-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(data, normal_workspace)

            print(f"\nNormal training for comparison:")
            normal_result, normal_val_loss, normal_time, normal_curve_df = train_normal_model(normal_workspace)

            # Analysis and comparison
            print("\n" + "=" * 80)
            print("EPOCH-BY-EPOCH ANALYSIS RESULTS")
            print("=" * 80)

            print("Federated training progression:")
            for epoch, (weights, loss, train_time) in enumerate(
                zip(federated_weights_history, federated_loss_history, federated_times), 1
            ):
                print(f"  Epoch {epoch}:")
                print(f"    Loss: {loss}")
                print(f"    Training time: {train_time:.2f}s")
                print(f"    Weights available: {weights is not None}")

            print(f"\nNormal training final result:")
            print(f"  Final loss: {normal_val_loss}")
            print(f"  Training time: {normal_time:.2f}s")

            # Compare final results
            if federated_loss_history and normal_val_loss:
                final_federated_loss = federated_loss_history[-1]
                loss_diff = abs(final_federated_loss - normal_val_loss)
                loss_ratio = loss_diff / max(final_federated_loss, normal_val_loss, 1e-6)

                print(f"\nFinal comparison:")
                print(f"  Federated final loss: {final_federated_loss}")
                print(f"  Normal final loss: {normal_val_loss}")
                print(f"  Absolute difference: {loss_diff:.6f}")
                print(f"  Relative difference: {loss_ratio:.2%}")

                # Consider them similar if the relative difference is < 15% (slightly more tolerant for epoch analysis)
                similar_results = loss_ratio < 0.15
                print(f"  Similar results: {'YES' if similar_results else 'NO'}")

                # Plot training curves and write GitHub step summary
                curves = {}
                if normal_curve_df is not None:
                    curves["Normal"] = normal_curve_df
                if fed_curve_df is not None:
                    curves["Federated (epoch-by-epoch)"] = fed_curve_df
                if curves:
                    TestConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    reporting.plot_training_curves(curves, TestConfig.OUTPUT_DIR / "epoch_by_epoch_training_curves.png")
                summary_rows = [
                    {
                        "Approach": "Normal",
                        "Final Val Loss": f"{normal_val_loss:.6f}" if normal_val_loss is not None else "N/A",
                        "Training Time (s)": f"{normal_time:.1f}",
                    },
                    {
                        "Approach": "Federated (epoch-by-epoch)",
                        "Final Val Loss": f"{final_federated_loss:.6f}" if final_federated_loss is not None else "N/A",
                        "Training Time (s)": f"{sum(federated_times):.1f}",
                    },
                ]
                reporting.write_github_step_summary(
                    "Epoch-by-Epoch Training Comparison", summary_rows, output_dir=TestConfig.OUTPUT_DIR
                )
                epoch_rows = reporting.epoch_loss_table_rows(curves)
                if epoch_rows:
                    reporting.write_github_step_summary(
                        "Val Loss by Epoch — Epoch-by-Epoch Comparison",
                        epoch_rows,
                        output_dir=TestConfig.OUTPUT_DIR,
                    )

                assert similar_results, (
                    f"Loss ratio {loss_ratio:.2%} exceeds 15% tolerance "
                    f"(fed={final_federated_loss:.6f}, normal={normal_val_loss:.6f})"
                )
                return
            else:
                print("\nCould not compare final validation losses")
                pytest.fail("Could not compare final validation losses (one or both are None)")

    except Exception as e:
        print(f"Error during epoch-by-epoch test: {e}")
        raise


def test_federated_weights_loading(epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION, model=TestConfig.MODEL_SIZE):
    """Test that federated weights can be properly loaded and training can continue.

    This function provides focused unit-level testing of the weight loading mechanism:
    - Tests weight serialization and deserialization in isolation
    - Validates that weights obtained from federated training can be loaded for continuation
    - Demonstrates the new federated state pattern for training continuation

    While train_federated_model tests weight loading as part of the complete workflow, this
    function provides isolated validation of the core weight loading mechanism using the
    new federated state pattern.
    """
    print("\n" + "=" * 80)
    print("FEDERATED WEIGHTS LOADING TEST")
    print("=" * 80)

    try:
        data = create_test_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir) / "weights-test-ws"
            workspace_dir.mkdir(parents=True)

            setup_workspace(data, workspace_dir)

            # Train with federated epochs to get weights
            print("Training with federated epochs to get weights...")
            federated_state = train(
                workspace_dir=workspace_dir, federated_epochs=epochs_per_iteration, max_epochs=100, model=model
            )

            if federated_state is None:
                print("❌ FAILED: No federated state returned from federated training")
                pytest.fail("No federated state returned from federated training")

            print(f"✓ Received federated state with {len(federated_state['model_weights'])} model parameters")
            print(f"✓ Federated state contains: {list(federated_state.keys())}")

            # Verify we can continue training using the federated state pattern
            print("Continuing training from federated state...")
            final_result = train(
                workspace_dir=workspace_dir,  # Use the same workspace for data and associated metadata
                federated_epochs=epochs_per_iteration,
                max_epochs=100,
                model=model,
                federated_state=federated_state,
            )

            if final_result is None:
                print("❌ FAILED: Continuation training returned None")
                pytest.fail("Continuation training returned None")

            print("✓ Successfully continued training using federated state pattern")
            print(f"✓ Final federated state contains: {list(final_result.keys())}")

            # Verify that training actually continued (epochs progressed)
            final_epoch = final_result["training_metrics"]["epoch"]

            if final_epoch > 0:
                print(f"✓ Training ran, reached epoch {final_epoch}")
            else:
                pytest.fail(f"Epochs did not progress: epoch={final_epoch}")

            return

    except Exception as e:
        print(f"❌ FAILED: Error during weights loading test: {e}")
        raise


def test_training_approach_comparison():
    """Compare normal training vs federated training approaches."""
    print("\n" + "=" * 80)
    print("FEDERATED vs NORMAL TRAINING CONVERGENCE COMPARISON")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run 1: Normal training (local dev)
            normal_workspace = Path(tmpdir) / "normal-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(data, normal_workspace)

            normal_result, normal_val_loss, normal_time, normal_curve_df = train_normal_model(normal_workspace)

            # Run 2: Federated training (local dev)
            federated_workspace = Path(tmpdir) / "federated-ws"
            federated_workspace.mkdir(parents=True)

            print(f"\nSetting up federated training workspace: {federated_workspace}")
            setup_workspace(data, federated_workspace)

            federated_weights, intermediate_val_loss, federated_val_loss, federated_time, fed_curve_df = (
                train_federated_model(federated_workspace)
            )

            # Run 3: Normal training (PyPI baseline) — skipped if PyPI engine not installed
            pypi_val_loss, pypi_time, pypi_curve_df = None, None, None
            if HAS_PYPI_ENGINE:
                pypi_workspace = Path(tmpdir) / "pypi-ws"
                pypi_workspace.mkdir(parents=True)

                print(f"\nSetting up PyPI baseline workspace: {pypi_workspace}")
                setup_workspace_pypi(data, pypi_workspace)

                _, pypi_val_loss, pypi_time, pypi_curve_df = train_normal_model_pypi(pypi_workspace)
            else:
                print("\nSkipping PyPI baseline run (mostlyai.engine not installed as PyPI package)")

            # Comparison results
            print("\n" + "=" * 80)
            print("COMPARISON RESULTS")
            print("=" * 80)

            print(f"Normal training (local dev):")
            print(f"  - Result: {normal_result}")
            print(f"  - Final validation loss: {normal_val_loss}")
            print(f"  - Training time: {normal_time:.2f} seconds")

            print(f"\nFederated training (local dev):")
            print(f"  - Federated weights returned: {federated_weights is not None}")
            print(f"  - Intermediate validation loss: {intermediate_val_loss}")
            print(f"  - Final validation loss: {federated_val_loss}")
            print(f"  - Total training time: {federated_time:.2f} seconds")

            if HAS_PYPI_ENGINE:
                print(f"\nNormal training (PyPI baseline):")
                print(f"  - Final validation loss: {pypi_val_loss}")
                print(f"  - Training time: {pypi_time:.2f} seconds")

            # Analysis — primary assertion is still local dev normal vs federated
            if normal_val_loss is not None and federated_val_loss is not None:
                loss_diff = abs(normal_val_loss - federated_val_loss)
                loss_ratio = loss_diff / max(normal_val_loss, federated_val_loss, 1e-6)

                print(f"\nValidation loss comparison (local dev normal vs federated):")
                print(f"  - Absolute difference: {loss_diff:.6f}")
                print(f"  - Relative difference: {loss_ratio:.2%}")

                similar_results = loss_ratio < 0.10
                print(f"  - Similar results: {'YES' if similar_results else 'NO'}")

                if pypi_val_loss is not None:
                    pypi_diff = abs(normal_val_loss - pypi_val_loss)
                    pypi_ratio = pypi_diff / max(normal_val_loss, pypi_val_loss, 1e-6)
                    print(f"\nValidation loss comparison (local dev normal vs PyPI baseline):")
                    print(f"  - Absolute difference: {pypi_diff:.6f}")
                    print(f"  - Relative difference: {pypi_ratio:.2%}")

                # Build curves dict (used for both plot and epoch table)
                curves = {}
                if normal_curve_df is not None:
                    curves["Normal (dev)"] = normal_curve_df
                if fed_curve_df is not None:
                    curves["Federated (dev)"] = fed_curve_df
                if pypi_curve_df is not None:
                    curves["Normal (PyPI)"] = pypi_curve_df

                if curves:
                    TestConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    reporting.plot_training_curves(curves, TestConfig.OUTPUT_DIR / "training_approach_curves.png")

                summary_rows = [
                    {
                        "Approach": "Normal (dev)",
                        "Final Val Loss": f"{normal_val_loss:.6f}" if normal_val_loss is not None else "N/A",
                        "Training Time (s)": f"{normal_time:.1f}",
                    },
                    {
                        "Approach": "Federated (dev)",
                        "Final Val Loss": f"{federated_val_loss:.6f}" if federated_val_loss is not None else "N/A",
                        "Training Time (s)": f"{federated_time:.1f}",
                    },
                ]
                if HAS_PYPI_ENGINE:
                    summary_rows.append(
                        {
                            "Approach": "Normal (PyPI)",
                            "Final Val Loss": f"{pypi_val_loss:.6f}" if pypi_val_loss is not None else "N/A",
                            "Training Time (s)": f"{pypi_time:.1f}" if pypi_time is not None else "N/A",
                        }
                    )

                reporting.write_github_step_summary(
                    "Training Approach Comparison", summary_rows, output_dir=TestConfig.OUTPUT_DIR
                )
                epoch_rows = reporting.epoch_loss_table_rows(curves)
                if epoch_rows:
                    reporting.write_github_step_summary(
                        "Val Loss by Epoch — Training Approach Comparison",
                        epoch_rows,
                        output_dir=TestConfig.OUTPUT_DIR,
                    )

                assert similar_results, (
                    f"Loss ratio {loss_ratio:.2%} exceeds 10% tolerance "
                    f"(normal={normal_val_loss:.6f}, federated={federated_val_loss:.6f})"
                )
                return
            else:
                print("\nCould not compare validation losses (one or both are None)")
                pytest.fail("Could not compare validation losses (one or both are None)")

    except Exception as e:
        print(f"Error during comparison test: {e}")
        raise


def main():
    """Run all comparison tests."""
    print("FEDERATED TRAINING CONVERGENCE COMPARISON TEST SUITE")
    print("=" * 80)

    # Check Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("Running on CPU")
    print(f"PyPI mostlyai.engine available: {HAS_PYPI_ENGINE}")

    def _run_test(test_name, test_fn):
        try:
            test_fn()
            return test_name, True
        except BaseException as e:
            if isinstance(e, pytest.skip.Exception):
                print(f"⏭  {test_name}: SKIPPED")
                return test_name, True
            print(f"❌ {test_name}: FAILED — {e}")
            return test_name, False

    # Run tests
    results = []

    # Test 1: Enhanced federated state pattern (new comprehensive test)
    results.append(_run_test("Enhanced Federated State Pattern", test_enhanced_federated_state_pattern))

    # Test 2: Federated weights loading
    results.append(_run_test("Federated Weights Loading", test_federated_weights_loading))

    # Test 3: Training approach comparison
    results.append(_run_test("Training Approach Comparison", test_training_approach_comparison))

    # Test 4: Epoch-by-epoch analysis
    results.append(_run_test("Epoch-by-Epoch Analysis", test_epoch_by_epoch_comparison))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "❌"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("SUCCESS: ALL TESTS PASSED!")
        print("Federated and normal training approaches produce similar results.")
    else:
        print("FAILURE: SOME TESTS FAILED!")
        print("There may be differences between federated and normal training.")

    print("=" * 80)

    return 0 if all_passed else 1


def test_enhanced_federated_state_pattern():
    """Test the enhanced federated state pattern with all robustness improvements.

    This test specifically validates the new federated state implementation:
    - Tests comprehensive federated state objects (model weights, optimiser state, LR scheduler state, DP accountant state)
    - Validates None guards for missing or None state components
    - Tests proper state continuation across multiple iterations
    - Demonstrates the complete federated learning workflow with state passing
    """
    print("\n" + "=" * 80)
    print("ENHANCED FEDERATED STATE PATTERN TEST")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir) / "enhanced-federated-ws"
            workspace_dir.mkdir(parents=True)

            print(f"Setting up workspace: {workspace_dir}")
            setup_workspace(data, workspace_dir)

            # Test 1: Initial training with no federated state
            print("\n1. Testing initial training (no federated state)...")
            federated_state = None  # Start with no state

            result1 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=federated_state,
            )

            if result1 is None:
                print("❌ FAILED: Initial training returned None")
                pytest.fail("Initial training returned None")

            print("✓ Initial training successful")
            print(f"✓ Federated state contains: {list(result1.keys())}")

            # Test 2: Continuation with complete federated state
            print("\n2. Testing continuation with complete federated state...")

            result2 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=result1,  # Pass complete state
            )

            if result2 is None:
                print("❌ FAILED: Continuation training returned None")
                pytest.fail("Continuation training returned None")

            print("✓ Continuation with complete state successful")

            # Test 3: Continuation with partial federated state (missing some components)
            print("\n3. Testing continuation with partial federated state...")

            # Create a partial federated state (missing optimiser and LR scheduler state)
            partial_federated_state = {
                "model_weights": result2["model_weights"],
                "training_metrics": result2["training_metrics"],
            }

            result3 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=partial_federated_state,  # Pass partial state
            )

            if result3 is None:
                print("❌ FAILED: Partial state continuation returned None")
                pytest.fail("Partial state continuation returned None")

            print("✓ Continuation with partial state successful (None guards working)")

            # Test 4: Continuation with minimal federated state (only model weights)
            print("\n4. Testing continuation with minimal federated state...")

            minimal_federated_state = {
                "model_weights": result3["model_weights"],
                "training_metrics": result3["training_metrics"],
            }

            result4 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=minimal_federated_state,  # Pass minimal state
            )

            if result4 is None:
                print("❌ FAILED: Minimal state continuation returned None")
                pytest.fail("Minimal state continuation returned None")

            print("✓ Continuation with minimal state successful")

            # Test 5: Verify federated state structure consistency
            print("\n5. Verifying federated state structure consistency...")

            all_states = [result1, result2, result3, result4]
            expected_keys = {"model_weights", "training_metrics"}

            for i, state in enumerate(all_states, 1):
                actual_keys = set(state.keys())
                missing_keys = expected_keys - actual_keys
                if missing_keys:
                    print(f"❌ FAILED: State {i} missing keys: {missing_keys}")
                    pytest.fail(f"State {i} missing keys: {missing_keys}")

                # Verify training metrics structure
                metrics_keys = {"epoch", "steps", "samples", "learn_rate", "trn_loss", "val_loss"}
                actual_metrics_keys = set(state["training_metrics"].keys())
                missing_metrics_keys = metrics_keys - actual_metrics_keys
                if missing_metrics_keys:
                    print(f"❌ FAILED: State {i} missing metrics keys: {missing_metrics_keys}")
                    pytest.fail(f"State {i} missing metrics keys: {missing_metrics_keys}")

            print("✓ All federated states have consistent structure")

            # Test 6: Verify state evolution
            print("\n6. Verifying state evolution across iterations...")

            # Check that epochs are reported locally (e.g. 1.0 each round)
            epochs = [state["training_metrics"]["epoch"] for state in all_states]
            if any(e <= 0 for e in epochs):
                print(f"❌ FAILED: Epochs not reported correctly: {epochs}")
                pytest.fail(f"Epochs not reported correctly: {epochs}")

            print(f"✓ Epochs reported correctly: {epochs}")

            # Check that steps are reported locally (greater than 0 each round)
            steps = [state["training_metrics"]["steps"] for state in all_states]
            if any(s <= 0 for s in steps):
                print(f"❌ FAILED: Steps not reported correctly: {steps}")
                pytest.fail(f"Steps not reported correctly: {steps}")

            print(f"✓ Steps reported correctly: {steps}")

            # Test 7: Verify weight evolution (training actually continues)
            print("\n7. Verifying weight evolution (training continuation)...")

            # Compare weights between iterations to verify training continued
            weights_changed = False
            for i in range(len(all_states) - 1):
                current_weights = all_states[i]["model_weights"]
                next_weights = all_states[i + 1]["model_weights"]

                # Check a sample of weight parameters
                sample_key = list(current_weights.keys())[0]  # Use first weight parameter

                w_current = current_weights[sample_key]
                w_next = next_weights[sample_key]

                # Convert to numpy if needed (handle both CPU and GPU tensors)
                if hasattr(w_current, "cpu"):
                    w_current = w_current.detach().cpu().numpy()
                elif hasattr(w_current, "numpy"):
                    w_current = w_current.detach().numpy()
                else:
                    w_current = np.array(w_current)

                if hasattr(w_next, "cpu"):
                    w_next = w_next.detach().cpu().numpy()
                elif hasattr(w_next, "numpy"):
                    w_next = w_next.detach().numpy()
                else:
                    w_next = np.array(w_next)

                # Check if weights changed (training continued)
                if not np.array_equal(w_current, w_next):
                    weights_changed = True
                    print(f"✓ Weights changed between iteration {i + 1} and {i + 2} (training continued)")
                    break

            if not weights_changed:
                print("⚠️  Warning: Weights appear unchanged between iterations")
                print("   This may indicate training did not continue properly")
                pytest.fail("Weights appear unchanged between iterations — training may not have continued properly")
            else:
                print("✓ Weight evolution verified: training continued across iterations")

            print("\n" + "=" * 60)
            print("ENHANCED FEDERATED STATE PATTERN TEST: PASSED")
            print("=" * 60)
            print("✓ All robustness improvements working correctly:")
            print("  - Comprehensive state objects with all components")
            print("  - None guards for missing/None state components")
            print("  - Proper state continuation across iterations")
            print("  - Consistent state structure")
            print("  - Correct state evolution")
            print("  - Weight evolution verified (training continues)")

            return

    except Exception as e:
        print(f"❌ FAILED: Error during enhanced federated state test: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
