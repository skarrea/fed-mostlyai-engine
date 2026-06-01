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
End-to-end test to compare normal training vs federated training convergence for language models.

This test evaluates whether:
1. Training a language model without federated epochs (normal training)
2. Training a language model with federated epochs, loading weights, and continuing training

Both approaches should converge to similar results.
"""

import sys
import tempfile
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mostlyai.engine import analyze, encode, generate, split, train  # noqa: E402
from mostlyai.engine._language.lstm import LSTMFromScratchConfig  # noqa: E402
from mostlyai.engine._workspace import Workspace  # noqa: E402
from mostlyai.engine.domain import ModelEncodingType  # noqa: E402


# ============================================
# TEST CONFIGURATION - Adjust these parameters
# ============================================
class TestConfig:
    """Centralised configuration for test parameters"""

    # Training parameters
    EPOCHS_PER_ITERATION = 1  # Number of epochs per federated iteration
    MAX_EPOCHS = 10  # Maximum number of epochs for training
    MODEL_SIZE = LSTMFromScratchConfig.model_id  # Use LSTM model for faster testing
    # MODEL_SIZE = "mistralai/Mistral-7B-v0.3"  # Use HF Mistral model for extensive testing

    # Data generation parameters
    TOTAL_SAMPLES = 200  # Total samples to create
    TEST_SAMPLES = 50  # Samples used for quality testing

    # Quality assessment
    GENERATE_HTML_REPORTS = True  # Set to True to generate HTML reports (slower)
    QUALITY_TOLERANCE = 0.15  # 15% tolerance for quality score comparison (language models are noisier)


# ============================================

# Optional imports for visualisation (uncomment if you want plots)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available - visualisation features disabled")

# Optional import for quality assessment
try:
    from mostlyai import qa  # noqa: F401

    HAS_QA_LIBRARY = True
except ImportError:
    HAS_QA_LIBRARY = False
    print("Note: mostlyai-qa library not available - quality assessment features disabled")


def create_test_data():
    """Create sample language data for testing."""
    no_of_records = TestConfig.TOTAL_SAMPLES
    data = pd.DataFrame(
        {
            "gender": ["m", "f", "x", pd.NA] * int(no_of_records / 4),
            "bio": list(chain(*[[f"Joe {i}", f"Anna {i}", pd.NA, pd.NA] for i in range(int(no_of_records / 4))])),
        }
    )
    return data


def setup_workspace(data, workspace_dir):
    """Set up a workspace with split, analyse, and encode steps for language model."""
    split(
        tgt_data=data,
        model_type="LANGUAGE",
        tgt_encoding_types={
            "gender": ModelEncodingType.language_categorical,
            "bio": ModelEncodingType.language_text,
        },
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)


def train_normal_model(workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, model=TestConfig.MODEL_SIZE):
    """Train a model using the normal training approach."""
    print(f"\n--- Normal Training (max_epochs={max_epochs}) ---")
    start_time = time.time()

    result = train(
        workspace_dir=workspace_dir,
        max_epochs=max_epochs,
        model=model,
    )

    training_time = time.time() - start_time
    print(f"Normal training completed in {training_time:.2f} seconds")
    print(f"Normal training result: {result}")

    # Get final validation loss from progress messages CSV
    workspace = Workspace(workspace_dir)
    progress_messages_path = workspace.model_progress_messages_path
    final_val_loss = None
    if progress_messages_path.exists():
        try:
            progress_df = pd.read_csv(progress_messages_path)
            if not progress_df.empty:
                final_val_loss = progress_df.iloc[-1].get("val_loss")
        except Exception as e:
            print(f"Warning: Could not read progress messages: {e}")

    return result, final_val_loss, training_time


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
        if progress_messages_path.exists():
            try:
                progress_df = pd.read_csv(progress_messages_path)
                if not progress_df.empty:
                    current_val_loss = progress_df.iloc[-1].get("val_loss")
            except Exception as e:
                print(f"      Warning: Could not read progress messages: {e}")

        # Store results
        weights_history.append(result)
        loss_history.append(current_val_loss)
        training_times.append(training_time)

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
            expected_keys = {"model_weights", "training_metrics", "optimizer_state", "lr_scheduler_state"}
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

    return final_weights, intermediate_loss, final_loss, total_training_time


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

    # Visualisation
    if HAS_MATPLOTLIB and len(all_values) > 0:
        try:
            plt.figure(figsize=(10, 4))
            plt.hist(all_values, bins=50, alpha=0.7, color="blue")
            plt.title(f"Weight Distribution - Epoch {epoch}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"language_weight_distribution_epoch_{epoch}.png")
            plt.close()
            print(f"      Saved weight distribution plot")
        except Exception as e:
            print(f"      Warning: Could not create plot: {e}")


def train_epoch_by_epoch(
    workspace_dir,
    max_epochs=TestConfig.MAX_EPOCHS,
    epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION,
):
    """Train one epoch at a time and analyse weights after each epoch.

    This function focuses on detailed analysis and monitoring of the federated training process:
    - Tracks complete history of weights, losses, and training times for each iteration
    - Includes comprehensive weight analysis with statistics, percentiles, and visualisation
    - Provides progress monitoring and detailed logging for debugging and understanding
    - Tests the core federated training pattern with fixed epochs per iteration
    """
    print(f"\n--- Epoch-by-Epoch Training (total_epochs={max_epochs}, epochs_per_iteration={epochs_per_iteration}) ---")

    weights_history = []
    loss_history = []
    training_times = []
    federated_state = None  # Start with no federated state

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
        if progress_messages_path.exists():
            try:
                progress_df = pd.read_csv(progress_messages_path)
                if not progress_df.empty:
                    current_val_loss = progress_df.iloc[-1].get("val_loss")
            except Exception as e:
                print(f"      Warning: Could not read progress messages: {e}")

        # Store results
        weights_history.append(result)
        loss_history.append(current_val_loss)
        training_times.append(training_time)

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

        if result:
            analyse_weights(result, iteration)

    return weights_history, loss_history, training_times


def test_epoch_by_epoch_comparison():
    """Compare epoch-by-epoch training between federated approaches for language models."""
    print("\n" + "=" * 80)
    print("LANGUAGE MODEL EPOCH-BY-EPOCH TRAINING ANALYSIS")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Created test data with {len(data)} samples")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Federated epoch-by-epoch training
            federated_workspace = Path(tmpdir) / "federated-epoch-ws"
            federated_workspace.mkdir(parents=True)

            print(f"\nSetting up federated epoch-by-epoch workspace: {federated_workspace}")
            setup_workspace(data, federated_workspace)

            print(f"\nFederated epoch-by-epoch training:")
            federated_weights_history, federated_loss_history, federated_times = train_epoch_by_epoch(
                federated_workspace
            )

            # Normal training for final comparison
            normal_workspace = Path(tmpdir) / "normal-epoch-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(data, normal_workspace)

            print(f"\nNormal training for comparison:")
            normal_result, normal_val_loss, normal_time = train_normal_model(normal_workspace)

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

                # Consider them similar if the relative difference is < 20% (language models are noisier)
                similar_results = loss_ratio < 0.20
                print(f"  Similar results: {'YES' if similar_results else 'NO'}")

                assert similar_results, (
                    f"Loss ratio {loss_ratio:.2%} exceeds 20% tolerance "
                    f"(fed={final_federated_loss:.6f}, normal={normal_val_loss:.6f})"
                )
                return
            else:
                print("\nCould not compare final validation losses")
                pytest.fail("Could not compare final validation losses (one or both are None)")

    except Exception as e:
        print(f"Error during epoch-by-epoch test: {e}")
        raise


def test_federated_weights_loading(
    epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION,
    model=TestConfig.MODEL_SIZE,
):
    """Test that federated weights can be properly loaded and training can continue for language models.

    This function provides focused unit-level testing of the weight loading mechanism:
    - Tests weight serialization and deserialization in isolation
    - Validates that weights obtained from federated training can be loaded for continuation
    - Demonstrates the new federated state pattern for training continuation
    """
    print("\n" + "=" * 80)
    print("LANGUAGE MODEL FEDERATED WEIGHTS LOADING TEST")
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
                workspace_dir=workspace_dir,
                federated_epochs=epochs_per_iteration,
                max_epochs=100,
                model=model,
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
    """Compare normal training vs federated training approaches for language models."""
    print("\n" + "=" * 80)
    print("LANGUAGE MODEL FEDERATED vs NORMAL TRAINING CONVERGENCE COMPARISON")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Created test data with {len(data)} samples")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test 1: Normal training
            normal_workspace = Path(tmpdir) / "normal-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(data, normal_workspace)

            normal_result, normal_val_loss, normal_time = train_normal_model(normal_workspace)

            # Test 2: Federated training with continuation
            federated_workspace = Path(tmpdir) / "federated-ws"
            federated_workspace.mkdir(parents=True)

            print(f"\nSetting up federated training workspace: {federated_workspace}")
            setup_workspace(data, federated_workspace)

            federated_weights, intermediate_val_loss, federated_val_loss, federated_time = train_federated_model(
                federated_workspace
            )

            # Comparison results
            print("\n" + "=" * 80)
            print("COMPARISON RESULTS")
            print("=" * 80)

            print(f"Normal training:")
            print(f"  - Result: {normal_result}")
            print(f"  - Final validation loss: {normal_val_loss}")
            print(f"  - Training time: {normal_time:.2f} seconds")

            print(f"\nFederated training:")
            print(f"  - Federated weights returned: {federated_weights is not None}")
            print(f"  - Intermediate validation loss: {intermediate_val_loss}")
            print(f"  - Final validation loss: {federated_val_loss}")
            print(f"  - Total training time: {federated_time:.2f} seconds")

            # Analysis
            if normal_val_loss is not None and federated_val_loss is not None:
                loss_diff = abs(normal_val_loss - federated_val_loss)
                loss_ratio = loss_diff / max(normal_val_loss, federated_val_loss, 1e-6)

                print(f"\nValidation loss comparison:")
                print(f"  - Absolute difference: {loss_diff:.6f}")
                print(f"  - Relative difference: {loss_ratio:.2%}")

                # Consider them similar if the relative difference is < 15%
                similar_results = loss_ratio < 0.15
                print(f"  - Similar results: {'YES' if similar_results else 'NO'}")

                assert similar_results, (
                    f"Loss ratio {loss_ratio:.2%} exceeds 15% tolerance "
                    f"(normal={normal_val_loss:.6f}, federated={federated_val_loss:.6f})"
                )
                return
            else:
                print("\nCould not compare validation losses (one or both are None)")
                pytest.fail("Could not compare validation losses (one or both are None)")

    except Exception as e:
        print(f"Error during comparison test: {e}")
        raise


def test_data_generation_quality():
    """Compare data generation quality between federated and normal training approaches for language models.

    This test generates synthetic data using both normal training and federated training,
    then compares the generated datasets directly to see if they are statistically similar.
    """
    print("\n" + "=" * 80)
    print("LANGUAGE MODEL DATA GENERATION QUALITY COMPARISON")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    try:
        data = create_test_data()
        print(f"Created test data with {len(data)} samples")

        # Use the same training data for both approaches
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up the workspace for normal training
            normal_workspace = Path(tmpdir) / "normal-gen-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(train_data, normal_workspace)

            # Train normally for specified epochs
            print("Training normal model...")
            train(
                workspace_dir=normal_workspace,
                max_epochs=TestConfig.MAX_EPOCHS,
                model=TestConfig.MODEL_SIZE,
            )

            # Generate synthetic data from the normal model
            print("Generating data from normal model...")
            generate(
                workspace_dir=normal_workspace,
                sample_size=TestConfig.TEST_SAMPLES,
            )
            # Read the generated data
            normal_synthetic = pd.read_parquet(normal_workspace / "SyntheticData")

            # Set up the workspace for federated training
            federated_workspace = Path(tmpdir) / "federated-gen-ws"
            federated_workspace.mkdir(parents=True)

            print(f"\nSetting up federated training workspace: {federated_workspace}")
            setup_workspace(train_data, federated_workspace)

            # Train with the federated approach (fixed epochs per iteration)
            print("Training federated model with fixed epochs per iteration...")
            federated_state = None
            for iteration in range(1, TestConfig.MAX_EPOCHS + 1):
                federated_state = train(
                    workspace_dir=federated_workspace,
                    federated_epochs=1,
                    max_epochs=TestConfig.MAX_EPOCHS,
                    model=TestConfig.MODEL_SIZE,
                    federated_state=federated_state,
                )
                print(f"  Completed iteration {iteration}/{TestConfig.MAX_EPOCHS}")

            # Generate synthetic data from the federated model
            print("Generating data from federated model...")
            generate(
                workspace_dir=federated_workspace,
                sample_size=TestConfig.TEST_SAMPLES,
            )
            # Read the generated data
            federated_synthetic = pd.read_parquet(federated_workspace / "SyntheticData")

            # ============================================================
            # PRINT THE ACTUAL DATA so we can see what each model produced
            # ============================================================

            # --- Original training data (first 20 rows) ---
            print("\n" + "=" * 80)
            print("ORIGINAL TRAINING DATA (first 20 rows)")
            print("=" * 80)
            with pd.option_context(
                "display.max_rows", 20, "display.max_columns", None, "display.width", 200, "display.max_colwidth", 80
            ):
                print(train_data.head(20).to_string(index=True))

            # --- Normal synthetic data (all rows) ---
            print("\n" + "=" * 80)
            print(f"NORMAL SYNTHETIC DATA ({len(normal_synthetic)} rows)")
            print("=" * 80)
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None, "display.width", 200, "display.max_colwidth", 80
            ):
                print(normal_synthetic.to_string(index=True))

            # --- Federated synthetic data (all rows) ---
            print("\n" + "=" * 80)
            print(f"FEDERATED SYNTHETIC DATA ({len(federated_synthetic)} rows)")
            print("=" * 80)
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None, "display.width", 200, "display.max_colwidth", 80
            ):
                print(federated_synthetic.to_string(index=True))

            # --- Side-by-side sample comparison (first 20 rows) ---
            print("\n" + "=" * 80)
            print("SIDE-BY-SIDE COMPARISON (first 20 rows)")
            print("=" * 80)
            # Build a combined DataFrame with prefixed columns for easy visual diff
            display_cols = [c for c in normal_synthetic.columns if c != "__primary_key"]
            n_preview = min(20, len(normal_synthetic), len(federated_synthetic))
            side_by_side = pd.DataFrame(index=range(n_preview))
            for col in display_cols:
                side_by_side[f"normal_{col}"] = normal_synthetic[col].iloc[:n_preview].values
                side_by_side[f"fed_{col}"] = federated_synthetic[col].iloc[:n_preview].values
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None, "display.width", 240, "display.max_colwidth", 60
            ):
                print(side_by_side.to_string(index=True))

            # ============================================================
            # STATISTICAL COMPARISON
            # ============================================================

            # Direct comparison between the two synthetic datasets
            print("\n" + "=" * 80)
            print("STATISTICAL COMPARISON OF SYNTHETIC DATASETS")
            print("=" * 80)

            # Basic comparison
            print(f"Normal synthetic data: {len(normal_synthetic)} samples")
            print(f"Federated synthetic data: {len(federated_synthetic)} samples")

            # Column comparison
            normal_columns = set(normal_synthetic.columns)
            federated_columns = set(federated_synthetic.columns)

            print(f"\nColumn analysis:")
            print(f"  Normal synthetic columns: {sorted(normal_columns)}")
            print(f"  Federated synthetic columns: {sorted(federated_columns)}")

            # Check if columns match
            columns_match = normal_columns == federated_columns
            print(f"  Column match: {'✓ YES' if columns_match else '❌ NO'}")

            if not columns_match:
                print(f"  Missing in federated: {normal_columns - federated_columns}")
                print(f"  Missing in normal: {federated_columns - normal_columns}")
                pytest.fail(
                    f"Column mismatch: missing in federated={normal_columns - federated_columns}, "
                    f"extra={federated_columns - normal_columns}"
                )

            all_similar = True

            # Categorical column comparison (gender)
            print(f"\nCategorical comparison:")
            for col in ["gender"]:
                if col in normal_synthetic.columns:
                    try:
                        # Get value distributions
                        normal_dist = normal_synthetic[col].value_counts(normalize=True)
                        fed_dist = federated_synthetic[col].value_counts(normalize=True)

                        # Calculate total variation distance
                        all_values_set = set(normal_dist.index) | set(fed_dist.index)
                        tv_distance = 0.5 * sum(
                            abs(normal_dist.get(val, 0) - fed_dist.get(val, 0)) for val in all_values_set
                        )

                        similar = tv_distance < TestConfig.QUALITY_TOLERANCE
                        print(f"  {col}:")
                        print(f"    Normal distribution: {normal_dist.to_dict()}")
                        print(f"    Federated distribution: {fed_dist.to_dict()}")
                        print(f"    Total variation distance: {tv_distance:.3f} - {'✓' if similar else '❌'}")
                        print(f"    Similar: {'✓ YES' if similar else '❌ NO'}")

                        if not similar:
                            all_similar = False

                    except Exception as e:
                        print(f"    Could not compare {col}: {e}")
                        all_similar = False

            # Text column comparison (bio) - compare non-null ratios and basic stats
            print(f"\nText column comparison:")
            for col in ["bio"]:
                if col in normal_synthetic.columns:
                    try:
                        normal_non_null = normal_synthetic[col].notna().mean()
                        fed_non_null = federated_synthetic[col].notna().mean()
                        non_null_diff = abs(normal_non_null - fed_non_null)

                        normal_avg_len = normal_synthetic[col].dropna().str.len().mean()
                        fed_avg_len = federated_synthetic[col].dropna().str.len().mean()
                        if normal_avg_len > 0 and fed_avg_len > 0:
                            avg_len_rel_diff = abs(normal_avg_len - fed_avg_len) / max(normal_avg_len, fed_avg_len)
                        else:
                            avg_len_rel_diff = 0.0

                        similar = non_null_diff < TestConfig.QUALITY_TOLERANCE
                        print(f"  {col}:")
                        print(f"    Normal non-null ratio: {normal_non_null:.3f}")
                        print(f"    Federated non-null ratio: {fed_non_null:.3f}")
                        print(f"    Non-null diff: {non_null_diff:.3f} - {'✓' if similar else '❌'}")
                        print(f"    Normal avg length: {normal_avg_len:.1f}")
                        print(f"    Federated avg length: {fed_avg_len:.1f}")
                        print(f"    Avg length rel diff: {avg_len_rel_diff:.3f}")
                        print(f"    Similar: {'✓ YES' if similar else '❌ NO'}")

                        if not similar:
                            all_similar = False

                    except Exception as e:
                        print(f"    Could not compare {col}: {e}")
                        all_similar = False

            print(f"\nOverall quality assessment:")
            print(f"  Overall similarity: {'✓ YES' if all_similar else '❌ NO'}")
            assert all_similar, "Language model normal and federated synthetic data are not statistically similar"
            return

    except Exception as e:
        print(f"Error during data generation quality test: {e}")
        raise


def test_enhanced_federated_state_pattern():
    """Test the enhanced federated state pattern with all robustness improvements for language models.

    This test specifically validates the new federated state implementation:
    - Tests comprehensive federated state objects (model weights, optimiser state, LR scheduler state)
    - Validates None guards for missing or None state components
    - Tests proper state continuation across multiple iterations
    - Demonstrates the complete federated learning workflow with state passing
    """
    print("\n" + "=" * 80)
    print("LANGUAGE MODEL ENHANCED FEDERATED STATE PATTERN TEST")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Created test data with {len(data)} samples")

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
                "optimizer_state": None,  # Missing optimiser state
                "lr_scheduler_state": None,  # Missing LR scheduler state
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
                "optimizer_state": None,
                "lr_scheduler_state": None,
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
            expected_keys = {"model_weights", "training_metrics", "optimizer_state", "lr_scheduler_state"}

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


def main():
    """Run all comparison tests for language models."""
    print("LANGUAGE MODEL FEDERATED TRAINING CONVERGENCE COMPARISON TEST SUITE")
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
    print(f"Matplotlib available: {HAS_MATPLOTLIB}")
    print(f"Model: {TestConfig.MODEL_SIZE}")

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

    # Test 1: Enhanced federated state pattern
    results.append(_run_test("Enhanced Federated State Pattern", test_enhanced_federated_state_pattern))

    # Test 2: Federated weights loading
    results.append(_run_test("Federated Weights Loading", test_federated_weights_loading))

    # Test 3: Training approach comparison
    results.append(_run_test("Training Approach Comparison", test_training_approach_comparison))

    # Test 4: Epoch-by-epoch analysis
    results.append(_run_test("Epoch-by-Epoch Analysis", test_epoch_by_epoch_comparison))

    # Test 5: Data generation quality comparison
    results.append(_run_test("Data Generation Quality", test_data_generation_quality))

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
        print("Language model federated and normal training approaches produce similar results.")
    else:
        print("FAILURE: SOME TESTS FAILED!")
        print("There may be differences between federated and normal training for language models.")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
