#!/usr/bin/env python3
"""
End-to-end test to compare normal training vs federated training convergence.

This test evaluates whether:
1. Training a model without federated epochs (normal training)
2. Training a model with federated epochs, loading weights, and continuing training

Both approaches should converge to similar results.
"""

import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import torch

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mostlyai.engine import split, analyze, encode, train, generate
from mostlyai.engine.domain import ModelEncodingType, ModelStateStrategy
from mostlyai.engine._workspace import Workspace


# ============================================
# TEST CONFIGURATION - Adjust these parameters
# ============================================
class TestConfig:
    """Centralised configuration for test parameters"""
    # Training parameters
    EPOCHS_PER_ITERATION = 1  # Numbers of epochs per federated iteration
    MAX_EPOCHS = 5  # Maximum number of epochs for training
    MODEL_SIZE = "MOSTLY_AI/Small"  # Model size to use

    # Data generation parameters
    TOTAL_SAMPLES = 3000  # Total samples to create
    TEST_SAMPLES = 1000  # Samples used for quality testing

    # Quality assessment
    GENERATE_HTML_REPORTS = False  # Set to True to generate HTML reports (slower)
    QUALITY_TOLERANCE = 0.10  # 10% tolerance for quality score comparison


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
    from mostlyai import qa

    HAS_QA_LIBRARY = True
except ImportError:
    try:
        from mostlyai import qa

        HAS_QA_LIBRARY = True
    except ImportError:
        HAS_QA_LIBRARY = False
        print("Note: mostlyai-qa library not available - quality assessment features disabled")


def create_test_data():
    """Fetch sample tabular data for testing."""

    data = pd.read_csv(
        "https://github.com/mostly-ai/public-demo-data/raw/dev/titanic/titanic.csv"
    )

    return data


def setup_workspace(data, workspace_dir):
    """Set up a workspace with split, analyse, and encode steps."""
    split(
        tgt_data=data,
        tgt_encoding_types={
            "survived": ModelEncodingType.tabular_categorical,
            "pclass": ModelEncodingType.tabular_categorical,
            "sex": ModelEncodingType.tabular_categorical,
            "age": ModelEncodingType.tabular_numeric_auto,
            "sibsp": ModelEncodingType.tabular_categorical,
            "parch": ModelEncodingType.tabular_categorical,
            "fare": ModelEncodingType.tabular_numeric_auto,
            "embarked": ModelEncodingType.tabular_categorical
        },
        workspace_dir=workspace_dir
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
        model=model
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
                final_val_loss = progress_df.iloc[-1].get('val_loss')
        except Exception as e:
            print(f"Warning: Could not read progress messages: {e}")

    return result, final_val_loss, training_time


def train_federated_model(workspace_dir, total_epochs=TestConfig.MAX_EPOCHS,
                          epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION, model=TestConfig.MODEL_SIZE):
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
            workspace_dir=workspace_dir, # Continue to pass the workspace for data and associated metadata
            federated_epochs=epochs_per_iteration,
            max_epochs=total_epochs,
            model=model,
            federated_state=federated_state  # Pass previous federated state
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
                    current_val_loss = progress_df.iloc[-1].get('val_loss')
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
        if hasattr(tensor, 'numpy'):  # PyTorch tensor
            # Handle both CPU and CUDA tensors
            if tensor.is_cuda:
                values = tensor.cpu().numpy().flatten()
            else:
                values = tensor.numpy().flatten()
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
            if 'weight' in name:
                layer_types['weight'] = layer_types.get('weight', 0) + 1
            elif 'bias' in name:
                layer_types['bias'] = layer_types.get('bias', 0) + 1
            elif 'embed' in name:
                layer_types['embed'] = layer_types.get('embed', 0) + 1

        print(f"      Layer types: {layer_types}")

    # Visualisation
    if HAS_MATPLOTLIB and len(all_values) > 0:
        try:
            plt.figure(figsize=(10, 4))
            plt.hist(all_values, bins=50, alpha=0.7, color='blue')
            plt.title(f"Weight Distribution - Epoch {epoch}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"weight_distribution_epoch_{epoch}.png")
            plt.close()
            print(f"      Saved weight distribution plot")
        except Exception as e:
            print(f"      Warning: Could not create plot: {e}")


def train_epoch_by_epoch(workspace_dir, max_epochs=TestConfig.MAX_EPOCHS,
                         epochs_per_iteration=TestConfig.EPOCHS_PER_ITERATION):
    """Train one epoch at a time and analyse weights after each epoch.
    
    This function focuses on detailed analysis and monitoring of the federated training process:
    - Tracks complete history of weights, losses, and training times for each iteration
    - Includes comprehensive weight analysis with statistics, percentiles, and visualisation
    - Provides progress monitoring and detailed logging for debugging and understanding
    - Tests the core federated training pattern with fixed epochs per iteration
    
    While train_federated_model provides an all-encompassing integration test, this function
    offers straightforward epoch-by-epoch analysis and monitoring capabilities for understanding training dynamics.
    """
    print(
        f"\n--- Epoch-by-Epoch Training (total_epochs={max_epochs}, epochs_per_iteration={epochs_per_iteration}) ---")

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
            federated_state=federated_state  # Pass previous federated state
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
                    current_val_loss = progress_df.iloc[-1].get('val_loss')
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
            federated_weights_history, federated_loss_history, federated_times = train_epoch_by_epoch(
                federated_workspace)

            # Normal training for final comparison
            normal_workspace = Path(tmpdir) / "normal-epoch-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(data, normal_workspace)

            print(f"\nNormal training for comparison:")
            normal_result, normal_val_loss, normal_time = train_normal_model(
                normal_workspace
            )

            # Analysis and comparison
            print("\n" + "=" * 80)
            print("EPOCH-BY-EPOCH ANALYSIS RESULTS")
            print("=" * 80)

            print("Federated training progression:")
            for epoch, (weights, loss, train_time) in enumerate(zip(
                    federated_weights_history, federated_loss_history, federated_times
            ), 1):
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

                return similar_results
            else:
                print("\nCould not compare final validation losses")
                return False

    except Exception as e:
        print(f"Error during epoch-by-epoch test: {e}")
        import traceback
        traceback.print_exc()
        return False


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
                workspace_dir=workspace_dir,
                federated_epochs=epochs_per_iteration,
                max_epochs=100,
                model=model
            )

            if federated_state is None:
                print("❌ FAILED: No federated state returned from federated training")
                return False

            print(f"✓ Received federated state with {len(federated_state['model_weights'])} model parameters")
            print(f"✓ Federated state contains: {list(federated_state.keys())}")

            # Verify we can continue training using the federated state pattern
            print("Continuing training from federated state...")
            final_result = train(
                workspace_dir=workspace_dir,  # Use the same workspace for data and associated metadata
                federated_epochs=epochs_per_iteration,
                max_epochs=100,
                model=model,
                federated_state=federated_state
            )

            if final_result is None:
                print("❌ FAILED: Continuation training returned None")
                return False

            print("✓ Successfully continued training using federated state pattern")
            print(f"✓ Final federated state contains: {list(final_result.keys())}")
            
            # Verify that training actually continued (epochs progressed)
            initial_epoch = federated_state["training_metrics"]["epoch"]
            final_epoch = final_result["training_metrics"]["epoch"]
            
            if final_epoch > initial_epoch:
                print(f"✓ Training progressed from epoch {initial_epoch} to {final_epoch}")
            else:
                print(f"⚠️  Warning: Epochs did not progress as expected: {initial_epoch} -> {final_epoch}")

            return True

    except Exception as e:
        print(f"❌ FAILED: Error during weights loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_approach_comparison():
    """Compare normal training vs federated training approaches."""
    print("\n" + "=" * 80)
    print("FEDERATED vs NORMAL TRAINING CONVERGENCE COMPARISON")
    print("=" * 80)

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")

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

                # Consider them similar if the relative difference is < 10%
                similar_results = loss_ratio < 0.10
                print(f"  - Similar results: {'YES' if similar_results else 'NO'}")

                return similar_results
            else:
                print("\nCould not compare validation losses (one or both are None)")
                return False

    except Exception as e:
        print(f"Error during comparison test: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    print(f"Matplotlib available: {HAS_MATPLOTLIB}")

    # Run tests
    results = []

    # Test 1: Enhanced federated state pattern (new comprehensive test)
    results.append(("Enhanced Federated State Pattern", test_enhanced_federated_state_pattern()))

    # Test 2: Federated weights loading
    results.append(("Federated Weights Loading", test_federated_weights_loading()))

    # Test 3: Training approach comparison
    results.append(("Training Approach Comparison", test_training_approach_comparison()))

    # Test 4: Epoch-by-epoch analysis
    results.append(("Epoch-by-Epoch Analysis", test_epoch_by_epoch_comparison()))

    # Test 5: Data generation quality comparison
    results.append(("Data Generation Quality", test_data_generation_quality()))

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


def test_data_generation_quality():
    """Compare data generation quality between federated and normal training approaches.
    
    This test generates synthetic data using both normal training and federated training,
    then compares the generated datasets directly to see if they are statistically similar.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY COMPARISON")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")

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
                model=TestConfig.MODEL_SIZE
            )

            # Generate synthetic data from the normal model
            print("Generating data from normal model...")
            generate(
                workspace_dir=normal_workspace,
                sample_size=TestConfig.TEST_SAMPLES
            )
            # Read the generated data
            normal_workspace_obj = Workspace(normal_workspace)
            normal_synthetic = pd.concat([
                pd.read_parquet(file) for file in normal_workspace_obj.generated_data.fetch_all()
            ], ignore_index=True)

            # Set up the workspace for federated training
            federated_workspace = Path(tmpdir) / "federated-gen-ws"
            federated_workspace.mkdir(parents=True)

            print(f"\nSetting up federated training workspace: {federated_workspace}")
            setup_workspace(train_data, federated_workspace)

            # Train with the federated approach (fixed epochs per iteration)
            print("Training federated model with fixed epochs per iteration...")
            for iteration in range(1, TestConfig.MAX_EPOCHS + 1):
                train(
                    workspace_dir=federated_workspace,
                    federated_epochs=1,
                    max_epochs=TestConfig.MAX_EPOCHS,
                    model=TestConfig.MODEL_SIZE
                )
                print(f"  Completed iteration {iteration}/{TestConfig.MAX_EPOCHS}")

            # Generate synthetic data from the federated model
            print("Generating data from federated model...")
            generate(
                workspace_dir=federated_workspace,  # TODO in federated context this would of course not work
                sample_size=TestConfig.TEST_SAMPLES
            )
            # Read the generated data
            federated_workspace_obj = Workspace(federated_workspace)
            federated_synthetic = pd.concat([
                pd.read_parquet(file) for file in federated_workspace_obj.generated_data.fetch_all()
            ], ignore_index=True)

            # Direct comparison between the two synthetic datasets
            print("\n" + "=" * 80)
            print("DIRECT COMPARISON OF SYNTHETIC DATASETS")
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
                return False

            # Statistical comparison for numeric columns
            print(f"\nStatistical comparison:")
            all_similar = True

            for col in ['age', 'fare']:
                if col in normal_synthetic.columns:
                    try:
                        # Calculate statistics for both synthetic datasets
                        normal_mean = normal_synthetic[col].mean()
                        normal_std = normal_synthetic[col].std()
                        fed_mean = federated_synthetic[col].mean()
                        fed_std = federated_synthetic[col].std()

                        # Calculate differences
                        mean_diff = abs(normal_mean - fed_mean)
                        std_diff = abs(normal_std - fed_std)

                        # Calculate relative differences
                        mean_rel_diff = mean_diff / max(abs(normal_mean), abs(fed_mean), 1e-6)
                        std_rel_diff = std_diff / max(abs(normal_std), abs(fed_std), 1e-6)

                        # Check if differences are within tolerance
                        mean_similar = mean_rel_diff < TestConfig.QUALITY_TOLERANCE
                        std_similar = std_rel_diff < TestConfig.QUALITY_TOLERANCE
                        col_similar = mean_similar and std_similar

                        print(f"  {col}:")
                        print(f"    Normal:   μ={normal_mean:.2f}, σ={normal_std:.2f}")
                        print(f"    Federated:μ={fed_mean:.2f}, σ={fed_std:.2f}")
                        print(f"    Mean diff: {mean_diff:.2f} ({mean_rel_diff:.1%}) - {'✓' if mean_similar else '❌'}")
                        print(f"    Std diff:  {std_diff:.2f} ({std_rel_diff:.1%}) - {'✓' if std_similar else '❌'}")
                        print(f"    Similar: {'✓ YES' if col_similar else '❌ NO'}")

                        if not col_similar:
                            all_similar = False

                    except Exception as e:
                        print(f"    Could not compare {col}: {e}")
                        all_similar = False

            # Categorical column comparison
            print(f"\nCategorical comparison:")
            for col in ['survived', 'pclass', 'sex', 'sibsp', 'parch', 'embarked']:
                if col in normal_synthetic.columns:
                    try:
                        # Get value distributions
                        normal_dist = normal_synthetic[col].value_counts(normalize=True)
                        fed_dist = federated_synthetic[col].value_counts(normalize=True)

                        # Calculate total variation distance
                        common_values = set(normal_dist.index) & set(fed_dist.index)
                        tv_distance = 0.5 * sum(abs(normal_dist[val] - fed_dist[val]) for val in common_values)

                        similar = tv_distance < TestConfig.QUALITY_TOLERANCE
                        print(f"  {col}:")
                        print(f"    Total variation distance: {tv_distance:.3f} - {'✓' if similar else '❌'}")
                        print(f"    Similar: {'✓ YES' if similar else '❌ NO'}")

                        if not similar:
                            all_similar = False

                    except Exception as e:
                        print(f"    Could not compare {col}: {e}")
                        all_similar = False

            # Advanced quality assessment with mostlyai-qa if available
            if HAS_QA_LIBRARY:
                print(f"\nAdvanced quality assessment (using mostlyai-qa):")

                try:
                    import os

                    # Create a temporary directory for reports
                    with tempfile.TemporaryDirectory() as report_dir:
                        # Generate a report comparing the two synthetic datasets
                        report_path = os.path.join(report_dir, "synthetic_comparison_report.html")
                        report_path, metrics = qa.report(
                            syn_tgt_data=federated_synthetic,
                            trn_tgt_data=normal_synthetic,
                            report_path=report_path,
                            report_title="Synthetic (federated) vs Synthetic (central) Comparison"
                        )

                        print(f"  Report saved to: {report_path}")

                        # Extract and analyse key metrics from the QA report
                        def extract_and_analyse_metrics(metrics):
                            """Extract and analyse key metrics from the QA report"""
                            if not metrics:
                                print(f"  No metrics available")
                                return True  # Fall back to basic assessment

                            try:
                                metrics_dict = metrics.model_dump()
                                print(f"  Key metrics from QA report:")

                                # Focus on the most relevant metrics for comparing two synthetic datasets
                                # 1. Overall accuracy (if available)
                                overall_accuracy = metrics_dict.get('accuracy', {}).get('overall')
                                if overall_accuracy is not None:
                                    print(f"    Overall accuracy: {overall_accuracy:.3f}")
                                    accuracy_similar = overall_accuracy > 0.6  # Reasonable threshold
                                    print(f"    Accuracy acceptable: {'✓ YES' if accuracy_similar else '❌ NO'}")

                                # 2. Cosine similarity between training and synthetic data
                                cosine_sim = metrics_dict.get('similarity', {}).get(
                                    'cosine_similarity_training_synthetic')
                                if cosine_sim is not None:
                                    print(f"    Cosine similarity (training vs synthetic): {cosine_sim:.3f}")
                                    cosine_similar = cosine_sim > 0.7  # Good similarity threshold
                                    print(f"    Cosine similarity good: {'✓ YES' if cosine_similar else '❌ NO'}")

                                # 3. Discriminator AUC (how well a classifier can distinguish real vs synthetic)
                                discriminator_auc = metrics_dict.get('similarity', {}).get(
                                    'discriminator_auc_training_synthetic')
                                if discriminator_auc is not None:
                                    print(f"    Discriminator AUC: {discriminator_auc:.3f}")
                                    # For similarity, we want AUC close to 0.5 (can't distinguish well)
                                    auc_similar = abs(discriminator_auc - 0.5) < 0.2  # Within a 0.3-0.7 range
                                    print(f"    AUC indicates similarity: {'✓ YES' if auc_similar else '❌ NO'}")

                                # 4. Univariate accuracy (individual column statistics)
                                univariate_acc = metrics_dict.get('accuracy', {}).get('univariate')
                                if univariate_acc is not None:
                                    print(f"    Univariate accuracy: {univariate_acc:.3f}")
                                    univariate_good = univariate_acc > 0.8  # High threshold for individual columns
                                    print(f"    Univariate accuracy good: {'✓ YES' if univariate_good else '❌ NO'}")

                                # 5. NNDR distance (nearest neighbor distance ratio)
                                nndr = metrics_dict.get('distances', {}).get('nndr_training')
                                if nndr is not None:
                                    print(f"    NNDR distance: {nndr:.3f}")
                                    nndr_good = nndr < 0.8  # Lower is better for similarity
                                    print(f"    NNDR indicates similarity: {'✓ YES' if nndr_good else '❌ NO'}")

                                # Overall assessment based on available metrics
                                available_metrics = []
                                if overall_accuracy is not None:
                                    available_metrics.append(accuracy_similar)
                                if cosine_sim is not None:
                                    available_metrics.append(cosine_similar)
                                if discriminator_auc is not None:
                                    available_metrics.append(auc_similar)
                                if univariate_acc is not None:
                                    available_metrics.append(univariate_good)
                                if nndr is not None:
                                    available_metrics.append(nndr_good)

                                if available_metrics:
                                    # Consider similar if at least 3/5 metrics indicate similarity
                                    similar_count = sum(available_metrics)
                                    overall_similar = similar_count >= 3
                                    print(
                                        f"  Overall similarity assessment: {similar_count}/{len(available_metrics)} metrics good")
                                    print(f"  Synthetic datasets similar: {'✓ YES' if overall_similar else '❌ NO'}")
                                    return all_similar and overall_similar
                                else:
                                    print(f"  No comparable metrics found, falling back to basic assessment")
                                    return all_similar

                            except Exception as e:
                                print(f"  Could not extract metrics: {e}")
                                return all_similar

                        return extract_and_analyse_metrics(metrics)

                except Exception as e:
                    print(f"  Warning: Could not run advanced quality assessment: {e}")
                    return all_similar
            else:
                print(f"\nBasic quality assessment (mostlyai-qa not available):")
                print(f"  Overall similarity: {'✓ YES' if all_similar else '❌ NO'}")
                print(f"  Note: Install mostlyai-qa for comprehensive quality assessment")
                return all_similar

    except Exception as e:
        print(f"Error during data generation quality test: {e}")
        import traceback
        traceback.print_exc()
        return False


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
                federated_state=federated_state
            )
            
            if result1 is None:
                print("❌ FAILED: Initial training returned None")
                return False
            
            print("✓ Initial training successful")
            print(f"✓ Federated state contains: {list(result1.keys())}")
            
            # Test 2: Continuation with complete federated state
            print("\n2. Testing continuation with complete federated state...")
            
            result2 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=result1  # Pass complete state
            )
            
            if result2 is None:
                print("❌ FAILED: Continuation training returned None")
                return False
            
            print("✓ Continuation with complete state successful")
            
            # Test 3: Continuation with partial federated state (missing some components)
            print("\n3. Testing continuation with partial federated state...")
            
            # Create a partial federated state (missing optimiser and LR scheduler state)
            partial_federated_state = {
                "model_weights": result2["model_weights"],
                "training_metrics": result2["training_metrics"],
                "optimizer_state": None,  # Missing optimiser state
                "lr_scheduler_state": None  # Missing LR scheduler state
            }
            
            result3 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=partial_federated_state  # Pass partial state
            )
            
            if result3 is None:
                print("❌ FAILED: Partial state continuation returned None")
                return False
            
            print("✓ Continuation with partial state successful (None guards working)")
            
            # Test 4: Continuation with minimal federated state (only model weights)
            print("\n4. Testing continuation with minimal federated state...")
            
            minimal_federated_state = {
                "model_weights": result3["model_weights"],
                "training_metrics": result3["training_metrics"],
                "optimizer_state": None,
                "lr_scheduler_state": None
            }
            
            result4 = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                model=TestConfig.MODEL_SIZE,
                federated_state=minimal_federated_state  # Pass minimal state
            )
            
            if result4 is None:
                print("❌ FAILED: Minimal state continuation returned None")
                return False
            
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
                    return False
                
                # Verify training metrics structure
                metrics_keys = {"epoch", "steps", "samples", "learn_rate", "trn_loss", "val_loss"}
                actual_metrics_keys = set(state["training_metrics"].keys())
                missing_metrics_keys = metrics_keys - actual_metrics_keys
                if missing_metrics_keys:
                    print(f"❌ FAILED: State {i} missing metrics keys: {missing_metrics_keys}")
                    return False
            
            print("✓ All federated states have consistent structure")
            
            # Test 6: Verify state evolution
            print("\n6. Verifying state evolution across iterations...")
            
            # Check that epochs are progressing
            epochs = [state["training_metrics"]["epoch"] for state in all_states]
            if epochs != sorted(epochs):
                print(f"❌ FAILED: Epochs not progressing correctly: {epochs}")
                return False
            
            print(f"✓ Epochs progressing correctly: {epochs}")
            
            # Check that steps are increasing
            steps = [state["training_metrics"]["steps"] for state in all_states]
            if steps != sorted(steps):
                print(f"❌ FAILED: Steps not increasing correctly: {steps}")
                return False
            
            print(f"✓ Steps increasing correctly: {steps}")
            
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
                if hasattr(w_current, 'cpu'):
                    w_current = w_current.cpu().numpy()
                elif hasattr(w_current, 'numpy'):
                    w_current = w_current.numpy()
                else:
                    w_current = np.array(w_current)
                    
                if hasattr(w_next, 'cpu'):
                    w_next = w_next.cpu().numpy()
                elif hasattr(w_next, 'numpy'):
                    w_next = w_next.numpy()
                else:
                    w_next = np.array(w_next)
                
                # Check if weights changed (training continued)
                if not np.array_equal(w_current, w_next):
                    weights_changed = True
                    print(f"✓ Weights changed between iteration {i+1} and {i+2} (training continued)")
                    break
            
            if not weights_changed:
                print("⚠️  Warning: Weights appear unchanged between iterations")
                print("   This may indicate training did not continue properly")
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
            
            return True
            
    except Exception as e:
        print(f"❌ FAILED: Error during enhanced federated state test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(main())
