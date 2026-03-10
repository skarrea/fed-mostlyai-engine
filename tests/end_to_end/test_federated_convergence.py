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
    TRAIN_EPOCHS = 5  # Number of epochs for training
    MODEL_SIZE = "MOSTLY_AI/Small"  # Model size to use

    # Data generation parameters
    TOTAL_SAMPLES = 300  # Total samples to create
    TRAIN_SAMPLES = 200  # Samples used for training
    TEST_SAMPLES = 100  # Samples used for quality testing

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
    """Create sample tabular data for testing."""
    np.random.seed(42)
    n_samples = 300  # Small dataset for faster testing

    data = pd.DataFrame({
        'id': [f"id_{i}" for i in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples).astype(int),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'score': np.random.uniform(0, 1, n_samples),
        'active': np.random.choice([True, False], n_samples)
    })

    return data


def setup_workspace(data, workspace_dir):
    """Set up a workspace with split, analyse, and encode steps."""
    split(
        tgt_data=data,
        tgt_primary_key="id",
        tgt_encoding_types={
            "age": ModelEncodingType.tabular_numeric_auto,
            "income": ModelEncodingType.tabular_numeric_auto,
            "category": ModelEncodingType.tabular_categorical,
            "score": ModelEncodingType.tabular_numeric_auto,
            "active": ModelEncodingType.tabular_categorical
        },
        workspace_dir=workspace_dir
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)


def train_normal_model(workspace_dir, max_epochs=5):
    """Train a model using the normal training approach."""
    print(f"\n--- Normal Training (max_epochs={max_epochs}) ---")
    start_time = time.time()

    result = train(
        workspace_dir=workspace_dir,
        max_epochs=max_epochs,
        max_training_time=10,  # 10 minutes max
        model="MOSTLY_AI/Large"
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


def train_federated_model(workspace_dir, federated_epochs=2, continue_epochs=3):
    """Train a model using the federated approach with continuation."""
    print(f"\n--- Federated Training (federated_epochs={federated_epochs}, continue_epochs={continue_epochs}) ---")

    # Phase 1: Federated training
    print(f"Phase 1: Federated training for {federated_epochs} epochs")
    start_time = time.time()

    federated_weights = train(
        workspace_dir=workspace_dir,
        federated_epochs=federated_epochs,
        max_epochs=100,  # High enough to not interfere
        max_training_time=10,
        model="MOSTLY_AI/Large"
    )

    federated_time = time.time() - start_time
    print(f"Federated training completed in {federated_time:.2f} seconds")
    print(f"Federated weights returned: {federated_weights is not None}")
    if federated_weights:
        print(f"Federated weights keys: {list(federated_weights.keys())[:3]}...")

    # Get intermediate validation loss
    workspace = Workspace(workspace_dir)
    progress_messages_path = workspace.model_progress_messages_path
    intermediate_val_loss = None
    if progress_messages_path.exists():
        try:
            progress_df = pd.read_csv(progress_messages_path)
            if not progress_df.empty:
                intermediate_val_loss = progress_df.iloc[-1].get('val_loss')
        except Exception as e:
            print(f"Warning: Could not read intermediate progress messages: {e}")

    # Phase 2: Continue training from federated weights
    print(f"Phase 2: Continuing training for {continue_epochs} more epochs")
    start_continue_time = time.time()

    # Load the federated weights by using resume strategy
    final_result = train(
        workspace_dir=workspace_dir,
        max_epochs=federated_epochs + continue_epochs,
        max_training_time=10,
        model="MOSTLY_AI/Large",
        model_state_strategy=ModelStateStrategy.resume  # This should load the saved weights
    )

    continue_time = time.time() - start_continue_time
    total_federated_time = federated_time + continue_time
    print(f"Continuation training completed in {continue_time:.2f} seconds")
    print(f"Total federated approach time: {total_federated_time:.2f} seconds")
    print(f"Final result: {final_result}")

    # Get final validation loss
    progress_messages_path = workspace.model_progress_messages_path
    final_val_loss = None
    if progress_messages_path.exists():
        try:
            progress_df = pd.read_csv(progress_messages_path)
            if not progress_df.empty:
                final_val_loss = progress_df.iloc[-1].get('val_loss')
        except Exception as e:
            print(f"Warning: Could not read final progress messages: {e}")

    return federated_weights, intermediate_val_loss, final_val_loss, total_federated_time


def analyse_weights(weights, epoch, detailed=False):
    """Analyse and print information about model weights."""
    if not weights:
        print(f"    No weights available for epoch {epoch}")
        return

    print(f"    Weight analysis for epoch {epoch}:")

    # Collect all weight values
    all_values = []
    for name, tensor in weights.items():
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
        for name in weights.keys():
            if 'weight' in name:
                layer_types['weight'] = layer_types.get('weight', 0) + 1
            elif 'bias' in name:
                layer_types['bias'] = layer_types.get('bias', 0) + 1
            elif 'embed' in name:
                layer_types['embed'] = layer_types.get('embed', 0) + 1

        print(f"      Layer types: {layer_types}")

    # Visualisation
    if HAS_MATPLOTLIB and all_values:
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


def train_epoch_by_epoch(workspace_dir, total_epochs=15):
    """Train one epoch at a time and analyse weights after each epoch."""
    print(f"\n--- Epoch-by-Epoch Training (total_epochs={total_epochs}) ---")

    weights_history = []
    loss_history = []
    training_times = []

    for epoch in range(1, total_epochs + 1):
        print(f"\n  Epoch {epoch}/{total_epochs}")
        start_time = time.time()

        # Train for exactly 'epoch' epochs (cumulative)
        weights = train(
            workspace_dir=workspace_dir,
            federated_epochs=epoch,  # Train up to this epoch
            max_epochs=total_epochs,
            max_training_time=30,  # Increased for longer training
            model="MOSTLY_AI/Large"
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
        weights_history.append(weights)
        loss_history.append(current_val_loss)
        training_times.append(training_time)

        # Print analysis
        print(f"    Training time: {training_time:.2f}s")
        print(f"    Weights returned: {weights is not None}")
        print(f"    Validation loss: {current_val_loss}")

        # Progress indicator for long training
        if total_epochs > 10:
            progress_percent = (epoch / total_epochs) * 100
            print(f"    Progress: {progress_percent:.0f}% complete")

        if weights:
            analyse_weights(weights, epoch)

    return weights_history, loss_history, training_times


def test_epoch_by_epoch_comparison():
    """Compare epoch-by-epoch training between federated approaches."""
    print("\n" + "=" * 80)
    print("EPOCH-BY-EPOCH TRAINING ANALYSIS")
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
                federated_workspace, total_epochs=15
            )

            # Normal training for final comparison
            normal_workspace = Path(tmpdir) / "normal-epoch-ws"
            normal_workspace.mkdir(parents=True)

            print(f"\nSetting up normal training workspace: {normal_workspace}")
            setup_workspace(data, normal_workspace)

            print(f"\nNormal training for comparison:")
            normal_result, normal_val_loss, normal_time = train_normal_model(
                normal_workspace, max_epochs=15
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


def test_federated_weights_loading():
    """Test that federated weights can be properly loaded and training can continue."""
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
            weights = train(
                workspace_dir=workspace_dir,
                federated_epochs=1,
                max_epochs=100,
                max_training_time=5,
                model="MOSTLY_AI/Large"
            )

            if weights is None:
                print("❌ FAILED: No weights returned from federated training")
                return False

            print(f"✓ Received {len(weights)} model parameters")

            # Verify we can continue training
            print("Continuing training from saved weights...")
            final_result = train(
                workspace_dir=workspace_dir,
                max_epochs=15,
                max_training_time=30,
                model="MOSTLY_AI/Large",
                model_state_strategy=ModelStateStrategy.resume
            )

            print("✓ Successfully continued training from federated weights")
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

    # Test 1: Federated weights loading
    results.append(("Federated Weights Loading", test_federated_weights_loading()))

    # Test 2: Training approach comparison
    results.append(("Training Approach Comparison", test_training_approach_comparison()))

    # Test 3: Epoch-by-epoch analysis
    results.append(("Epoch-by-Epoch Analysis", test_epoch_by_epoch_comparison()))

    # Test 4: Data generation quality comparison
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
    print(f"Configuration: {TestConfig.TRAIN_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    try:
        data = create_test_data()
        print(f"Created test data with {len(data)} samples")

        # Use the same training data for both approaches
        train_data = data.iloc[:TestConfig.TRAIN_SAMPLES].copy()

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
                max_epochs=TestConfig.TRAIN_EPOCHS,
                max_training_time=15,
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

            # Train with the federated approach (epoch by epoch)
            print("Training federated model epoch-by-epoch...")
            for epoch in range(1, TestConfig.TRAIN_EPOCHS + 1):
                train(
                    workspace_dir=federated_workspace,
                    federated_epochs=epoch,
                    max_epochs=TestConfig.TRAIN_EPOCHS,
                    max_training_time=15,
                    model=TestConfig.MODEL_SIZE
                )
                print(f"  Completed epoch {epoch}/{TestConfig.TRAIN_EPOCHS}")

            # Generate synthetic data from the federated model
            print("Generating data from federated model...")
            generate(
                workspace_dir=federated_workspace,
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

            for col in ['age', 'income', 'score']:
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
            for col in ['category', 'active']:
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
                                    auc_similar = abs(discriminator_auc - 0.5) < 0.2  # Within 0.3-0.7 range
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


if __name__ == "__main__":
    sys.exit(main())
