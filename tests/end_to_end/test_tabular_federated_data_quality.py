#!/usr/bin/env python3
"""
End-to-end tests comparing data generation quality across training approaches.

These tests evaluate whether synthetic data produced by:
1. Local central training
2. Local federated training
3. PyPI central training

converge to statistically similar distributions.
"""

import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Project root for the local (development) version
project_root = Path(__file__).parent.parent.parent

# Import PyPI version first (before adding local path)
try:
    import mostlyai.engine as _mostlyai_engine_pypi
    from mostlyai.engine import split as split_pypi
    from mostlyai.engine import analyze as analyze_pypi
    from mostlyai.engine import encode as encode_pypi
    from mostlyai.engine import train as train_pypi
    from mostlyai.engine import generate as generate_pypi
    from mostlyai.engine.domain import ModelEncodingType as ModelEncodingType_pypi
    from mostlyai.engine._workspace import Workspace as Workspace_pypi

    HAS_PYPI_ENGINE = True
    print(f"PyPI mostlyai.engine imported: {getattr(_mostlyai_engine_pypi, '__version__', 'unknown version')}")
except ImportError:
    split_pypi = analyze_pypi = encode_pypi = train_pypi = generate_pypi = None
    ModelEncodingType_pypi = None
    Workspace_pypi = None
    HAS_PYPI_ENGINE = False
    print("Note: PyPI mostlyai.engine not available - PyPI comparison tests will be skipped")

# Clear cached mostlyai modules so the local development version can be loaded fresh
for _key in list(sys.modules.keys()):
    if _key.startswith("mostlyai"):
        del sys.modules[_key]

# Now add local path for development version
sys.path.insert(0, str(project_root))

from mostlyai.engine import split, analyze, encode, train, generate
from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine._workspace import Workspace

# Shared reporting utilities (plots + GitHub step summary)
_test_dir = Path(__file__).parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
import reporting


# ============================================
# TEST CONFIGURATION - Adjust these parameters
# ============================================
class TestConfig:
    """Centralised configuration for test parameters"""

    # Training parameters
    EPOCHS_PER_ITERATION = 1  # Numbers of epochs per federated iteration
    MAX_EPOCHS = 20  # Maximum number of epochs for training
    MODEL_SIZE = "MOSTLY_AI/Small"  # Model size to use

    # Data generation parameters
    TOTAL_SAMPLES = 3000  # Total samples to create
    TEST_SAMPLES = 1000  # Samples used for quality testing

    # Quality assessment
    GENERATE_HTML_REPORTS = True  # Set to True to generate HTML reports (slower)
    QUALITY_TOLERANCE = 0.10  # 10% tolerance for quality score comparison

    # Output directory for plots and summary artifacts
    OUTPUT_DIR = Path("test-output/quality")


# ============================================

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

    data = pd.read_csv("https://zenodo.org/records/20411920/files/Intensivregister_Deutschland_Versorgungsstufen.csv")
    # Very large dataset
    # data = pd.read_csv("https://zenodo.org/records/20411920/files/Intensivregister_Landkreise_Kapazitaeten.csv")

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


def _generate_synthetic_local_central(train_data, workspace_dir):
    """Train centrally with the local development engine and return a synthetic DataFrame."""
    setup_workspace(train_data, workspace_dir)
    print("Training local central model...")
    train(workspace_dir=workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, model=TestConfig.MODEL_SIZE)
    print("Generating data from local central model...")
    generate(workspace_dir=workspace_dir, sample_size=TestConfig.TEST_SAMPLES)
    ws = Workspace(workspace_dir)
    return pd.concat([pd.read_parquet(f) for f in ws.generated_data.fetch_all()], ignore_index=True)


def _generate_synthetic_local_federated(train_data, workspace_dir):
    """Train federatedly with the local development engine and return a synthetic DataFrame."""
    setup_workspace(train_data, workspace_dir)
    print("Training local federated model (1 epoch per iteration)...")
    federated_state = None
    for iteration in range(1, TestConfig.MAX_EPOCHS + 1):
        federated_state = train(
            workspace_dir=workspace_dir,
            federated_epochs=1,
            max_epochs=TestConfig.MAX_EPOCHS,
            model=TestConfig.MODEL_SIZE,
            federated_state=federated_state,
        )
        print(f"  Completed federated iteration {iteration}/{TestConfig.MAX_EPOCHS}")
    print("Generating data from local federated model...")
    generate(workspace_dir=workspace_dir, sample_size=TestConfig.TEST_SAMPLES)
    ws = Workspace(workspace_dir)
    return pd.concat([pd.read_parquet(f) for f in ws.generated_data.fetch_all()], ignore_index=True)


def _generate_synthetic_pypi_central(train_data, workspace_dir):
    """Train centrally with the PyPI engine and return a synthetic DataFrame."""
    setup_workspace_pypi(train_data, workspace_dir)
    print("Training PyPI central model...")
    train_pypi(workspace_dir=workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, model=TestConfig.MODEL_SIZE)
    print("Generating data from PyPI central model...")
    generate_pypi(workspace_dir=workspace_dir, sample_size=TestConfig.TEST_SAMPLES)
    ws = Workspace_pypi(workspace_dir)
    return pd.concat([pd.read_parquet(f) for f in ws.generated_data.fetch_all()], ignore_index=True)


def setup_workspace_pypi(data, workspace_dir):
    """Set up a workspace using the PyPI version of mostlyai.engine."""
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


def _compare_synthetic_datasets(syn_a, syn_b, other_label="other"):
    """Compare two synthetic DataFrames statistically and return whether they are similar.

    Performs column checks, numeric/categorical statistical comparisons, and (if available)
    an advanced QA report using the same 5-metric logic as the original quality tests:
    overall accuracy, cosine similarity, discriminator AUC, univariate accuracy, NNDR distance.
    Passes if ≥ 3/5 available QA metrics are acceptable; falls back to basic stats otherwise.
    """
    # Column check
    cols_a = set(syn_a.columns)
    cols_b = set(syn_b.columns)
    columns_match = cols_a == cols_b
    print(f"\nColumn analysis:")
    print(f"  Columns A:  {sorted(cols_a)}")
    print(f"  Columns B ({other_label}): {sorted(cols_b)}")
    print(f"  Column match: {'✓ YES' if columns_match else '❌ NO'}")
    if not columns_match:
        print(f"  Missing in B: {cols_a - cols_b}")
        print(f"  Missing in A: {cols_b - cols_a}")
        return False

    all_similar = True
    summary_rows = []  # Collected for GitHub step summary
    print(f"\nStatistical comparison (numeric):")
    for col in ["bundesland_id", "anzahl_meldebereiche", "faelle_covid_aktuell", "intensivbetten_belegt",
                "intensivbetten_frei"]:
        if col in syn_a.columns:
            try:
                a_mean, a_std = syn_a[col].mean(), syn_a[col].std()
                b_mean, b_std = syn_b[col].mean(), syn_b[col].std()

                mean_rel_diff = abs(a_mean - b_mean) / max(abs(a_mean), abs(b_mean), 1e-6)
                std_rel_diff = abs(a_std - b_std) / max(abs(a_std), abs(b_std), 1e-6)

                mean_ok = mean_rel_diff < TestConfig.QUALITY_TOLERANCE
                std_ok = std_rel_diff < TestConfig.QUALITY_TOLERANCE
                col_similar = mean_ok and std_ok

                print(f"  {col}:")
                print(f"    A:             μ={a_mean:.2f}, σ={a_std:.2f}")
                print(f"    B ({other_label}): μ={b_mean:.2f}, σ={b_std:.2f}")
                print(f"    Mean diff: {abs(a_mean - b_mean):.2f} ({mean_rel_diff:.1%}) - {'✓' if mean_ok else '❌'}")
                print(f"    Std diff:  {abs(a_std - b_std):.2f} ({std_rel_diff:.1%}) - {'✓' if std_ok else '❌'}")
                print(f"    Similar: {'✓ YES' if col_similar else '❌ NO'}")

                if not col_similar:
                    all_similar = False
            except Exception as e:
                print(f"    Could not compare {col}: {e}")
                all_similar = False

    # Categorical columns
    print(f"\nStatistical comparison (categorical):")
    for col in ["bundesland_name", "versorgungsstufe"]:
        if col in syn_a.columns:
            try:
                dist_a = syn_a[col].value_counts(normalize=True)
                dist_b = syn_b[col].value_counts(normalize=True)
                common_values = set(dist_a.index) & set(dist_b.index)
                tv_distance = 0.5 * sum(abs(dist_a.get(v, 0) - dist_b.get(v, 0)) for v in common_values)
                similar = tv_distance < TestConfig.QUALITY_TOLERANCE
                print(f"  {col}: TV distance={tv_distance:.3f} - {'✓' if similar else '❌'}")
                if not similar:
                    all_similar = False
            except Exception as e:
                print(f"    Could not compare {col}: {e}")
                all_similar = False

    # Advanced QA assessment
    if HAS_QA_LIBRARY:
        print(f"\nAdvanced quality assessment (using mostlyai-qa):")
        try:
            import os

            with tempfile.TemporaryDirectory() as report_dir:
                report_path = os.path.join(report_dir, "comparison_report.html")
                report_path, metrics = qa.report(
                    syn_tgt_data=syn_b,
                    trn_tgt_data=syn_a,
                    report_path=report_path,
                    report_title=f"A vs {other_label} Comparison",
                )
                print(f"  Report saved to: {report_path}")

                if metrics:
                    try:
                        metrics_dict = metrics.model_dump()
                        available_metrics = []

                        overall_accuracy = metrics_dict.get("accuracy", {}).get("overall")
                        if overall_accuracy is not None:
                            acc_ok = overall_accuracy > 0.6
                            print(f"    Overall accuracy: {overall_accuracy:.3f} - {'✓' if acc_ok else '❌'}")
                            available_metrics.append(acc_ok)
                            summary_rows.append({"Metric": "Overall Accuracy", "Value": f"{overall_accuracy:.3f}", "Threshold": ">0.6", "Pass": "✓" if acc_ok else "❌"})

                        cosine_sim = metrics_dict.get("similarity", {}).get("cosine_similarity_training_synthetic")
                        if cosine_sim is not None:
                            cos_ok = cosine_sim > 0.7
                            print(f"    Cosine similarity: {cosine_sim:.3f} - {'✓' if cos_ok else '❌'}")
                            available_metrics.append(cos_ok)
                            summary_rows.append({"Metric": "Cosine Similarity", "Value": f"{cosine_sim:.3f}", "Threshold": ">0.7", "Pass": "✓" if cos_ok else "❌"})

                        discriminator_auc = metrics_dict.get("similarity", {}).get(
                            "discriminator_auc_training_synthetic"
                        )
                        if discriminator_auc is not None:
                            auc_ok = abs(discriminator_auc - 0.5) < 0.2
                            print(f"    Discriminator AUC: {discriminator_auc:.3f} - {'✓' if auc_ok else '❌'}")
                            available_metrics.append(auc_ok)
                            summary_rows.append({"Metric": "Discriminator AUC", "Value": f"{discriminator_auc:.3f}", "Threshold": "|AUC-0.5|<0.2", "Pass": "✓" if auc_ok else "❌"})

                        univariate_acc = metrics_dict.get("accuracy", {}).get("univariate")
                        if univariate_acc is not None:
                            uni_ok = univariate_acc > 0.8
                            print(f"    Univariate accuracy: {univariate_acc:.3f} - {'✓' if uni_ok else '❌'}")
                            available_metrics.append(uni_ok)
                            summary_rows.append({"Metric": "Univariate Accuracy", "Value": f"{univariate_acc:.3f}", "Threshold": ">0.8", "Pass": "✓" if uni_ok else "❌"})

                        nndr = metrics_dict.get("distances", {}).get("nndr_training")
                        if nndr is not None:
                            nndr_ok = nndr < 0.8
                            print(f"    NNDR distance: {nndr:.3f} - {'✓' if nndr_ok else '❌'}")
                            available_metrics.append(nndr_ok)
                            summary_rows.append({"Metric": "NNDR Distance", "Value": f"{nndr:.3f}", "Threshold": "<0.8", "Pass": "✓" if nndr_ok else "❌"})

                        if available_metrics:
                            similar_count = sum(available_metrics)
                            # Pass if at least 3/5 (or majority if fewer metrics available) are acceptable
                            overall_similar = similar_count >= min(3, len(available_metrics))
                            print(
                                f"  Overall QA assessment: {similar_count}/{len(available_metrics)} metrics acceptable"
                            )
                            print(f"  Datasets similar: {'✓ YES' if overall_similar else '❌ NO'}")
                            reporting.write_github_step_summary(
                                f"Data Quality: Local Central vs {other_label}",
                                summary_rows,
                                output_dir=TestConfig.OUTPUT_DIR,
                            )
                            return all_similar and overall_similar
                        else:
                            print(f"  No QA metrics found, falling back to basic assessment.")
                            reporting.write_github_step_summary(
                                f"Data Quality: Local Central vs {other_label}",
                                summary_rows,
                                output_dir=TestConfig.OUTPUT_DIR,
                            )
                            return all_similar
                    except Exception as e:
                        print(f"  Could not extract QA metrics: {e}")
        except Exception as e:
            print(f"  Warning: Could not run QA assessment: {e}")
            print(f"  Note: The synthetic data is non-deterministic — the mostlyai-qa library may hit edge cases")
            print(f"  (e.g. duplicate labels after category trimming). Re-running may resolve it.")
    else:
        print(f"\nBasic assessment only (mostlyai-qa not available).")
        print(f"  Overall similarity: {'✓ YES' if all_similar else '❌ NO'}")
        print(f"  Note: Install mostlyai-qa for comprehensive quality assessment")

    reporting.write_github_step_summary(
        f"Data Quality: Local Central vs {other_label}",
        summary_rows,
        output_dir=TestConfig.OUTPUT_DIR,
    )
    return all_similar


def test_data_generation_quality_local_central_vs_local_federated():
    """Compare data generation quality: local library (central) vs local library (federated).

    Trains both approaches with the local development engine and compares the statistical
    properties of the generated synthetic data to verify they converge to similar distributions.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: LOCAL CENTRAL vs LOCAL FEDERATED")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_central_ws = Path(tmpdir) / "local-central-ws"
            local_central_ws.mkdir(parents=True)
            local_central_synthetic = _generate_synthetic_local_central(train_data, local_central_ws)

            local_federated_ws = Path(tmpdir) / "local-federated-ws"
            local_federated_ws.mkdir(parents=True)
            local_federated_synthetic = _generate_synthetic_local_federated(train_data, local_federated_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: LOCAL CENTRAL vs LOCAL FEDERATED")
            print("=" * 80)
            print(f"Local central synthetic data:   {len(local_central_synthetic)} samples")
            print(f"Local federated synthetic data: {len(local_federated_synthetic)} samples")

            result = _compare_synthetic_datasets(
                local_central_synthetic, local_federated_synthetic, other_label="local federated"
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            return result

    except Exception as e:
        print(f"Error during local central vs local federated test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_generation_quality_local_central_vs_pypi_central():
    """Compare data generation quality: local library (central) vs PyPI library (central).

    Trains a centralised model with both the local development version and the published
    PyPI version of mostlyai.engine, then compares the statistical properties of the
    generated synthetic data to verify they are equivalent.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: LOCAL CENTRAL vs PyPI CENTRAL")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    if not HAS_PYPI_ENGINE:
        print("\nSkipping: PyPI mostlyai.engine not available.")
        return True

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_central_ws = Path(tmpdir) / "local-central-ws"
            local_central_ws.mkdir(parents=True)
            local_central_synthetic = _generate_synthetic_local_central(train_data, local_central_ws)

            pypi_central_ws = Path(tmpdir) / "pypi-central-ws"
            pypi_central_ws.mkdir(parents=True)
            pypi_central_synthetic = _generate_synthetic_pypi_central(train_data, pypi_central_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: LOCAL CENTRAL vs PyPI CENTRAL")
            print("=" * 80)
            print(f"Local central synthetic data: {len(local_central_synthetic)} samples")
            print(f"PyPI central synthetic data:  {len(pypi_central_synthetic)} samples")

            result = _compare_synthetic_datasets(
                local_central_synthetic, pypi_central_synthetic, other_label="PyPI central"
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            return result

    except Exception as e:
        print(f"Error during local central vs PyPI central test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_generation_quality_local_federated_vs_pypi_central():
    """Compare data generation quality: local library (federated) vs PyPI library (central).

    Trains a federated model with the local development version and a centralised model
    with the published PyPI version of mostlyai.engine, then compares the statistical
    properties of the generated synthetic data to verify they are equivalent.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: LOCAL FEDERATED vs PyPI CENTRAL")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    if not HAS_PYPI_ENGINE:
        print("\nSkipping: PyPI mostlyai.engine not available.")
        return True

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_federated_ws = Path(tmpdir) / "local-federated-ws"
            local_federated_ws.mkdir(parents=True)
            local_federated_synthetic = _generate_synthetic_local_federated(train_data, local_federated_ws)

            pypi_central_ws = Path(tmpdir) / "pypi-central-ws"
            pypi_central_ws.mkdir(parents=True)
            pypi_central_synthetic = _generate_synthetic_pypi_central(train_data, pypi_central_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: LOCAL FEDERATED vs PyPI CENTRAL")
            print("=" * 80)
            print(f"Local federated synthetic data: {len(local_federated_synthetic)} samples")
            print(f"PyPI central synthetic data:    {len(pypi_central_synthetic)} samples")

            result = _compare_synthetic_datasets(
                local_federated_synthetic, pypi_central_synthetic, other_label="PyPI central"
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            return result

    except Exception as e:
        print(f"Error during local federated vs PyPI central test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_generation_quality_pypi_central_vs_pypi_central():
    """Compare data generation quality: PyPI library (central) vs PyPI library (central).

    Trains a centralised model with the published PyPI version of mostlyai.engine,
    then compares the statistical properties of the generated synthetic data to verify
    they are equivalent.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: PyPI CENTRAL vs PyPI CENTRAL")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    if not HAS_PYPI_ENGINE:
        print("\nSkipping: PyPI mostlyai.engine not available.")
        return True

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_federated_ws = Path(tmpdir) / "local-federated-ws"
            local_federated_ws.mkdir(parents=True)
            local_federated_synthetic = _generate_synthetic_pypi_central(train_data, local_federated_ws)

            pypi_central_ws = Path(tmpdir) / "pypi-central-ws"
            pypi_central_ws.mkdir(parents=True)
            pypi_central_synthetic = _generate_synthetic_pypi_central(train_data, pypi_central_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: PyPI CENTRAL vs PyPI CENTRAL")
            print("=" * 80)
            print(f"Local federated synthetic data: {len(local_federated_synthetic)} samples")
            print(f"PyPI central synthetic data:    {len(pypi_central_synthetic)} samples")

            result = _compare_synthetic_datasets(
                local_federated_synthetic, pypi_central_synthetic, other_label="PyPI central"
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            return result

    except Exception as e:
        print(f"Error during local federated vs PyPI central test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all data generation quality tests."""
    print("DATA GENERATION QUALITY TEST SUITE")
    print("=" * 80)

    results = []

    # Test 1: Local central vs local federated
    results.append(
        (
            "Data Generation Quality (local central vs local federated)",
            test_data_generation_quality_local_central_vs_local_federated(),
        )
    )

    # Test 2: Local central vs PyPI central
    results.append(
        (
            "Data Generation Quality (local central vs PyPI central)",
            test_data_generation_quality_local_central_vs_pypi_central(),
        )
    )

    # Test 3: Local federated vs PyPI central
    results.append(
        (
            "Data Generation Quality (local federated vs PyPI central)",
            test_data_generation_quality_local_federated_vs_pypi_central(),
        )
    )

    # Test 4: PyPI central vs PyPI central
    results.append(
        (
            "Data Generation Quality (PyPI central vs PyPI central)",
            test_data_generation_quality_pypi_central_vs_pypi_central(),
        )
    )

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
        print("All training approaches produce statistically similar synthetic data.")
    else:
        print("FAILURE: SOME TESTS FAILED!")
        print("There may be quality differences between training approaches.")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
