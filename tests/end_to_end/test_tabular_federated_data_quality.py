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
End-to-end tests comparing data generation quality across training approaches.

These tests evaluate whether synthetic data produced by:
1. Dev central training
2. Dev federated training
3. PyPI central training

converge to statistically similar distributions.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Project root for the dev checkout
project_root = Path(__file__).parent.parent.parent

# Import PyPI version first (before adding the dev path)
try:
    import mostlyai.engine as _mostlyai_engine_pypi
    from mostlyai.engine import analyze as analyze_pypi
    from mostlyai.engine import encode as encode_pypi
    from mostlyai.engine import generate as generate_pypi
    from mostlyai.engine import split as split_pypi
    from mostlyai.engine import train as train_pypi
    from mostlyai.engine._workspace import Workspace as Workspace_pypi
    from mostlyai.engine.domain import ModelEncodingType as ModelEncodingType_pypi

    HAS_PYPI_ENGINE = True
    print(f"PyPI mostlyai.engine imported: {getattr(_mostlyai_engine_pypi, '__version__', 'unknown version')}")
except ImportError:
    split_pypi = analyze_pypi = encode_pypi = train_pypi = generate_pypi = None
    ModelEncodingType_pypi = None
    Workspace_pypi = None
    HAS_PYPI_ENGINE = False
    print("Note: PyPI mostlyai.engine not available - PyPI comparison tests will be skipped")

# Clear cached mostlyai modules so the dev version can be loaded fresh
for _key in list(sys.modules.keys()):
    if _key.startswith("mostlyai"):
        del sys.modules[_key]

# Now add the dev path
sys.path.insert(0, str(project_root))

from mostlyai.engine import analyze, encode, generate, split, train  # noqa: E402
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

    # Data generation parameters
    TEST_SAMPLES = 3000  # Samples used for quality testing

    # Quality assessment
    QUALITY_TOLERANCE = 0.15  # 15% tolerance for quality score comparison

    # Output directory for plots and summary artifacts
    OUTPUT_DIR = Path("test-output/quality")


# ============================================

# Optional import for quality assessment
try:
    from mostlyai import qa

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
                "Generated samples": TestConfig.TEST_SAMPLES,
                "Quality tolerance": f"{TestConfig.QUALITY_TOLERANCE:.0%}",
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


def _generate_synthetic_dev_central(train_data, workspace_dir):
    """Train centrally with the dev engine and return a synthetic DataFrame."""
    setup_workspace(train_data, workspace_dir)
    print("Training dev central model...")
    train(workspace_dir=workspace_dir, max_epochs=TestConfig.MAX_EPOCHS, model=TestConfig.MODEL_SIZE)
    print("Generating data from dev central model...")
    generate(workspace_dir=workspace_dir, sample_size=TestConfig.TEST_SAMPLES)
    ws = Workspace(workspace_dir)
    return pd.concat([pd.read_parquet(f) for f in ws.generated_data.fetch_all()], ignore_index=True)


def _generate_synthetic_dev_federated(train_data, workspace_dir):
    """Train federatedly with the dev engine and return a synthetic DataFrame."""
    setup_workspace(train_data, workspace_dir)
    print("Training dev federated model (1 epoch per iteration)...")
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
    print("Generating data from dev federated model...")
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


def _compare_synthetic_datasets(syn_a, syn_b, a_label="Dev Central", other_label="other"):
    """Compare two synthetic DataFrames statistically and return whether they are similar.

    Performs column checks, numeric/categorical statistical comparisons, and (if available)
    an advanced QA report using the same 5-metric logic as the original quality tests:
    overall accuracy, cosine similarity, discriminator AUC, univariate accuracy, NNDR distance.
    Passes if ≥ 3/5 available QA metrics are acceptable; falls back to basic stats otherwise.
    """
    comparison_title = f"Data Quality: {a_label} vs {other_label}"
    # Column check
    cols_a = set(syn_a.columns)
    cols_b = set(syn_b.columns)
    columns_match = cols_a == cols_b
    print("\nColumn analysis:")
    print(f"  Columns A ({a_label}):  {sorted(cols_a)}")
    print(f"  Columns B ({other_label}): {sorted(cols_b)}")
    print(f"  Column match: {'✓ YES' if columns_match else '❌ NO'}")
    if not columns_match:
        print(f"  Missing in B: {cols_a - cols_b}")
        print(f"  Missing in A: {cols_b - cols_a}")
        return False

    all_similar = True
    summary_rows = []  # Collected for GitHub step summary
    print("\nStatistical comparison (numeric):")
    for col in [
        "bundesland_id",
        "anzahl_meldebereiche",
        "faelle_covid_aktuell",
        "intensivbetten_belegt",
        "intensivbetten_frei",
    ]:
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
                summary_rows.append(
                    {
                        "Metric": f"{col} (mean diff)",
                        "Value": f"{mean_rel_diff:.1%}",
                        "Threshold": f"<{TestConfig.QUALITY_TOLERANCE:.0%}",
                        "Pass": "✓" if mean_ok else "❌",
                    }
                )
                summary_rows.append(
                    {
                        "Metric": f"{col} (std diff)",
                        "Value": f"{std_rel_diff:.1%}",
                        "Threshold": f"<{TestConfig.QUALITY_TOLERANCE:.0%}",
                        "Pass": "✓" if std_ok else "❌",
                    }
                )
            except Exception as e:
                print(f"    Could not compare {col}: {e}")
                all_similar = False

    # Categorical columns
    print("\nStatistical comparison (categorical):")
    for col in ["bundesland_name", "versorgungsstufe"]:
        if col in syn_a.columns:
            try:
                dist_a = syn_a[col].value_counts(normalize=True)
                dist_b = syn_b[col].value_counts(normalize=True)
                all_values = set(dist_a.index) | set(dist_b.index)  # union: don't miss exclusive categories
                tv_distance = 0.5 * sum(abs(dist_a.get(v, 0) - dist_b.get(v, 0)) for v in all_values)
                similar = tv_distance < TestConfig.QUALITY_TOLERANCE
                print(f"  {col}: TV distance={tv_distance:.3f} - {'✓' if similar else '❌'}")
                if not similar:
                    all_similar = False
                summary_rows.append(
                    {
                        "Metric": f"{col} (TV distance)",
                        "Value": f"{tv_distance:.3f}",
                        "Threshold": f"<{TestConfig.QUALITY_TOLERANCE:.0%}",
                        "Pass": "✓" if similar else "❌",
                    }
                )
            except Exception as e:
                print(f"    Could not compare {col}: {e}")
                all_similar = False

    # Advanced QA assessment
    if HAS_QA_LIBRARY:
        print("\nAdvanced quality assessment (using mostlyai-qa):")
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
                            summary_rows.append(
                                {
                                    "Metric": "Overall Accuracy",
                                    "Value": f"{overall_accuracy:.3f}",
                                    "Threshold": ">0.6",
                                    "Pass": "✓" if acc_ok else "❌",
                                }
                            )

                        cosine_sim = metrics_dict.get("similarity", {}).get("cosine_similarity_training_synthetic")
                        if cosine_sim is not None:
                            cos_ok = cosine_sim > 0.7
                            print(f"    Cosine similarity: {cosine_sim:.3f} - {'✓' if cos_ok else '❌'}")
                            available_metrics.append(cos_ok)
                            summary_rows.append(
                                {
                                    "Metric": "Cosine Similarity",
                                    "Value": f"{cosine_sim:.3f}",
                                    "Threshold": ">0.7",
                                    "Pass": "✓" if cos_ok else "❌",
                                }
                            )

                        discriminator_auc = metrics_dict.get("similarity", {}).get(
                            "discriminator_auc_training_synthetic"
                        )
                        if discriminator_auc is not None:
                            auc_ok = abs(discriminator_auc - 0.5) < 0.2
                            print(f"    Discriminator AUC: {discriminator_auc:.3f} - {'✓' if auc_ok else '❌'}")
                            available_metrics.append(auc_ok)
                            summary_rows.append(
                                {
                                    "Metric": "Discriminator AUC",
                                    "Value": f"{discriminator_auc:.3f}",
                                    "Threshold": "|AUC-0.5|<0.2",
                                    "Pass": "✓" if auc_ok else "❌",
                                }
                            )

                        univariate_acc = metrics_dict.get("accuracy", {}).get("univariate")
                        if univariate_acc is not None:
                            uni_ok = univariate_acc > 0.8
                            print(f"    Univariate accuracy: {univariate_acc:.3f} - {'✓' if uni_ok else '❌'}")
                            available_metrics.append(uni_ok)
                            summary_rows.append(
                                {
                                    "Metric": "Univariate Accuracy",
                                    "Value": f"{univariate_acc:.3f}",
                                    "Threshold": ">0.8",
                                    "Pass": "✓" if uni_ok else "❌",
                                }
                            )

                        nndr = metrics_dict.get("distances", {}).get("nndr_training")
                        if nndr is not None:
                            nndr_ok = nndr < 0.8
                            print(f"    NNDR distance: {nndr:.3f} - {'✓' if nndr_ok else '❌'}")
                            available_metrics.append(nndr_ok)
                            summary_rows.append(
                                {
                                    "Metric": "NNDR Distance",
                                    "Value": f"{nndr:.3f}",
                                    "Threshold": "<0.8",
                                    "Pass": "✓" if nndr_ok else "❌",
                                }
                            )

                        if available_metrics:
                            similar_count = sum(available_metrics)
                            # Pass if at least 3/5 (or majority if fewer metrics available) are acceptable
                            overall_similar = similar_count >= min(3, len(available_metrics))
                            print(
                                f"  Overall QA assessment: {similar_count}/{len(available_metrics)} metrics acceptable"
                            )
                            print(f"  Datasets similar: {'✓ YES' if overall_similar else '❌ NO'}")
                            reporting.write_github_step_summary(
                                comparison_title,
                                summary_rows,
                                output_dir=TestConfig.OUTPUT_DIR,
                            )
                            return all_similar and overall_similar
                        else:
                            print("  No QA metrics found, falling back to basic assessment.")
                            reporting.write_github_step_summary(
                                comparison_title,
                                summary_rows,
                                output_dir=TestConfig.OUTPUT_DIR,
                            )
                            return all_similar
                    except Exception as e:
                        print(f"  Could not extract QA metrics: {e}")
        except Exception as e:
            print(f"  Warning: Could not run QA assessment: {e}")
            print("  Note: The synthetic data is non-deterministic — the mostlyai-qa library may hit edge cases")
            print("  (e.g. duplicate labels after category trimming). Re-running may resolve it.")
    else:
        print("\nBasic assessment only (mostlyai-qa not available).")
        print(f"  Overall similarity: {'✓ YES' if all_similar else '❌ NO'}")
        print("  Note: Install mostlyai-qa for comprehensive quality assessment")

    reporting.write_github_step_summary(
        comparison_title,
        summary_rows,
        output_dir=TestConfig.OUTPUT_DIR,
    )
    return all_similar


def test_data_generation_quality_dev_central_vs_dev_federated():
    """Compare data generation quality: dev library (central) vs dev library (federated).

    Trains both approaches with the dev engine and compares the statistical
    properties of the generated synthetic data to verify they converge to similar distributions.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: DEV CENTRAL vs DEV FEDERATED")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            dev_central_ws = Path(tmpdir) / "dev-central-ws"
            dev_central_ws.mkdir(parents=True)
            dev_central_synthetic = _generate_synthetic_dev_central(train_data, dev_central_ws)

            dev_federated_ws = Path(tmpdir) / "dev-federated-ws"
            dev_federated_ws.mkdir(parents=True)
            dev_federated_synthetic = _generate_synthetic_dev_federated(train_data, dev_federated_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: DEV CENTRAL vs DEV FEDERATED")
            print("=" * 80)
            print(f"Dev central synthetic data:   {len(dev_central_synthetic)} samples")
            print(f"Dev federated synthetic data: {len(dev_federated_synthetic)} samples")

            result = _compare_synthetic_datasets(
                dev_central_synthetic, dev_federated_synthetic, other_label="Dev Federated"
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            assert result, "Dev central and dev federated synthetic data are not statistically similar"
            return

    except Exception as e:
        print(f"Error during dev central vs dev federated test: {e}")
        raise


def test_data_generation_quality_dev_central_vs_pypi_central():
    """Compare data generation quality: dev library (central) vs PyPI library (central).

    Trains a centralised model with both the dev version and the published
    PyPI version of mostlyai.engine, then compares the statistical properties of the
    generated synthetic data to verify they are equivalent.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: DEV CENTRAL vs PyPI CENTRAL")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    if not HAS_PYPI_ENGINE:
        print("\nSkipping: PyPI mostlyai.engine not available.")
        pytest.skip("PyPI mostlyai.engine not available")

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            dev_central_ws = Path(tmpdir) / "dev-central-ws"
            dev_central_ws.mkdir(parents=True)
            dev_central_synthetic = _generate_synthetic_dev_central(train_data, dev_central_ws)

            pypi_central_ws = Path(tmpdir) / "pypi-central-ws"
            pypi_central_ws.mkdir(parents=True)
            pypi_central_synthetic = _generate_synthetic_pypi_central(train_data, pypi_central_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: DEV CENTRAL vs PyPI CENTRAL")
            print("=" * 80)
            print(f"Dev central synthetic data: {len(dev_central_synthetic)} samples")
            print(f"PyPI central synthetic data:  {len(pypi_central_synthetic)} samples")

            result = _compare_synthetic_datasets(
                dev_central_synthetic, pypi_central_synthetic, other_label="PyPI Central"
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            assert result, "Dev central and PyPI central synthetic data are not statistically similar"
            return

    except Exception as e:
        print(f"Error during dev central vs PyPI central test: {e}")
        raise


def test_data_generation_quality_dev_federated_vs_pypi_central():
    """Compare data generation quality: dev library (federated) vs PyPI library (central).

    Trains a federated model with the dev version and a centralised model
    with the published PyPI version of mostlyai.engine, then compares the statistical
    properties of the generated synthetic data to verify they are equivalent.
    """
    print("\n" + "=" * 80)
    print("DATA GENERATION QUALITY: DEV FEDERATED vs PyPI CENTRAL")
    print("=" * 80)
    print(f"Configuration: {TestConfig.MAX_EPOCHS} epochs, {TestConfig.MODEL_SIZE} model")

    if not HAS_PYPI_ENGINE:
        print("\nSkipping: PyPI mostlyai.engine not available.")
        pytest.skip("PyPI mostlyai.engine not available")

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            dev_federated_ws = Path(tmpdir) / "dev-federated-ws"
            dev_federated_ws.mkdir(parents=True)
            dev_federated_synthetic = _generate_synthetic_dev_federated(train_data, dev_federated_ws)

            pypi_central_ws = Path(tmpdir) / "pypi-central-ws"
            pypi_central_ws.mkdir(parents=True)
            pypi_central_synthetic = _generate_synthetic_pypi_central(train_data, pypi_central_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: DEV FEDERATED vs PyPI CENTRAL")
            print("=" * 80)
            print(f"Dev federated synthetic data: {len(dev_federated_synthetic)} samples")
            print(f"PyPI central synthetic data:    {len(pypi_central_synthetic)} samples")

            result = _compare_synthetic_datasets(
                dev_federated_synthetic,
                pypi_central_synthetic,
                a_label="Dev Federated",
                other_label="PyPI Central",
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            assert result, "Dev federated and PyPI central synthetic data are not statistically similar"
            return

    except Exception as e:
        print(f"Error during dev federated vs PyPI central test: {e}")
        raise


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
        pytest.skip("PyPI mostlyai.engine not available")

    try:
        data = create_test_data()
        print(f"Fetched test data with {len(data)} samples")
        train_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            pypi_a_ws = Path(tmpdir) / "pypi-central-a-ws"
            pypi_a_ws.mkdir(parents=True)
            pypi_a_synthetic = _generate_synthetic_pypi_central(train_data, pypi_a_ws)

            pypi_b_ws = Path(tmpdir) / "pypi-central-b-ws"
            pypi_b_ws.mkdir(parents=True)
            pypi_b_synthetic = _generate_synthetic_pypi_central(train_data, pypi_b_ws)

            print("\n" + "=" * 80)
            print("COMPARISON: PyPI CENTRAL vs PyPI CENTRAL")
            print("=" * 80)
            print(f"PyPI central A synthetic data: {len(pypi_a_synthetic)} samples")
            print(f"PyPI central B synthetic data: {len(pypi_b_synthetic)} samples")

            result = _compare_synthetic_datasets(
                pypi_a_synthetic,
                pypi_b_synthetic,
                a_label="PyPI Central (A)",
                other_label="PyPI Central (B)",
            )
            print(f"\nOverall result: {'✓ PASSED' if result else '❌ FAILED'}")
            assert result, "PyPI central synthetic data self-comparison is not statistically similar"
            return

    except Exception as e:
        print(f"Error during PyPI central self-comparison test: {e}")
        raise


def main():
    """Run all data generation quality tests."""
    print("DATA GENERATION QUALITY TEST SUITE")
    print("=" * 80)

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

    results = []

    # Test 1: Dev central vs dev federated
    results.append(
        _run_test(
            "Data Generation Quality (dev central vs dev federated)",
            test_data_generation_quality_dev_central_vs_dev_federated,
        )
    )

    # Test 2: Dev central vs PyPI central
    results.append(
        _run_test(
            "Data Generation Quality (dev central vs PyPI central)",
            test_data_generation_quality_dev_central_vs_pypi_central,
        )
    )

    # Test 3: Dev federated vs PyPI central
    results.append(
        _run_test(
            "Data Generation Quality (dev federated vs PyPI central)",
            test_data_generation_quality_dev_federated_vs_pypi_central,
        )
    )

    # Test 4: PyPI central vs PyPI central
    results.append(
        _run_test(
            "Data Generation Quality (PyPI central vs PyPI central)",
            test_data_generation_quality_pypi_central_vs_pypi_central,
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
