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
Shared reporting utilities for end-to-end tests.

Provides:
- plot_training_curves(): save a val_loss / trn_loss PNG for multiple training runs
- write_github_step_summary(): emit a markdown table to $GITHUB_STEP_SUMMARY and/or an artifact file
"""

import os
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend — safe in CI and headless environments
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_training_curves(curves: dict, output_path) -> bool:
    """Plot validation-loss training curves for multiple runs and save to a PNG.

    Args:
        curves: Mapping of label → pd.DataFrame with at minimum a ``val_loss`` column.
                An ``epoch`` column is used for the x-axis; if absent a 1-based index is
                synthesised.  An optional ``trn_loss`` column is plotted as a faint dashed
                line when present.
        output_path: Destination PNG path (parent directories are created automatically).

    Returns:
        True on success, False when matplotlib is not available.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [p["color"] for p in prop_cycle]

    for idx, (label, df) in enumerate(curves.items()):
        if df is None or df.empty:
            continue

        colour = colours[idx % len(colours)]

        epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

        if "val_loss" in df.columns:
            ax.plot(
                epochs, df["val_loss"], marker="o", markersize=3, linewidth=1.5, color=colour, label=f"{label} (val)"
            )

        if "trn_loss" in df.columns and df["trn_loss"].notna().any():
            ax.plot(
                epochs, df["trn_loss"], linestyle="--", linewidth=1.0, alpha=0.5, color=colour, label=f"{label} (train)"
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curve plot: {output_path}")
    return True


def epoch_loss_table_rows(curves: dict) -> list:
    """Build a list of row dicts for a per-epoch val_loss comparison table.

    Args:
        curves: Mapping of label → pd.DataFrame with ``epoch`` and ``val_loss`` columns.

    Returns:
        List of dicts keyed by ``"Epoch"`` plus one key per label.
    """
    if not curves:
        return []

    series = {}
    for label, df in curves.items():
        if df is None or df.empty or "val_loss" not in df.columns:
            continue
        epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
        series[label] = pd.Series(
            [f"{v:.6f}" if pd.notna(v) else "—" for v in df["val_loss"]],
            index=list(epochs),
            name=label,
        )

    if not series:
        return []

    merged = pd.concat(series.values(), axis=1, keys=series.keys())
    merged = merged.fillna("—")  # replace NaN from misaligned epoch counts with em-dash
    merged.index.name = "Epoch"
    merged = merged.reset_index()
    merged["Epoch"] = merged["Epoch"].astype(int)
    return merged.to_dict(orient="records")


def write_dataset_info(df: "pd.DataFrame", source_url: str, config: dict, output_dir=None) -> None:
    """Write a dataset metadata table to $GITHUB_STEP_SUMMARY and/or an artifact file.

    Produces a ``## Test Dataset`` section with provenance, shape, and training config
    so readers can understand what data was used without digging into the test source.

    Args:
        df:         The loaded dataset DataFrame (used for row/column counts and basic stats).
        source_url: Public URL the data was fetched from.
        config:     Dict of training config key/value pairs to include (e.g. model, epochs).
        output_dir: Optional directory.  When given, appends to ``{output_dir}/summary.md``.
    """
    rows = [
        {"Property": "Source", "Value": source_url},
        {"Property": "Rows", "Value": f"{len(df):,}"},
        {"Property": "Columns", "Value": str(len(df.columns))},
        {"Property": "Column names", "Value": ", ".join(df.columns.tolist())},
    ]
    for k, v in config.items():
        rows.append({"Property": k, "Value": str(v)})

    write_github_step_summary("Test Dataset", rows, output_dir=output_dir)


def write_summary_header(title: str, output_dir=None) -> None:
    """Write a top-level H1 heading to $GITHUB_STEP_SUMMARY and/or an artifact file.

    Intended to be called once at the start of a test run to provide context for all
    subsequent tables appended to the same summary page.

    Args:
        title:      Heading text rendered as a ``#`` (H1) heading.
        output_dir: Optional directory.  When given, appends to ``{output_dir}/summary.md``.
    """
    import datetime

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    markdown = f"# {title}\n\n_Generated: {timestamp}_\n\n"

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a", encoding="utf-8") as fh:
            fh.write(markdown)

    if output_dir:
        summary_file = Path(output_dir) / "summary.md"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(markdown)


def write_github_step_summary(title: str, rows: list, output_dir=None) -> None:
    """Write a markdown table to $GITHUB_STEP_SUMMARY and optionally to an artifact file.

    The table is appended (not overwritten) so multiple test sections can accumulate
    in the same summary page.  A ``summary.md`` artifact file is written alongside plots
    when *output_dir* is provided — useful for uploading as a GitHub Actions artifact.

    Args:
        title:      Section heading rendered as an ``##`` heading.
        rows:       List of dicts; all dicts should share the same keys (used as column
                    headers in order of the first dict).
        output_dir: Optional directory.  When given, appends to ``{output_dir}/summary.md``.
    """
    if not rows:
        return

    headers = list(rows[0].keys())
    # Left-align first column (labels), right-align the rest (numbers)
    sep_parts = [":---" if i == 0 else "---:" for i in range(len(headers))]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep_parts) + " |",
    ]
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h, "—")
            cells.append("—" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    markdown = "\n".join(lines) + "\n"

    # Always echo to stdout so it is visible in the raw log
    print("\n" + markdown)

    # Write to GitHub Actions job summary (visible on the workflow run page)
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a", encoding="utf-8") as fh:
            fh.write(markdown)
        print("  → appended to $GITHUB_STEP_SUMMARY")

    # Write to an artifact file for download / archival
    if output_dir:
        summary_file = Path(output_dir) / "summary.md"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(markdown)
        print(f"  → appended to {summary_file}")
