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

import numpy as np
import pandas as pd

from mostlyai.engine._common import ANALYZE_MIN_MAX_TOP_N, read_json, write_json
from mostlyai.engine.analysis import (
    _analyze_col,
    _analyze_partition,
    _analyze_reduce_seq_len,
    _analyze_seq_len,
    analyze,
    analyze_partial,
    analyze_reduce,
)
from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine.splitting import split


def test_analyze_cnt(tmp_path):
    events = pd.DataFrame({"context_key": [1, 1, 1, 2, 3, 4], "x": [1, 1, 1, 1, 1, 1]})
    no_of_records = events["context_key"].nunique()

    events.to_parquet(tmp_path / "part.000000-trn.parquet")
    stats_path = tmp_path / "stats"
    stats_path.mkdir()
    _analyze_partition(
        tmp_path / "part.000000-trn.parquet",
        stats_path,
        tgt_context_key="context_key",
        tgt_encoding_types={"x": ModelEncodingType.tabular_numeric_digit},
    )
    stats = read_json(stats_path / "part.000000-trn.json")
    assert stats["no_of_training_records"] == no_of_records
    assert stats["no_of_validation_records"] == 0


def test_analyze_seq_len(tmp_path):
    tgt_context_keys = pd.Series(np.repeat(range(21), range(21)), name="account_id")
    partition_stats = _analyze_seq_len(tgt_context_keys=tgt_context_keys, ctx_primary_keys=tgt_context_keys)
    write_json(partition_stats, tmp_path / "stats1.json")
    partition_stats = read_json(tmp_path / "stats1.json")
    global_stats = _analyze_reduce_seq_len([partition_stats])
    write_json(global_stats, tmp_path / "stats.json")
    global_stats = read_json(tmp_path / "stats.json")
    assert global_stats["max"] >= 12 and global_stats["max"] <= 15
    assert isinstance(global_stats["max"], int)
    global_stats = _analyze_reduce_seq_len([partition_stats for i in range(9)])
    assert global_stats["max"] == 20

    tgt_context_keys = pd.Series(np.repeat(range(21), range(21)), name="account_id")
    partition_stats = _analyze_seq_len(
        tgt_context_keys=tgt_context_keys,
        ctx_primary_keys=pd.concat([tgt_context_keys, pd.Series([100])]),
    )
    global_stats = _analyze_reduce_seq_len([partition_stats], value_protection=False)
    assert global_stats["min"] == 0
    assert global_stats["max"] == 20


def test_analyze_root_key(tmp_path):
    tgt_context_keys = pd.Series(np.repeat(range(40), 2), name="tgt_context_key")
    tgt_values = pd.Series(list(range(80)), name="tgt_values")
    tgt = pd.concat([tgt_context_keys, tgt_values], axis=1)

    ctx_root_keys = pd.Series(np.repeat(range(20), 2), name="ctx_root_keys")
    ctx_primary_keys = pd.Series(list(range(40)), name="ctx_primary_key")
    ctx_values = pd.Series(list(range(40)), name="ctx_values")
    ctx = pd.concat([ctx_root_keys, ctx_primary_keys, ctx_values], axis=1)

    tgt_partition_path, ctx_partition_path = (
        tmp_path / "tgt.000000-trn.parquet",
        tmp_path / "ctx.000000-trn.parquet",
    )
    tgt.to_parquet(tgt_partition_path), ctx.to_parquet(ctx_partition_path)

    tgt_stats_path, ctx_stats_path = tmp_path / "tgt_stats", tmp_path / "ctx_stats"
    tgt_stats_path.mkdir(), ctx_stats_path.mkdir()

    # root key column is in tgt table
    _analyze_partition(
        tgt_partition_file=tgt_partition_path,
        tgt_stats_path=tgt_stats_path,
        tgt_encoding_types={tgt_values.name: ModelEncodingType.tabular_numeric_digit},
        tgt_context_key=tgt_context_keys.name,
        ctx_partition_file=ctx_partition_path,
        ctx_stats_path=ctx_stats_path,
        ctx_encoding_types={ctx_values.name: ModelEncodingType.tabular_numeric_digit},
        ctx_primary_key=ctx_primary_keys.name,
        ctx_root_key=ctx_root_keys.name,
    )
    ctx_stats = read_json(ctx_stats_path / "part.000000-trn.json")
    assert ctx_stats["columns"][ctx_values.name]["max_n"] == list(range(40))[::-2][:ANALYZE_MIN_MAX_TOP_N]
    assert ctx_stats["columns"][ctx_values.name]["min_n"] == list(range(40))[::2][:ANALYZE_MIN_MAX_TOP_N]

    # root key column is in ctx table
    _analyze_partition(
        tgt_partition_file=tgt_partition_path,
        tgt_stats_path=tgt_stats_path,
        tgt_encoding_types={tgt_values.name: ModelEncodingType.tabular_numeric_digit},
        tgt_context_key=tgt_context_keys.name,
        ctx_partition_file=ctx_partition_path,
        ctx_stats_path=ctx_stats_path,
        ctx_encoding_types={ctx_values.name: ModelEncodingType.tabular_numeric_digit},
        ctx_primary_key=ctx_primary_keys.name,
        ctx_root_key=ctx_root_keys.name,
    )
    tgt_stats = read_json(tgt_stats_path / "part.000000-trn.json")
    assert tgt_stats["columns"][tgt_values.name]["max_n"] == list(range(80))[::-1][:ANALYZE_MIN_MAX_TOP_N]
    assert tgt_stats["columns"][tgt_values.name]["min_n"] == list(range(80))[::1][:ANALYZE_MIN_MAX_TOP_N]
    ctx_stats = read_json(ctx_stats_path / "part.000000-trn.json")
    assert ctx_stats["columns"][ctx_values.name]["max_n"] == list(range(40))[::-2][:ANALYZE_MIN_MAX_TOP_N]
    assert ctx_stats["columns"][ctx_values.name]["min_n"] == list(range(40))[::2][:ANALYZE_MIN_MAX_TOP_N]


class TestAnalyzeCol:
    def test_empty_values(self):
        values = pd.Series([], name="values")
        root_keys = pd.Series([], name="root_keys")
        stats = _analyze_col(values=values, encoding_type=ModelEncodingType.tabular_categorical, root_keys=root_keys)
        assert stats == {"encoding_type": ModelEncodingType.tabular_categorical.value}

    def test_flat_values(self):
        values = pd.Series([1, 2, 3], name="values")
        root_keys = pd.Series([1, 2, 3], name="root_keys")
        stats = _analyze_col(
            values=values, encoding_type=ModelEncodingType.tabular_categorical.value, root_keys=root_keys
        )
        assert stats == {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "cnt_values": {"1": 1, "2": 1, "3": 1},
            "has_nan": False,
        }

    def test_sequential_values(self):
        values = pd.Series([[1, 2, 3], [], [3], [], [2]], name="values")
        root_keys = pd.Series([1, 2, 3, 4, 5], name="root_keys")
        stats = _analyze_col(values=values, encoding_type=ModelEncodingType.tabular_categorical, root_keys=root_keys)
        assert stats == {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "cnt_values": {"1": 1, "2": 2, "3": 2},
            "has_nan": False,
            "seq_len": {"cnt_lengths": {0: 2, 1: 2, 3: 1}},
        }


def _split_tgt_flat(workspace_dir):
    tgt_df = pd.DataFrame(
        {
            "id": list(range(100)),
            "cat": ["A", "B"] * 50,
        }
    )
    split(
        tgt_data=tgt_df,
        tgt_encoding_types={"cat": ModelEncodingType.tabular_categorical},
        tgt_primary_key="id",
        workspace_dir=workspace_dir,
    )


def test_analyze_partial_then_reduce_parity(tmp_path):
    # build two identical workspaces from the same data
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    _split_tgt_flat(ws_a)
    _split_tgt_flat(ws_b)

    # path A: analyze() directly
    analyze(workspace_dir=ws_a)
    stats_a = read_json(ws_a / "ModelStore" / "tgt-stats" / "stats.json")

    # path B: analyze_partial() then analyze_reduce() (no aggregate)
    analyze_partial(workspace_dir=ws_b)
    analyze_reduce(workspace_dir=ws_b)
    stats_b = read_json(ws_b / "ModelStore" / "tgt-stats" / "stats.json")

    assert stats_a == stats_b


def test_analyze_partial_return_matches_part_files(tmp_path):
    ws = tmp_path / "ws"
    _split_tgt_flat(ws)
    partials = analyze_partial(workspace_dir=ws)
    assert set(partials.keys()) == {"tgt", "ctx"}
    assert partials["ctx"] == []
    assert len(partials["tgt"]) >= 1
    # returned dicts must equal the content written to the part.*.json files
    # (normalize via a JSON round-trip, since JSON serializes int dict keys as strings)
    import json

    part_dir = ws / "ModelStore" / "tgt-stats"
    on_disk = [read_json(p) for p in sorted(part_dir.glob("part.*.json"))]
    in_memory = json.loads(json.dumps(partials["tgt"]))
    assert in_memory == on_disk


def test_analyze_reduce_local_cleans_up_part_files(tmp_path):
    ws = tmp_path / "ws"
    _split_tgt_flat(ws)
    analyze_partial(workspace_dir=ws)
    part_dir = ws / "ModelStore" / "tgt-stats"
    assert list(part_dir.glob("part.*.json"))  # present before reduce
    analyze_reduce(workspace_dir=ws)
    assert not list(part_dir.glob("part.*.json"))  # removed after reduce


def test_analyze_reduce_federated_superset_adds_names(tmp_path):
    ws = tmp_path / "ws"
    _split_tgt_flat(ws)
    partials = analyze_partial(workspace_dir=ws)
    # federation-wide names: local A/B plus an extra "C" not seen locally (minimal: names only, no counts)
    aggregated = [
        {"columns": {"cat": {"encoding_type": ModelEncodingType.tabular_categorical, "cnt_values": {"A": 0, "B": 0, "C": 0}}}}
    ]
    # include this node's own partials in the aggregate, as required
    aggregated += partials["tgt"]
    analyze_reduce(workspace_dir=ws, value_protection=False, aggregated_tgt_stats=aggregated)
    stats = read_json(ws / "ModelStore" / "tgt-stats" / "stats.json")
    codes = stats["columns"]["cat"]["codes"]
    assert "A" in codes and "B" in codes
    assert "C" in codes  # federation-wide name added with zero local count


def test_analyze_reduce_federated_subset_drops_names(tmp_path):
    ws = tmp_path / "ws"
    _split_tgt_flat(ws)
    analyze_partial(workspace_dir=ws)
    # only "A" is part of the federation-wide vocabulary; local "B" must be dropped
    aggregated = [
        {"columns": {"cat": {"encoding_type": ModelEncodingType.tabular_categorical, "cnt_values": {"A": 0}}}}
    ]
    analyze_reduce(workspace_dir=ws, value_protection=False, aggregated_tgt_stats=aggregated)
    stats = read_json(ws / "ModelStore" / "tgt-stats" / "stats.json")
    codes = stats["columns"]["cat"]["codes"]
    assert "A" in codes
    assert "B" not in codes
