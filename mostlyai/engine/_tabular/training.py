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

import logging
import time
import warnings
from collections.abc import Callable
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import disable_progress_bar, load_dataset
from opacus import GradSampleModule, PrivacyEngine
from opacus.accountants import GaussianAccountant, PRVAccountant, RDPAccountant
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from mostlyai.engine._common import (
    CTXFLT,
    CTXSEQ,
    RIDX_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    TGT,
    ProgressCallback,
    ProgressCallbackWrapper,
    get_cardinalities,
    get_columns_from_cardinalities,
    get_ctx_sequence_length,
    get_empirical_probs_for_predictor_init,
    get_max_data_points_per_sample,
    get_sequence_length_stats,
    get_sub_columns_from_cardinalities,
    get_sub_columns_nested_from_cardinalities,
)
from mostlyai.engine._memory import get_available_ram_for_heuristics
from mostlyai.engine._tabular.argn import (
    FlatModel,
    ModelSize,
    SequentialModel,
    get_model_units,
    get_no_of_model_parameters,
)
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._training_utils import (
    EarlyStopper,
    ModelCheckpoint,
    ProgressMessage,
    check_early_training_exit,
    gpu_memory_cleanup,
)
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir
from mostlyai.engine.domain import DifferentialPrivacyConfig, ModelStateStrategy

_LOG = logging.getLogger(__name__)


##################
### HEURISTICS ###
##################


def _physical_batch_size_heuristic(
    mem_available_gb: float,
    no_of_records: int,
    no_tgt_data_points: int,
    no_ctx_data_points: int,
    no_of_model_params: int,
) -> int:
    """
    Calculate the physical batch size.

    Args:
        mem_available_gb (float): Available memory in GB.
        no_of_records (int): Number of records in the training dataset.
        no_tgt_data_points (int): Number of target data points per sample.
        no_ctx_data_points (int): Number of context data points per sample.
        no_of_model_params (int): Number of model parameters.

    Returns:
        Batch size (int)
    """
    data_points = no_tgt_data_points + no_ctx_data_points
    min_batch_size = 8
    # scale batch_size corresponding to available memory
    if mem_available_gb >= 32:
        mem_scale = 2.0
    elif mem_available_gb >= 8:
        mem_scale = 1.0
    else:
        mem_scale = 0.5
    # set max_batch_size corresponding to available memory, model params and data points
    if no_of_model_params > 1_000_000_000 or data_points > 100_000:
        max_batch_size = int(8 * mem_scale)
    elif no_of_model_params > 100_000_000 or data_points > 10_000:
        max_batch_size = int(32 * mem_scale)
    elif no_of_model_params > 10_000_000 or data_points > 1_000:
        max_batch_size = int(128 * mem_scale)
    elif no_of_model_params > 1_000_000 or data_points > 100:
        max_batch_size = int(512 * mem_scale)
    else:
        max_batch_size = int(2048 * mem_scale)
    # ensure a minimum number of batches to avoid excessive padding
    min_batches = 64
    batch_size = 2 ** int(np.log2(no_of_records / min_batches)) if no_of_records > 0 else min_batch_size
    return int(np.clip(a=batch_size, a_min=min_batch_size, a_max=max_batch_size))


def _learn_rate_heuristic(batch_size: int) -> float:
    learn_rate = np.round(0.001 * np.sqrt(batch_size / 32), 5)
    return learn_rate


####################
### DATA LOADERS ###
####################


class BatchCollator:
    """
    Collate a batch of samples into a dictionary of tensors.
    For sequence data, it will sample subsequences with lengths up to max_sequence_window.
    """

    def __init__(
        self,
        is_sequential: bool,
        max_sequence_window: int | None,
        device: torch.device,
        *,
        use_nested_ctxseq: bool = True,
    ):
        self.is_sequential = is_sequential
        self.max_sequence_window = max_sequence_window
        self.device = device
        # Opacus per-sample gradients do not support NestedTensor on CPU/CUDA; use padded
        # dense tensors for CTXSEQ when training with DP (see test_tabular_sequential DP path).
        self.use_nested_ctxseq = use_nested_ctxseq

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch = pd.DataFrame(batch)
        if self.is_sequential and self.max_sequence_window:
            batch = self._slice_sequences(batch, self.max_sequence_window)
        batch = self._convert_to_tensors(batch)
        return batch

    def _convert_to_tensors(self, batch: pd.DataFrame) -> dict[str, torch.Tensor]:
        tensors = {}
        for column in batch.columns:
            if column.startswith(TGT) and self.is_sequential:
                # construct column tensor in single step
                tensors[column] = torch.unsqueeze(
                    torch.tensor(
                        # pad batch-wise to the longest sequence length with 0s
                        np.array(list(zip_longest(*batch[column], fillvalue=0))).T,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    dim=-1,
                )
            elif column.startswith(TGT) and not self.is_sequential:
                # construct column tensor in single step
                tensors[column] = torch.unsqueeze(
                    torch.tensor(batch[column].values, dtype=torch.int64, device=self.device),
                    dim=-1,
                )
            elif column.startswith(CTXFLT):
                # construct column tensor in single step
                tensors[column] = torch.unsqueeze(
                    torch.tensor(batch[column].values, dtype=torch.int64, device=self.device),
                    dim=-1,
                )
            elif column.startswith(CTXSEQ):
                if self.use_nested_ctxseq:
                    # construct row tensors and convert the list to nested column tensor
                    tensors[column] = torch.unsqueeze(
                        torch.nested.as_nested_tensor(
                            [torch.tensor(row, dtype=torch.int64, device=self.device) for row in batch[column]],
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        dim=-1,
                    )
                else:
                    # padded batch (variable-length rows); -1 marks padding (matches SequentialContextEmbedders)
                    tensors[column] = torch.unsqueeze(
                        torch.tensor(
                            np.array(list(zip_longest(*batch[column], fillvalue=-1))).T,
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        dim=-1,
                    )
        return tensors

    @staticmethod
    def _slice_sequences(batch: pd.DataFrame, max_sequence_window: int) -> pd.DataFrame:
        # we pad sequences with one step
        # thus, to respect the max_sequence_window provided by the user, we need to add 1 to it
        max_sequence_window += 1
        # determine sequence lengths of current batch
        tgt_columns = [col for col in batch.columns if col.startswith(TGT)]
        seq_lens = batch[tgt_columns[0]].copy().str.len().values

        # determine sampling logic for current batch
        flip = np.random.random()
        if flip < 0.3:  # 30%
            # pick start of the sequence to focus on the beginning
            sel_idxs = [np.arange(0, min(max_sequence_window, seq_len)) for seq_len in seq_lens]
        elif 0.3 <= flip < 0.4:  # 10%
            # pick end of the sequence to focus on the end
            sel_idxs = [np.arange(max(0, seq_len - max_sequence_window), seq_len) for seq_len in seq_lens]
        else:  # 60%
            # random continuous window to focus on any part
            start_idxs = np.random.randint(low=1 - max_sequence_window, high=seq_lens, size=len(seq_lens))
            # ensure that sequences that fit into max_sequence_length are completely covered
            start_idxs[seq_lens <= max_sequence_window] = 0
            # calculate final start and end indexes
            end_idxs = start_idxs + max_sequence_window
            start_idxs = np.maximum(0, start_idxs)
            sel_idxs = [
                np.arange(start, min(seq_len, end)) for start, end, seq_len in zip(start_idxs, end_idxs, seq_lens)
            ]

        # loop over each record within batch and pick values for each tgt column
        tgt_col_idxs = [batch.columns.get_loc(c) for c in tgt_columns]
        rows = []
        for row_idx, batch_row in enumerate(batch.itertuples(index=False)):
            cells = []
            for col_idx, batch_cell in enumerate(batch_row):
                if col_idx in tgt_col_idxs:
                    cells.append([batch_cell[i] for i in sel_idxs[row_idx]])
                else:
                    cells.append(batch_cell)
            rows.append(cells)

        return pd.DataFrame(rows, columns=batch.columns, index=batch.index)


#####################
### TRAINING LOOP ###
#####################


class TabularModelCheckpoint(ModelCheckpoint):
    def _save_model_weights(self, model: torch.nn.Module):
        if isinstance(model, GradSampleModule):
            state_dict = model._module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, self.workspace.model_tabular_weights_path)

    def _clear_model_weights(self) -> None:
        self.workspace.model_tabular_weights_path.unlink(missing_ok=True)

    def model_weights_path_exists(self) -> bool:
        return self.workspace.model_tabular_weights_path.exists()


def _calculate_sample_losses(
    model: FlatModel | SequentialModel | GradSampleModule, data: dict[str, torch.Tensor]
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
        output, _ = model(data, mode="trn")
    criterion = nn.CrossEntropyLoss(reduction="none")

    tgt_cols = (
        list(model.tgt_cardinalities.keys())
        if not isinstance(model, GradSampleModule)
        else model._module.tgt_cardinalities.keys()
    )
    if isinstance(model, SequentialModel) or (
        isinstance(model, GradSampleModule) and isinstance(model._module, SequentialModel)
    ):
        sidx_cols = {k for k in data if k.startswith(SIDX_SUB_COLUMN_PREFIX)}
        slen_cols = {k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
        ridx_cols = [k for k in data if k.startswith(RIDX_SUB_COLUMN_PREFIX)]

        # mask for data columns
        data_mask = torch.zeros_like(data[ridx_cols[0]], dtype=torch.int64)
        for ridx_col in ridx_cols:
            data_mask |= data[ridx_col] != 0  # mask loss for padded rows, which have RIDX=0
        data_mask = data_mask.squeeze(-1)
        # mask for slen columns; only first step is unmasked
        slen_mask = torch.zeros_like(data_mask)
        slen_mask[:, 0] = 1
        # mask for ridx columns: this takes the sequence padding into account to learn the stopping with ridx=0
        ridx_mask = torch.nn.functional.pad(data_mask, (1, 0), value=1)[:, :-1]
        # mask for sidx columns
        sidx_mask = torch.zeros_like(data_mask)

        # calculate per column losses
        losses_by_column = []
        for col in tgt_cols:
            if col in sidx_cols:
                mask = sidx_mask
            elif col in slen_cols:
                mask = slen_mask
            elif col in ridx_cols:
                mask = ridx_mask
            else:
                mask = data_mask
            column_loss = criterion(output[col].transpose(1, 2), data[col].squeeze(2))
            masked_loss = torch.sum(column_loss * mask, dim=1) / torch.clamp(torch.sum(mask), min=1)
            losses_by_column.append(masked_loss)
    else:
        losses_by_column = [criterion(output[col], data[col].squeeze(1)) for col in tgt_cols]
    # sum up column level losses to get overall losses at sample level
    losses = torch.sum(torch.stack(losses_by_column, dim=0), dim=0)
    return losses


# gradient tracking is not needed for validation steps, disable it to save memory
@torch.no_grad()
def _calculate_val_loss(
    model: FlatModel | SequentialModel,
    val_dataloader: DataLoader,
) -> float:
    val_sample_losses: list[torch.Tensor] = []
    model.eval()
    for step_data in val_dataloader:
        step_losses = _calculate_sample_losses(model, step_data)
        val_sample_losses.extend(step_losses.detach())
    model.train()
    val_sample_losses: torch.Tensor = torch.stack(val_sample_losses, dim=0)
    val_loss_avg = torch.mean(val_sample_losses).item()
    return val_loss_avg


def _calculate_average_trn_loss(trn_sample_losses: list[torch.Tensor], n: int | None = None) -> float | None:
    if len(trn_sample_losses) == 0:
        return None
    trn_losses_latest = torch.stack(trn_sample_losses, dim=0)
    if n is not None:
        trn_losses_latest = trn_losses_latest[-n:]
    trn_loss = torch.mean(trn_losses_latest).item()
    return trn_loss


################
### TRAINING ###
################


_MODEL_SIZES = {
    "MOSTLY_AI/Small": ModelSize.S,
    "MOSTLY_AI/Medium": ModelSize.M,
    "MOSTLY_AI/Large": ModelSize.L,
}


class Trainer:
    """Per-call tabular trainer.

    Each invocation of the module-level ``train()`` constructs a fresh ``Trainer``.
    No Python-level state is intended to persist across calls. All disk reads /
    writes happen inside ``_setup()`` and the public ``train()`` / ``validate_only()``
    methods so that the constructor stays side-effect free.
    """

    def __init__(
        self,
        *,
        model: str = "MOSTLY_AI/Medium",
        max_training_time: float = 14400.0,
        max_epochs: float = 100.0,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        max_sequence_window: int = 100,
        enable_flexible_generation: bool = True,
        differential_privacy: DifferentialPrivacyConfig | dict | None = None,
        upload_model_data_callback: Callable | None = None,
        model_state_strategy: ModelStateStrategy | str = ModelStateStrategy.reset,
        device: torch.device | str | None = None,
        workspace_dir: str | Path = "engine-ws",
        update_progress: ProgressCallback | None = None,
        federated_epochs: int | None = None,
        federated_state: dict | None = None,
        validate_only: bool = False,
        fixed_learning_rate: float | None = None,
    ):
        # raw configuration
        self.model_id = model
        if self.model_id not in _MODEL_SIZES:
            raise ValueError(f"model {self.model_id} not supported")
        self.model_size = _MODEL_SIZES[self.model_id]
        self.max_training_time = max(0.0, max_training_time) * 60  # seconds
        self.max_epochs = max(0.0, max_epochs)
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_sequence_window = max_sequence_window
        self.enable_flexible_generation = enable_flexible_generation
        self.differential_privacy = differential_privacy
        self.with_dp = differential_privacy is not None
        self.upload_model_data_callback = upload_model_data_callback or (lambda *args, **kwargs: None)
        if isinstance(model_state_strategy, str):
            model_state_strategy = ModelStateStrategy(model_state_strategy)
        self.model_state_strategy = model_state_strategy
        self.device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        self.workspace = Workspace(ensure_workspace_dir(workspace_dir))
        self.update_progress = update_progress
        self.federated_epochs = federated_epochs
        self.federated_state = federated_state
        self.validate_only_flag = validate_only
        self.fixed_learning_rate = fixed_learning_rate

        # placeholders (filled in _setup)
        self.model: SequentialModel | FlatModel | GradSampleModule | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: LRScheduler | None = None
        self.privacy_engine: PrivacyEngine | None = None
        self.trn_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.model_checkpoint: TabularModelCheckpoint | None = None
        self.early_stopper: EarlyStopper | None = None
        self.dp_total_delta = None
        self.dp_value_protection_epsilon = None
        self.dp_max_epsilon = None
        self.dp_config = None

        # state derived in _load_stats / _compute_training_params
        self.tgt_stats = None
        self.ctx_stats = None
        self.has_context = False
        self.is_sequential = False
        self.trn_cnt = 0
        self.val_cnt = 0
        self.tgt_cardinalities = {}
        self.ctx_cardinalities = {}
        self.trn_column_order = None
        self.tgt_seq_len_median = None
        self.tgt_seq_len_max = None
        self.ctx_seq_len_median = None
        self.empirical_probs_for_predictor_init = None
        self.trn_batch_size = None
        self.val_batch_size = None
        self.trn_steps = None
        self.val_steps = None
        self.initial_lr = None

        # counters
        self.epoch = 0.0
        self.steps = 0
        self.samples = 0
        self.current_lr: float | None = None
        self.val_loss: float | None = None
        self.total_time_init = 0.0
        self.early_exit = False
        self._is_setup = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup(self, for_validation_only: bool = False) -> None:
        """Build everything required for a single train/validate call.

        Order is critical:
          1. process-level torch settings
          2. federated reset-on-state precedence
          3. _load_stats -> early-exit check
          4. _compute_training_params
          5. _build_model
          6. _restore_state_from_disk_or_federated
          7. _build_dataloaders
          8. (training only) _build_optimizer -> _setup_differential_privacy -> _apply_fixed_learning_rate
        """
        if self._is_setup:
            return

        # 1. process-level torch settings (must be re-applied each call)
        torch.set_default_dtype(torch.float32)
        torch.set_flush_denormal(True)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # 2. federated reset-on-state precedence
        if self.federated_state is not None:
            self.model_state_strategy = ModelStateStrategy.reset

        _LOG.info(f"numpy={version('numpy')}, pandas={version('pandas')}")
        _LOG.info(f"torch={version('torch')}, opacus={version('opacus')}")
        _LOG.info(f"device={self.device}")
        _LOG.info(f"model_size={self.model_size}")
        _LOG.info(f"with_dp={self.with_dp}")
        _LOG.info(f"model_state_strategy={self.model_state_strategy}")

        # 3. load stats + early-exit check
        self._load_stats()
        if for_validation_only:
            val_files = self.workspace.encoded_data_val.fetch_all()
            self.early_exit = len(val_files) == 0 or self.val_cnt == 0
        else:
            self.early_exit = check_early_training_exit(
                workspace=self.workspace, trn_cnt=self.trn_cnt, val_cnt=self.val_cnt
            )
        if self.early_exit:
            _LOG.warning("not enough data; early exit")
            self._is_setup = True
            return

        # 4. compute training params (returns heuristic inputs for training; None for validate-only)
        heuristic_inputs = self._compute_training_params(for_validation_only=for_validation_only)

        # 5. build model (writes model_configs only when not validate-only)
        self._build_model(for_validation_only=for_validation_only)

        # 6. finalize batch sizes / steps / initial LR (training only)
        if not for_validation_only:
            no_of_model_params = get_no_of_model_parameters(
                self.model._module if isinstance(self.model, GradSampleModule) else self.model
            )
            mem_available_gb, no_tgt_data_points, no_ctx_data_points = heuristic_inputs
            self._finalize_training_params(
                no_of_model_params=no_of_model_params,
                mem_available_gb=mem_available_gb,
                no_tgt_data_points=no_tgt_data_points,
                no_ctx_data_points=no_ctx_data_points,
            )

        # 7. restore state (model weights only, from federated_state or disk)
        self._restore_state_from_disk_or_federated(for_validation_only=for_validation_only)

        # 8. dataloaders
        self._build_dataloaders(for_validation_only=for_validation_only)

        # 9. optimizer / DP / fixed LR (training only).
        # Order matters: _build_optimizer must run before _setup_differential_privacy
        # (Opacus wraps the optimizer), and _apply_fixed_learning_rate must run last so
        # the pinned LR survives DP wrapping.
        if not for_validation_only:
            self._build_optimizer()
            if self.with_dp:
                self._setup_differential_privacy()
            if self.fixed_learning_rate is not None:
                self._apply_fixed_learning_rate()

        self._is_setup = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_stats(self) -> None:
        ws = self.workspace
        self.has_context = ws.ctx_stats.path.exists()
        self.tgt_stats = ws.tgt_stats.read()
        self.ctx_stats = ws.ctx_stats.read()
        self.is_sequential = self.tgt_stats["is_sequential"]
        self.trn_cnt = self.tgt_stats["no_of_training_records"]
        self.val_cnt = self.tgt_stats["no_of_validation_records"]
        self.tgt_cardinalities = get_cardinalities(self.tgt_stats)
        self.ctx_cardinalities = get_cardinalities(self.ctx_stats) if self.has_context else {}

        if self.enable_flexible_generation:
            self.trn_column_order = None
        else:
            self.trn_column_order = get_columns_from_cardinalities(self.tgt_cardinalities)

        tgt_seq_len_stats = get_sequence_length_stats(self.tgt_stats)
        self.tgt_seq_len_median = tgt_seq_len_stats["median"]
        self.tgt_seq_len_max = tgt_seq_len_stats["max"]
        self.max_sequence_window = int(np.clip(self.max_sequence_window, a_min=1, a_max=self.tgt_seq_len_max))
        self.ctx_seq_len_median = get_ctx_sequence_length(self.ctx_stats, key="median")
        _LOG.info(f"is_sequential={self.is_sequential}, trn_cnt={self.trn_cnt}, val_cnt={self.val_cnt}")
        _LOG.info(f"max_sequence_window={self.max_sequence_window}")
        # observability: log cardinality deciles when available (matches pre-refactor logging)
        if len(self.tgt_cardinalities) > 0:
            tgt_card_values = np.array(list(self.tgt_cardinalities.values()))
            tgt_cardinalities_deciles = np.quantile(tgt_card_values, np.linspace(0, 1, 11)).tolist()
            _LOG.info(f"tgt_cardinalities_deciles={tgt_cardinalities_deciles}")
        if len(self.ctx_cardinalities) > 0:
            ctx_card_values = np.array(list(self.ctx_cardinalities.values()))
            ctx_cardinalities_deciles = np.quantile(ctx_card_values, np.linspace(0, 1, 11)).tolist()
            _LOG.info(f"ctx_cardinalities_deciles={ctx_cardinalities_deciles}")

    def _compute_training_params(self, for_validation_only: bool = False) -> tuple[float, int, int] | None:
        """Compute heuristic inputs that do not depend on `no_of_model_params`.

        For training: returns (mem_available_gb, no_tgt_data_points, no_ctx_data_points)
        and pre-computes `empirical_probs_for_predictor_init` for model construction.
        For validation-only: computes only `val_batch_size` / `val_steps` and returns None.
        """
        if for_validation_only:
            base_bs = self.batch_size if self.batch_size is not None else 32
            self.val_batch_size = max(1, min(base_bs, self.val_cnt))
            self.val_steps = max(1, self.val_cnt // self.val_batch_size)
            if self.is_sequential:
                self.val_batch_size = max(1, self.val_batch_size // 2)
            return None

        # empirical_probs only used for model construction (training path)
        if not self.with_dp:
            trn_files = self.workspace.encoded_data_trn.fetch_all()
            if trn_files:
                self.empirical_probs_for_predictor_init = get_empirical_probs_for_predictor_init(
                    trn_files[0], self.tgt_cardinalities, self.is_sequential
                )

        mem_available_gb = get_available_ram_for_heuristics() / 1024**3
        no_tgt_data_points = get_max_data_points_per_sample(self.tgt_stats)
        no_ctx_data_points = get_max_data_points_per_sample(self.ctx_stats)
        return mem_available_gb, no_tgt_data_points, no_ctx_data_points

    def _build_model(self, for_validation_only: bool = False) -> None:
        model_kwargs = {
            "tgt_cardinalities": self.tgt_cardinalities,
            "ctx_cardinalities": self.ctx_cardinalities,
            "ctxseq_len_median": self.ctx_seq_len_median,
            "model_size": self.model_size,
            "column_order": self.trn_column_order,
            "device": self.device,
            "with_dp": self.with_dp,
            "empirical_probs_for_predictor_init": self.empirical_probs_for_predictor_init,
        }
        if self.is_sequential:
            argn = SequentialModel(
                **model_kwargs,
                tgt_seq_len_median=self.tgt_seq_len_median,
                tgt_seq_len_max=self.tgt_seq_len_max,
            )
        else:
            argn = FlatModel(**model_kwargs)
        argn.to(self.device)
        self.model = argn
        no_of_model_params = get_no_of_model_parameters(argn)
        _LOG.info(f"model class: {argn.__class__.__name__}")
        _LOG.info(f"no_of_model_params={no_of_model_params}")

        # persist model_configs (idempotent) only when actually training; validate_only
        # must be read-only against the workspace in federated contexts.
        if not for_validation_only:
            model_units = get_model_units(argn)
            model_configs = {
                "model_id": self.model_id,
                "model_units": model_units,
                "enable_flexible_generation": self.enable_flexible_generation,
            }
            self.workspace.model_configs.write(model_configs)

    def _finalize_training_params(
        self,
        no_of_model_params: int,
        mem_available_gb: float,
        no_tgt_data_points: int,
        no_ctx_data_points: int,
    ) -> None:
        """Finalize batch sizes / steps / initial LR (training-only).

        Order matches pre-refactor behavior: compute val_steps against the
        pre-halved val_batch_size, then halve val_batch_size for sequential models.
        """
        if self.batch_size is None:
            self.batch_size = _physical_batch_size_heuristic(
                mem_available_gb=mem_available_gb,
                no_of_records=self.trn_cnt,
                no_tgt_data_points=no_tgt_data_points,
                no_ctx_data_points=no_ctx_data_points,
                no_of_model_params=no_of_model_params,
            )
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = 1
        self.batch_size = max(1, min(self.batch_size, self.trn_cnt))
        self.gradient_accumulation_steps = max(
            1, min(self.gradient_accumulation_steps, self.trn_cnt // self.batch_size)
        )
        self.trn_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.trn_steps = max(1, self.trn_cnt // self.trn_batch_size)
        self.val_batch_size = max(1, min(self.batch_size, self.val_cnt))
        self.val_steps = max(1, self.val_cnt // self.val_batch_size)
        if self.initial_lr is None:
            self.initial_lr = _learn_rate_heuristic(self.trn_batch_size)
        # halve val_batch_size last (after val_steps is computed) for sequential models;
        # this matches pre-refactor order.
        if self.is_sequential:
            self.val_batch_size = max(1, self.val_batch_size // 2)
        _LOG.info(f"trn_batch_size={self.trn_batch_size}, val_batch_size={self.val_batch_size}")
        _LOG.info(f"trn_steps={self.trn_steps}, val_steps={self.val_steps}")
        _LOG.info(
            f"batch_size={self.batch_size}, "
            f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
            f"initial_lr={self.initial_lr}"
        )

    def _restore_state_from_disk_or_federated(self, for_validation_only: bool = False) -> None:
        self.model_checkpoint = TabularModelCheckpoint(workspace=self.workspace)

        # Federated path: load weights only from federated_state; never touch disk for state.
        if self.federated_state is not None and self.federated_state.get("model_weights") is not None:
            _LOG.info("loading model weights from federated_state")
            weights = self.federated_state["model_weights"]
            # weights may be numpy arrays (from federated round-trip) or tensors
            state_dict = {}
            for k, v in weights.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v
                else:
                    state_dict[k] = torch.as_tensor(np.array(v))
            target = self.model._module if isinstance(self.model, GradSampleModule) else self.model
            target.load_state_dict(state_dict)
            return

        # Disk path
        if not self.model_checkpoint.model_weights_path_exists() and self.federated_state is None:
            _LOG.info(f"model weights not found on disk; forcing strategy from {self.model_state_strategy} to RESET")
            self.model_state_strategy = ModelStateStrategy.reset

        if for_validation_only:
            # In validation mode, only attempt to load weights from disk if explicitly resume/reuse;
            # never clear or touch other checkpoint files.
            if (
                self.model_state_strategy in (ModelStateStrategy.resume, ModelStateStrategy.reuse)
                and self.model_checkpoint.model_weights_path_exists()
            ):
                torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
                load_model_weights(
                    model=self.model,
                    path=self.workspace.model_tabular_weights_path,
                    device=self.device,
                )
            return

        if (
            self.model_state_strategy in (ModelStateStrategy.resume, ModelStateStrategy.reuse)
            and self.federated_state is None
        ):
            _LOG.info("load existing model weights from disk")
            torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
            load_model_weights(
                model=self.model,
                path=self.workspace.model_tabular_weights_path,
                device=self.device,
            )
        else:
            _LOG.info("remove existing checkpoint files")
            self.model_checkpoint.clear_checkpoint()

    def _build_dataloaders(self, for_validation_only: bool = False) -> None:
        batch_collator = BatchCollator(
            is_sequential=self.is_sequential,
            max_sequence_window=self.max_sequence_window,
            device=self.device,
            use_nested_ctxseq=not self.with_dp,
        )
        disable_progress_bar()
        val_files = self.workspace.encoded_data_val.fetch_all()
        val_dataset = load_dataset("parquet", data_files=[str(p) for p in val_files])["train"]
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=self.val_batch_size,
            collate_fn=batch_collator,
        )
        if for_validation_only:
            return
        trn_files = self.workspace.encoded_data_trn.fetch_all()
        trn_dataset = load_dataset("parquet", data_files=[str(p) for p in trn_files])["train"]
        self.trn_dataloader = DataLoader(
            dataset=trn_dataset,
            shuffle=True,
            batch_size=self.trn_batch_size if self.with_dp else self.batch_size,
            collate_fn=batch_collator,
        )

    def _build_optimizer(self) -> None:
        # Counters (epoch/steps/samples/total_time_init) are restored from the progress CSV by
        # Trainer.train() when applicable. initial_lr stays at the heuristic value unless
        # explicitly overridden by a resumed progress message there.
        self.early_stopper = EarlyStopper(val_loss_patience=4)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.initial_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=2,
            min_lr=0.1 * self.initial_lr,
        )
        # restore optimizer / lr_scheduler from disk only when resuming centrally
        if (
            self.federated_state is None
            and self.model_state_strategy == ModelStateStrategy.resume
            and self.model_checkpoint.optimizer_and_lr_scheduler_paths_exist()
        ):
            _LOG.info("restore optimizer and LR scheduler states from disk")
            self.optimizer.load_state_dict(
                torch.load(self.workspace.model_optimizer_path, map_location=self.device, weights_only=True)
            )
            self.lr_scheduler.load_state_dict(
                torch.load(self.workspace.model_lr_scheduler_path, map_location=self.device, weights_only=True)
            )
        self.current_lr = self.initial_lr

    def _setup_differential_privacy(self) -> None:
        if isinstance(self.differential_privacy, DifferentialPrivacyConfig):
            self.dp_config = self.differential_privacy.model_dump()
        else:
            self.dp_config = DifferentialPrivacyConfig(**self.differential_privacy).model_dump()
        self.dp_max_epsilon = self.dp_config.get("max_epsilon") or float("inf")
        self.dp_total_delta = self.dp_config.get("delta", 1e-5)
        self.dp_value_protection_epsilon = (self.ctx_stats.get("value_protection_epsilon_spent") or 0.0) + (
            self.tgt_stats.get("value_protection_epsilon_spent") or 0.0
        )
        dp_accountant = "rdp"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Secure RNG turned off*")
            self.privacy_engine = PrivacyEngine(accountant=dp_accountant)

        # restore accountant from federated_state, else from disk if resume
        if self.federated_state is not None and self.federated_state.get("dp_accountant_state") is not None:
            _LOG.info("restore DP accountant state from federated_state")
            torch.serialization.add_safe_globals([getattr, PRVAccountant, RDPAccountant, GaussianAccountant])
            self.privacy_engine.accountant.load_state_dict(self.federated_state["dp_accountant_state"])
        elif (
            self.federated_state is None
            and self.model_state_strategy == ModelStateStrategy.resume
            and self.workspace.model_dp_accountant_path.exists()
        ):
            _LOG.info("restore DP accountant state from disk")
            torch.serialization.add_safe_globals([getattr, PRVAccountant, RDPAccountant, GaussianAccountant])
            self.privacy_engine.accountant.load_state_dict(
                torch.load(self.workspace.model_dp_accountant_path, map_location=self.device, weights_only=True)
            )
        # Opacus wraps model/optimizer/dataloader
        self.model, self.optimizer, self.trn_dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trn_dataloader,
            noise_multiplier=self.dp_config.get("noise_multiplier"),
            max_grad_norm=self.dp_config.get("max_grad_norm"),
            poisson_sampling=True,
        )
        self.trn_dataloader = wrap_data_loader(
            data_loader=self.trn_dataloader, max_batch_size=self.batch_size, optimizer=self.optimizer
        )

    def _apply_fixed_learning_rate(self) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.fixed_learning_rate
        self.current_lr = self.fixed_learning_rate
        _LOG.info(f"fixed_learning_rate={self.fixed_learning_rate}: local LR scheduler will be skipped this round")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _calculate_val_loss_internal(self) -> float:
        return _calculate_val_loss(model=self.model, val_dataloader=self.val_dataloader)

    def _run_training_loop(self, progress) -> None:
        progress_message = None
        start_trn_time = time.time()
        last_msg_time = time.time()
        trn_data_iter = iter(self.trn_dataloader)
        trn_sample_losses: list[torch.Tensor] = []
        do_stop = False
        skip_scheduler = self.fixed_learning_rate is not None

        while not do_stop:
            is_checkpoint = 0
            self.steps += 1
            self.epoch = self.steps / self.trn_steps

            stop_accumulating_grads = False
            accumulated_steps = 0
            if not self.with_dp:
                self.optimizer.zero_grad(set_to_none=True)
            while not stop_accumulating_grads:
                try:
                    step_data = next(trn_data_iter)
                except StopIteration:
                    trn_data_iter = iter(self.trn_dataloader)
                    step_data = next(trn_data_iter)
                step_losses = _calculate_sample_losses(self.model, step_data)
                step_loss = torch.mean(step_losses) / (1 if self.with_dp else self.gradient_accumulation_steps)
                if self.with_dp:
                    self.optimizer.zero_grad(set_to_none=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
                    if self.with_dp:
                        warnings.filterwarnings("ignore", category=UserWarning, message="Full backward hook is firing*")
                    step_loss.backward()
                accumulated_steps += 1
                self.samples += step_losses.shape[0]
                if self.with_dp:
                    self.optimizer.step()
                    stop_accumulating_grads = not self.optimizer._is_last_step_skipped
                elif accumulated_steps % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    stop_accumulating_grads = True
                step_losses = step_losses.detach()
                trn_sample_losses.extend(step_losses)

            if skip_scheduler:
                # LR is pinned by _apply_fixed_learning_rate(); local scheduler never steps.
                pass
            else:
                self.current_lr = self.optimizer.param_groups[0]["lr"]
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

            do_validation = on_epoch_end = self.epoch.is_integer()
            if do_validation:
                self.val_loss = self._calculate_val_loss_internal()
                if pd.isna(self.val_loss):
                    _LOG.warning("validation loss NaN - reload last checkpoint")
                    load_model_weights(
                        model=self.model,
                        path=self.workspace.model_tabular_weights_path,
                        device=self.device,
                    )
                trn_loss = _calculate_average_trn_loss(trn_sample_losses)
                dp_total_epsilon = (
                    self.privacy_engine.get_epsilon(self.dp_total_delta) + self.dp_value_protection_epsilon
                    if self.with_dp
                    else None
                )
                has_exceeded_dp_max_epsilon = dp_total_epsilon > self.dp_max_epsilon if self.with_dp else False
                if not has_exceeded_dp_max_epsilon:
                    is_checkpoint = self.model_checkpoint.save_checkpoint_if_best(
                        val_loss=self.val_loss,
                        model=self.model,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        dp_accountant=self.privacy_engine.accountant if self.with_dp else None,
                    )
                else:
                    _LOG.info("early stopping: DP epsilon exceeded max epsilon")
                progress_message = ProgressMessage(
                    epoch=self.epoch,
                    is_checkpoint=is_checkpoint,
                    steps=self.steps,
                    samples=self.samples,
                    trn_loss=trn_loss,
                    val_loss=self.val_loss,
                    total_time=self.total_time_init + time.time() - start_trn_time,
                    learn_rate=self.current_lr,
                    dp_eps=dp_total_epsilon,
                    dp_delta=self.dp_total_delta,
                )
                do_stop = self.early_stopper(val_loss=self.val_loss) or has_exceeded_dp_max_epsilon
                if not skip_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(metrics=self.val_loss)

            elapsed = time.time() - start_trn_time
            estimated = (self.max_epochs * self.trn_steps) * (elapsed / self.steps)
            if self.max_training_time < estimated:
                progress_total_count = self.max_training_time
                progress_processed = elapsed
            else:
                progress_total_count = self.max_epochs * self.trn_steps
                progress_processed = self.steps
            last_msg_interval = 5 * 60
            last_msg_elapsed = time.time() - last_msg_time
            if progress_message is None and (last_msg_elapsed > last_msg_interval or self.steps == 1):
                running_trn_loss = _calculate_average_trn_loss(
                    trn_sample_losses, n=self.val_steps * self.val_batch_size
                )
                dp_total_epsilon = (
                    self.privacy_engine.get_epsilon(self.dp_total_delta) + self.dp_value_protection_epsilon
                    if self.with_dp
                    else None
                )
                progress_message = ProgressMessage(
                    epoch=self.epoch,
                    is_checkpoint=is_checkpoint,
                    steps=self.steps,
                    samples=self.samples,
                    trn_loss=running_trn_loss,
                    val_loss=None,
                    total_time=self.total_time_init + time.time() - start_trn_time,
                    learn_rate=self.current_lr,
                    dp_eps=dp_total_epsilon,
                    dp_delta=self.dp_total_delta,
                )
            if progress_message:
                last_msg_time = time.time()
            res = progress.update(
                completed=int(progress_processed),
                total=int(progress_total_count),
                message=progress_message,
            )
            if do_validation:
                self.upload_model_data_callback()
            progress_message = None
            if (res or {}).get("stopExecution", False):
                _LOG.info("received STOP EXECUTION signal")
                do_stop = True

            if on_epoch_end:
                trn_sample_losses = []

            epoch_limit = self.federated_epochs if self.federated_epochs is not None else self.max_epochs
            if self.epoch >= epoch_limit:
                do_stop = True

            total_training_time = self.total_time_init + time.time() - start_trn_time
            if total_training_time > self.max_training_time:
                do_stop = True

        # save final if no checkpoint saved yet
        self._save_final_if_needed(progress, start_trn_time, trn_sample_losses)
        # store last trn_loss for federated_state
        self._final_trn_loss = _calculate_average_trn_loss(trn_sample_losses)

    def _save_final_if_needed(self, progress, start_trn_time, trn_sample_losses) -> None:
        if self.model_checkpoint.has_saved_once():
            return
        _LOG.info("saving model weights, as none were saved so far")
        self.model_checkpoint.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            dp_accountant=self.privacy_engine.accountant if self.with_dp else None,
        )
        total_training_time = self.total_time_init + time.time() - start_trn_time
        if total_training_time > self.max_training_time:
            _LOG.info("skip validation loss calculation due to time-capped early stopping")
            self.val_loss = None
        else:
            _LOG.info("calculate validation loss")
            self.val_loss = self._calculate_val_loss_internal()
        dp_total_epsilon = (
            self.privacy_engine.get_epsilon(self.dp_total_delta) + self.dp_value_protection_epsilon
            if self.with_dp
            else None
        )
        trn_loss = _calculate_average_trn_loss(trn_sample_losses)
        progress_message = ProgressMessage(
            epoch=self.epoch,
            is_checkpoint=1,
            steps=self.steps,
            samples=self.samples,
            trn_loss=trn_loss,
            val_loss=self.val_loss,
            total_time=total_training_time,
            learn_rate=self.current_lr,
            dp_eps=dp_total_epsilon,
            dp_delta=self.dp_total_delta,
        )
        progress.update(completed=self.steps, total=self.steps, message=progress_message)
        self.upload_model_data_callback()

    def _build_federated_state_dict(self, *, validate_only_mode: bool = False) -> dict:
        module = self.model._module if isinstance(self.model, GradSampleModule) else self.model
        model_weights = {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}

        if validate_only_mode:
            final_trn_loss = None
            epoch = 0.0
            steps = 0
            samples = 0
            learn_rate = None
        else:
            final_trn_loss = getattr(self, "_final_trn_loss", None)
            epoch = self.epoch
            steps = self.steps
            samples = self.samples
            learn_rate = self.current_lr

        state = {
            "model_weights": model_weights,
            "training_metrics": {
                "epoch": epoch,
                "steps": steps,
                "samples": samples,
                "learn_rate": learn_rate,
                "trn_loss": final_trn_loss,
                "val_loss": self.val_loss,
            },
        }
        if self.with_dp and self.privacy_engine is not None:
            state["dp_accountant_state"] = self.privacy_engine.accountant.state_dict()
        elif "dp_accountant_state" in (self.federated_state or {}):
            # validate_only path with DP: pass through unchanged
            state["dp_accountant_state"] = self.federated_state["dp_accountant_state"]
        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> dict | None:
        _LOG.info("TRAIN_TABULAR started")
        t0 = time.time()
        with ProgressCallbackWrapper(
            self.update_progress, progress_messages_path=self.workspace.model_progress_messages_path
        ) as progress:
            # Federated reset-on-state precedence (applied here *before* any disk peek so
            # the resume CSV is never consulted when a federated_state was supplied;
            # _setup() repeats this check as defense-in-depth).
            if self.federated_state is not None and self.model_state_strategy != ModelStateStrategy.reset:
                self.model_state_strategy = ModelStateStrategy.reset
            # Peek last progress message *before* setup so that _build_optimizer can pick up
            # the resumed initial_lr (matches pre-refactor behavior). The federated path is
            # already excluded by the precedence above (strategy is now `reset`).
            resume_msg = None
            if self.model_state_strategy == ModelStateStrategy.resume:
                resume_msg = progress.get_last_progress_message()
                if resume_msg:
                    self.epoch = resume_msg.get("epoch", 0.0)
                    self.steps = resume_msg.get("steps", 0)
                    self.samples = resume_msg.get("samples", 0)
                    self.total_time_init = resume_msg.get("total_time", 0.0)
                    lr_from_msg = resume_msg.get("learn_rate", None)
                    if lr_from_msg is not None:
                        self.initial_lr = lr_from_msg
                    _LOG.info(f"start training progress from epoch={self.epoch}, steps={self.steps}")
            self._setup(for_validation_only=False)
            if self.early_exit:
                return None
            # Reset the progress CSV iff no resume counters were applied; this mirrors
            # the pre-refactor behavior (fresh runs start with a clean CSV; resumed runs
            # keep the existing one).
            if resume_msg is None:
                progress.reset_progress_messages()

            self._run_training_loop(progress)
        _LOG.info(f"TRAIN_TABULAR finished in {time.time() - t0:.2f}s")
        if self.federated_epochs is not None:
            return self._build_federated_state_dict(validate_only_mode=False)
        return None

    def validate_only(self) -> dict | None:
        _LOG.info("VALIDATE_ONLY_TABULAR started")
        t0 = time.time()
        with ProgressCallbackWrapper(
            self.update_progress, progress_messages_path=self.workspace.model_progress_messages_path
        ) as _progress:
            # intentionally do NOT call _progress.reset_progress_messages()
            self._setup(for_validation_only=True)
            if self.early_exit:
                return None
            assert self.model is not None, "model must be built for validate_only"
            assert self.val_dataloader is not None, "val_dataloader must be built for validate_only"
            self.val_loss = self._calculate_val_loss_internal()
        _LOG.info(f"VALIDATE_ONLY_TABULAR finished in {time.time() - t0:.2f}s, val_loss={self.val_loss}")
        # Note: model_weights returned here are the coordinator-supplied weights,
        # unchanged modulo the CPU-numpy serialization round-trip.
        return self._build_federated_state_dict(validate_only_mode=True)


@gpu_memory_cleanup
def train(
    *,
    model: str = "MOSTLY_AI/Medium",
    max_training_time: float = 14400.0,
    max_epochs: float = 100.0,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    max_sequence_window: int = 100,
    enable_flexible_generation: bool = True,
    differential_privacy: DifferentialPrivacyConfig | dict | None = None,
    upload_model_data_callback: Callable | None = None,
    model_state_strategy: ModelStateStrategy | str = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    federated_epochs: int | None = None,
    federated_state: dict | None = None,
    validate_only: bool = False,
    fixed_learning_rate: float | None = None,
) -> dict | None:
    """Train a tabular ARGN model, or run validation-only against coordinator weights.

    This is the only public entry point. It instantiates a fresh ``Trainer`` and
    dispatches to either ``trainer.train()`` or ``trainer.validate_only()``.

    Federated invocation contract (workspace ephemerality):

    Each call is fully self-contained. Federated orchestrators (Flower, NVFlare, ...)
    are responsible for re-staging the following artifacts into ``workspace_dir``
    *before* every call:

    * ``tgt_stats`` and ``ctx_stats`` (always required)
    * encoded training parquet files (only required when ``validate_only=False``)
    * encoded validation parquet files (always required)

    The engine assumes nothing else on disk. In particular, when ``federated_state``
    is supplied:

    * model weights are loaded from ``federated_state["model_weights"]`` only
      (never from disk),
    * the DP accountant is restored from ``federated_state["dp_accountant_state"]``
      if present (never from disk),
    * optimizer and LR-scheduler state are not read from disk,
    * ``progress-messages.csv`` is not consulted for resume values,
    * ``model_state_strategy`` is forced to ``reset``.

    With ``validate_only=True``: only val parquet files and stats are required.
    No checkpoint files, training files, optimizer files, scheduler files, DP
    accountant files, or writable progress CSV are needed. The call returns a
    ``federated_state`` dict whose ``model_weights`` are the coordinator-supplied
    weights (unchanged modulo CPU-numpy serialization) and whose
    ``training_metrics.val_loss`` reflects the local validation loss.

    With ``fixed_learning_rate`` set, the local ``ReduceLROnPlateau`` scheduler
    is bypassed for this round; the coordinator decides the next LR globally.
    """
    trainer = Trainer(
        model=model,
        max_training_time=max_training_time,
        max_epochs=max_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_sequence_window=max_sequence_window,
        enable_flexible_generation=enable_flexible_generation,
        differential_privacy=differential_privacy,
        upload_model_data_callback=upload_model_data_callback,
        model_state_strategy=model_state_strategy,
        device=device,
        workspace_dir=workspace_dir,
        update_progress=update_progress,
        federated_epochs=federated_epochs,
        federated_state=federated_state,
        validate_only=validate_only,
        fixed_learning_rate=fixed_learning_rate,
    )
    if validate_only:
        return trainer.validate_only()
    return trainer.train()
