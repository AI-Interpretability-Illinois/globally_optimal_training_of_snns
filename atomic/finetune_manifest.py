"""Checkpoint manifest validation (no CVX / STE imports)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def assert_finetune_manifest_matches_runtime(
    manifest: Mapping[str, Any],
    *,
    readout_mode: str,
    P_in: int,
    P_rec: int,
    P_last: int,
    L: int,
    num_classes: int,
    K_parallel: int = 1,
) -> None:
    if int(manifest["L"]) != int(L):
        raise ValueError(f"Checkpoint L={manifest['L']} does not match runtime L={L}.")
    if int(manifest["P_rec"]) != int(P_rec):
        raise ValueError(f"Checkpoint P_rec={manifest['P_rec']} does not match runtime P_rec={P_rec}.")
    if int(manifest["P_last"]) != int(P_last):
        raise ValueError(f"Checkpoint P_last={manifest['P_last']} does not match runtime P_last={P_last}.")
    if int(manifest["P_in"]) != int(P_in):
        raise ValueError(f"Checkpoint P_in={manifest['P_in']} does not match runtime P_in={P_in}.")
    if int(manifest["num_classes"]) != int(num_classes):
        raise ValueError(
            f"Checkpoint num_classes={manifest['num_classes']} does not match runtime num_classes={num_classes}."
        )
    if str(manifest["readout_mode"]) != str(readout_mode):
        raise ValueError(
            f"Checkpoint readout_mode={manifest['readout_mode']!r} does not match runtime {readout_mode!r}."
        )
    man_k = manifest.get("K_parallel")
    if man_k is None:
        if int(K_parallel) != 1:
            raise ValueError(
                "Checkpoint manifest has no K_parallel field; reloading with K_parallel>1 requires a manifest "
                "written by a current fine_tune export. Legacy checkpoints are treated as K_parallel=1 only."
            )
    elif int(man_k) != int(K_parallel):
        raise ValueError(
            f"Checkpoint K_parallel={man_k} does not match runtime K_parallel={K_parallel}."
        )


def validate_finetune_weight_shapes_against_manifest(manifest: Mapping[str, Any], weights: Sequence[Any]) -> None:
    shapes = manifest.get("snn_weight_shapes")
    if shapes is None:
        return
    if len(shapes) != len(weights):
        raise ValueError(
            f"Manifest snn_weight_shapes has length {len(shapes)} but loaded {len(weights)} weight tensors."
        )
    for i, (exp, arr) in enumerate(zip(shapes, weights)):
        got = [int(x) for x in arr.shape]
        want = [int(x) for x in exp]
        if want != got:
            raise ValueError(f"Weight w{i} shape mismatch: manifest {want}, file {got}.")
