#!/usr/bin/env python3
"""
Random DFA construction utilities for symbolic-sequence benchmarks.

Goals
-----
1. Build *reachable*, *minimal* random DFAs with exact target state count.
2. Reject trivial automata so that train lengths T=5..8 show non-trivial behavior.
3. Provide balanced dataset generation for:
   - last-timestep accept/reject tasks
   - autoregressive state+label tasks
4. Preserve the same OOD protocol used elsewhere: 2x, 5x, 10x train length.

This file is intentionally standalone and uses only the Python standard library plus NumPy.
It is designed as a starting point for the user's DFA data loader / benchmark suite.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core DFA structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFA:
    """Deterministic finite automaton over integer alphabet symbols 0..|Sigma|-1."""

    alphabet: Tuple[int, ...]
    transitions: Tuple[Tuple[int, ...], ...]  # transitions[q][a] -> q'
    start_state: int
    accepting: Tuple[bool, ...]
    name: str = ""
    metadata: Dict[str, float | int | str] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.transitions)

    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)

    def step(self, state: int, symbol: int) -> int:
        return int(self.transitions[state][symbol])

    def run(self, symbols: Sequence[int]) -> Tuple[List[int], List[int]]:
        """
        Returns
        -------
        states_after_step : list[int]
            q_t after consuming x_t for each timestep.
        labels_after_step : list[int]
            1 if state is accepting after consuming x_t, else 0.
        """
        q = int(self.start_state)
        states: List[int] = []
        labels: List[int] = []
        for sym in symbols:
            q = self.step(q, int(sym))
            states.append(q)
            labels.append(1 if self.accepting[q] else 0)
        return states, labels

    def accepts(self, symbols: Sequence[int]) -> int:
        q = int(self.start_state)
        for sym in symbols:
            q = self.step(q, int(sym))
        return 1 if self.accepting[q] else 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _entropy_from_probs(probs: Sequence[float]) -> float:
    e = 0.0
    for p in probs:
        if p > 0.0:
            e -= float(p) * math.log(float(p), 2.0)
    return e


def _binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return _entropy_from_probs([p, 1.0 - p])


def _one_hot(index: int, size: int) -> np.ndarray:
    x = np.zeros(size, dtype=np.float32)
    x[int(index)] = 1.0
    return x


def _serialize_dfa(dfa: DFA) -> Dict[str, object]:
    return {
        "name": dfa.name,
        "alphabet": list(dfa.alphabet),
        "transitions": [list(r) for r in dfa.transitions],
        "start_state": int(dfa.start_state),
        "accepting": [bool(x) for x in dfa.accepting],
        "metadata": dict(dfa.metadata),
    }


# ---------------------------------------------------------------------------
# Reachability / minimization
# ---------------------------------------------------------------------------


def reachable_states(dfa: DFA) -> List[int]:
    seen = {int(dfa.start_state)}
    q = deque([int(dfa.start_state)])
    while q:
        s = q.popleft()
        for a in dfa.alphabet:
            t = dfa.step(s, a)
            if t not in seen:
                seen.add(t)
                q.append(t)
    return sorted(seen)


def trim_to_reachable(dfa: DFA) -> DFA:
    reach = reachable_states(dfa)
    remap = {old: new for new, old in enumerate(reach)}
    trans = []
    acc = []
    for old in reach:
        trans.append(tuple(remap[dfa.step(old, a)] for a in dfa.alphabet))
        acc.append(bool(dfa.accepting[old]))
    return DFA(
        alphabet=tuple(dfa.alphabet),
        transitions=tuple(trans),
        start_state=remap[int(dfa.start_state)],
        accepting=tuple(acc),
        name=dfa.name,
        metadata=dict(dfa.metadata),
    )


def minimize_dfa(dfa: DFA) -> DFA:
    """Classic partition refinement (Moore/Hopcroft-style) for complete DFA."""
    dfa = trim_to_reachable(dfa)
    n = dfa.num_states
    if n <= 1:
        return dfa

    accepting = {i for i, is_acc in enumerate(dfa.accepting) if is_acc}
    rejecting = set(range(n)) - accepting
    partitions: List[set[int]] = []
    if accepting:
        partitions.append(set(accepting))
    if rejecting:
        partitions.append(set(rejecting))

    changed = True
    while changed:
        changed = False
        new_parts: List[set[int]] = []
        state_to_block = {}
        for bi, block in enumerate(partitions):
            for s in block:
                state_to_block[s] = bi

        for block in partitions:
            buckets: Dict[Tuple[int, ...], set[int]] = defaultdict(set)
            for s in block:
                sig = tuple(state_to_block[dfa.step(s, a)] for a in dfa.alphabet)
                buckets[sig].add(s)
            if len(buckets) == 1:
                new_parts.append(block)
            else:
                changed = True
                new_parts.extend(buckets.values())
        partitions = new_parts

    block_of = {}
    for bi, block in enumerate(partitions):
        for s in block:
            block_of[s] = bi

    rep = [min(block) for block in partitions]
    trans = []
    acc = []
    for r in rep:
        trans.append(tuple(block_of[dfa.step(r, a)] for a in dfa.alphabet))
        acc.append(bool(dfa.accepting[r]))
    return DFA(
        alphabet=tuple(dfa.alphabet),
        transitions=tuple(trans),
        start_state=block_of[int(dfa.start_state)],
        accepting=tuple(acc),
        name=dfa.name,
        metadata=dict(dfa.metadata),
    )


# ---------------------------------------------------------------------------
# Random DFA construction
# ---------------------------------------------------------------------------


def _random_reachable_complete_dfa(num_states: int, alphabet_size: int, rng: np.random.Generator) -> DFA:
    """
    Construct a complete DFA whose transition graph is guaranteed reachable from state 0.
    Then random accepting labels are added.
    """
    if num_states < 2:
        raise ValueError("num_states must be at least 2.")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2.")

    # Start with random complete table.
    trans = np.asarray(
        rng.integers(0, num_states, size=(num_states, alphabet_size), endpoint=False),
        dtype=np.int64,
    )

    # Force reachability by planting a randomized spanning tree from state 0.
    for q in range(1, num_states):
        parent = int(rng.integers(0, q))
        sym = int(rng.integers(0, alphabet_size))
        trans[parent, sym] = q

    # Accepting set roughly balanced, but not all / none.
    acc = rng.random(num_states) < 0.5
    if np.all(acc):
        acc[int(rng.integers(0, num_states))] = False
    if not np.any(acc):
        acc[int(rng.integers(0, num_states))] = True

    return DFA(
        alphabet=tuple(range(alphabet_size)),
        transitions=tuple(tuple(int(x) for x in row) for row in trans),
        start_state=0,
        accepting=tuple(bool(x) for x in acc.tolist()),
        name=f"randdfa_q{num_states}_a{alphabet_size}",
        metadata={},
    )


@dataclass
class ShortHorizonProfile:
    lengths: Tuple[int, ...]
    final_accept_rate: Dict[int, float]
    final_accept_entropy: Dict[int, float]
    prefix_accept_entropy: Dict[int, float]
    reached_state_fraction: Dict[int, float]
    end_state_max_share: Dict[int, float]
    mean_label_flip_rate: Dict[int, float]
    aggregate_score: float


def _sample_strings(rng: np.random.Generator, *, alphabet_size: int, length: int, n: int) -> np.ndarray:
    return np.asarray(rng.integers(0, alphabet_size, size=(n, length), endpoint=False), dtype=np.int64)


def profile_short_horizon_behavior(
    dfa: DFA,
    *,
    lengths: Sequence[int] = (5, 6, 7, 8),
    samples_per_length: int = 4096,
    seed: int = 0,
) -> ShortHorizonProfile:
    rng = np.random.default_rng(seed)
    final_rate: Dict[int, float] = {}
    final_ent: Dict[int, float] = {}
    prefix_ent: Dict[int, float] = {}
    state_frac: Dict[int, float] = {}
    end_share: Dict[int, float] = {}
    flip_rate: Dict[int, float] = {}
    score = 0.0

    for L in lengths:
        X = _sample_strings(rng, alphabet_size=dfa.alphabet_size, length=int(L), n=int(samples_per_length))
        end_labels = np.zeros(len(X), dtype=np.int64)
        end_states = np.zeros(len(X), dtype=np.int64)
        prefix_labels = np.zeros((len(X), int(L)), dtype=np.int64)
        for i, seq in enumerate(X):
            states, labels = dfa.run(seq)
            end_labels[i] = int(labels[-1])
            end_states[i] = int(states[-1])
            prefix_labels[i, :] = np.asarray(labels, dtype=np.int64)

        p = float(end_labels.mean())
        final_rate[int(L)] = p
        final_ent[int(L)] = _binary_entropy(p)
        prefix_ent[int(L)] = float(np.mean([_binary_entropy(float(prefix_labels[:, t].mean())) for t in range(int(L))]))
        reached = len(set(int(x) for x in end_states.tolist()))
        state_frac[int(L)] = reached / float(max(1, dfa.num_states))
        counts = Counter(int(x) for x in end_states.tolist())
        end_share[int(L)] = max(counts.values()) / float(len(end_states))
        flips = np.not_equal(prefix_labels[:, 1:], prefix_labels[:, :-1]).mean() if L > 1 else 0.0
        flip_rate[int(L)] = float(flips)

        # Aggregate score: encourage balanced labels, non-collapsed end-state distribution,
        # and some label dynamics. This is deliberately simple and interpretable.
        score += (
            1.5 * final_ent[int(L)]
            + 1.0 * prefix_ent[int(L)]
            + 1.0 * state_frac[int(L)]
            + 0.5 * flip_rate[int(L)]
            - 1.5 * max(0.0, end_share[int(L)] - 0.75)
        )

    return ShortHorizonProfile(
        lengths=tuple(int(x) for x in lengths),
        final_accept_rate=final_rate,
        final_accept_entropy=final_ent,
        prefix_accept_entropy=prefix_ent,
        reached_state_fraction=state_frac,
        end_state_max_share=end_share,
        mean_label_flip_rate=flip_rate,
        aggregate_score=float(score / max(1, len(tuple(lengths)))),
    )


@dataclass
class RandomDFASelectionConfig:
    lengths: Tuple[int, ...] = (5, 6, 7, 8)
    samples_per_length: int = 4096
    min_final_accept_rate: float = 0.15
    max_final_accept_rate: float = 0.85
    min_final_accept_entropy: float = 0.55
    min_prefix_accept_entropy: float = 0.25
    min_reached_state_fraction: float = 0.35
    max_end_state_share: float = 0.90
    min_aggregate_score: float = 1.10
    max_tries: int = 2000


def is_nontrivial_short_horizon(profile: ShortHorizonProfile, cfg: RandomDFASelectionConfig) -> bool:
    for L in cfg.lengths:
        p = profile.final_accept_rate[int(L)]
        if not (cfg.min_final_accept_rate <= p <= cfg.max_final_accept_rate):
            return False
        if profile.final_accept_entropy[int(L)] < cfg.min_final_accept_entropy:
            return False
        if profile.prefix_accept_entropy[int(L)] < cfg.min_prefix_accept_entropy:
            return False
        if profile.reached_state_fraction[int(L)] < cfg.min_reached_state_fraction:
            return False
        if profile.end_state_max_share[int(L)] > cfg.max_end_state_share:
            return False
    return profile.aggregate_score >= cfg.min_aggregate_score


@dataclass
class RandomDFASpec:
    num_states: int
    alphabet_size: int
    train_T: int
    ood_factors: Tuple[int, ...] = (2, 5, 10)
    seed: int = 0
    instance_id: int = 0
    profile: Optional[ShortHorizonProfile] = None


def construct_random_dfa(
    *,
    num_states: int,
    alphabet_size: int,
    seed: int,
    selection_cfg: RandomDFASelectionConfig | None = None,
    name: Optional[str] = None,
) -> DFA:
    """
    Rejection-sample random complete reachable DFAs until:
      1. the minimized DFA has exactly `num_states` states,
      2. short-horizon behavior (lengths 5..8 by default) is non-trivial.
    """
    cfg = selection_cfg or RandomDFASelectionConfig()
    master = np.random.default_rng(seed)

    best_dfa: Optional[DFA] = None
    best_profile: Optional[ShortHorizonProfile] = None

    for _try in range(int(cfg.max_tries)):
        cand_seed = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(cand_seed)
        dfa = _random_reachable_complete_dfa(num_states, alphabet_size, rng)
        dfa = minimize_dfa(dfa)
        if dfa.num_states != int(num_states):
            continue

        prof = profile_short_horizon_behavior(
            dfa,
            lengths=cfg.lengths,
            samples_per_length=int(cfg.samples_per_length),
            seed=cand_seed + 17,
        )
        if best_profile is None or prof.aggregate_score > best_profile.aggregate_score:
            best_profile, best_dfa = prof, dfa
        if is_nontrivial_short_horizon(prof, cfg):
            meta = {
                "selection_seed": int(seed),
                "construction_seed": int(cand_seed),
                "aggregate_score": float(prof.aggregate_score),
            }
            meta.update({f"accept_rate_L{L}": float(prof.final_accept_rate[L]) for L in prof.lengths})
            meta.update({f"prefix_entropy_L{L}": float(prof.prefix_accept_entropy[L]) for L in prof.lengths})
            return DFA(
                alphabet=dfa.alphabet,
                transitions=dfa.transitions,
                start_state=dfa.start_state,
                accepting=dfa.accepting,
                name=name or f"randdfa_q{num_states}_a{alphabet_size}_seed{seed}",
                metadata=meta,
            )

    if best_dfa is None:
        raise RuntimeError(
            f"Failed to construct any minimal DFA with exact {num_states} states and alphabet {alphabet_size}."
        )
    prof = best_profile
    assert prof is not None
    meta = {
        "selection_seed": int(seed),
        "aggregate_score": float(prof.aggregate_score),
        "warning": "Returned best candidate after max_tries without satisfying all non-triviality thresholds.",
    }
    return DFA(
        alphabet=best_dfa.alphabet,
        transitions=best_dfa.transitions,
        start_state=best_dfa.start_state,
        accepting=best_dfa.accepting,
        name=name or f"randdfa_q{num_states}_a{alphabet_size}_seed{seed}_fallback",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _generate_balanced_strings(
    dfa: DFA,
    *,
    length: int,
    n_examples: int,
    seed: int,
    oversample_factor: int = 4,
    max_rounds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate approximately balanced final accept/reject dataset."""
    target_pos = n_examples // 2
    target_neg = n_examples - target_pos
    pos: List[np.ndarray] = []
    neg: List[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for _ in range(int(max_rounds)):
        need = max(target_pos - len(pos), target_neg - len(neg))
        if need <= 0:
            break
        batch_n = max(256, int(oversample_factor * need))
        X = _sample_strings(rng, alphabet_size=dfa.alphabet_size, length=int(length), n=batch_n)
        y = np.asarray([dfa.accepts(x) for x in X], dtype=np.int64)
        for x, yi in zip(X, y):
            if yi == 1 and len(pos) < target_pos:
                pos.append(x.copy())
            elif yi == 0 and len(neg) < target_neg:
                neg.append(x.copy())
            if len(pos) >= target_pos and len(neg) >= target_neg:
                break

    if len(pos) < target_pos or len(neg) < target_neg:
        raise RuntimeError(
            f"Could not build balanced dataset for DFA {dfa.name} at length={length}."
        )
    X = np.vstack(pos + neg)
    y = np.asarray([1] * target_pos + [0] * target_neg, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def strings_to_last_timestep_tensors(dfa: DFA, strings: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    T = len(strings[0]) if strings else 0
    X = np.zeros((len(strings), T, dfa.alphabet_size), dtype=np.float32)
    y = np.zeros(len(strings), dtype=np.int64)
    for i, seq in enumerate(strings):
        for t, sym in enumerate(seq):
            X[i, t, int(sym)] = 1.0
        y[i] = int(dfa.accepts(seq))
    return X, y


def strings_to_autoregressive_tensors(dfa: DFA, strings: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not strings:
        return (
            np.zeros((0, 0, dfa.num_states + dfa.alphabet_size), dtype=np.float32),
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.int64),
        )
    T = len(strings[0])
    X = np.zeros((len(strings), T, dfa.num_states + dfa.alphabet_size), dtype=np.float32)
    y_state = np.zeros((len(strings), T), dtype=np.int64)
    y_label = np.zeros((len(strings), T), dtype=np.int64)
    start_oh = _one_hot(dfa.start_state, dfa.num_states)
    for i, seq in enumerate(strings):
        states, labels = dfa.run(seq)
        prev = int(dfa.start_state)
        for t, sym in enumerate(seq):
            X[i, t, : dfa.num_states] = _one_hot(prev, dfa.num_states) if t > 0 else start_oh
            X[i, t, dfa.num_states + int(sym)] = 1.0
            y_state[i, t] = int(states[t])
            y_label[i, t] = int(labels[t])
            prev = int(states[t])
    return X, y_state, y_label


def make_last_timestep_dataset(
    dfa: DFA,
    *,
    T: int,
    n_examples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_str, y = _generate_balanced_strings(dfa, length=int(T), n_examples=int(n_examples), seed=seed)
    X, y_last = strings_to_last_timestep_tensors(dfa, [x.tolist() for x in X_str])
    return X, y_last, X_str


def make_autoregressive_dataset(
    dfa: DFA,
    *,
    T: int,
    n_examples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_str, _ = _generate_balanced_strings(dfa, length=int(T), n_examples=int(n_examples), seed=seed)
    X, y_state, y_label = strings_to_autoregressive_tensors(dfa, [x.tolist() for x in X_str])
    return X, y_state, y_label, X_str


# ---------------------------------------------------------------------------
# Suite generation for the requested benchmark grid
# ---------------------------------------------------------------------------


def requested_grid() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (2, 4, 6, 8, 10), (2, 4, 6, 8, 10)


def build_random_dfa_suite(
    *,
    train_T_values: Sequence[int] = (5, 6, 7, 8),
    state_choices: Sequence[int] = (2, 4, 6, 8, 10),
    alphabet_choices: Sequence[int] = (2, 4, 6, 8, 10),
    ood_factors: Sequence[int] = (2, 5, 10),
    instances_per_pair: int = 1,
    seed: int = 0,
    selection_cfg: RandomDFASelectionConfig | None = None,
) -> List[Dict[str, object]]:
    master = np.random.default_rng(seed)
    out: List[Dict[str, object]] = []
    idx = 0
    for T in train_T_values:
        for q in state_choices:
            for a in alphabet_choices:
                for inst in range(int(instances_per_pair)):
                    dfa_seed = int(master.integers(0, 2**31 - 1))
                    dfa = construct_random_dfa(
                        num_states=int(q),
                        alphabet_size=int(a),
                        seed=dfa_seed,
                        selection_cfg=selection_cfg,
                        name=f"randomdfa_T{int(T)}_q{int(q)}_a{int(a)}_i{inst}",
                    )
                    prof = profile_short_horizon_behavior(
                        dfa,
                        lengths=(int(T),),
                        samples_per_length=4096,
                        seed=dfa_seed + 99,
                    )
                    out.append(
                        {
                            "id": idx,
                            "train_T": int(T),
                            "ood_lengths": [int(T) * int(m) for m in ood_factors],
                            "num_states": int(dfa.num_states),
                            "alphabet_size": int(dfa.alphabet_size),
                            "instance_id": int(inst),
                            "dfa": _serialize_dfa(dfa),
                            "short_horizon_profile": asdict(prof),
                        }
                    )
                    idx += 1
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_preview(args: argparse.Namespace) -> None:
    cfg = RandomDFASelectionConfig(
        lengths=tuple(int(x) for x in args.lengths),
        samples_per_length=int(args.samples_per_length),
        max_tries=int(args.max_tries),
    )
    dfa = construct_random_dfa(
        num_states=int(args.num_states),
        alphabet_size=int(args.alphabet_size),
        seed=int(args.seed),
        selection_cfg=cfg,
    )
    prof = profile_short_horizon_behavior(
        dfa,
        lengths=tuple(int(x) for x in args.lengths),
        samples_per_length=int(args.samples_per_length),
        seed=int(args.seed) + 123,
    )
    payload = {
        "dfa": _serialize_dfa(dfa),
        "profile": asdict(prof),
    }
    print(json.dumps(payload, indent=2))


def _cmd_build_suite(args: argparse.Namespace) -> None:
    cfg = RandomDFASelectionConfig(
        lengths=tuple(int(x) for x in args.lengths),
        samples_per_length=int(args.samples_per_length),
        max_tries=int(args.max_tries),
    )
    suite = build_random_dfa_suite(
        train_T_values=tuple(int(x) for x in args.train_T_values),
        state_choices=tuple(int(x) for x in args.state_choices),
        alphabet_choices=tuple(int(x) for x in args.alphabet_choices),
        ood_factors=tuple(int(x) for x in args.ood_factors),
        instances_per_pair=int(args.instances_per_pair),
        seed=int(args.seed),
        selection_cfg=cfg,
    )
    out = Path(args.output).expanduser().resolve()
    out.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"Wrote {len(suite)} benchmark specs to {out}")


def _cmd_dataset_example(args: argparse.Namespace) -> None:
    cfg = RandomDFASelectionConfig(
        lengths=tuple(int(x) for x in args.lengths),
        samples_per_length=int(args.samples_per_length),
        max_tries=int(args.max_tries),
    )
    dfa = construct_random_dfa(
        num_states=int(args.num_states),
        alphabet_size=int(args.alphabet_size),
        seed=int(args.seed),
        selection_cfg=cfg,
    )
    X_ar, y_state, y_label, _ = make_autoregressive_dataset(
        dfa, T=int(args.train_T), n_examples=int(args.n_examples), seed=int(args.seed) + 1
    )
    X_last, y_last, _ = make_last_timestep_dataset(
        dfa, T=int(args.train_T), n_examples=int(args.n_examples), seed=int(args.seed) + 2
    )
    print(json.dumps(
        {
            "dfa": _serialize_dfa(dfa),
            "autoregressive_shapes": {
                "X": list(X_ar.shape),
                "y_state": list(y_state.shape),
                "y_label": list(y_label.shape),
            },
            "last_timestep_shapes": {"X": list(X_last.shape), "y": list(y_last.shape)},
        },
        indent=2,
    ))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Random DFA benchmark constructor.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prev = sub.add_parser("preview", help="Construct one DFA and print its short-horizon profile.")
    ap_prev.add_argument("--num_states", type=int, required=True)
    ap_prev.add_argument("--alphabet_size", type=int, required=True)
    ap_prev.add_argument("--seed", type=int, default=0)
    ap_prev.add_argument("--lengths", type=int, nargs="*", default=[5, 6, 7, 8])
    ap_prev.add_argument("--samples_per_length", type=int, default=4096)
    ap_prev.add_argument("--max_tries", type=int, default=2000)
    ap_prev.set_defaults(func=_cmd_preview)

    ap_suite = sub.add_parser("build_suite", help="Build the requested benchmark grid and save JSON.")
    ap_suite.add_argument("--train_T_values", type=int, nargs="*", default=[5, 6, 7, 8])
    ap_suite.add_argument("--state_choices", type=int, nargs="*", default=[2, 4, 6, 8, 10])
    ap_suite.add_argument("--alphabet_choices", type=int, nargs="*", default=[2, 4, 6, 8, 10])
    ap_suite.add_argument("--ood_factors", type=int, nargs="*", default=[2, 5, 10])
    ap_suite.add_argument("--instances_per_pair", type=int, default=1)
    ap_suite.add_argument("--seed", type=int, default=0)
    ap_suite.add_argument("--lengths", type=int, nargs="*", default=[5, 6, 7, 8])
    ap_suite.add_argument("--samples_per_length", type=int, default=4096)
    ap_suite.add_argument("--max_tries", type=int, default=2000)
    ap_suite.add_argument("--output", type=str, default="random_dfa_suite.json")
    ap_suite.set_defaults(func=_cmd_build_suite)

    ap_ex = sub.add_parser("dataset_example", help="Construct one DFA and print tensor shapes.")
    ap_ex.add_argument("--num_states", type=int, required=True)
    ap_ex.add_argument("--alphabet_size", type=int, required=True)
    ap_ex.add_argument("--train_T", type=int, default=6)
    ap_ex.add_argument("--n_examples", type=int, default=256)
    ap_ex.add_argument("--seed", type=int, default=0)
    ap_ex.add_argument("--lengths", type=int, nargs="*", default=[5, 6, 7, 8])
    ap_ex.add_argument("--samples_per_length", type=int, default=4096)
    ap_ex.add_argument("--max_tries", type=int, default=2000)
    ap_ex.set_defaults(func=_cmd_dataset_example)

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
