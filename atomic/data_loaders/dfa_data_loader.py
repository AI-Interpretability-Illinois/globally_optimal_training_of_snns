#!/usr/bin/env python3
"""
DFA Test Bench for Convex SNN evaluation.

Provides:
  1. Generic DFA simulator
  2. Built-in DFA library (Tomita, parity, counters, XOR variants)
  3. .att file loader (OpenFst AT&T format, for MLRegTest)
  4. MLRegTest data file loader
  5. make_dfa_dataset() matching snn_p2.py interface: → (X, y, num_classes)
  6. make_dfa_autoregressive_dataset() for (prev_state, input_t) → (next_state, per-step
     accept) on built-in Tomita, bounded Dyck (``dyck1_d*``), and dynamic random DFAs
     ``random_{num_states}_{alphabet_size}``; not ``dyck1_unbounded``

Usage from snn_p2.py:
  --task dfa:tomita_3         # built-in DFA
  --task dfa:parity_3         # parity mod 3
  --task dfa:att:/path/to.att # load .att file
  --task dfa:mlregtest:/path/to/langname  # load MLRegTest data files
  --task dfa:random_6_4         # dynamic random DFA with 6 states, alphabet size 4

Usage standalone:
  python dfa_tasks.py --list                          # list all built-in DFAs
  python dfa_tasks.py --name tomita_3 --T 20 --n 100  # generate + stats
  python dfa_tasks.py --att /path/to/file.att --T 30   # from .att file
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from random_dfa_library import (
        DFA as RandomDFA,
        RandomDFASelectionConfig,
        construct_random_dfa,
    )
except Exception:
    RandomDFA = None  # type: ignore[assignment]
    RandomDFASelectionConfig = None  # type: ignore[assignment]
    construct_random_dfa = None  # type: ignore[assignment]


# ============================================================
# DFA class
# ============================================================

@dataclass
class DFA:
    """
    Deterministic Finite Automaton.

    States are integers, alphabet symbols are strings.
    """
    name: str
    states: Set[int]
    alphabet: List[str]               # ordered list of symbols
    transitions: Dict[Tuple[int, str], int]  # (state, symbol) → next_state
    start: int
    accept: Set[int]

    generation_metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    """Audit trail for synthetic DFAs (e.g. Markov stationary under uniform inputs)."""

    # Derived (computed on first use)
    _sym2idx: Optional[Dict[str, int]] = field(default=None, repr=False)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_symbols(self) -> int:
        return len(self.alphabet)

    @property
    def sym2idx(self) -> Dict[str, int]:
        if self._sym2idx is None:
            self._sym2idx = {s: i for i, s in enumerate(self.alphabet)}
        return self._sym2idx

    def step(self, state: int, symbol: str) -> int:
        """Single transition. Returns -1 (dead/reject) if undefined."""
        return self.transitions.get((state, symbol), -1)

    def run(self, string: List[str]) -> Tuple[int, bool]:
        """Run DFA on a string. Returns (final_state, accepted)."""
        state = self.start
        for sym in string:
            state = self.step(state, sym)
            if state == -1:
                return -1, False
        return state, state in self.accept

    def accepts(self, string: List[str]) -> bool:
        _, acc = self.run(string)
        return acc

    def generate_strings(
        self,
        length: int,
        n: int,
        seed: int = 0,
        balanced: bool = True,
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Generate n strings of given length, uniformly random over alphabet.
        If balanced=True, resample to get ~50/50 positive/negative.

        Returns: (strings, labels) where labels ∈ {0, 1}.
        """
        rng = np.random.default_rng(seed)
        sigma = self.alphabet

        if not balanced:
            strings = []
            labels = []
            for _ in range(n):
                s = [sigma[i] for i in rng.integers(0, len(sigma), size=length)]
                strings.append(s)
                labels.append(1 if self.accepts(s) else 0)
            return strings, np.array(labels, dtype=np.int64)

        # Balanced: generate excess, split by label, downsample
        pos_strings, neg_strings = [], []
        max_attempts = n * 20
        attempts = 0
        target = n // 2

        while (len(pos_strings) < target or len(neg_strings) < target) and attempts < max_attempts:
            batch = max(256, n)
            syms = rng.integers(0, len(sigma), size=(batch, length))
            for row in syms:
                s = [sigma[i] for i in row]
                if self.accepts(s):
                    if len(pos_strings) < target:
                        pos_strings.append(s)
                else:
                    if len(neg_strings) < target:
                        neg_strings.append(s)
                attempts += 1
                if len(pos_strings) >= target and len(neg_strings) >= target:
                    break

        # If one class is under-represented, pad with what we have
        pos_strings = pos_strings[:target]
        neg_strings = neg_strings[:n - len(pos_strings)]

        strings = pos_strings + neg_strings
        labels = [1] * len(pos_strings) + [0] * len(neg_strings)

        # Shuffle
        perm = rng.permutation(len(strings))
        strings = [strings[i] for i in perm]
        labels_arr = np.array(labels, dtype=np.int64)[perm]

        return strings, labels_arr

    def strings_to_onehot(
        self, strings: List[List[str]], T: int
    ) -> np.ndarray:
        """
        Convert list of symbol-strings to one-hot: (n, T, |Σ|).
        Pads/truncates to length T.
        """
        n = len(strings)
        d = len(self.alphabet)
        X = np.zeros((n, T, d), dtype=np.float32)
        for i, s in enumerate(strings):
            for t in range(min(len(s), T)):
                idx = self.sym2idx.get(s[t], -1)
                if idx >= 0:
                    X[i, t, idx] = 1.0
        return X

    def summary(self) -> str:
        return (
            f"DFA '{self.name}': {self.num_states} states, "
            f"|Σ|={self.num_symbols} ({self.alphabet}), "
            f"accept={self.accept}"
        )


def dfa_to_json_dict(dfa: DFA) -> Dict[str, Union[str, int, List[int], List[str], List[Dict[str, Union[int, str]]]]]:
    """Serializable dict (transitions as edge records) for analysis pipelines."""
    edges: List[Dict[str, Union[int, str]]] = []
    for (q, sym), nq in sorted(dfa.transitions.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        edges.append({"from": int(q), "symbol": str(sym), "to": int(nq)})
    d: Dict[str, Union[str, int, List[int], List[str], List[Dict[str, Union[int, str]]], Dict[str, Union[str, int, float]]]] = {
        "name": str(dfa.name),
        "start": int(dfa.start),
        "accept": sorted(int(x) for x in dfa.accept),
        "alphabet": list(dfa.alphabet),
        "num_states": int(dfa.num_states),
        "num_symbols": int(dfa.num_symbols),
        "transitions": edges,
    }
    if dfa.generation_metadata:
        d["generation_metadata"] = dict(dfa.generation_metadata)
    return d


def save_dfa_json(path: Union[str, Path], dfa: DFA) -> None:
    """Write :func:`dfa_to_json_dict` to a JSON file (parent dirs created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dfa_to_json_dict(dfa), indent=2) + "\n", encoding="utf-8")


# ============================================================
# .att file loader (OpenFst AT&T text format)
# ============================================================

def load_att(path: str, name: Optional[str] = None) -> DFA:
    """
    Load a DFA from OpenFst AT&T text format (.att).

    Format:
      src dest ilabel olabel [weight]   ← transitions
      final_state [weight]              ← accept states
    """
    states = set()
    transitions = {}
    accept = set()
    alphabet_set = set()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                # Transition: src dest ilabel [olabel] [weight]
                src = int(parts[0])
                dst = int(parts[1])
                ilabel = parts[2]
                states.add(src)
                states.add(dst)
                if ilabel not in ("<eps>", "ε"):
                    alphabet_set.add(ilabel)
                    transitions[(src, ilabel)] = dst
            elif len(parts) == 1 or len(parts) == 2:
                # Accept state: state [weight]
                accept.add(int(parts[0]))
                states.add(int(parts[0]))

    alphabet = sorted(alphabet_set)
    dfa_name = name or os.path.splitext(os.path.basename(path))[0]

    return DFA(
        name=dfa_name,
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start=0,  # OpenFst convention: state 0 is start
        accept=accept,
    )


# ============================================================
# MLRegTest data loader
# ============================================================

def load_mlregtest_data(
    base_path: str,
    split: str = "Train",
) -> Tuple[List[List[str]], np.ndarray, List[str]]:
    """
    Load MLRegTest data file.

    Files are named: languagename_Train.txt, languagename_Dev.txt, etc.
    Each line: TRUE\tstring  or  FALSE\tstring
    where string is space-separated symbols.

    Parameters
    ----------
    base_path : str
        Either the full path to a specific file, or the path prefix
        (e.g. "data/Large/04.04.SL.2.1.1") to which "_Train.txt" etc. is appended.
    split : str
        One of: Train, Dev, TestSR, TestSA, TestLR, TestLA

    Returns
    -------
    strings : list of list of str
    labels : np.ndarray of int64
    alphabet : sorted list of unique symbols
    """
    if os.path.isfile(base_path):
        fpath = base_path
    else:
        fpath = f"{base_path}_{split}.txt"

    strings = []
    labels = []
    alphabet_set = set()

    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: TRUE\tsym1 sym2 sym3 ...  or  FALSE\tsym1 sym2 ...
            # Some files use tab, some use space as delimiter
            if "\t" in line:
                label_str, sym_str = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                label_str, sym_str = parts

            label = 1 if label_str.strip().upper() in ("TRUE", "1") else 0
            syms = sym_str.strip().split()
            strings.append(syms)
            labels.append(label)
            alphabet_set.update(syms)

    alphabet = sorted(alphabet_set)
    return strings, np.array(labels, dtype=np.int64), alphabet



# ============================================================
# Dynamic random DFA support
# ============================================================

_RANDOM_SPEC_RE = re.compile(
    r"^random_(?P<num_states>\d+)_(?P<alphabet_size>\d+)"
    r"(?:_seed(?P<seed>\d+))?"
    r"(?P<tag>_dense|_bias_accept|_bias_reject)?"
    r"(?:_k(?P<peak>\d+))?$"
)


def parse_random_dfa_spec(name: str) -> Optional[Tuple[int, int, int, str, float]]:
    """
    Parse dynamic random DFA specs of the form:
        random_{num_states}_{alphabet_size}
        random_{num_states}_{alphabet_size}_seed{seed}
        random_..._dense  — uniform stationary (column-balanced) under uniform input
        random_..._bias_accept  — π peaks on an **accepting** state: max_{q∈F} π(q) ≥ k/Q (default k=2)
        random_..._bias_reject — π peaks on a **rejecting** state: max_{q∉F} π(q) ≥ k/Q (default k=2)
        random_..._bias_accept_k3 — same with integer peak factor k=3 (2–3× uniform mass 1/Q)

    Returns
    -------
    (num_states, alphabet_size, seed, variant, peak_factor) or None.

    ``variant`` ∈ {``iid``, ``uniform_stationary``, ``bias_accept``, ``bias_reject``}.
    ``peak_factor`` applies to bias variants only (else 2.0, unused).
    """
    m = _RANDOM_SPEC_RE.fullmatch(str(name))
    if m is None:
        return None
    q = int(m.group("num_states"))
    a = int(m.group("alphabet_size"))
    seed_g = m.group("seed")
    seed = int(seed_g) if seed_g is not None else 0
    tag = m.group("tag")
    peak_g = m.group("peak")
    if tag is None:
        if peak_g is not None:
            raise ValueError(f"Invalid spec (peak suffix without mode tag): {name!r}")
        return q, a, seed, "iid", 2.0
    if tag == "_dense":
        if peak_g is not None:
            raise ValueError(f"_k peak factor is not valid with _dense: {name!r}")
        return q, a, seed, "uniform_stationary", 2.0
    if tag == "_bias_accept":
        k = float(int(peak_g)) if peak_g is not None else 2.0
        return q, a, seed, "bias_accept", k
    if tag == "_bias_reject":
        k = float(int(peak_g)) if peak_g is not None else 2.0
        return q, a, seed, "bias_reject", k
    raise ValueError(f"internal: unhandled tag {tag!r}")


def _ensure_random_dfa_support() -> None:
    if construct_random_dfa is None or RandomDFASelectionConfig is None:
        raise ImportError(
            "random_dfa_library.py is required for specs like "
            "'random_{num_states}_{alphabet_size}'."
        )


def _convert_random_dfa_to_string_dfa(rdfa: "RandomDFA", name: str) -> DFA:
    """
    Convert the standalone integer-alphabet RandomDFA into this file's string-alphabet DFA.
    """
    alphabet = [str(int(a)) for a in rdfa.alphabet]
    transitions: Dict[Tuple[int, str], int] = {}
    states = set(range(int(rdfa.num_states)))
    for q in range(int(rdfa.num_states)):
        for a in range(int(rdfa.alphabet_size)):
            transitions[(q, str(a))] = int(rdfa.step(q, a))
    accept = {q for q, is_acc in enumerate(rdfa.accepting) if bool(is_acc)}
    gmeta: Dict[str, Union[str, int, float]] = {}
    for k, v in rdfa.metadata.items():
        if isinstance(v, (str, int, float)):
            gmeta[str(k)] = v
        else:
            raise TypeError(f"random DFA metadata must be str|int|float for JSON; got {k}={v!r}")
    return DFA(
        name=name,
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start=int(rdfa.start_state),
        accept=accept,
        generation_metadata=gmeta,
    )


@lru_cache(maxsize=256)
def _cached_random_dfa(name: str) -> DFA:
    """
    Construct and cache a random DFA instance keyed by the literal spec string.

    The default constructor enforces non-trivial short-horizon behavior on lengths 5..8,
    which is the intended train-length regime for these synthetic benchmarks.
    Each ``_dense`` spec uses ``uniform_stationary`` tables (column-balanced, uniform π).

    ``_bias_accept`` / ``_bias_reject`` use i.i.d. transitions with rejection until, under uniform
    input, **some** state in the target set (accept / reject) has Markov stationary weight
    at least ``stationary_peak_factor / |Q|`` (default factor 2 ⇒ twice the uniform ``1/|Q|`` mass;
    use spec suffix ``_k3`` for three times).

    Plain specs (no suffix) keep legacy i.i.d. tables with no stationary filter.
    """
    parsed = parse_random_dfa_spec(name)
    if parsed is None:
        raise KeyError(f"{name!r} is not a random DFA spec.")
    q, a, seed, variant, peak_factor = parsed
    _ensure_random_dfa_support()
    if variant == "uniform_stationary":
        transition_mode = "uniform_stationary"
        stationary_target = "none"
    elif variant == "bias_accept":
        transition_mode = "iid"
        stationary_target = "accept_mass"
    elif variant == "bias_reject":
        transition_mode = "iid"
        stationary_target = "reject_mass"
    elif variant == "iid":
        transition_mode = "iid"
        stationary_target = "none"
    else:
        raise KeyError(f"unhandled random variant {variant!r}")
    sel_cfg = RandomDFASelectionConfig(
        lengths=(5, 6, 7, 8),
        transition_mode=transition_mode,
        stationary_target=stationary_target,
        stationary_peak_factor=float(peak_factor),
    )
    rdfa = construct_random_dfa(
        num_states=int(q),
        alphabet_size=int(a),
        seed=int(seed),
        selection_cfg=sel_cfg,
        name=name,
    )
    return _convert_random_dfa_to_string_dfa(rdfa, name=name)

# ============================================================
# Built-in DFA library
# ============================================================

def _make_dfa(name, n_states, alphabet, trans_fn, accept_fn) -> DFA:
    """Helper: build DFA from functions."""
    states = set(range(n_states))
    transitions = {}
    for q in range(n_states):
        for a in alphabet:
            nq = trans_fn(q, a)
            if nq is not None and 0 <= nq < n_states:
                transitions[(q, a)] = nq
    accept = {q for q in range(n_states) if accept_fn(q)}
    return DFA(name=name, states=states, alphabet=alphabet,
               transitions=transitions, start=0, accept=accept)


# ── Tomita grammars (classic DFA benchmark, binary alphabet) ──

def tomita_1() -> DFA:
    """Tomita 1: 1* (all 1s)"""
    # States: 0=start/accept, 1=dead
    return _make_dfa("tomita_1", 2, ["0", "1"],
        lambda q, a: 0 if q == 0 and a == "1" else 1,
        lambda q: q == 0)

def tomita_2() -> DFA:
    """Tomita 2: (10)* """
    # States: 0=start, 1=saw 1, 2=dead
    def trans(q, a):
        if q == 0: return 1 if a == "1" else 2
        if q == 1: return 0 if a == "0" else 2
        return 2
    return _make_dfa("tomita_2", 3, ["0", "1"], trans, lambda q: q == 0)

def tomita_3() -> DFA:
    """Tomita 3: no odd-length runs of 0s after odd-length runs of 1s.
    Complement of strings containing an odd number of consecutive 0s
    immediately following an odd number of consecutive 1s."""
    # This is the classic 5-state Tomita 3
    # States: 0 (start), 1, 2, 3, 4 (dead)
    T = {
        (0, "0"): 0, (0, "1"): 1,
        (1, "0"): 3, (1, "1"): 2,
        (2, "0"): 3, (2, "1"): 2,  # even 1s
        (3, "0"): 0, (3, "1"): 4,  # odd 1s, then one 0
        (4, "0"): 4, (4, "1"): 4,  # dead
    }
    return DFA(name="tomita_3", states={0,1,2,3,4}, alphabet=["0","1"],
               transitions=T, start=0, accept={0,1,2,3})

def tomita_4() -> DFA:
    """Tomita 4: strings not containing '000'."""
    # States: 0=start, 1=one 0, 2=two 0s, 3=dead (saw 000)
    def trans(q, a):
        if q == 3: return 3
        if a == "1": return 0
        return q + 1 if q < 3 else 3
    return _make_dfa("tomita_4", 4, ["0", "1"], trans, lambda q: q != 3)

def tomita_5() -> DFA:
    """Tomita 5: even number of 0s AND even number of 1s."""
    # States encode (count_0 mod 2, count_1 mod 2) as q = 2*c0 + c1
    def trans(q, a):
        c0, c1 = q // 2, q % 2
        if a == "0": c0 ^= 1
        else: c1 ^= 1
        return 2 * c0 + c1
    return _make_dfa("tomita_5", 4, ["0", "1"], trans, lambda q: q == 0)

def tomita_6() -> DFA:
    """Tomita 6: (count of 0s - count of 1s) mod 3 == 0."""
    # 3 states: difference mod 3
    def trans(q, a):
        if a == "0": return (q + 1) % 3
        else: return (q - 1) % 3
    return _make_dfa("tomita_6", 3, ["0", "1"], trans, lambda q: q == 0)

def tomita_7() -> DFA:
    """Tomita 7: 0*1*0*1* (at most 4 alternation blocks)."""
    # States: phases of the pattern
    # 0: reading 0s (phase 1)
    # 1: reading 1s (phase 2)
    # 2: reading 0s (phase 3)
    # 3: reading 1s (phase 4)
    # 4: dead (too many alternations)
    def trans(q, a):
        if q == 4: return 4
        if q == 0: return 0 if a == "0" else 1
        if q == 1: return 2 if a == "0" else 1
        if q == 2: return 2 if a == "0" else 3
        if q == 3: return 4 if a == "0" else 3
        return 4
    return _make_dfa("tomita_7", 5, ["0", "1"], trans,
                     lambda q: q != 4)


# ── Parity / modular counting ──

def parity_binary(k: int = 2) -> DFA:
    """Accept strings where count of 1s ≡ 0 (mod k). Binary alphabet."""
    def trans(q, a):
        if a == "1": return (q + 1) % k
        return q
    return _make_dfa(f"parity_{k}", k, ["0", "1"], trans, lambda q: q == 0)

def modular_count(k: int, sigma_size: int = 2) -> DFA:
    """Accept strings where (sum of symbol indices) ≡ 0 (mod k)."""
    alphabet = [str(i) for i in range(sigma_size)]
    def trans(q, a):
        return (q + int(a)) % k
    return _make_dfa(f"mod{k}_sigma{sigma_size}", k, alphabet, trans,
                     lambda q: q == 0)


# ── XOR variants ──

def first_last_xor() -> DFA:
    """Accept iff first symbol XOR last symbol = 1. Binary alphabet."""
    # 4 states: (first_bit, last_bit_so_far)
    # After seeing first symbol: state = first_bit * 2 + current_last
    # Actually need to track: first_bit and last_bit
    # State encoding: 0-3 for (first, last) pairs, plus initial state
    T = {}
    # State 4 = initial (no symbols seen)
    # After first symbol a: state = 2*a + a (first=a, last=a)
    T[(4, "0")] = 0  # first=0, last=0
    T[(4, "1")] = 3  # first=1, last=1
    # From state (first, last): reading a → (first, a)
    for first in (0, 1):
        for last in (0, 1):
            q = first * 2 + last
            for a_str in ("0", "1"):
                a = int(a_str)
                T[(q, a_str)] = first * 2 + a

    return DFA(name="first_last_xor", states={0,1,2,3,4}, alphabet=["0","1"],
               transitions=T, start=4,
               accept={1, 2})  # first≠last → XOR=1


def two_step_xor_dfa() -> DFA:
    """
    Simplified two_step_xor as a DFA.
    Binary alphabet, label = x[0] XOR x[T-1].
    Same as first_last_xor.
    """
    dfa = first_last_xor()
    dfa.name = "two_step_xor"
    return dfa


# ── Bracket matching (Dyck-1 bounded depth) ──

def dyck1_bounded(max_depth: int = 3) -> DFA:
    """
    Balanced parentheses with max nesting depth.
    Alphabet: '(' and ')'.
    """
    # States 0..max_depth = current depth, max_depth+1 = dead
    dead = max_depth + 1
    def trans(q, a):
        if q == dead: return dead
        if a == "(":
            return q + 1 if q < max_depth else dead
        else:  # ")"
            return q - 1 if q > 0 else dead
    return _make_dfa(f"dyck1_d{max_depth}", dead + 1, ["(", ")"],
                     trans, lambda q: q == 0)


def is_balanced_unbounded(s: List[str]) -> bool:
    """
    Dyck-1 membership with unbounded depth (not regular / not a DFA).
    """
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
        else:
            return False
    return depth == 0


def generate_dyck1_unbounded_strings(
    length: int,
    n: int,
    seed: int = 0,
    balanced: bool = True,
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Generate Dyck-1 examples at fixed length using stack-based acceptance.

    Notes:
      - This is NOT a DFA generator (unbounded depth).
      - For odd `length`, positives do not exist.
    """
    rng = np.random.default_rng(seed)
    sigma = ["(", ")"]

    if not balanced or length % 2 == 1:
        syms = rng.integers(0, 2, size=(n, length))
        strings = [[sigma[i] for i in row] for row in syms]
        labels = np.array([1 if is_balanced_unbounded(s) else 0 for s in strings], dtype=np.int64)
        return strings, labels

    # Balanced sampling (roughly 50/50) by rejection.
    target_pos = n // 2
    target_neg = n - target_pos
    pos_strings: List[List[str]] = []
    neg_strings: List[List[str]] = []

    max_attempts = n * 200
    attempts = 0
    while (len(pos_strings) < target_pos or len(neg_strings) < target_neg) and attempts < max_attempts:
        batch = max(512, n)
        syms = rng.integers(0, 2, size=(batch, length))
        for row in syms:
            s = [sigma[i] for i in row]
            if is_balanced_unbounded(s):
                if len(pos_strings) < target_pos:
                    pos_strings.append(s)
            else:
                if len(neg_strings) < target_neg:
                    neg_strings.append(s)
            attempts += 1
            if len(pos_strings) >= target_pos and len(neg_strings) >= target_neg:
                break

    # Fallback fill if rejection didn't hit target exactly.
    while len(pos_strings) < target_pos and attempts < max_attempts * 2:
        row = rng.integers(0, 2, size=(length,))
        s = [sigma[i] for i in row]
        if is_balanced_unbounded(s):
            pos_strings.append(s)
        attempts += 1

    while len(neg_strings) < target_neg:
        row = rng.integers(0, 2, size=(length,))
        s = [sigma[i] for i in row]
        if not is_balanced_unbounded(s):
            neg_strings.append(s)

    strings = pos_strings[:target_pos] + neg_strings[:target_neg]
    labels = np.array([1] * min(len(pos_strings), target_pos) + [0] * target_neg, dtype=np.int64)

    perm = rng.permutation(len(strings))
    strings = [strings[i] for i in perm]
    labels = labels[perm]
    return strings, labels


# ── Specific pattern containment (SL-class) ──

def contains_substring(pattern: str, alphabet: Optional[List[str]] = None) -> DFA:
    """
    Accept strings containing `pattern` as a substring.
    Uses simple NFA→DFA for pattern matching (KMP-style).
    """
    if alphabet is None:
        alphabet = sorted(set(pattern))

    # Build KMP failure function
    pat = list(pattern)
    m = len(pat)

    # States 0..m-1 = matched prefix length, m = full match (accept)
    # Simplified: for each state and symbol, compute next state
    fail = [0] * (m + 1)
    for i in range(1, m):
        j = fail[i - 1]
        while j > 0 and pat[j] != pat[i]:
            j = fail[j - 1]
        if pat[j] == pat[i]:
            j += 1
        fail[i] = j

    def trans(q, a):
        if q == m:  # already matched, absorb
            return m
        state = q
        while state > 0 and (state >= m or pat[state] != a):
            state = fail[state - 1]
        if state < m and pat[state] == a:
            state += 1
        return state

    return _make_dfa(f"contains_{''.join(pattern)}", m + 1, alphabet,
                     trans, lambda q: q == m)


def forbids_substring(pattern: str, alphabet: Optional[List[str]] = None) -> DFA:
    """Accept strings NOT containing `pattern`. Complement of contains_substring."""
    dfa = contains_substring(pattern, alphabet)
    dfa.name = f"forbids_{''.join(pattern)}"
    # Flip accept states
    dfa.accept = dfa.states - dfa.accept
    return dfa


# ── Star-free / counter-free examples ──

def ends_with(pattern: str, alphabet: Optional[List[str]] = None) -> DFA:
    """Accept strings ending with `pattern`."""
    if alphabet is None:
        alphabet = sorted(set(pattern))
    # Same as contains_substring but only accept at end
    # Actually: need to check last len(pattern) symbols
    # Use a sliding window via DFA states = matched suffix length
    dfa = contains_substring(pattern, alphabet)
    # Modify: only accept in state m (already correct for contains)
    # But for "ends_with", we need the string to END in the pattern,
    # not just contain it. The contains DFA absorbs at state m forever.
    # Fix: don't absorb at m, keep matching
    pat = list(pattern)
    m = len(pat)

    # Rebuild without absorption
    fail = [0] * (m + 1)
    for i in range(1, m):
        j = fail[i - 1]
        while j > 0 and pat[j] != pat[i]:
            j = fail[j - 1]
        if pat[j] == pat[i]:
            j += 1
        fail[i] = j

    transitions = {}
    for q in range(m + 1):
        for a in alphabet:
            state = q
            if state == m:
                state = fail[m - 1]  # un-absorb: go back
                # now continue matching from fail state
            while state > 0 and (state >= m or pat[state] != a):
                state = fail[state - 1]
            if state < m and pat[state] == a:
                state += 1
            transitions[(q, a)] = state

    dfa.transitions = transitions
    dfa.accept = {m}
    dfa.name = f"ends_with_{''.join(pattern)}"
    return dfa


# ============================================================
# Registry of built-in DFAs
# ============================================================

BUILTIN_DFAS = {
    # Tomita grammars
    "tomita_1": tomita_1,
    "tomita_2": tomita_2,
    "tomita_3": tomita_3,
    "tomita_4": tomita_4,
    "tomita_5": tomita_5,
    "tomita_6": tomita_6,
    "tomita_7": tomita_7,

    # Parity / modular counting
    "parity_2": lambda: parity_binary(2),
    "parity_3": lambda: parity_binary(3),
    "parity_5": lambda: parity_binary(5),
    "parity_7": lambda: parity_binary(7),
    "mod3_sigma4": lambda: modular_count(3, 4),
    "mod5_sigma4": lambda: modular_count(5, 4),

    # XOR variants
    "first_last_xor": first_last_xor,
    "two_step_xor": two_step_xor_dfa,

    # Bracket matching
    "dyck1_d2": lambda: dyck1_bounded(2),
    "dyck1_d3": lambda: dyck1_bounded(3),
    "dyck1_d4": lambda: dyck1_bounded(4),

    # Substring containment (SL class)
    "contains_00": lambda: contains_substring("00", ["0", "1"]),
    "contains_11": lambda: contains_substring("11", ["0", "1"]),
    "contains_010": lambda: contains_substring("010", ["0", "1"]),
    "contains_101": lambda: contains_substring("101", ["0", "1"]),
    "forbids_000": lambda: forbids_substring("000", ["0", "1"]),
    "forbids_111": lambda: forbids_substring("111", ["0", "1"]),

    # Ends-with
    "ends_01": lambda: ends_with("01", ["0", "1"]),
    "ends_10": lambda: ends_with("10", ["0", "1"]),
}


def get_dfa(name: str) -> DFA:
    """Get a DFA by name: built-in, dynamic random spec, or .att path."""
    if name in BUILTIN_DFAS:
        return BUILTIN_DFAS[name]()

    if parse_random_dfa_spec(name) is not None:
        return _cached_random_dfa(name)

    if name.startswith("att:"):
        path = name[4:]
        return load_att(path)

    raise KeyError(
        f"Unknown DFA '{name}'. Available: {sorted(BUILTIN_DFAS.keys())}\n"
        f"Dynamic random specs supported: 'random_{{num_states}}_{{alphabet_size}}', "
        f"'random_{{num_states}}_{{alphabet_size}}_seed{{seed}}', "
        f"optional suffix '_dense', '_bias_accept', '_bias_reject', optional '_kN' (bias peak factor, default k=2).\n"
        f"Or use 'att:/path/to/file.att' for custom DFAs.\n"
        f"Special non-DFA language specs supported in make_dfa_dataset: ['dyck1_unbounded']."
    )


# ============================================================
# Dataset generation (snn_p2.py interface)
# ============================================================

def autoregressive_dfa_spec_is_supported(dfa_spec: str) -> bool:
    """
    Autoregressive (state, input) -> (next state, label) is defined for:
      - built-in Tomita grammars,
      - bounded Dyck-1 DFAs,
      - dynamic random DFA specs ``random_{num_states}_{alphabet_size}``.

    It is not defined for ``dyck1_unbounded``, MLRegTest string files, or generic .att loaders.
    """
    s = str(dfa_spec)
    if s == "dyck1_unbounded" or s.startswith("mlregtest:"):
        return False
    if s.startswith("att:"):
        return False
    if parse_random_dfa_spec(s) is not None:
        return True
    if s in BUILTIN_DFAS and (s.startswith("tomita_") or s.startswith("dyck1_d")):
        return True
    return False


def _build_state_index_table(dfa: DFA) -> Tuple[Dict[int, int], int]:
    """
    Map each reachable DFA state and the dead/undefined state (-1) to 0..K-1.
    K = |states| + 1 (last index is the dead / undefined -- step returned -1).
    """
    ordered = sorted(int(s) for s in dfa.states)
    m: Dict[int, int] = {s: i for i, s in enumerate(ordered)}
    dead = len(ordered)
    m[-1] = dead
    n_cls = dead + 1
    return m, n_cls


def get_dfa_start_state_index(dfa: DFA) -> int:
    m, _ = _build_state_index_table(dfa)
    s = int(dfa.start)
    if s not in m:
        raise KeyError(f"Start state {s} not in DFA state index table.")
    return m[s]


def strings_to_autoregressive_tensors(
    dfa: DFA,
    strings: List[List[str]],
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    For each string (length T, symbols in dfa.alphabet), build timestep inputs
    (prev_state_onehot, current_symbol_onehot) and targets (next state class,
    per-step accept of the next state).

    Returns
    -------
    X : (n, T, d_in) with d_in = n_state_classes + |alphabet|
    y_state : (n, T) int64 class index for state after the symbol at t
    y_label : (n, T) int64 in {0,1} (next state in accept set)
    n_state_classes, d_in
    """
    d_sym = int(len(dfa.alphabet))
    sym2idx = dfa.sym2idx
    st2, n_state_cls = _build_state_index_table(dfa)
    d_in = n_state_cls + d_sym
    n = len(strings)
    X = np.zeros((n, T, d_in), dtype=np.float32)
    y_state = np.zeros((n, T), dtype=np.int64)
    y_label = np.zeros((n, T), dtype=np.int64)
    for i, s in enumerate(strings):
        if len(s) != T:
            raise ValueError(f"Autoregressive format expects length T={T}, got {len(s)}.")
        state = int(dfa.start)
        for t in range(T):
            sym = s[t]
            if sym not in sym2idx:
                raise KeyError(f"Symbol {sym!r} not in alphabet {dfa.alphabet}.")
            prev_idx = st2[state] if state in st2 else st2[-1]
            X[i, t, :n_state_cls] = 0.0
            X[i, t, prev_idx] = 1.0
            X[i, t, n_state_cls + sym2idx[sym]] = 1.0
            nxt = dfa.step(state, sym)
            if nxt == -1:
                y_state[i, t] = st2[-1]
                y_label[i, t] = 0
                state = -1
            else:
                y_state[i, t] = st2[nxt]
                y_label[i, t] = 1 if nxt in dfa.accept else 0
                state = nxt
    return X, y_state, y_label, n_state_cls, d_in


def make_dfa_autoregressive_dataset(
    dfa_spec: str,
    n: int,
    T: int,
    seed: int = 0,
    balanced: bool = True,
    return_strings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str, Optional[List[List[str]]]]:
    """
    Balanced (or not) DFA strings with (prev_state, symbol) -> (next state, accept bit).

    *Only* built-in ``tomita_*``, ``dyck1_d*`` (bounded Dyck), and dynamic
    random specs ``random_{num_states}_{alphabet_size}``; raises for
    ``dyck1_unbounded`` or external file specs.

    Returns
    -------
    X, y_state, y_label, n_state_classes, d_in, dfa_spec
    """
    if not autoregressive_dfa_spec_is_supported(dfa_spec):
        raise ValueError(
            f"autoregressive dataset not supported for dfa_spec={dfa_spec!r}. "
            "Use a built-in tomita_* / dyck1_d* (bounded) name or a dynamic random spec 'random_{num_states}_{alphabet_size}'."
        )
    dfa = get_dfa(dfa_spec)
    strings, _labels = dfa.generate_strings(T, n, seed=seed, balanced=balanced)
    X, y_s, y_l, n_sc, d_in = strings_to_autoregressive_tensors(dfa, strings, T)
    s_out: Optional[List[List[str]]] = strings if return_strings else None
    return X, y_s, y_l, n_sc, d_in, dfa_spec, s_out


def make_dfa_dataset(
    dfa_spec: str,
    n: int,
    T: int,
    seed: int = 0,
    balanced: bool = True,
    mlregtest_root: Optional[str] = None,
    mlregtest_split: str = "Train",
    mlregtest_test_split: str = "TestSR",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate dataset from a DFA specification.

    For autoregressive (prev state + symbol) supervision, use
    :func:`make_dfa_autoregressive_dataset` (Tomita and bounded Dyck only).

    Parameters
    ----------
    dfa_spec : str
        One of:
        - Built-in name: "tomita_3", "parity_5", etc.
        - ATT file: "att:/path/to/file.att"
        - MLRegTest: "mlregtest:/path/to/langname_prefix"
    n : int
        Number of samples to generate.
    T : int
        Sequence length.
    seed : int
        Random seed.
    balanced : bool
        If True, aim for ~50/50 class balance.

    Returns
    -------
    X : np.ndarray, shape (n, T, d_in) with d_in = |alphabet|
    y : np.ndarray, shape (n,) ∈ {0, 1}
    num_classes : int = 2
    """
    if dfa_spec == "dyck1_unbounded":
        strings, labels = generate_dyck1_unbounded_strings(
            length=T,
            n=n,
            seed=seed,
            balanced=balanced,
        )
        sym2idx = {"(": 0, ")": 1}
        X = np.zeros((len(strings), T, 2), dtype=np.float32)
        for i, s in enumerate(strings):
            for t in range(min(len(s), T)):
                X[i, t, sym2idx[s[t]]] = 1.0
        return X, labels, 2

    if dfa_spec.startswith("mlregtest:"):
        # Load pre-generated data from MLRegTest files
        prefix = dfa_spec[len("mlregtest:"):]
        return _load_mlregtest_as_dataset(prefix, n, T, seed,
                                          mlregtest_split, mlregtest_test_split)

    # Get DFA and generate from it
    dfa = get_dfa(dfa_spec)
    strings, labels = dfa.generate_strings(T, n, seed=seed, balanced=balanced)
    X = dfa.strings_to_onehot(strings, T)
    return X, labels, 2


def _load_mlregtest_as_dataset(
    prefix: str,
    n: int,
    T: int,
    seed: int,
    train_split: str = "Train",
    test_split: str = "TestSR",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load MLRegTest data, pad/truncate to T, subsample to n."""
    strings, labels, alphabet = load_mlregtest_data(prefix, train_split)
    sym2idx = {s: i for i, s in enumerate(sorted(set().union(*[set(s) for s in strings])))}
    d = len(sym2idx)

    # Subsample
    rng = np.random.default_rng(seed)
    if len(strings) > n:
        idx = rng.choice(len(strings), size=n, replace=False)
        strings = [strings[i] for i in idx]
        labels = labels[idx]
    elif len(strings) < n:
        # Oversample with replacement
        idx = rng.choice(len(strings), size=n, replace=True)
        strings = [strings[i] for i in idx]
        labels = labels[idx]

    # Convert to one-hot (n, T, d)
    X = np.zeros((len(strings), T, d), dtype=np.float32)
    for i, s in enumerate(strings):
        for t in range(min(len(s), T)):
            idx = sym2idx.get(s[t], -1)
            if idx >= 0:
                X[i, t, idx] = 1.0

    return X, labels, 2


# ============================================================
# Standalone CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DFA Test Bench")
    parser.add_argument("--list", action="store_true",
                        help="List all built-in DFAs")
    parser.add_argument("--name", type=str, default=None,
                        help="Built-in DFA name or dynamic random spec random_{num_states}_{alphabet_size}")
    parser.add_argument("--att", type=str, default=None,
                        help="Path to .att file")
    parser.add_argument("--T", type=int, default=20,
                        help="Sequence length")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="Emit (prev_state, symbol) -> (next_state, accept) sequences for built-in "
        "tomita_*, dyck1_d* (bounded), and random_{num_states}_{alphabet_size}.",
    )
    args = parser.parse_args()

    if args.list:
        print(f"Built-in DFAs ({len(BUILTIN_DFAS)}):\n")
        for name, factory in sorted(BUILTIN_DFAS.items()):
            dfa = factory()
            print(f"  {name:<20} {dfa.num_states} states, |Σ|={dfa.num_symbols} "
                  f"Σ={dfa.alphabet}, accept={dfa.accept}")
        return

    if args.autoregressive:
        if args.att:
            raise ValueError("--autoregressive requires --name (built-in tomita_*, dyck1_d*, or random_{q}_{a}), not --att")
        if not args.name:
            raise ValueError("--autoregressive requires --name tomita_*, dyck1_d*, or random_{q}_{a}")
        X, y_state, y_label, n_sc, d_in, spec, _s = make_dfa_autoregressive_dataset(
            args.name, args.n, args.T, seed=args.seed, balanced=True, return_strings=False
        )
        dfa = get_dfa(args.name)
        print(dfa.summary())
        print(
            f"\nAutoregressive (balanced) dataset: spec={spec!r} n_state_classes={n_sc} d_in={d_in}"
        )
        print(f"  X shape: {X.shape}  y_state: {y_state.shape}  y_label: {y_label.shape}")
        for i in range(min(3, X.shape[0])):
            print(f"  ex{i} y_state row0..2: {y_state[i, : min(3, args.T)]}  y_label: {y_label[i, : min(3, args.T)]}")
        return

    # Build DFA
    if args.att:
        dfa = load_att(args.att)
    elif args.name:
        dfa = get_dfa(args.name)
    else:
        parser.print_help()
        return

    print(dfa.summary())

    # Generate data and show statistics
    strings, labels = dfa.generate_strings(args.T, args.n, seed=args.seed, balanced=True)
    X = dfa.strings_to_onehot(strings, args.T)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"\nGenerated {len(strings)} strings of length {args.T}")
    print(f"  Positive: {n_pos} ({100*n_pos/len(labels):.1f}%)")
    print(f"  Negative: {n_neg} ({100*n_neg/len(labels):.1f}%)")
    print(f"  X shape: {X.shape} (n, T, |Σ|)")

    # Show a few examples
    print(f"\nExamples:")
    for i in range(min(8, len(strings))):
        s = "".join(strings[i])
        print(f"  {'✓' if labels[i] else '✗'} {s}")

    # Test acceptance rate on random strings (unbalanced)
    rng = np.random.default_rng(args.seed + 42)
    test_n = 10000
    syms = rng.integers(0, len(dfa.alphabet), size=(test_n, args.T))
    acc = sum(
        dfa.accepts([dfa.alphabet[s] for s in row])
        for row in syms
    )
    print(f"\nNatural acceptance rate (T={args.T}): {100*acc/test_n:.1f}% "
          f"({acc}/{test_n})")


if __name__ == "__main__":
    main()
