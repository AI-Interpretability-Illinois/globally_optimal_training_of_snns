#!/usr/bin/env python3
"""
DFA Test Bench for Convex SNN evaluation.

Provides:
  1. Generic DFA simulator
  2. Built-in DFA library (Tomita, parity, counters, XOR variants)
  3. .att file loader (OpenFst AT&T format, for MLRegTest)
  4. MLRegTest data file loader
  5. make_dfa_dataset() matching snn_p2.py interface: → (X, y, num_classes)

Usage from snn_p2.py:
  --task dfa:tomita_3         # built-in DFA
  --task dfa:parity_3         # parity mod 3
  --task dfa:att:/path/to.att # load .att file
  --task dfa:mlregtest:/path/to/langname  # load MLRegTest data files

Usage standalone:
  python dfa_tasks.py --list                          # list all built-in DFAs
  python dfa_tasks.py --name tomita_3 --T 20 --n 100  # generate + stats
  python dfa_tasks.py --att /path/to/file.att --T 30   # from .att file
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


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
    """Get a DFA by name: built-in, .att path, or mlregtest path."""
    if name in BUILTIN_DFAS:
        return BUILTIN_DFAS[name]()

    if name.startswith("att:"):
        path = name[4:]
        return load_att(path)

    raise KeyError(
        f"Unknown DFA '{name}'. Available: {sorted(BUILTIN_DFAS.keys())}\n"
        f"Or use 'att:/path/to/file.att' for custom DFAs.\n"
        f"Special non-DFA language specs supported in make_dfa_dataset: ['dyck1_unbounded']."
    )


# ============================================================
# Dataset generation (snn_p2.py interface)
# ============================================================

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
                        help="Built-in DFA name")
    parser.add_argument("--att", type=str, default=None,
                        help="Path to .att file")
    parser.add_argument("--T", type=int, default=20,
                        help="Sequence length")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.list:
        print(f"Built-in DFAs ({len(BUILTIN_DFAS)}):\n")
        for name, factory in sorted(BUILTIN_DFAS.items()):
            dfa = factory()
            print(f"  {name:<20} {dfa.num_states} states, |Σ|={dfa.num_symbols} "
                  f"Σ={dfa.alphabet}, accept={dfa.accept}")
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
