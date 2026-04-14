from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

SUPPORTED_BASES = (2, 3, 5, 7, 10)
SUPPORTED_OPS = ("add", "sub", "mul", "div")
DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def int_to_base_str(x: int, base: int) -> str:
    if base < 2 or base > len(DIGITS):
        raise ValueError(f"base must be in [2, {len(DIGITS)}], got {base}.")
    if x < 0:
        return "-" + int_to_base_str(-x, base)
    if x == 0:
        return "0"
    out: List[str] = []
    while x > 0:
        out.append(DIGITS[x % base])
        x //= base
    return "".join(reversed(out))


def digits_lsd(x: int, base: int, min_len: int = 1) -> List[int]:
    if base < 2:
        raise ValueError(f"base must be >=2, got {base}.")
    if min_len < 1:
        raise ValueError(f"min_len must be >=1, got {min_len}.")
    if x < 0:
        raise ValueError("digits_lsd expects non-negative integer input.")
    ds: List[int] = []
    if x == 0:
        ds = [0]
    else:
        while x > 0:
            ds.append(x % base)
            x //= base
    while len(ds) < min_len:
        ds.append(0)
    return ds


def digits_to_str_msd(digits: Sequence[int]) -> str:
    ds = list(digits)
    i = 0
    while i < len(ds) - 1 and ds[i] == 0:
        i += 1
    ds = ds[i:]
    return "".join(DIGITS[d] for d in ds)


@dataclass
class ArithmeticTokenSample:
    op: str
    base: int
    operand1: str
    operand2: str
    inputs: np.ndarray
    target_tokens: np.ndarray
    result: str


def _validate_generation_args(op: str, base: int, n_digits: int, n_samples: int) -> None:
    if op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported op={op}. Supported ops: {SUPPORTED_OPS}.")
    if base not in SUPPORTED_BASES:
        raise ValueError(f"Unsupported base={base}. Supported bases: {SUPPORTED_BASES}.")
    if n_digits < 1:
        raise ValueError(f"n_digits must be >=1, got {n_digits}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >=1, got {n_samples}.")


def build_add_sequence_tokens(a: int, b: int, base: int, n_digits: int) -> Tuple[np.ndarray, np.ndarray, str]:
    a_ds = digits_lsd(a, base, min_len=n_digits)
    b_ds = digits_lsd(b, base, min_len=n_digits)
    rows: List[List[int]] = []
    out: List[int] = []
    carry = 0
    for t in range(n_digits):
        carry_in = carry
        s = a_ds[t] + b_ds[t] + carry_in
        out_digit = s % base
        carry = s // base
        rows.append([a_ds[t], b_ds[t], carry_in])
        out.append(out_digit)
    rows.append([0, 0, carry])
    out.append(carry)
    result = digits_to_str_msd(reversed(out))
    return np.asarray(rows, dtype=np.int64), np.asarray(out, dtype=np.int64), result


def build_sub_sequence_tokens(a: int, b: int, base: int, n_digits: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if a < b:
        raise ValueError("Subtraction requires a >= b.")
    a_ds = digits_lsd(a, base, min_len=n_digits)
    b_ds = digits_lsd(b, base, min_len=n_digits)
    rows: List[List[int]] = []
    out: List[int] = []
    borrow = 0
    for t in range(n_digits):
        borrow_in = borrow
        d = a_ds[t] - b_ds[t] - borrow_in
        if d < 0:
            d += base
            borrow = 1
        else:
            borrow = 0
        rows.append([a_ds[t], b_ds[t], borrow_in])
        out.append(d)
    result = digits_to_str_msd(reversed(out))
    return np.asarray(rows, dtype=np.int64), np.asarray(out, dtype=np.int64), result


def build_mul_sequence_tokens(a: int, b: int, base: int, n_digits: int) -> Tuple[np.ndarray, np.ndarray, str]:
    a_ds = digits_lsd(a, base, min_len=n_digits)
    b_ds = digits_lsd(b, base, min_len=n_digits)
    rows: List[List[int]] = []
    out: List[int] = []
    carry = 0
    for k in range(2 * n_digits - 1):
        carry_in = carry
        column_sum = carry_in
        rep_a, rep_b = 0, 0
        rep_found = False
        for i in range(max(0, k - (n_digits - 1)), min(n_digits - 1, k) + 1):
            j = k - i
            column_sum += a_ds[i] * b_ds[j]
            if not rep_found:
                rep_a, rep_b = a_ds[i], b_ds[j]
                rep_found = True
        out_digit = column_sum % base
        carry = column_sum // base
        rows.append([rep_a, rep_b, carry_in])
        out.append(out_digit)
    rows.append([0, 0, carry])
    out.append(carry)
    result = digits_to_str_msd(reversed(out))
    return np.asarray(rows, dtype=np.int64), np.asarray(out, dtype=np.int64), result


def build_div_sequence_tokens(a: int, b_digit: int, base: int, n_digits: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if not (1 <= b_digit < base):
        raise ValueError("Division divisor must be a single non-zero base digit.")
    a_lsd = digits_lsd(a, base, min_len=n_digits)
    a_msd = list(reversed(a_lsd))
    rows: List[List[int]] = []
    q_digits: List[int] = []
    remainder = 0
    for d in a_msd:
        rem_in = remainder
        cur = rem_in * base + d
        q = cur // b_digit
        remainder = cur % b_digit
        rows.append([d, b_digit, rem_in])
        q_digits.append(q)
    rows.append([0, b_digit, remainder])
    out = q_digits + [remainder]
    q_str = digits_to_str_msd(q_digits)
    result = f"{q_str}|r{DIGITS[remainder]}"
    return np.asarray(rows, dtype=np.int64), np.asarray(out, dtype=np.int64), result


def verify_sample_seq(sample: ArithmeticTokenSample) -> None:
    """Hard correctness check for a generated arithmetic token sample."""
    a = int(sample.operand1, sample.base)
    if sample.op != "div":
        b = int(sample.operand2, sample.base)

    if sample.op == "add":
        expected = int_to_base_str(a + b, sample.base)
    elif sample.op == "sub":
        expected = int_to_base_str(a - b, sample.base)
    elif sample.op == "mul":
        expected = int_to_base_str(a * b, sample.base)
    elif sample.op == "div":
        b_digit = int(sample.operand2, sample.base)
        q = a // b_digit
        r = a % b_digit
        expected = f"{int_to_base_str(q, sample.base)}|r{int_to_base_str(r, sample.base)}"
    else:
        raise ValueError(f"Unsupported op: {sample.op}")

    if sample.result != expected:
        raise ValueError(f"Result mismatch for {sample.op}: got {sample.result}, expected {expected}.")
    if sample.inputs.shape[0] != sample.target_tokens.shape[0]:
        raise ValueError("Input/target timestep length mismatch.")


def _build_single_sample(
    *,
    op: str,
    base: int,
    n_digits: int,
    a: int,
    b: int,
    rng: np.random.Generator,
) -> ArithmeticTokenSample:
    if op == "sub" and a < b:
        a, b = b, a

    if op == "add":
        inputs, target_tokens, result = build_add_sequence_tokens(a, b, base, n_digits)
        op2_display = int_to_base_str(b, base)
    elif op == "sub":
        inputs, target_tokens, result = build_sub_sequence_tokens(a, b, base, n_digits)
        op2_display = int_to_base_str(b, base)
    elif op == "mul":
        inputs, target_tokens, result = build_mul_sequence_tokens(a, b, base, n_digits)
        op2_display = int_to_base_str(b, base)
    elif op == "div":
        b_digit = int(rng.integers(1, base))
        inputs, target_tokens, result = build_div_sequence_tokens(a, b_digit, base, n_digits)
        op2_display = int_to_base_str(b_digit, base)
    else:
        raise ValueError(f"Unsupported op: {op}")

    return ArithmeticTokenSample(
        op=op,
        base=base,
        operand1=int_to_base_str(a, base),
        operand2=op2_display,
        inputs=inputs,
        target_tokens=target_tokens,
        result=result,
    )


def generate_samples_for_op_base_seq(
    op: str,
    base: int,
    n_digits: int,
    n_samples: int,
    seed: int,
) -> List[ArithmeticTokenSample]:
    _validate_generation_args(op=op, base=base, n_digits=n_digits, n_samples=n_samples)
    rng = np.random.default_rng(seed)
    samples: List[ArithmeticTokenSample] = []
    max_val = base ** n_digits
    for _ in range(n_samples):
        a = int(rng.integers(0, max_val))
        b = int(rng.integers(0, max_val))
        samples.append(
            _build_single_sample(
                op=op,
                base=base,
                n_digits=n_digits,
                a=a,
                b=b,
                rng=rng,
            )
        )
    return samples


@dataclass
class ArithmeticDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    op: str
    base: int
    n_digits: int


def _stack(samples: Sequence[ArithmeticTokenSample], base: int) -> tuple[np.ndarray, np.ndarray]:
    if len(samples) == 0:
        raise ValueError("Cannot stack empty sample list.")
    x = np.stack([s.inputs.astype(np.float32) / float(base - 1) for s in samples], axis=0)
    y = np.stack([s.target_tokens.astype(np.int64) for s in samples], axis=0)
    return x, y


def _build_split(
    *,
    op: str,
    base: int,
    n_digits: int,
    n_samples: int,
    seed: int,
    verify_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = generate_samples_for_op_base_seq(
        op=op,
        base=base,
        n_digits=n_digits,
        n_samples=n_samples,
        seed=seed,
    )
    for sample in samples[:verify_count]:
        verify_sample_seq(sample)
    return _stack(samples=samples, base=base)


def load_arithmetic_dataset(
    *,
    op: str,
    base: int,
    n_digits: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    debug_verify_samples: int = 8,
) -> ArithmeticDataset:
    _validate_generation_args(op=op, base=base, n_digits=n_digits, n_samples=max(n_train, n_val, n_test))
    if debug_verify_samples < 0:
        raise ValueError(f"debug_verify_samples must be >=0, got {debug_verify_samples}.")

    x_train, y_train = _build_split(
        op=op,
        base=base,
        n_digits=n_digits,
        n_samples=n_train,
        seed=seed + 11,
        verify_count=min(debug_verify_samples, n_train),
    )
    x_val, y_val = _build_split(
        op=op,
        base=base,
        n_digits=n_digits,
        n_samples=n_val,
        seed=seed + 29,
        verify_count=min(debug_verify_samples, n_val),
    )
    x_test, y_test = _build_split(
        op=op,
        base=base,
        n_digits=n_digits,
        n_samples=n_test,
        seed=seed + 47,
        verify_count=min(debug_verify_samples, n_test),
    )

    return ArithmeticDataset(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        num_classes=int(base),
        op=op,
        base=base,
        n_digits=n_digits,
    )
