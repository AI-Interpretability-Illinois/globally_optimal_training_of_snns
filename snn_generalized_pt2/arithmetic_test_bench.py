#!/usr/bin/env python3
"""
Per-timestep arithmetic sequence test-bench.

At each timestep the input is:
    (operand_1_digit, operand_2_digit, carry_or_remainder_in)

The target is now a TOKEN emitted at EVERY timestep.
This converts the benchmark from final-step sequence classification to
per-step sequence transduction / token classification.

Token convention:
- add/sub/mul: tokens are output digits in LSD-first order
- div: quotient digits are emitted in MSD-first order for each processed digit,
       followed by one final token for the remainder digit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

SUPPORTED_BASES = (2, 3, 5, 7, 10)
SUPPORTED_OPS = ("add", "sub", "mul", "div")
DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def int_to_base_str(x: int, base: int) -> str:
    if x < 0:
        return "-" + int_to_base_str(-x, base)
    if x == 0:
        return "0"
    out = []
    while x > 0:
        out.append(DIGITS[x % base])
        x //= base
    return "".join(reversed(out))


def digits_lsd(x: int, base: int, min_len: int = 1) -> List[int]:
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
    inputs: np.ndarray          # (T, 3)
    target_tokens: np.ndarray   # (T,), integer tokens in [0, base-1]
    result: str                 # human-readable result string


# ------------------------------------------------------------------
# Fixed-length per-timestep builders
# ------------------------------------------------------------------

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
    # Process 2*n_digits-1 product columns, then one final carry column.
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
    # Final carry digit; for valid n-digit multiplication it is guaranteed < base.
    rows.append([0, 0, carry])
    out.append(carry)
    result = digits_to_str_msd(reversed(out))
    return np.asarray(rows, dtype=np.int64), np.asarray(out, dtype=np.int64), result


def build_div_sequence_tokens(a: int, b_digit: int, base: int, n_digits: int) -> Tuple[np.ndarray, np.ndarray, str]:
    # Long division with single-digit divisor. Output one quotient digit per input digit,
    # then one final remainder token.
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
    # Final remainder step.
    rows.append([0, b_digit, remainder])
    out = q_digits + [remainder]
    q_str = digits_to_str_msd(q_digits)
    result = f"{q_str}|r{DIGITS[remainder]}"
    return np.asarray(rows, dtype=np.int64), np.asarray(out, dtype=np.int64), result


def generate_samples_for_op_base_seq(
    op: str,
    base: int,
    n_digits: int,
    n_samples: int,
    seed: int,
) -> List[ArithmeticTokenSample]:
    rng = np.random.default_rng(seed)
    out: List[ArithmeticTokenSample] = []
    max_val = base ** n_digits

    for _ in range(n_samples):
        a = int(rng.integers(0, max_val))
        b = int(rng.integers(0, max_val))

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

        out.append(
            ArithmeticTokenSample(
                op=op,
                base=base,
                operand1=int_to_base_str(a, base),
                operand2=op2_display,
                inputs=inputs,
                target_tokens=target_tokens,
                result=result,
            )
        )
    return out


def verify_sample_seq(sample: ArithmeticTokenSample) -> None:
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
    assert sample.result == expected, f"{sample.op} mismatch: {sample.result} vs {expected}"
    assert sample.inputs.shape[0] == sample.target_tokens.shape[0], "input/target length mismatch"


def run_bench(
    bases: Sequence[int],
    ops: Sequence[str],
    n_digits: int,
    n_samples: int,
    seed: int,
    print_examples: int,
) -> Dict[Tuple[int, str], List[ArithmeticTokenSample]]:
    all_sets: Dict[Tuple[int, str], List[ArithmeticTokenSample]] = {}
    for base in bases:
        for op in ops:
            samples = generate_samples_for_op_base_seq(
                op=op,
                base=base,
                n_digits=n_digits,
                n_samples=n_samples,
                seed=seed + 31 * base + 7 * len(op),
            )
            for s in samples:
                verify_sample_seq(s)
            all_sets[(base, op)] = samples
            print(f"[ok] base={base} op={op} samples={len(samples)} verified")

            for s in samples[:print_examples]:
                print(
                    f"  sample base={base} op={op}: "
                    f"a={s.operand1}, b={s.operand2}, T={s.inputs.shape[0]}, result={s.result}"
                )
                show_t = min(8, s.inputs.shape[0])
                for t in range(show_t):
                    o1, o2, c = s.inputs[t].tolist()
                    tgt = int(s.target_tokens[t])
                    print(f"    t={t:02d}: in=({o1},{o2},{c}) target_token={tgt}")
                if s.inputs.shape[0] > show_t:
                    print("    ...")
    return all_sets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-timestep arithmetic sequence test-bench")
    p.add_argument("--bases", type=int, nargs="+", default=list(SUPPORTED_BASES))
    p.add_argument("--ops", type=str, nargs="+", default=list(SUPPORTED_OPS))
    p.add_argument("--n_digits", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_examples", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for b in args.bases:
        assert b in SUPPORTED_BASES, f"Unsupported base: {b}"
    for op in args.ops:
        assert op in SUPPORTED_OPS, f"Unsupported op: {op}"
    run_bench(
        bases=args.bases,
        ops=args.ops,
        n_digits=args.n_digits,
        n_samples=args.n_samples,
        seed=args.seed,
        print_examples=args.print_examples,
    )


if __name__ == "__main__":
    main()
