import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class SampleRecord:
    datapoint: int
    label: int
    sequence: List[str]
    violation_t: int
    activations: Dict[Tuple[int, int], str]  # (layer, t) -> canonical signature string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Symbolic analysis over activation-index reports. "
            "Runs DFA per sequence, finds first 1->0 transition timestep, and "
            "compares reject-layer activation groups versus accept at matched timesteps."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/Users/hima_3114/Desktop/Paper_1/experiments/snn_generalized_pt2/activation_indices",
        help="Activation report file or directory (recursive).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/hima_3114/Desktop/Paper_1/experiments/snn_generalized_pt2/symbolic_analysis_outputs",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--focus_label",
        type=int,
        default=0,
        help="Reject label (default 0).",
    )
    parser.add_argument(
        "--min_reject_prob",
        type=float,
        default=0.8,
        help="Threshold on P(active-group|reject) for pure reject group.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.5,
        help="Threshold on P(reject)-P(accept) for pure reject group.",
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=None,
        help="Debug limit: maximum number of datapoints loaded per file.",
    )
    return parser.parse_args()


def parse_sequence(raw: str) -> List[str]:
    txt = raw.strip()
    if txt == "":
        return []
    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1]
    txt = txt.replace(",", " ")
    toks = [tok for tok in txt.split() if tok != ""]
    return toks


def canonicalize_signature(raw: str) -> str:
    # Activation indices emitted by snn_p2.py are already ordered from np.flatnonzero.
    # Keep this O(len(raw)) instead of reparsing/sorting huge signatures repeatedly.
    return raw.strip()


def parse_header_line(line: str) -> Tuple[str, str] | None:
    # Header format written by snn_p2.py:
    # # key: value
    txt = line.strip()
    if not txt.startswith("#"):
        return None
    body = txt[1:].strip()
    if ":" not in body:
        return None
    k, v = body.split(":", 1)
    return k.strip(), v.strip()


def parse_datapoint_line(line: str) -> Tuple[int, List[str], int, int] | None:
    # Format:
    # datapoint=<i> sequence=(...) -> label=<y> violation_t=<t>
    txt = line.strip()
    if not txt.startswith("datapoint="):
        return None
    if " sequence=" not in txt or " -> label=" not in txt:
        raise ValueError(f"Malformed datapoint line: {line}")
    left, right = txt.split(" -> label=", 1)
    dp_part, seq_part = left.split(" sequence=", 1)
    dp = int(dp_part.replace("datapoint=", "").strip())
    seq = parse_sequence(seq_part.strip())

    if " violation_t=" in right:
        label_txt, vt_txt = right.split(" violation_t=", 1)
        label = int(label_txt.strip())
        violation_t = int(vt_txt.strip())
    else:
        label = int(right.strip())
        violation_t = -1
    return dp, seq, label, violation_t


def parse_lt_activation_line(line: str) -> Tuple[Tuple[int, int], str] | None:
    # Format:
    # (l,t),idx;idx;...
    txt = line.strip()
    if not txt.startswith("("):
        return None
    if ")," not in txt:
        raise ValueError(f"Malformed activation line: {line}")
    lt_txt, sig_txt = txt.split("),", 1)
    lt_txt = lt_txt + ")"
    inner = lt_txt[1:-1]
    a, b = inner.split(",")
    lt = (int(a.strip()), int(b.strip()))
    return lt, canonicalize_signature(sig_txt)


def parse_activation_report(path: Path, max_datapoints: int | None) -> Tuple[Dict[str, str], Dict[int, SampleRecord]]:
    metadata: Dict[str, str] = {}
    samples: Dict[int, SampleRecord] = {}
    current_dp: int | None = None
    loaded = 0

    with path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "":
                continue

            kv = parse_header_line(line)
            if kv is not None:
                metadata[kv[0]] = kv[1]
                continue

            parsed_dp = parse_datapoint_line(line)
            if parsed_dp is not None:
                dp, seq, label, vt = parsed_dp
                if dp in samples:
                    raise ValueError(f"{path}: duplicate datapoint block {dp}")
                samples[dp] = SampleRecord(
                    datapoint=dp,
                    label=label,
                    sequence=seq,
                    violation_t=vt,
                    activations={},
                )
                current_dp = dp
                loaded += 1
                if max_datapoints is not None and loaded >= max_datapoints:
                    break
                continue

            lt_parsed = parse_lt_activation_line(line)
            if lt_parsed is not None:
                if current_dp is None:
                    raise ValueError(f"{path}: activation line before datapoint header: {line}")
                lt, sig = lt_parsed
                if lt in samples[current_dp].activations:
                    raise ValueError(f"{path}: duplicate lt {lt} for datapoint {current_dp}")
                samples[current_dp].activations[lt] = sig
                continue

            raise ValueError(f"{path}: unrecognized line: {line}")

    if not samples:
        raise ValueError(f"{path}: no datapoint blocks parsed")
    return metadata, samples


def first_accept_to_reject_timestep_with_dfa(dfa, sequence: List[str]) -> int:
    state = dfa.start
    prev_accept = state in dfa.accept
    for t, sym in enumerate(sequence):
        state = dfa.step(state, sym)
        cur_accept = state in dfa.accept if state != -1 else False
        if prev_accept and (not cur_accept):
            return t
        prev_accept = cur_accept
    return -1


def collect_layers_and_timesteps(samples: Dict[int, SampleRecord]) -> Tuple[List[int], List[int]]:
    all_lt = set()
    for rec in samples.values():
        all_lt.update(rec.activations.keys())
    if not all_lt:
        raise ValueError("No (layer,t) activations found")
    layers = sorted({l for l, _ in all_lt})
    timesteps = sorted({t for _, t in all_lt})
    return layers, timesteps


def matched_accept_probability(
    signature: str,
    layer: int,
    reject_t_counts: Counter[int],
    accept_records: List[SampleRecord],
) -> float:
    total_reject_events = sum(reject_t_counts.values())
    if total_reject_events == 0:
        return 0.0

    accept_by_t: Dict[int, List[str]] = defaultdict(list)
    for rec in accept_records:
        for (l, t), sig in rec.activations.items():
            if l == layer:
                accept_by_t[t].append(sig)

    p_accept = 0.0
    for t, c in reject_t_counts.items():
        sigs = accept_by_t.get(t, [])
        if not sigs:
            continue
        frac = sum(1 for s in sigs if s == signature) / len(sigs)
        p_accept += (c / total_reject_events) * frac
    return p_accept


def analyze_dfa_violation_groups(
    metadata: Dict[str, str],
    samples: Dict[int, SampleRecord],
    focus_label: int,
    min_reject_prob: float,
    min_delta: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    dfa_name = metadata.get("dfa_name")
    if dfa_name is None or dfa_name == "":
        task = metadata.get("task", "")
        if task.startswith("dfa:"):
            dfa_name = task[4:]
        else:
            raise ValueError("DFA analysis requires metadata field 'dfa_name' or task prefixed with 'dfa:'.")

    from dfa_tasks import get_dfa

    dfa = get_dfa(dfa_name)

    # Recompute violation_t from DFA to ensure consistency with specification.
    computed_violation: Dict[int, int] = {}
    for dp, rec in samples.items():
        computed_violation[dp] = first_accept_to_reject_timestep_with_dfa(dfa, rec.sequence)

    layers, _ = collect_layers_and_timesteps(samples)
    reject_records = [r for r in samples.values() if r.label == focus_label]
    accept_records = [r for r in samples.values() if r.label != focus_label]

    layer_rows: List[Dict[str, object]] = []
    neuron_rows: List[Dict[str, object]] = []
    missing_vt = 0

    for layer in layers:
        reject_signatures: List[str] = []
        reject_t_counts: Counter[int] = Counter()
        per_signature_count: Counter[str] = Counter()
        accept_sig_counter_by_t: Dict[int, Counter[str]] = defaultdict(Counter)

        for rec in reject_records:
            vt = computed_violation.get(rec.datapoint, -1)
            if vt < 0:
                missing_vt += 1
                continue
            key = (layer, vt)
            if key not in rec.activations:
                continue
            sig = rec.activations[key]
            reject_signatures.append(sig)
            per_signature_count[sig] += 1
            reject_t_counts[vt] += 1
        for rec in accept_records:
            for (ll, t), sig in rec.activations.items():
                if ll == layer:
                    accept_sig_counter_by_t[t][sig] += 1

        if not reject_signatures:
            continue

        dominant_sig, dominant_count = per_signature_count.most_common(1)[0]
        reject_total = len(reject_signatures)
        p_reject = dominant_count / reject_total
        p_accept = 0.0
        for t, c in reject_t_counts.items():
            sig_counter_t = accept_sig_counter_by_t.get(t)
            if not sig_counter_t:
                continue
            total_t = sum(sig_counter_t.values())
            frac = sig_counter_t.get(dominant_sig, 0) / total_t
            p_accept += (c / reject_total) * frac
        delta = p_reject - p_accept
        lift = p_reject / max(p_accept, 1e-12)

        layer_rows.append(
            {
                "layer": layer,
                "dominant_signature": dominant_sig,
                "reject_count": dominant_count,
                "reject_total": reject_total,
                "p_reject": p_reject,
                "p_accept_matched_t": p_accept,
                "delta": delta,
                "lift": lift,
                "num_unique_reject_signatures": len(per_signature_count),
                "is_pure_reject_group": int((p_reject >= min_reject_prob) and (delta >= min_delta)),
            }
        )

        # Per-neuron probabilities inside this layer at violation time
        reject_neuron_count: Counter[int] = Counter()
        for rec in reject_records:
            vt = computed_violation.get(rec.datapoint, -1)
            if vt < 0:
                continue
            key = (layer, vt)
            sig = rec.activations.get(key, "")
            if sig == "":
                continue
            for tok in sig.split(";"):
                reject_neuron_count[int(tok)] += 1

        signature_neuron_cache: Dict[str, set[int]] = {}
        accept_neuron_prob_by_t: Dict[int, Dict[int, float]] = {}
        for t, sig_counter_t in accept_sig_counter_by_t.items():
            total_t = sum(sig_counter_t.values())
            neuron_count_t: Counter[int] = Counter()
            for sig_str, cnt in sig_counter_t.items():
                if sig_str not in signature_neuron_cache:
                    signature_neuron_cache[sig_str] = (
                        {int(tok) for tok in sig_str.split(";")} if sig_str else set()
                    )
                for nid in signature_neuron_cache[sig_str]:
                    neuron_count_t[nid] += cnt
            accept_neuron_prob_by_t[t] = {
                nid: c / total_t for nid, c in neuron_count_t.items()
            }

        for neuron_id, c in reject_neuron_count.items():
            p_r = c / reject_total
            # p_accept for neuron at matched t
            weighted = 0.0
            for t, cnt_t in reject_t_counts.items():
                prob_t = accept_neuron_prob_by_t.get(t, {}).get(neuron_id, 0.0)
                frac_t = prob_t
                weighted += (cnt_t / reject_total) * frac_t
            d = p_r - weighted
            neuron_rows.append(
                {
                    "layer": layer,
                    "neuron_id": neuron_id,
                    "p_reject": p_r,
                    "p_accept_matched_t": weighted,
                    "delta": d,
                    "lift": p_r / max(weighted, 1e-12),
                    "is_pure_reject_neuron": int((p_r >= min_reject_prob) and (d >= min_delta)),
                }
            )

    layer_rows.sort(key=lambda r: (int(r["layer"]), -float(r["delta"])))
    neuron_rows.sort(key=lambda r: (int(r["layer"]), -float(r["delta"]), -float(r["p_reject"])))

    meta = {
        "dfa_name": dfa_name,
        "num_datapoints": len(samples),
        "num_reject": len(reject_records),
        "num_accept": len(accept_records),
        "missing_violation_t_reject_records": missing_vt,
    }
    return layer_rows, neuron_rows, meta


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            for k, v in serial.items():
                if isinstance(v, (list, tuple, set, dict)):
                    serial[k] = str(v)
            writer.writerow(serial)


def collect_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.rglob("*.txt")) + sorted(input_path.rglob("*.csv"))
        # Keep only files likely to be activation reports with datapoint blocks/header format.
        return [p for p in paths if p.name.startswith("202") or "activation_indices" in p.name]
    raise ValueError(f"{input_path}: path does not exist")


def analyze_file(path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    metadata, samples = parse_activation_report(path, max_datapoints=args.max_datapoints)
    layer_rows, neuron_rows, meta = analyze_dfa_violation_groups(
        metadata=metadata,
        samples=samples,
        focus_label=args.focus_label,
        min_reject_prob=args.min_reject_prob,
        min_delta=args.min_delta,
    )

    tag = path.stem
    layer_out = out_dir / f"{tag}_dfa_violation_layer_groups.csv"
    neuron_out = out_dir / f"{tag}_dfa_violation_reject_neurons.csv"

    write_csv(
        layer_out,
        layer_rows,
        columns=[
            "layer",
            "dominant_signature",
            "reject_count",
            "reject_total",
            "p_reject",
            "p_accept_matched_t",
            "delta",
            "lift",
            "num_unique_reject_signatures",
            "is_pure_reject_group",
        ],
    )
    write_csv(
        neuron_out,
        neuron_rows,
        columns=[
            "layer",
            "neuron_id",
            "p_reject",
            "p_accept_matched_t",
            "delta",
            "lift",
            "is_pure_reject_neuron",
        ],
    )

    pure_layers = sum(int(r["is_pure_reject_group"]) for r in layer_rows)
    pure_neurons = sum(int(r["is_pure_reject_neuron"]) for r in neuron_rows)
    print(f"\n[{path}]")
    print(
        f"dfa={meta['dfa_name']} datapoints={meta['num_datapoints']} "
        f"reject={meta['num_reject']} accept={meta['num_accept']}"
    )
    print(
        f"missing_violation_t_reject_records={meta['missing_violation_t_reject_records']} "
        f"pure_layers={pure_layers}/{len(layer_rows)} pure_neurons={pure_neurons}/{len(neuron_rows)}"
    )
    print(f"saved: {layer_out}")
    print(f"saved: {neuron_out}")
    if layer_rows:
        best = max(layer_rows, key=lambda r: float(r["delta"]))
        print(
            f"top_layer={best['layer']} p_reject={float(best['p_reject']):.4f} "
            f"p_accept={float(best['p_accept_matched_t']):.4f} delta={float(best['delta']):.4f}"
        )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_input_files(input_path)
    if not files:
        raise ValueError(f"No candidate activation report files found under {input_path}")
    print(f"found_files={len(files)}")
    for p in files:
        analyze_file(p, out_dir, args)


if __name__ == "__main__":
    main()
