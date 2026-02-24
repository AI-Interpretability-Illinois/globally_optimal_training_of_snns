#!/usr/bin/env python3
"""
Parse RNN_logs.txt and generate:
1. Graphs showing train_acc over epochs for CVX vs STE for each run
2. Consolidated table with dataset, L, T, train_size, CVX test, STE test
"""

import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

def parse_logs(log_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse log file and extract run data and training curves."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    runs = []
    current_run = None
    current_cvx_epochs = []
    current_ste_epochs = []
    in_run = False
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect new run start - look for [info] with task=
        if '[info]' in line and 'task=' in line and 'device=' in line:
            # Save previous run if exists
            if current_run is not None and in_run:
                current_run['cvx_train_acc'] = sorted(current_cvx_epochs)
                current_run['ste_train_acc'] = sorted(current_ste_epochs)
                runs.append(current_run)
            
            # Start new run
            current_run = {}
            current_cvx_epochs = []
            current_ste_epochs = []
            in_run = True
            
            # Extract from info line
            task_match = re.search(r'task=(\S+)', line)
            t_match = re.search(r'T=(\d+)', line)
            l_match = re.search(r'L=(\d+)', line)
            
            if task_match:
                current_run['task'] = task_match.group(1)
            if t_match:
                current_run['T'] = int(t_match.group(1))
            if l_match:
                current_run['L'] = int(l_match.group(1))
        
        # Extract n_train_total and val_frac from separate info line
        if '[info]' in line and 'n_train_total=' in line:
            n_match = re.search(r'n_train_total=(\d+)', line)
            val_match = re.search(r'val_frac=([\d.]+)', line)
            if n_match and current_run:
                current_run['n_train_total'] = int(n_match.group(1))
            if val_match and current_run:
                current_run['val_frac'] = float(val_match.group(1))
        
        # Extract CVX training accuracies
        if '[CVX]' in line and 'train_acc=' in line and current_run:
            epoch_match = re.search(r'ep=(\d+)/', line)
            acc_match = re.search(r'train_acc=([\d.]+)', line)
            if epoch_match and acc_match:
                epoch = int(epoch_match.group(1))
                acc = float(acc_match.group(1))
                current_cvx_epochs.append((epoch, acc))
        
        # Extract STE training accuracies
        if '[STE]' in line and 'train_acc=' in line and current_run:
            epoch_match = re.search(r'ep=(\d+)/', line)
            acc_match = re.search(r'train_acc=([\d.]+)', line)
            if epoch_match and acc_match:
                epoch = int(epoch_match.group(1))
                acc = float(acc_match.group(1))
                current_ste_epochs.append((epoch, acc))
        
        # Extract final test results (prefer FINAL summary)
        if '=== FINAL' in line and i + 2 < len(lines):
            cvx_line = lines[i+1].strip()
            ste_line = lines[i+2].strip()
            cvx_match = re.search(r'CVX test_acc = ([\d.]+)', cvx_line)
            ste_match = re.search(r'STE test_acc = ([\d.]+)', ste_line)
            if cvx_match and ste_match and current_run:
                current_run['cvx_test'] = float(cvx_match.group(1))
                current_run['ste_test'] = float(ste_match.group(1))
        elif '[seed' in line and 'CVX test=' in line and 'STE test=' in line and current_run and 'cvx_test' not in current_run:
            cvx_match = re.search(r'CVX test=([\d.]+)', line)
            ste_match = re.search(r'STE test=([\d.]+)', line)
            if cvx_match and ste_match:
                current_run['cvx_test'] = float(cvx_match.group(1))
                current_run['ste_test'] = float(ste_match.group(1))
        
        i += 1
    
    # Save last run
    if current_run is not None and in_run:
        current_run['cvx_train_acc'] = sorted(current_cvx_epochs)
        current_run['ste_train_acc'] = sorted(current_ste_epochs)
        runs.append(current_run)
    
    # Build summary table
    table_data = []
    for run in runs:
        if 'task' in run and 'L' in run and 'T' in run:
            train_size = int(run.get('n_train_total', 0) * (1 - run.get('val_frac', 0.2)))
            table_data.append({
                'Dataset': run['task'],
                'L': run['L'],
                'T': run['T'],
                'train_size': train_size,
                'CVX_test': run.get('cvx_test', np.nan),
                'STE_test': run.get('ste_test', np.nan),
            })
    
    return runs, table_data

def plot_training_curves(runs: List[Dict], output_dir: Path):
    """Generate training curve plots for each run."""
    output_dir.mkdir(exist_ok=True)
    
    for idx, run in enumerate(runs):
        if 'task' not in run or 'L' not in run or 'T' not in run:
            continue
        
        cvx_epochs = run.get('cvx_train_acc', [])
        ste_epochs = run.get('ste_train_acc', [])
        
        if not cvx_epochs and not ste_epochs:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot CVX
        if cvx_epochs:
            epochs_cvx, accs_cvx = zip(*cvx_epochs)
            ax.plot(epochs_cvx, accs_cvx, 'o-', label='CVX', linewidth=2, markersize=4, alpha=0.7, color='green')
        
        # Plot STE
        if ste_epochs:
            epochs_ste, accs_ste = zip(*ste_epochs)
            ax.plot(epochs_ste, accs_ste, 's-', label='STE', linewidth=2, markersize=4, alpha=0.7, color='red')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train Accuracy', fontsize=12)
        task = run['task']
        L = run['L']
        T = run['T']
        train_size = int(run.get('n_train_total', 0) * (1 - run.get('val_frac', 0.2)))
        title = f'{task} - L={L}, T={T}, train_size={train_size}'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Save figure
        filename = f'train_acc_{task}_L{L}_T{T}_train{train_size}_{idx}.png'
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def create_summary_table(table_data: List[Dict], output_file: Path):
    """Create consolidated summary table."""
    if not table_data:
        print("No data to create summary table!")
        return
    
    df = pd.DataFrame(table_data)
    
    # Sort by dataset, then L, then T
    if len(df) > 0:
        df = df.sort_values(['Dataset', 'L', 'T', 'train_size'])
        
        # Format test accuracies to 4 decimal places
        df['CVX_test'] = df['CVX_test'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        df['STE_test'] = df['STE_test'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        
        # Save as CSV
        output_file.parent.mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nSummary table saved to: {output_file}")
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

def main():
    log_file = Path(__file__).parent / 'RNN_logs.txt'
    output_dir = Path(__file__).parent / 'rnn_analysis_output'
    table_file = output_dir / 'summary_table.csv'
    
    print(f"Parsing {log_file}...")
    try:
        runs, table_data = parse_logs(str(log_file))
        print(f"\nFound {len(runs)} runs")
        print(f"Found {len(table_data)} entries for summary table")
        
        if len(runs) > 0:
            print(f"\nFirst run sample: task={runs[0].get('task')}, L={runs[0].get('L')}, T={runs[0].get('T')}")
            print(f"  CVX epochs: {len(runs[0].get('cvx_train_acc', []))}")
            print(f"  STE epochs: {len(runs[0].get('ste_train_acc', []))}")
        
        if len(runs) == 0:
            print("ERROR: No runs found! Check log file format.")
            return
    except Exception as e:
        print(f"ERROR parsing logs: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nGenerating training curve plots...")
    try:
        plot_training_curves(runs, output_dir)
    except Exception as e:
        print(f"ERROR generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nCreating summary table...")
    try:
        create_summary_table(table_data, table_file)
    except Exception as e:
        print(f"ERROR creating table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
