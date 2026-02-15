import itertools
import math
import time
import numpy as np
import torch
import cvxpy as cp
from typing import List, Tuple

class Convex_SNN():

    def __init__(self, layers: int, widths: np.array, input_dim: int, beta: float):
        self.layers = layers
        self.widths = widths
        self.beta = beta
        self.log_path = None
        self.reconstructed_weights = None
        self.convex_w = None
        self.convex_primal_val = None
        self.convex_dual_val = None
        self.convex_duality_gap = None
        self.output_weights = None
        self.n_samples_last = None
        self.training_accuracy = None
        self.testing_accuracy = None
        self.D_arrangements = None
        self.D_arrangements_hash_table = None

    
    def forward(self, x_list : List[np.ndarray], initial_states: List[np.ndarray] = None) -> torch.Tensor:
        '''
        We assume that x_list is a list of tensors of shape (n,d) for each timestep t in range(T)
        Now we also assume that the reconsrutted weights are stored in self.reconstructed_weights 
        where for each layer i we have that self.reconstructed_weights[i] is a list with P_in , P_rec
        '''
        n = x_list[0].shape[0]
        h_map = {}
        timesteps = len(x_list)
        membrane_potentials = [np.zeros((n, self.widths[l - 1]), dtype=np.float32) for l in range(1, self.layers + 1)]
        for l in range(1, self.layers + 1):
            p_in = self.reconstructed_weights[l - 1][0]
            decay_rate = np.diag(self.reconstructed_weights[l - 1][1])
            u_threshold = np.diag(self.reconstructed_weights[l - 1][2])
            p_rec = np.vstack((decay_rate, u_threshold))
            
            for t in range(1, timesteps + 1): 
                if l == 1:
                    x_in = x_list[t - 1]
                else:
                    x_in = h_map[(l - 1, t)]
                if t == 1:
                    if initial_states is not None:
                        h_prev = initial_states[l - 1]
                    else:
                        h_prev = np.zeros((n, self.widths[l - 1]), dtype=np.float32)
                else:
                    h_prev = h_map[(l, t - 1)]
                rec_feat = np.hstack([-h_prev, membrane_potentials[l - 1]])
                membrane_potentials[l - 1] = (x_in @ p_in + rec_feat @ p_rec).astype(np.float32)
                h_map[(l, t)] = (membrane_potentials[l - 1] >= 0).astype(np.float32)
        if self.output_weights is None:
            raise ValueError("output_weights not available. Run reconstruct_weights first.")
        p_out = self.output_weights
        output = np.sign(h_map[self.layers, timesteps] @ p_out).astype(np.float32)
        return output

    def A_operator(
        self,
        x: np.ndarray,
        input: bool = False,
        sampled: bool = False,
        num_samples: int = 0,
    ) -> np.ndarray:
        
        '''
        Flow: 
        - If input is True then we do sign(Xw) -> if sampled == True then we only return num_sampled arrangments .. else its a full enumeration
        - We return the hyperplane arrangements in the form of a numpy array of shape (n,d)
        '''

        z = x.astype(np.float32)
        n, p = z.shape

        if sampled:
            max_samples = 2 ** n
            if num_samples > max_samples:
                raise ValueError(
                    f"num_samples={num_samples} exceeds max_samples=2**n={max_samples}."
                )
            if num_samples <= 0:
                raise ValueError("num_samples must be > 0 when sampled=True.")
            u = np.random.normal(size=(p, num_samples)).astype(np.float32)
            patterns = (z @ u >= 0).astype(np.int32)
            uniq = {}
            for idx in range(patterns.shape[1]):
                key = tuple(patterns[:, idx].tolist())
                if key not in uniq:
                    uniq[key] = patterns[:, idx].astype(np.float32)
            if not uniq:
                raise ValueError("A_operator sampling produced no spike patterns from input.")
            return np.stack(list(uniq.values()), axis=1)

        if p == 0:
            return np.ones((n, 1), dtype=np.float32)
        if p == 1:
            u = np.array([[1.0]], dtype=np.float32)
            s = (z @ u >= 0).astype(np.float32).reshape(-1)
            s_neg = (z @ (-u) >= 0).astype(np.float32).reshape(-1)
            uniq = {tuple(s.tolist()): s, tuple(s_neg.tolist()): s_neg}
            return np.stack(list(uniq.values()), axis=1)
        if n < p - 1:
            raise ValueError("A_operator requires n >= p-1 for full enumeration.")

        subset_size = p - 1
        uniq = {}
        for subset in itertools.combinations(range(n), subset_size):
            zs = z[list(subset), :]
            _, _, vt = np.linalg.svd(zs)
            u = vt[-1]
            if np.allclose(u, 0):
                continue
            s = (z @ u >= 0).astype(np.float32).reshape(-1)
            s_neg = (z @ (-u) >= 0).astype(np.float32).reshape(-1)
            key = tuple(s.tolist())
            if key not in uniq:
                uniq[key] = s
            key_neg = tuple(s_neg.tolist())
            if key_neg not in uniq:
                uniq[key_neg] = s_neg
        if not uniq:
            raise ValueError("A_operator produced no spike patterns from input.")
        return np.stack(list(uniq.values()), axis=1)

    def A_SNN_operator(
        self,
        D_x: List[np.ndarray],
        x_len: np.ndarray,
        y_len: np.ndarray,
        operator: List[str],
        sampled: bool = False,
        num_samples: int = 0,
    ) -> np.ndarray:
        '''
        Flow: Check if D_arrangets_hash_table exists .. if it does then we continue .. else we initialize empty dict .. 
        we then have a perfect hashing-> (l,t) -> index -> [parents] parents format (y(index chosen), y(index's chosen), nothing much here)
        '''

        if len(operator) > len(D_x):
            raise ValueError("operator length cannot exceed D_x length.")

        def unique_columns(mat: np.ndarray) -> np.ndarray:
            uniq = {}
            for idx in range(mat.shape[1]):
                key = tuple(mat[:, idx].tolist())
                if key not in uniq:
                    uniq[key] = mat[:, idx].astype(np.float32)
            if not uniq:
                raise ValueError("No unique patterns produced in A_SNN_operator.")
            return np.stack(list(uniq.values()), axis=1)

        subset_iters = []
        combo_sizes = []
        for i, op in enumerate(operator):
            if op not in {"+", "-"}:
                raise ValueError(f"Unsupported op '{op}', expected '+' or '-'.")
            cols = D_x[i]
            k = int(y_len[i])
            if k <= 0:
                raise ValueError("y_len must be positive.")
            if k > cols.shape[1]:
                raise ValueError("y_len cannot exceed number of columns in D_x.")
            combo_sizes.append(math.comb(cols.shape[1], k))
            subset_iters.append(itertools.combinations(range(cols.shape[1]), k))

        total_combos = 1
        for size in combo_sizes:
            total_combos *= size
        

        max_combos = 2 ** D_x[0].shape[0]
        if not sampled and total_combos > max_combos:
            raise ValueError(
                "Combinatorial blowup in A_SNN_operator: "
                f"total_combos={total_combos} max_combos={max_combos} per_block={combo_sizes}"
            )

        uniq_patterns = {}
        if sampled:
            if num_samples <= 0:
                raise ValueError("num_samples must be > 0 when sampled=True.")
            rng = np.random.default_rng(0)
            for _ in range(num_samples):
                blocks = []
                for i, op in enumerate(operator):
                    cols = D_x[i]
                    k = int(y_len[i])
                    idxs = rng.choice(cols.shape[1], size=k, replace=False)
                    block = cols[:, idxs]
                    if op == "-":
                        block = -block
                    blocks.append(block)
                for i in range(len(operator), len(D_x)):
                    blocks.append(D_x[i])
                z = np.hstack(blocks).astype(np.float32)
                u = rng.normal(size=(z.shape[1], 1)).astype(np.float32)
                s = (z @ u >= 0).astype(np.float32).reshape(-1)
                key = tuple(s.tolist())
                if key not in uniq_patterns:
                    uniq_patterns[key] = s
        else:
            for combo in itertools.product(*subset_iters):
                blocks = []
                for i, idxs in enumerate(combo):
                    block = D_x[i][:, idxs]
                    if operator[i] == "-":
                        block = -block
                    blocks.append(block)
                for i in range(len(operator), len(D_x)):
                    blocks.append(D_x[i])
                z = np.hstack(blocks).astype(np.float32)
                patterns = self.A_operator(z, input=False, sampled=False, num_samples=0)
                for col in range(patterns.shape[1]):
                    key = tuple(patterns[:, col].tolist())
                    if key not in uniq_patterns:
                        uniq_patterns[key] = patterns[:, col].astype(np.float32)
        if not uniq_patterns:
            raise ValueError("A_SNN_operator produced no spike patterns.")
        return np.stack(list(uniq_patterns.values()), axis=1)

    def generate_hyperplane_arrangements(
        self,
        x_list: List[np.ndarray],
        initial_firing_patterns: List[np.ndarray] = None,
        sampled: bool = False,
        num_samples: int = 0,
    ) -> torch.Tensor:
        '''
        From the given X we generate and return the last layer hyperplane arrangement
        we store the other arrangements in the class attribute D_arrangements[(l,t)] for each layer l and timestep t
        This time we store the acnestory -> using a separate dictiionary with parent -> child  .. we can use a b-tree to do so as well but hashing is fast + simple 

        '''

        timesteps = len(x_list)
        n = x_list[0].shape[0]
        mem_potential = np.ones((n,1))
        operator = ["-", "+"]
        x = np.ones(3)
        y = np.ones(3)
        self.D_arrangements = {}

        for l in range(1, self.layers + 1):
            D_1 = 0
            D_2 = 0
            for t in range(1, timesteps + 1):
                if l == 1:
                    D_1 = self.A_operator(
                        x_list[t - 1],
                        input=True,
                        sampled=sampled,
                        num_samples=num_samples,
                    )
                    self.D_arrangements[(l - 1, t)] = D_1
                else:
                    D_1 = self.D_arrangements[(l - 1, t)]
                if t == 1:
                    if initial_firing_patterns is not None:
                        D_2 = self.A_operator(
                            initial_firing_patterns[l - 1],
                            input=True,
                            sampled=sampled,
                            num_samples=num_samples,
                        )
                    else:
                        D_2 = np.zeros((n, self.widths[l - 1]), dtype=np.float32)
                    self.D_arrangements[(l, t - 1)] = D_2
                else:
                    D_2 = self.D_arrangements[(l, t - 1)]
                x[0] = D_1.shape[1]
                x[1] = D_2.shape[1]
                y[0] = self.widths[l - 1]
                y[1] = self.widths[l - 1]
                A_s = self.A_SNN_operator(
                    [D_1, D_2, mem_potential],
                    x,
                    y,
                    operator,
                    sampled=sampled,
                    num_samples=num_samples,
                )
                self.D_arrangements[(l, t)] = A_s

        return self.D_arrangements[self.layers, timesteps]

    def solve_convex_lasso(self,
        d_last: np.ndarray,
        y: np.ndarray,
        solver: str,
        solver_opts: dict,
    ) -> Tuple[np.ndarray, float, float, float]:
        beta_hat = self.beta / np.sqrt(self.widths[self.layers - 1])
        w = cp.Variable(d_last.shape[1])
        objective = 0.5 * cp.sum_squares(d_last @ w - y) + beta_hat * cp.norm1(w)
        primal_problem = cp.Problem(cp.Minimize(objective))
        primal_problem.solve(solver=solver, **solver_opts)
        primal_val = float(primal_problem.value)
    
        u = cp.Variable(d_last.shape[0])
        dual_objective = -0.5 * cp.sum_squares(u) - y @ u
        constraints = [cp.norm_inf(d_last.T @ u) <= beta_hat]
        dual_problem = cp.Problem(cp.Maximize(dual_objective), constraints)
        dual_problem.solve(solver=solver, **solver_opts)
        dual_val = float(dual_problem.value)
        duality_gap = primal_val - dual_val
        self.convex_w =  w.value
        self.convex_primal_val = primal_val
        self.convex_dual_val = dual_val
        self.convex_duality_gap = duality_gap



    def reconstruct_weights(
        self,
        x_list: List[np.ndarray],
        initial_states: List[np.ndarray] = None,
        svm_c: float = 1.0,
        iters: int = 2,
    ) -> None:
        if self.convex_w is None:
            raise ValueError("convex_w not available. Run solve_convex_lasso first.")
        if self.D_arrangements is None:
            raise ValueError("D_arrangements not available. Run generate_hyperplane_arrangements first.")
        if iters < 1:
            raise ValueError("iters must be >= 1.")

        timesteps = len(x_list)
        width_last = int(self.widths[self.layers - 1])
        d_last = self.D_arrangements[(self.layers, timesteps)]
        top_idx = np.argsort(-np.abs(self.convex_w))[: min(width_last, d_last.shape[1])]
        if top_idx.size < width_last:
            top_idx = np.resize(top_idx, width_last)

        def closest_patterns(d_curr: np.ndarray, target: np.ndarray) -> np.ndarray:
            h_out = np.zeros((d_curr.shape[0], target.shape[1]), dtype=np.float32)
            for j in range(target.shape[1]):
                diff = np.sum(np.abs(d_curr - target[:, [j]]), axis=0)
                h_out[:, j] = d_curr[:, np.argmin(diff)]
            return h_out

        h_map = {}
        h_map[(self.layers, timesteps)] = d_last[:, top_idx]
        p_out = self.convex_w[top_idx]
        if p_out.size < width_last:
            p_out = np.resize(p_out, width_last)
        self.output_weights = p_out.astype(np.float32)
        for t in range(timesteps - 1, 0, -1):
            d_curr = self.D_arrangements[(self.layers, t)]
            h_map[(self.layers, t)] = closest_patterns(d_curr, h_map[(self.layers, t + 1)])
        for l in range(self.layers - 1, 0, -1):
            for t in range(timesteps, 0, -1):
                d_curr = self.D_arrangements[(l, t)]
                h_map[(l, t)] = closest_patterns(d_curr, h_map[(l + 1, t)])

        reconstructed = []
        n = x_list[0].shape[0]
        for l in range(1, self.layers + 1):
            width = int(self.widths[l - 1])
            in_dim = x_list[0].shape[1] if l == 1 else int(self.widths[l - 2])
            p_in = np.zeros((in_dim, width), dtype=np.float32)
            decay = np.zeros((width,), dtype=np.float32)
            u_thresh = np.zeros((width,), dtype=np.float32)

            mems = [np.zeros((n, width), dtype=np.float32) for _ in range(timesteps + 1)]
            for _ in range(iters):
                for j in range(width):
                    x_blocks = []
                    y_blocks = []
                    for t in range(1, timesteps + 1):
                        if l == 1:
                            x_in = x_list[t - 1]
                        else:
                            x_in = h_map[(l - 1, t)]
                        if t == 1:
                            if initial_states is not None:
                                h_prev = initial_states[l - 1]
                            else:
                                h_prev = np.zeros((n, width), dtype=np.float32)
                            mem_prev = mems[t - 1]
                        else:
                            h_prev = h_map[(l, t - 1)]
                            mem_prev = mems[t - 1]
                        rec_feat = np.column_stack([-h_prev[:, j], mem_prev[:, j]])
                        x_blocks.append(np.hstack([x_in, rec_feat]))
                        y_blocks.append(2 * h_map[(l, t)][:, j] - 1)
                    x_stack = np.vstack(x_blocks)
                    y_stack = np.concatenate(y_blocks, axis=0)

                    w = cp.Variable(x_stack.shape[1])
                    xi = cp.Variable(x_stack.shape[0])
                    objective = 0.5 * cp.sum_squares(w) + svm_c * cp.sum(xi)
                    constraints = [cp.multiply(y_stack, x_stack @ w) >= 1 - xi, xi >= 0]
                    problem = cp.Problem(cp.Minimize(objective), constraints)
                    problem.solve(solver="SCS")
                    w_val = w.value
                    p_in[:, j] = w_val[:in_dim]
                    decay[j] = w_val[in_dim]
                    u_thresh[j] = w_val[in_dim + 1]

                for t in range(1, timesteps + 1):
                    if l == 1:
                        x_in = x_list[t - 1]
                    else:
                        x_in = h_map[(l - 1, t)]
                    if t == 1:
                        if initial_states is not None:
                            h_prev = initial_states[l - 1]
                        else:
                            h_prev = np.zeros((n, width), dtype=np.float32)
                    else:
                        h_prev = h_map[(l, t - 1)]
                    mems[t] = (
                        x_in @ p_in
                        + (-h_prev) * decay[None, :]
                        + mems[t - 1] * u_thresh[None, :]
                    )

            reconstructed.append([p_in, decay, u_thresh])

        self.reconstructed_weights = reconstructed



    def train(
        self,
        x_list: List[np.ndarray],
        y: np.ndarray,
        initial_states: List[np.ndarray] = None,
        sampled: bool = False,
        num_samples: int = 0,
        log_path: str = None,
    ) -> Tuple[float, List[float]]:
        '''
        First do generate hyperplanes -> solve convex lasso -> reconstruct weights

        '''

        self.log_path = log_path
        t0 = time.perf_counter()
        self.n_samples_last = x_list[0].shape[0]
        self.generate_hyperplane_arrangements(
            x_list,
            initial_states,
            sampled=sampled,
            num_samples=num_samples,
        )
        print(f"[ConvexSNN] stage=arrangements elapsed_s={time.perf_counter() - t0:.2f}")

        D_last = self.D_arrangements[self.layers, len(x_list)]
        self.solve_convex_lasso(D_last, y, solver='SCS', solver_opts={'max_iters': 1000000})
        print(f"[ConvexSNN] stage=convex_solve elapsed_s={time.perf_counter() - t0:.2f}")

        self.reconstruct_weights(x_list, initial_states=initial_states)
        print(f"[ConvexSNN] stage=reconstruct elapsed_s={time.perf_counter() - t0:.2f}")

        self.training_accuracy = self.test(x_list, y)
        print(f"[ConvexSNN] stage=test elapsed_s={time.perf_counter() - t0:.2f}")
        self.print_results(training=True)


    def test(self, x_list: List[np.ndarray], y: np.ndarray) -> Tuple[float, List[float]]:
        '''
        Test the network on the given data
        '''
        output = self.forward(x_list, initial_states=None)
        accruacy = np.mean(output == y)
        return accruacy


    def print_results(self, training: bool = False) -> None:
        '''
        Print the results of the training / testing 
        '''
        log_lines = []
        if training:
            print(f"Convex SNN Training Results:\n")
            print(f"Training accuracy: {self.training_accuracy}")
            print(f"Convex Primal Value: {self.convex_primal_val}")
            print(f"Convex Dual Value: {self.convex_dual_val}")
            print(f"Convex Duality Gap: {self.convex_duality_gap}")
            log_lines.extend(
                [
                    "Convex SNN Training Results:",
                    f"Training accuracy: {self.training_accuracy}",
                    f"Convex Primal Value: {self.convex_primal_val}",
                    f"Convex Dual Value: {self.convex_dual_val}",
                    f"Convex Duality Gap: {self.convex_duality_gap}",
                ]
            )
            if self.convex_w is not None:
                tol = 1e-4
                abs_w = np.abs(self.convex_w)
                nonzero = int(np.sum(abs_w > tol))
                print(f"Convex w nonzeros (tol={tol:.1e}): {nonzero}")
                log_lines.append(f"Convex w nonzeros (tol={tol:.1e}): {nonzero}")
                if self.n_samples_last is not None:
                    n = int(self.n_samples_last)
                    if n > 0:
                        abs_sorted = np.sort(abs_w)
                        if abs_sorted.size > n:
                            tol_on = abs_sorted[-n]
                        else:
                            tol_on = abs_sorted[0]
                        nonzero_on = int(np.sum(abs_w > tol_on))
                        print(
                            "Convex w tol for O(n) nonzeros "
                            f"(n={n}): tol={tol_on:.3e} nonzeros={nonzero_on}"
                        )
                        log_lines.append(
                            "Convex w tol for O(n) nonzeros "
                            f"(n={n}): tol={tol_on:.3e} nonzeros={nonzero_on}"
                        )
            if self.reconstructed_weights is not None:
                for idx, (_, decay_rate, u_threshold) in enumerate(self.reconstructed_weights, start=1):
                    decay_str = np.array2string(decay_rate, separator=",", threshold=np.inf)
                    thresh_str = np.array2string(u_threshold, separator=",", threshold=np.inf)
                    print(f"Decay rate (layer {idx}): {decay_str}")
                    print(f"U threshold (layer {idx}): {thresh_str}")
                    log_lines.append(f"Decay rate (layer {idx}): {decay_str}")
                    log_lines.append(f"U threshold (layer {idx}): {thresh_str}")
            if self.output_weights is not None:
                p_out_str = np.array2string(self.output_weights, separator=",", threshold=np.inf)
                print(f"P_out weights: {p_out_str}")
                log_lines.append(f"P_out weights: {p_out_str}")
            gap_tol = 1e7
            print(f"Gap <= {gap_tol:.1e}: {self.convex_duality_gap <= gap_tol}")
            log_lines.append(f"Gap <= {gap_tol:.1e}: {self.convex_duality_gap <= gap_tol}")
        else:
            print(f"Convex SNN Testing Results:\n")
            print(f"Testing accuracy: {self.testing_accuracy}") 
            log_lines.extend(
                [
                    "Convex SNN Testing Results:",
                    f"Testing accuracy: {self.testing_accuracy}",
                ]
            )
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                for line in log_lines:
                    handle.write(line + "\n")
                handle.write("\n")
        

    
