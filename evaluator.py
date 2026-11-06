import ast
import sys
import copy
import builtins
import random
import hashlib
from collections import Counter


class DifferentialVerifier:
    def __init__(self, n_fuzz: int = 100, seed: int = 0):
        self.n_fuzz = n_fuzz
        self.seed = seed

    def _infer_spec(self, test_cases):
        if not test_cases:
            return []
        inputs0 = test_cases[0][0]
        args0 = inputs0 if isinstance(inputs0, (list, tuple)) else (inputs0,)
        spec = []
        for i in range(len(args0)):
            values = []
            for inputs, _ in test_cases:
                args = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
                if i < len(args):
                    values.append(args[i])
            spec.append(values)
        return spec

    def _random_arg(self, seen_values, rng):
        if not seen_values:
            return 0
        t = type(seen_values[0])
        if t is bool:
            return rng.choice([True, False])
        if t is int:
            base = rng.choice(seen_values)
            return rng.choice([
                rng.randint(-100, 100),
                0, 1, -1,
                base + 1, base - 1,
                rng.randint(-10**6, 10**6),
            ])
        if t is float:
            return rng.uniform(-100.0, 100.0)
        if t is str:
            return ''.join(rng.choice('abcxyz') for _ in range(rng.randint(0, 8)))
        if t is list:
            elem_values = [x for v in seen_values for x in v] if seen_values else [0]
            return [self._random_arg(elem_values, rng) for _ in range(rng.randint(0, 10))]
        if t is tuple:
            elem_values = [x for v in seen_values for x in v] if seen_values else [0]
            return tuple(self._random_arg(elem_values, rng) for _ in range(rng.randint(0, 6)))
        return rng.choice(seen_values)

    def generate_fuzz_cases(self, test_cases):
        rng = random.Random(self.seed)
        spec = self._infer_spec(test_cases)
        cases = []
        for _ in range(self.n_fuzz):
            args = tuple(self._random_arg(vals, rng) for vals in spec)
            cases.append(args if len(args) > 1 else args[0])
        return cases

    def _run_one(self, code_str, func_name, args):
        env = {}
        exec(code_str, env)
        func = env[func_name]
        if isinstance(args, (list, tuple)):
            return func(*args)
        return func(args)

    def verify(self, drafts, func_name, test_cases):
        fuzz_inputs = self.generate_fuzz_cases(test_cases)

        outputs = {}
        for d in drafts:
            outs = []
            for inp in fuzz_inputs:
                try:
                    val = self._run_one(d["original_code"], func_name, inp)
                    outs.append(("ok", repr(val)))
                except BaseException:
                    outs.append(("err", None))
            outputs[d["id"]] = outs

        consensus = []
        for i in range(len(fuzz_inputs)):
            votes = Counter(
                outs[i][1] for outs in outputs.values()
                if outs[i][0] == "ok"
            )
            consensus.append(votes.most_common(1)[0][0] if votes else None)

        result = {}
        for did, outs in outputs.items():
            divergences = sum(
                1 for i, (status, val) in enumerate(outs)
                if status != "ok" or val != consensus[i]
            )
            result[did] = (divergences == 0, divergences)
        return result


class UltimateEvaluatorV3:
    def __init__(self, max_steps: int = 10000, n_fuzz: int = 100):
        self.ast_memory = Counter()
        self.branch_memory = Counter()
        self.task_memory = Counter()
        self.archive = {}
        self.max_steps = max_steps
        self.builtin_names = set(dir(builtins))
        self.n_fuzz = n_fuzz
        self._batch_seed = 0

    def run_and_trace(self, code_str: str, func_name: str, test_cases: list):
        total_steps = 0
        branch_signature = []

        for inputs, expected in test_cases:
            steps = 0
            trace_log = []
            last_loc = None
            call_depth = 0

            def tracer(frame, event, arg):
                nonlocal steps, last_loc, call_depth
                code_name = frame.f_code.co_name

                if event == "call":
                    if code_name == func_name and call_depth == 0:
                        call_depth = 1
                    elif call_depth > 0:
                        call_depth += 1

                if call_depth == 0:
                    return tracer

                if event in ("line", "call"):
                    steps += 1
                    if steps > self.max_steps:
                        raise TimeoutError("Step limit exceeded")

                    cur_loc = (code_name, frame.f_lineno)
                    if last_loc is not None:
                        is_jump = (cur_loc[0] != last_loc[0]) or (cur_loc[1] != last_loc[1] + 1)
                        if is_jump:
                            trace_log.append((cur_loc[0] != func_name, cur_loc[1] <= last_loc[1]))
                    last_loc = cur_loc

                elif event == "return" and call_depth > 0:
                    call_depth -= 1

                return tracer

            env = {}
            try:
                exec(code_str, env)
                if func_name not in env or not callable(env[func_name]):
                    return False, float('inf'), None
                func = env[func_name]

                args_copy = copy.deepcopy(inputs)
                sys.settrace(tracer)
                if isinstance(args_copy, (list, tuple)):
                    actual = func(*args_copy)
                else:
                    actual = func(args_copy)
                sys.settrace(None)
            except BaseException:
                sys.settrace(None)
                return False, float('inf'), None

            if actual != expected or type(actual) is not type(expected):
                return False, float('inf'), None

            total_steps += steps
            branch_signature.append(tuple(trace_log))

        return True, total_steps, tuple(branch_signature)

    def _get_all_stmt_blocks(self, root_node: ast.AST) -> list[list[ast.stmt]]:
        blocks = []
        for node in ast.walk(root_node):
            for attr in ("body", "orelse", "finalbody"):
                val = getattr(node, attr, None)
                if isinstance(val, list) and val and all(isinstance(x, ast.stmt) for x in val):
                    blocks.append(val)
        return blocks

    def _targets_name(self, target, name):
        if isinstance(target, ast.Name):
            return target.id == name
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._targets_name(t, name) for t in target.elts)
        if isinstance(target, ast.Starred):
            return self._targets_name(target.value, name)
        return False

    def _is_safe_to_inline(self, expr_val, remaining_stmts, var_name):
        for node in ast.walk(expr_val):
            if isinstance(node, (ast.Call, ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr)):
                return False

        for s in remaining_stmts:
            for node in ast.walk(s):
                if isinstance(node, ast.Assign):
                    if any(self._targets_name(t, var_name) for t in node.targets):
                        return False
                if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    if self._targets_name(node.target, var_name):
                        return False
                if isinstance(node, ast.For) and self._targets_name(node.target, var_name):
                    return False
                if isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars and self._targets_name(item.optional_vars, var_name):
                            return False
                if isinstance(node, ast.ExceptHandler) and node.name == var_name:
                    return False
                if isinstance(node, ast.Delete):
                    if any(self._targets_name(t, var_name) for t in node.targets):
                        return False
                if isinstance(node, (ast.Global, ast.Nonlocal)) and var_name in node.names:
                    return False
                if isinstance(node, ast.Lambda):
                    if any(a.arg == var_name for a in node.args.args):
                        return False
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == var_name or any(a.arg == var_name for a in node.args.args):
                        return False
        return True

    def guillotine_prune(self, code_str: str, func_name: str, test_cases: list) -> ast.AST:
        raw_tree = ast.parse(code_str)
        target_funcs = [n for n in raw_tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
        if not target_funcs:
            return raw_tree

        clean_mod = ast.Module(body=[copy.deepcopy(target_funcs[-1])], type_ignores=[])
        ast.fix_missing_locations(clean_mod)
        alive_isolated, _, _ = self.run_and_trace(ast.unparse(clean_mod), func_name, test_cases)
        tree = clean_mod if alive_isolated else copy.deepcopy(raw_tree)

        def is_still_alive(candidate_tree):
            try:
                ast.fix_missing_locations(candidate_tree)
                new_code = ast.unparse(candidate_tree)
                alive, _, _ = self.run_and_trace(new_code, func_name, test_cases)
                return alive
            except BaseException:
                return False

        changed = True
        while changed:
            changed = False
            func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
            blocks = self._get_all_stmt_blocks(func_node)

            for b_idx, block in enumerate(blocks):
                if len(block) <= 1:
                    continue
                for i in range(len(block)):
                    test_tree = copy.deepcopy(tree)
                    test_func = next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
                    test_block = self._get_all_stmt_blocks(test_func)[b_idx]
                    del test_block[i]
                    if is_still_alive(test_tree):
                        tree = test_tree
                        changed = True
                        break
                if changed:
                    break
            if changed:
                continue

            for b_idx, block in enumerate(blocks):
                if len(block) > 2:
                    for i in range(len(block) - 1):
                        for j in range(i + 1, len(block)):
                            test_tree = copy.deepcopy(tree)
                            test_func = next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
                            test_block = self._get_all_stmt_blocks(test_func)[b_idx]
                            del test_block[j]
                            del test_block[i]
                            if is_still_alive(test_tree):
                                tree = test_tree
                                changed = True
                                break
                        if changed:
                            break
                if changed:
                    break
            if changed:
                continue

            for b_idx, block in enumerate(blocks):
                for i, stmt in enumerate(block):
                    if isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                        candidate_branches = [getattr(stmt, attr, []) for attr in ("body", "orelse", "finalbody")]
                        for branch in candidate_branches:
                            if not branch:
                                continue
                            test_tree = copy.deepcopy(tree)
                            test_func = next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
                            test_block = self._get_all_stmt_blocks(test_func)[b_idx]
                            test_block[i:i+1] = copy.deepcopy(branch)
                            if is_still_alive(test_tree):
                                tree = test_tree
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                continue

            for b_idx, block in enumerate(blocks):
                for i, stmt in enumerate(block[:-1]):
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        var_name = stmt.targets[0].id
                        expr_val = stmt.value

                        if not self._is_safe_to_inline(expr_val, block[i+1:], var_name):
                            continue

                        test_tree = copy.deepcopy(tree)
                        test_func = next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
                        test_block = self._get_all_stmt_blocks(test_func)[b_idx]

                        class Inliner(ast.NodeTransformer):
                            def visit_Name(self, node):
                                if node.id == var_name and isinstance(node.ctx, ast.Load):
                                    return copy.deepcopy(expr_val)
                                return node

                        for j in range(i + 1, len(test_block)):
                            test_block[j] = Inliner().visit(test_block[j])
                        del test_block[i]

                        if is_still_alive(test_tree):
                            tree = test_tree
                            changed = True
                            break
                if changed:
                    break
            if changed:
                continue

            complex_types = (
                ast.Call, ast.IfExp, ast.BinOp, ast.UnaryOp, ast.BoolOp,
                ast.Compare, ast.Subscript, ast.ListComp, ast.GeneratorExp
            )
            expr_nodes = [n for n in ast.walk(func_node) if isinstance(n, complex_types)]
            for target_idx in range(len(expr_nodes)):
                candidates = []
                for c in ast.iter_child_nodes(expr_nodes[target_idx]):
                    if isinstance(c, ast.expr):
                        candidates.append(c)
                        for gc in ast.iter_child_nodes(c):
                            if isinstance(gc, ast.expr):
                                candidates.append(gc)

                for cand_idx in range(len(candidates)):
                    test_tree = copy.deepcopy(tree)
                    test_func = next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
                    test_expr_nodes = [n for n in ast.walk(test_func) if isinstance(n, complex_types)]
                    target_node = test_expr_nodes[target_idx]

                    test_candidates = []
                    for c in ast.iter_child_nodes(target_node):
                        if isinstance(c, ast.expr):
                            test_candidates.append(c)
                            for gc in ast.iter_child_nodes(c):
                                if isinstance(gc, ast.expr):
                                    test_candidates.append(gc)

                    replacement = copy.deepcopy(test_candidates[cand_idx])

                    class ExactReplacer(ast.NodeTransformer):
                        def generic_visit(self, n):
                            if n is target_node:
                                return replacement
                            return super().generic_visit(n)

                    ExactReplacer().visit(test_func)
                    if is_still_alive(test_tree):
                        tree = test_tree
                        changed = True
                        break
                if changed:
                    break

        return tree

    def get_clean_skeleton(self, tree: ast.AST, func_name: str):
        func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
        clean_func = copy.deepcopy(func_node)
        clean_func.name = "func"

        if (clean_func.body and isinstance(clean_func.body[0], ast.Expr) and
                isinstance(clean_func.body[0].value, ast.Constant) and isinstance(clean_func.body[0].value.value, str)):
            clean_func.body.pop(0)

        for node in ast.walk(clean_func):
            if isinstance(node, ast.Name) and node.id not in self.builtin_names:
                node.id = "var"
            elif isinstance(node, ast.arg):
                node.arg = "arg"
        return ast.dump(clean_func), sum(1 for _ in ast.walk(clean_func))

    def _get_cell(self, tree: ast.AST, func_name: str, branch_sig, steps: int):
        func_node = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name),
            None,
        )
        if func_node is None:
            return ("nofunc",)

        has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(func_node))

        fn_name = func_node.name
        has_recursion = False
        for n in ast.walk(func_node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id == fn_name:
                    has_recursion = True
                    break
                if isinstance(n.func, ast.Attribute) and n.func.attr == fn_name:
                    has_recursion = True
                    break

        n_branches = sum(1 for n in ast.walk(func_node) if isinstance(n, (ast.If, ast.IfExp)))
        n_branches_bucket = min(n_branches, 5)

        trace_len = sum(len(b) for b in branch_sig) if branch_sig else 0
        trace_bucket = min(trace_len // 10, 10)

        return (has_loop, has_recursion, n_branches_bucket, trace_bucket)

    def rank_drafts(self, drafts: list, func_name: str, test_cases: list):
        self._batch_seed = random.randint(0, 10**9)

        results = []
        for idx, code in enumerate(drafts):
            alive, _, _ = self.run_and_trace(code, func_name, test_cases)
            if not alive:
                results.append({"id": idx, "alive": False, "original_code": code})
                continue

            pruned_tree = self.guillotine_prune(code, func_name, test_cases)
            pruned_code = ast.unparse(pruned_tree)
            alive_after, clean_steps, branch_sig = self.run_and_trace(pruned_code, func_name, test_cases)
            if not alive_after:
                results.append({"id": idx, "alive": False, "original_code": code})
                continue

            ast_sig, node_count = self.get_clean_skeleton(pruned_tree, func_name)
            cell = self._get_cell(pruned_tree, func_name, branch_sig, clean_steps)

            try:
                orig_nodes = sum(1 for _ in ast.walk(ast.parse(code).body[0]))
                pruned_nodes = sum(1 for _ in ast.walk(pruned_tree.body[0]))
                suspicious = pruned_nodes < orig_nodes * 0.5
            except Exception:
                suspicious = False

            results.append({
                "id": idx,
                "alive": True,
                "original_code": code,
                "clean_code": pruned_code,
                "ast_sig": ast_sig,
                "branch_sig": branch_sig,
                "nodes": node_count,
                "steps": clean_steps,
                "cell": cell,
                "suspicious": suspicious,
            })

        alive_drafts = [
            {"id": r["id"], "original_code": r["original_code"]}
            for r in results if r["alive"]
        ]
        if len(alive_drafts) >= 3:
            verifier = DifferentialVerifier(n_fuzz=self.n_fuzz, seed=self._batch_seed)
            verdict = verifier.verify(alive_drafts, func_name, test_cases)
        else:
            verdict = {d["id"]: (True, 0) for d in alive_drafts}

        for r in results:
            if r["alive"]:
                trusted, div = verdict.get(r["id"], (False, 10**9))
                r["trusted"] = trusted
                r["divergence"] = div

        task_sig = hashlib.blake2b(repr(test_cases).encode(), digest_size=8).hexdigest()
        self.task_memory[task_sig] += 1
        task_freq = self.task_memory[task_sig]

        for r in results:
            if not r["alive"] or not r.get("trusted"):
                r["novelty"] = 0
                r["task_sig"] = task_sig
                continue
            key = (task_sig, r["cell"])
            entry = self.archive.get(key)
            if entry is None:
                self.archive[key] = {"result": r, "score": r["steps"]}
                r["novelty"] = 1
            else:
                r["novelty"] = 0
                if r["steps"] < entry["score"]:
                    self.archive[key] = {"result": r, "score": r["steps"]}
            r["task_sig"] = task_sig

        for r in results:
            if r["alive"]:
                h_ast = hashlib.blake2b(r["ast_sig"].encode(), digest_size=8).hexdigest()
                h_br = hashlib.blake2b(repr(r["branch_sig"]).encode(), digest_size=8).hexdigest()
                self.ast_memory[h_ast] += 1
                self.branch_memory[h_br] += 1
                r["_hash_ast"] = h_ast
                r["_hash_br"] = h_br
                r["frequency"] = min(self.ast_memory[h_ast], self.branch_memory[h_br])

        return sorted(
            results,
            key=lambda r: (
                not r["alive"],
                not r.get("trusted", False),
                -r.get("novelty", 0),
                task_freq,
                r.get("frequency", float('inf')),
                r.get("nodes", float('inf')),
                r.get("steps", float('inf')),
            )
        )

import pickle
import os


class TrainingSignal:
    def __init__(self, evaluator, state_path: str = "training_state.pkl", tracker=None):
        self.evaluator = evaluator
        self.state_path = state_path
        self.tracker = tracker or ProgressTracker()

    def log_batch(self, results, batch_id: int):
        alive = sum(1 for r in results if r.get("alive"))
        trusted = sum(1 for r in results if r.get("alive") and r.get("trusted"))
        novelty = sum(1 for r in results if r.get("alive") and r.get("trusted")
                      and r.get("novelty", 0) == 1)
        return self.tracker.record(
            batch_id=batch_id,
            alive=alive,
            trusted=trusted,
            novelty=novelty,
            archive_size=len(self.evaluator.archive),
            tasks_seen=len(self.evaluator.task_memory),
        )

    def compute_reward(self, result: dict) -> float:
        if not result.get("alive"):
            return -1.0
        if not result.get("trusted", False):
            return 0.0
        reward = 1.0
        if result.get("novelty", 0) == 1:
            reward += 0.5
        steps = result.get("steps", 0)
        if steps > 0:
            reward += min(0.3, 100.0 / steps)
        return reward

    def select_diverse_batch(self, results, k: int = 10):
        alive = [r for r in results if r.get("alive") and r.get("trusted", False)]
        seen_cells = set()
        picked = []
        for r in alive:
            if r.get("novelty", 0) == 1 and r.get("cell") not in seen_cells:
                seen_cells.add(r.get("cell"))
                picked.append(r)
                if len(picked) >= k:
                    return picked
        rest = [r for r in alive if r not in picked]
        rest.sort(key=lambda r: -self.compute_reward(r))
        for r in rest:
            if r.get("cell") not in seen_cells:
                seen_cells.add(r.get("cell"))
                picked.append(r)
                if len(picked) >= k:
                    break
        return picked[:k]

    def save(self):
        state = {
            "archive_cells": list(getattr(self.evaluator, "archive", {}).keys()),
            "ast_memory": dict(getattr(self.evaluator, "ast_memory", {})),
            "branch_memory": dict(getattr(self.evaluator, "branch_memory", {})),
            "task_memory": dict(getattr(self.evaluator, "task_memory", {})),
        }
        with open(self.state_path, "wb") as f:
            pickle.dump(state, f)

    def load(self):
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path, "rb") as f:
            state = pickle.load(f)
        from collections import Counter
        self.evaluator.archive = {cell: {"result": None, "score": 0}
                           for cell in state.get("archive_cells", [])
                           if isinstance(cell, tuple) and len(cell) == 2 and isinstance(cell[0], str)}
        self.evaluator.ast_memory = Counter(state.get("ast_memory", {}))
        self.evaluator.branch_memory = Counter(state.get("branch_memory", {}))
        if hasattr(self.evaluator, "task_memory"):
            self.evaluator.task_memory = Counter(state.get("task_memory", {}))
        return True

import json


class ProgressTracker:
    def __init__(self, state_path: str = "progress.json"):
        self.state_path = state_path
        self.history = []
        self.load()

    def load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                self.history = json.load(f)

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def record(self, batch_id, alive, trusted, novelty, archive_size, tasks_seen):
        entry = {
            "batch": batch_id,
            "alive": alive,
            "trusted": trusted,
            "novelty": novelty,
            "archive_size": archive_size,
            "tasks_seen": tasks_seen,
        }
        self.history.append(entry)
        self.save()
        return entry

    def trend(self):
        if len(self.history) < 2:
            return "недостаточно данных (нужно ≥2 батча)"
        last = self.history[-1]["novelty"]
        prev = self.history[-2]["novelty"]
        if last > prev:
            return f"растёт ({prev} → {last}) ✅ модель находит новые способы"
        elif last < prev:
            return f"падает ({prev} → {last}) ⚠️ возможен коллапс"
        else:
            return f"стоит на {last} — возможно, насыщение или хак"

    def report(self):
        print(f"\n{'batch':<7} {'alive':<7} {'trusted':<9} {'novelty':<9} {'archive':<9} {'tasks'}")
        print("-" * 60)
        for e in self.history:
            print(f"{e['batch']:<7} {e['alive']:<7} {e['trusted']:<9} "
                  f"{e['novelty']:<9} {e['archive_size']:<9} {e['tasks_seen']}")

class Curriculum:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def examples_for_task(self, task_sig: str, k: int = 3):
        examples = []
        seen_cells = set()
        for (ts, cell), entry in self.evaluator.archive.items():
            if ts == task_sig and entry.get("result") is not None:
                if cell not in seen_cells:
                    seen_cells.add(cell)
                    examples.append(entry["result"])
                    if len(examples) >= k:
                        break
        return examples

    def select_next_task(self, tasks: list):
        scored = []
        for task in tasks:
            task_sig = hashlib.blake2b(
                repr(task["test_cases"]).encode(), digest_size=8
            ).hexdigest()
            filled = sum(1 for (ts, _) in self.evaluator.archive.keys() if ts == task_sig)
            scored.append((filled, task))
        scored.sort(key=lambda x: x[0])
        return scored[0][1] if scored else None