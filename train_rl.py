import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ast
import random
import re
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from transformers import LogitsProcessor

from evaluator import UltimateEvaluatorV3, TrainingSignal, ProgressTracker
from memory_cleanup import memory_cleanup, print_vram

MODEL_NAME = "./Qwen3-4B"
DATASET_NAME = "google-research-datasets/mbpp"
DATASET_CONFIG = "full"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 3000
NUM_GENERATIONS = 4
NUM_STEPS = 1000
LEARNING_RATE = 5e-6
SEED = 42
THINKING_BUDGET = 2200

STATE_PATH = "training_state.pkl"
PROGRESS_PATH = "progress.json"
OUTPUT_DIR = "outputs"

class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, budget):
        self.budget = budget
        self.end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.num_end_tokens = len(self.end_ids)
        self.first_end_id = self.end_ids[0]
        self.initial_len = None
        self.has_closed = None
        self.force_attempted = None

    def __call__(self, input_ids, scores):
        if self.initial_len is None:
            self.initial_len = input_ids.shape[1]
            self.has_closed = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            self.force_attempted = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        current_len = input_ids.shape[1] - self.initial_len

        if input_ids.shape[1] >= self.num_end_tokens:
            last_tokens = input_ids[:, -self.num_end_tokens:]
            end_tensor = torch.tensor(self.end_ids, device=input_ids.device).unsqueeze(0)
            closed_now = (last_tokens == end_tensor).all(dim=1)
            self.has_closed = self.has_closed | closed_now

        if current_len >= self.budget:
            for i in range(input_ids.shape[0]):
                if not self.has_closed[i] and not self.force_attempted[i]:
                    scores[i, :] = -float('inf')
                    scores[i, self.first_end_id] = 0
                    self.force_attempted[i] = True

        return scores

def build_task(example, tokenizer):
    if example.get("test_setup_code"): return None
    test_list = example.get("test_list") or []
    if not test_list: return None
    ref_name = None
    for t in test_list:
        m = re.match(r"^assert\s+(\w+)\(", t.strip())
        if m: ref_name = m.group(1); break
    if not ref_name: return None

    renamed = [re.sub(rf"\b{re.escape(ref_name)}\b", "func", t) for t in test_list]
    test_cases = []
    for t in renamed:
        try:
            tree = ast.parse(t)
            node = tree.body[0]
            if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq): return None
            call = node.test.left
            if not isinstance(call, ast.Call): return None
            args = [ast.literal_eval(a) for a in call.args]
            expected = ast.literal_eval(node.test.comparators[0])
            test_cases.append([args, expected])
        except Exception:
            return None
    if len(test_cases) < 2: return None

    task_text = example.get("text") or ""
    if not task_text.strip(): return None
    
    task_text = re.sub(rf"\b{re.escape(ref_name)}\b", "func", task_text)

    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Think step by step inside <think> tags. Keep your reasoning around 2048 tokens, then give only the function definition inside a ```python``` block."},
        {"role": "user", "content": f"Task:\n{task_text.strip()}\n\nYour function must be named `func` and pass these tests:\n" + "\n".join(renamed)}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return {"prompt": prompt, "test_cases": test_cases}

def extract_code(text):
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(def\s+func\s*\(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

class RewardState:
    def __init__(self):
        self.batch_counter = 0

def make_reward_fn(evaluator, tracker, state):
    def reward_fn(prompts, completions, test_cases_json=None, **kwargs):
        num_prompts = len(prompts)
        if completions and isinstance(completions[0], list):
            groups = completions
        else:
            if num_prompts <= 0 or len(completions) % num_prompts != 0:
                return [0.0] * len(completions)
            gs = len(completions) // num_prompts
            groups = [completions[i * gs:(i + 1) * gs] for i in range(num_prompts)]
        
        if test_cases_json is None: return [0.0] * len(completions)
        tc_per_prompt = [ast.literal_eval(t) for t in test_cases_json] if len(test_cases_json) == num_prompts else [ast.literal_eval(test_cases_json[0])] * num_prompts

        all_rewards = []
        
        for gi, group in enumerate(groups):
            tc = tc_per_prompt[gi] if gi < len(tc_per_prompt) else tc_per_prompt[0]
            codes = [extract_code(c) for c in group]
            valid_indices = [i for i, c in enumerate(codes) if c]
            valid_codes = [codes[i] for i in valid_indices]
            
            group_rewards = [-1.0] * len(group)
            if not valid_codes:
                all_rewards.extend(group_rewards)
                continue

            try:
                results = evaluator.rank_drafts(valid_codes, "func", tc)
            except Exception as e:
                print(f"[Reward] rank_drafts error: {e}")
                all_rewards.extend(group_rewards)
                continue

            seen_cells = {}
            for r in results:
                orig_idx = valid_indices[r["id"]]
                reward = -1.0
                if r.get("alive") and r.get("trusted"):
                    reward = 1.0
                    if r.get("novelty", 0) == 1: reward += 1.0
                    reward += 0.1
                    
                    cell = r.get("cell")
                    count = seen_cells.get(cell, 0) + 1
                    seen_cells[cell] = count
                    penalty = -1.0 * (count - 1) / (NUM_GENERATIONS - 1)
                    reward += penalty
                    
                group_rewards[orig_idx] = reward

            all_rewards.extend(group_rewards)

        state.batch_counter += 1
        return all_rewards
    return reward_fn

def patch_generate_for_thinking(model, tokenizer, budget):
    original_generate = model.generate
    def patched(*args, **kwargs):
        processor = ThinkingBudgetLogitsProcessor(tokenizer, budget)
        existing = kwargs.get('logits_processor') or []
        kwargs['logits_processor'] = list(existing) + [processor]
        return original_generate(*args, **kwargs)
    model.generate = patched
    return original_generate

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ УНИКАЛЬНОСТИ (Qwen3-4B, GRPO, vLLM OFF, budget=2200)")
    print("=" * 60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.bfloat16, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_alpha=16, lora_dropout=0.0, bias="none", use_gradient_checkpointing="unsloth", random_state=SEED, max_seq_length=MAX_SEQ_LENGTH
    )

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    parts = [ds[k] for k in ("train", "test", "validation") if k in ds]
    all_ds = concatenate_datasets(parts) if parts else ds["train"]
    
    tasks = [build_task(ex, tokenizer) for ex in all_ds]
    tasks = [t for t in tasks if t is not None]
    random.seed(SEED)
    random.shuffle(tasks)
    print(f"Валидных задач: {len(tasks)}")

    evaluator = UltimateEvaluatorV3(max_steps=10000, n_fuzz=100)
    tracker = ProgressTracker(PROGRESS_PATH)
    signal = TrainingSignal(evaluator, STATE_PATH, tracker)
    signal.load()

    train_rows = [{"prompt": t["prompt"], "test_cases_json": repr(t["test_cases"])} for t in tasks]
    trl_ds = Dataset.from_list(train_rows)
    
    reward_fn = make_reward_fn(evaluator, tracker, RewardState())
    
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=NUM_GENERATIONS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        max_completion_length=MAX_NEW_TOKENS,
        num_generations=NUM_GENERATIONS,
        temperature=0.9,
        top_p=0.95,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=trl_ds,
    )

    patch_generate_for_thinking(trainer.model, tokenizer, THINKING_BUDGET)

    print("Старт обучения...")
    try:
        trainer.train(resume_from_checkpoint=True)
    except torch.cuda.OutOfMemoryError:
        print("OOM! Снизь num_generations до 2 или MAX_NEW_TOKENS до 2000.")
        raise
    finally:
        signal.save()
        tracker.save()

if __name__ == "__main__":
    main()
