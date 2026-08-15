"""
LoRA SFT for Qwen3 screening models using chat messages jsonl.
Loss is computed only on assistant tokens.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Example:
    messages: list[dict[str, str]]
    row_id: int
    label: str


class ChatJsonlDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.examples: list[Example] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.examples.append(
                    Example(
                        messages=obj["messages"],
                        row_id=int(obj.get("row_id", -1)),
                        label=str(obj.get("label", "")),
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


class ChatCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[Example]) -> dict[str, torch.Tensor]:
        input_ids_batch = []
        labels_batch = []
        attention_batch = []

        for ex in features:
            prompt_messages = ex.messages[:-1]
            full_messages = ex.messages

            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            full_ids = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            if len(labels) < len(full_ids):
                labels = labels + [-100] * (len(full_ids) - len(labels))
            labels = labels[: len(full_ids)]

            input_ids_batch.append(full_ids)
            labels_batch.append(labels)
            attention_batch.append([1] * len(full_ids))

        max_len = max(len(x) for x in input_ids_batch)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        def pad(seq: list[int], pad_value: int) -> list[int]:
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor(
            [pad(x, pad_id) for x in input_ids_batch], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [pad(x, 0) for x in attention_batch], dtype=torch.long
        )
        labels = torch.tensor(
            [pad(x, -100) for x in labels_batch], dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_DIR / "models" / "Qwen3-1.7B"),
    )
    p.add_argument(
        "--train-file",
        type=str,
        default=str(PROJECT_DIR / "sft_data" / "train.jsonl"),
    )
    p.add_argument(
        "--val-file",
        type=str,
        default=str(PROJECT_DIR / "sft_data" / "val.jsonl"),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_DIR / "sft_runs" / "qwen3_1_7b_lora"),
    )
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer/model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = ChatJsonlDataset(Path(args.train_file))
    val_ds = ChatJsonlDataset(Path(args.val_file))
    print(f"Train size={len(train_ds)} Val size={len(val_ds)}")

    collator = ChatCollator(tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        bf16=True,
        lr_scheduler_type="cosine",
        report_to=[],
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    metrics: dict[str, Any] = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save final adapter (avoid peft/transformers best-checkpoint reload bug).
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "run_args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
