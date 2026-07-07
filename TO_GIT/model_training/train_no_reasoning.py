"""
BrickAgent — Fine-Tuning Qwen2.5-32B-Instruct on PSC (H100)
=============================================================
Adapted from the proven Rhino Agent training recipe:
  - Qwen2.5 architecture (standard attention → FA2 optimized)
  - 4-bit QLoRA (fast, memory-efficient)
  - LoRA R=64, Alpha=128 (same as Rhino agent)
  - Assistant-only label masking
  - Boundary-aware packing

Expected: ~25-35 sec/step on H100 (vs 80 sec with Qwen3.5-27B 16-bit)

Usage:
    sbatch run_train_no_reasoning.sh
    python train_no_reasoning.py
"""

import os
import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Paths — edit these to match your PSC layout
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.environ.get(
    "BRICK_BASE_DIR",
    os.path.join(os.environ.get("PROJECT", os.path.expanduser("~")), "brickagent"),
)
HF_CACHE   = os.path.join(BASE_DIR, "hf-cache")

# Saves to the repo's checkpoints/no_reasoning/ (Stage 1: structure generator,
# trained before any physics/CoT reasoning was added to the curriculum).
# Override with BRICKAGENT_CHECKPOINTS_DIR to use a different checkpoints root.
OUTPUT_DIR = os.environ.get(
    "BRICKAGENT_NO_REASONING_DIR",
    os.path.join(
        os.environ.get(
            "BRICKAGENT_CHECKPOINTS_DIR",
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")),
        ),
        "no_reasoning",
    ),
)

os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ═══════════════════════════════════════════════════════════════
#  Config — modeled after Rhino Agent recipe
# ═══════════════════════════════════════════════════════════════
STAGE = 1
MODEL_NAME = "unsloth/Qwen2.5-32B-Instruct"
DISABLE_THINKING = False  # Qwen2.5 doesn't have thinking mode

STAGE_CONFIG = {
    1: {
        "name": "Stage 1 — Structure Generator",
        "data_file": "stage1_structure_generator.jsonl",
        "epochs": 1,
        "max_seq_length": 8192,
        "lr": 2e-4,
        "batch_size": 2,       # 4-bit fits batch=2 on H100 80GB
        "grad_accum": 8,       # effective batch = 2 * 8 = 16
        "warmup_ratio": 0.05,
    },
    2: {
        "name": "Stage 2 — Physics-Aware Builder",
        "data_file": "stage2_physics_builder.jsonl",
        "epochs": 2,
        "max_seq_length": 16384,
        "lr": 1e-4,
        "batch_size": 1,
        "grad_accum": 16,
        "warmup_ratio": 0.03,
    },
}

# LoRA config — R=64, Alpha=128 from Rhino agent, dropout=0 for Unsloth fast path
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════
def train():
    # IMPORTANT: Import unsloth FIRST
    import unsloth  # noqa: F401

    CFG = STAGE_CONFIG[STAGE]
    MAX_SEQ = CFG["max_seq_length"]

    print(f"{'=' * 60}")
    print(f"  {CFG['name']}")
    print(f"{'=' * 60}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Model:  {MODEL_NAME}")

    # ── HF Login ──────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)
        print("  HF login: OK")
    else:
        print("  HF login: skipped (no HF_TOKEN)")

    # ── Load Model (4-bit QLoRA — same as Rhino agent) ────────
    from unsloth import FastLanguageModel

    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=None,
        load_in_4bit=True,       # 4-bit like Rhino agent
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total / 1e9:.2f}B")
    print(f"  Trainable params: {trainable / 1e6:.1f}M ({trainable / total * 100:.2f}%)")
    print(f"  VRAM used:        {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ══════════════════════════════════════════════════════════
    #  Tokenize from local JSONL dataset
    # ══════════════════════════════════════════════════════════
    from datasets import load_dataset, Dataset

    DATA_FILE = os.path.join(BASE_DIR, CFG["data_file"])
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}\n"
            f"  Copy it to: {BASE_DIR}/"
        )

    print(f"\nLoading {DATA_FILE}...")
    raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"  Loaded {len(raw_ds):,} samples")

    # ── Tokenize with assistant-only label masking ────────────
    # Only the assistant response is trained on.
    # System prompt + user message are seen via attention but masked from loss.
    _inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
    IGNORE_INDEX = -100

    if _inner_tok.pad_token_id is None:
        _inner_tok.pad_token_id = _inner_tok.eos_token_id
    PAD_ID = _inner_tok.pad_token_id
    EOS_ID = _inner_tok.eos_token_id

    # Token IDs for locating assistant sections
    _asst_marker_ids = _inner_tok("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    _end_marker_ids = _inner_tok("<|im_end|>", add_special_tokens=False)["input_ids"]

    print(f"  Tokenizing with assistant-only label masking...")
    print(f"  Assistant marker tokens: {_asst_marker_ids}")
    print(f"  End marker tokens: {_end_marker_ids}")

    def tokenize_with_labels(sample):
        messages = sample["messages"]

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_ids = _inner_tok(
            full_text, truncation=True, max_length=MAX_SEQ, add_special_tokens=False,
        )["input_ids"]

        labels = [IGNORE_INDEX] * len(full_ids)

        # Find <|im_start|>assistant markers and unmask until <|im_end|>
        asst_len = len(_asst_marker_ids)
        end_len = len(_end_marker_ids)

        i = 0
        while i < len(full_ids) - asst_len + 1:
            if full_ids[i:i + asst_len] == _asst_marker_ids:
                # Start unmasking from the token after "assistant\n"
                j = i + asst_len
                while j < len(full_ids):
                    # Check for <|im_end|> to stop unmasking
                    if j <= len(full_ids) - end_len and full_ids[j:j + end_len] == _end_marker_ids:
                        for k in range(end_len):
                            labels[j + k] = full_ids[j + k]
                        j += end_len
                        break
                    labels[j] = full_ids[j]
                    j += 1
                i = j
            else:
                i += 1

        return {"input_ids": full_ids, "labels": labels}

    tokenized_ds = raw_ds.map(
        tokenize_with_labels,
        remove_columns=raw_ds.column_names,
        num_proc=4,       # PSC nodes have plenty of CPU cores
        desc="Tokenizing",
    )

    # ── Token length stats ────────────────────────────────────
    sample_indices = np.random.choice(
        len(tokenized_ds), min(200, len(tokenized_ds)), replace=False,
    )
    lengths = [len(tokenized_ds[int(i)]["input_ids"]) for i in sample_indices]
    avg_len = np.mean(lengths)
    print(f"  Token lengths — mean: {avg_len:.0f}, "
          f"p95: {np.percentile(lengths, 95):.0f}, max: {max(lengths)}")

    sample_labels = [tokenized_ds[int(i)]["labels"] for i in sample_indices[:50]]
    trained_ratio = np.mean([
        sum(1 for t in lab if t != IGNORE_INDEX) / max(len(lab), 1)
        for lab in sample_labels
    ])
    print(f"  Trained token ratio: {trained_ratio:.1%} (assistant only)")
    print(f"  Avg samples per pack: {MAX_SEQ / max(avg_len, 1):.1f}")

    # ── Train/Val split ───────────────────────────────────────
    tokenized_ds = tokenized_ds.shuffle(seed=42)
    split = tokenized_ds.train_test_split(test_size=0.02, seed=42)
    train_samples = split["train"]
    val_samples = split["test"]

    print(f"\n  Train samples: {len(train_samples):,}")
    print(f"  Val samples:   {len(val_samples):,}")

    # ── Boundary-aware packing ────────────────────────────────
    print(f"\n  Packing into {MAX_SEQ}-token chunks (boundary-aware)...")

    def pack_dataset(ds, max_len, pad_id, ignore_idx, eos_id):
        packed_ids, packed_labels = [], []
        cur_ids, cur_labels = [], []

        for i in range(len(ds)):
            s_ids = ds[i]["input_ids"] + [eos_id]
            s_lab = ds[i]["labels"] + [eos_id]      # ← THE FIX: was [ignore_idx]

            if len(cur_ids) + len(s_ids) <= max_len:
                cur_ids.extend(s_ids)
                cur_labels.extend(s_lab)
            else:
                if cur_ids:
                    pad_len = max_len - len(cur_ids)
                    cur_ids.extend([pad_id] * pad_len)
                    cur_labels.extend([ignore_idx] * pad_len)
                    packed_ids.append(cur_ids)
                    packed_labels.append(cur_labels)

                if len(s_ids) <= max_len:
                    cur_ids, cur_labels = list(s_ids), list(s_lab)
                else:
                    packed_ids.append(s_ids[:max_len])
                    packed_labels.append(s_lab[:max_len])
                    cur_ids, cur_labels = [], []

        if cur_ids:
            pad_len = max_len - len(cur_ids)
            cur_ids.extend([pad_id] * pad_len)
            cur_labels.extend([ignore_idx] * pad_len)
            packed_ids.append(cur_ids)
            packed_labels.append(cur_labels)

        return packed_ids, packed_labels

    num_epochs = CFG["epochs"]
    all_ids, all_labels = [], []

    for epoch in range(num_epochs):
        epoch_indices = list(range(len(train_samples)))
        np.random.seed(42 + epoch)
        np.random.shuffle(epoch_indices)
        shuffled = train_samples.select(epoch_indices)

        ep_ids, ep_labels = pack_dataset(
            shuffled, MAX_SEQ, PAD_ID, IGNORE_INDEX, EOS_ID,
        )
        print(f"  Epoch {epoch + 1}: {len(ep_ids):,} packed chunks")
        all_ids.extend(ep_ids)
        all_labels.extend(ep_labels)

    val_ids, val_labels = pack_dataset(
        val_samples, MAX_SEQ, PAD_ID, IGNORE_INDEX, EOS_ID,
    )

    final_idx = list(range(len(all_ids)))
    np.random.seed(42)
    np.random.shuffle(final_idx)
    all_ids = [all_ids[i] for i in final_idx]
    all_labels = [all_labels[i] for i in final_idx]

    train_dataset = Dataset.from_dict({"input_ids": all_ids, "labels": all_labels})
    val_dataset = Dataset.from_dict({"input_ids": val_ids, "labels": val_labels})

    packs_per_epoch = len(all_ids) // max(num_epochs, 1)
    print(f"\n  Packed: {packs_per_epoch:,}/epoch × {num_epochs} "
          f"= {len(all_ids):,} train, {len(val_dataset):,} val")
    print(f"  Utilization: ~{avg_len * len(train_samples) / (packs_per_epoch * MAX_SEQ) * 100:.0f}%")

    # ══════════════════════════════════════════════════════════
    #  SFTTrainer
    # ══════════════════════════════════════════════════════════
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=CFG["batch_size"],
        per_device_eval_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"],
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=CFG["warmup_ratio"],
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,
        max_seq_length=MAX_SEQ,
        packing=False,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        report_to="none",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=5,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # ── Pre-training info ─────────────────────────────────────
    eff_batch = CFG["batch_size"] * CFG["grad_accum"]
    total_steps = len(train_dataset) // eff_batch

    print(f"\n  VRAM before training: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"  Effective batch:     {eff_batch}")
    print(f"  Total steps:         ~{total_steps:,}")
    print(f"  Est. time:           ~{total_steps * 30 / 3600:.1f}h @ ~30s/step")

    # ── Auto-resume from latest checkpoint ────────────────────
    checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = sorted(
            [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            ckpt_files = os.listdir(checkpoint)
            print(f"\n  Resuming from: {checkpoint}")
            print(f"  Contents: {ckpt_files}")

            # Fix sharded adapter filenames if needed
            has_adapter = "adapter_model.safetensors" in ckpt_files
            sharded = [f for f in ckpt_files
                       if f.startswith("adapter_model-")
                       and f.endswith(".safetensors")]
            if not has_adapter and sharded:
                src = os.path.join(checkpoint, sharded[0])
                dst = os.path.join(checkpoint, "adapter_model.safetensors")
                print(f"  Symlink: adapter_model.safetensors -> {sharded[0]}")
                os.symlink(src, dst)

    if checkpoint is None:
        print(f"\n  Training from scratch.")

    # ── TRAIN ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Starting training...")
    print(f"{'=' * 60}")

    try:
        trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        if checkpoint and "checkpoint" in str(e).lower():
            print(f"\n  Checkpoint incompatible, starting fresh...")
            trainer_stats = trainer.train()
        else:
            raise

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE — {CFG['name']}")
    print(f"{'=' * 60}")
    print(f"  Runtime:     {trainer_stats.metrics['train_runtime'] / 3600:.1f}h")
    print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.1f}")
    print(f"  Final loss:  {trainer_stats.metrics['train_loss']:.4f}")
    print(f"  Peak VRAM:   {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ── Save LoRA adapter ─────────────────────────────────────
    lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\n  LoRA adapter saved: {lora_path}")


if __name__ == "__main__":
    train()
