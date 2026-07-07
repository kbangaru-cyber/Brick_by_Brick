"""
BrickAgent — Stage 6: Reasoning SFT (Model A)
===============================================
Continues from Stage 5 adapter. Trains on Stage 6 with-reasoning data.
"""
import os
import torch
import numpy as np

BASE_DIR = os.environ.get(
    "BRICK_BASE_DIR",
    "/ocean/projects/cis260075p/bangarug/brickagent",
)
HF_CACHE   = os.path.join(BASE_DIR, "hf-cache")

# Saves to the repo's checkpoints/physics_reasoning/ (Stage 6 "full": Model A,
# reasons about physics/stability via <think>+<plan> before <build>+<review>).
# Override with BRICKAGENT_CHECKPOINTS_DIR to use a different checkpoints root.
OUTPUT_DIR = os.environ.get(
    "BRICKAGENT_PHYSICS_REASONING_DIR",
    os.path.join(
        os.environ.get(
            "BRICKAGENT_CHECKPOINTS_DIR",
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")),
        ),
        "physics_reasoning",
    ),
)

os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "unsloth/Qwen2.5-32B-Instruct"
DATA_FILE_REL = "data/stage6_full.jsonl"

CFG = {
    "name": "Stage 6 — Reasoning SFT (Model A)",
    "data_file": DATA_FILE_REL,
    "epochs": 1,
    "max_seq_length": 16384,
    "lr": 5e-5,
    "batch_size": 1,
    "grad_accum": 16,
    "warmup_ratio": 0.03,
}

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def train():
    import unsloth  # noqa: F401

    MAX_SEQ = CFG["max_seq_length"]
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    IS_MAIN = LOCAL_RANK == 0

    if WORLD_SIZE > 1:
        torch.cuda.set_device(LOCAL_RANK)

    base_grad_accum = CFG["grad_accum"]
    scaled_grad_accum = max(1, base_grad_accum // WORLD_SIZE)
    effective_batch = CFG["batch_size"] * scaled_grad_accum * WORLD_SIZE

    if IS_MAIN:
        print("=" * 60)
        print(f"  {CFG['name']}")
        print("=" * 60)
        print(f"  GPUs:   {WORLD_SIZE} x {torch.cuda.get_device_name(LOCAL_RANK)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1e9:.0f} GB each")
        print(f"  Output: {OUTPUT_DIR}")
        print(f"  Model:  {MODEL_NAME}")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    from unsloth import FastLanguageModel

    STAGE5_ADAPTER = os.path.join(BASE_DIR, "stage5_qwen25_32b")
    ckpts = []
    if os.path.isdir(STAGE5_ADAPTER):
        ckpts = sorted(
            [d for d in os.listdir(STAGE5_ADAPTER) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
    if ckpts:
        PREV_ADAPTER = os.path.join(STAGE5_ADAPTER, ckpts[-1])
    elif os.path.isdir(os.path.join(STAGE5_ADAPTER, "lora_adapter")):
        PREV_ADAPTER = os.path.join(STAGE5_ADAPTER, "lora_adapter")
    else:
        PREV_ADAPTER = None

    if PREV_ADAPTER and os.path.exists(PREV_ADAPTER):
        print(f"\n  Loading Stage 5 adapter: {PREV_ADAPTER}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PREV_ADAPTER,
            max_seq_length=MAX_SEQ,
            dtype=None,
            load_in_4bit=True, load_in_16bit=False, full_finetuning=False,
        )
        FastLanguageModel.for_training(model)
        model.gradient_checkpointing_enable()
        for n, p in model.named_parameters():
            if 'lora' in n.lower():
                p.requires_grad = True
        lora_trainable = sum(1 for n, p in model.named_parameters()
                             if 'lora' in n.lower() and p.requires_grad)
        print(f"  Stage 5 adapter loaded — {lora_trainable} trainable LoRA params")
    else:
        print(f"\n  No Stage 5 adapter found. Loading fresh {MODEL_NAME}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ,
            dtype=None,
            load_in_4bit=True, load_in_16bit=False, full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS,
            bias="none", use_gradient_checkpointing="unsloth",
            random_state=3407, max_seq_length=MAX_SEQ,
        )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total / 1e9:.2f}B")
    print(f"  Trainable params: {trainable / 1e6:.1f}M ({trainable / total * 100:.2f}%)")

    from datasets import load_dataset, load_from_disk

    PACK_CACHE_DIR = os.path.join(OUTPUT_DIR, "packed_cache")
    train_cache = os.path.join(PACK_CACHE_DIR, "train")
    val_cache = os.path.join(PACK_CACHE_DIR, "val")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"\n  Packed cache found — loading...")
        train_dataset = load_from_disk(train_cache)
        val_dataset = load_from_disk(val_cache)
        print(f"  Loaded {len(train_dataset):,} train, {len(val_dataset):,} val")
    else:
        DATA_FILE = os.path.join(BASE_DIR, CFG["data_file"])
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

        print(f"\nLoading {DATA_FILE}...")
        raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")
        print(f"  Loaded {len(raw_ds):,} samples")

        _inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
        IGNORE_INDEX = -100
        if _inner_tok.pad_token_id is None:
            _inner_tok.pad_token_id = _inner_tok.eos_token_id
        PAD_ID = _inner_tok.pad_token_id
        EOS_ID = _inner_tok.eos_token_id

        _asst_marker_ids = _inner_tok("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
        _end_marker_ids = _inner_tok("<|im_end|>", add_special_tokens=False)["input_ids"]

        def tokenize_with_labels(sample):
            messages = sample["messages"]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_ids = _inner_tok(full_text, truncation=True, max_length=MAX_SEQ, add_special_tokens=False)["input_ids"]
            labels = [IGNORE_INDEX] * len(full_ids)
            asst_len = len(_asst_marker_ids)
            end_len = len(_end_marker_ids)
            i = 0
            while i < len(full_ids) - asst_len + 1:
                if full_ids[i:i + asst_len] == _asst_marker_ids:
                    j = i + asst_len
                    while j < len(full_ids):
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

        tokenized_ds = raw_ds.map(tokenize_with_labels, remove_columns=raw_ds.column_names, num_proc=4, desc="Tokenizing")

        lengths = [len(tokenized_ds[int(i)]["input_ids"]) for i in np.random.choice(len(tokenized_ds), min(200, len(tokenized_ds)), replace=False)]
        print(f"  Token lengths — mean: {np.mean(lengths):.0f}, p95: {np.percentile(lengths, 95):.0f}, max: {max(lengths)}")

        tokenized_ds = tokenized_ds.shuffle(seed=42)
        split = tokenized_ds.train_test_split(test_size=0.02, seed=42)
        train_samples = split["train"]
        val_samples = split["test"]
        print(f"\n  Train: {len(train_samples):,} | Val: {len(val_samples):,}")

        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        print(f"\n  Packing into {MAX_SEQ}-token chunks...")

        def pack_to_disk(ds, max_len, pad_id, ignore_idx, eos_id, out_dir, desc="Packing"):
            import shutil
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            cur_ids, cur_labels = [], []
            batch_ids, batch_labels = [], []
            total_chunks = 0
            part_num = 0
            FLUSH_EVERY = 5000

            for i in tqdm(range(len(ds)), desc=f"  {desc}"):
                s_ids = ds[i]["input_ids"] + [eos_id]
                s_lab = ds[i]["labels"] + [eos_id]
                if len(cur_ids) + len(s_ids) <= max_len:
                    cur_ids.extend(s_ids)
                    cur_labels.extend(s_lab)
                else:
                    if cur_ids:
                        pad_len = max_len - len(cur_ids)
                        cur_ids.extend([pad_id] * pad_len)
                        cur_labels.extend([ignore_idx] * pad_len)
                        batch_ids.append(cur_ids)
                        batch_labels.append(cur_labels)
                    if len(s_ids) <= max_len:
                        cur_ids, cur_labels = list(s_ids), list(s_lab)
                    else:
                        batch_ids.append(s_ids[:max_len])
                        batch_labels.append(s_lab[:max_len])
                        cur_ids, cur_labels = [], []
                if len(batch_ids) >= FLUSH_EVERY:
                    table = pa.table({"input_ids": batch_ids, "labels": batch_labels})
                    pq.write_table(table, os.path.join(out_dir, f"part_{part_num:04d}.parquet"))
                    total_chunks += len(batch_ids)
                    part_num += 1
                    batch_ids, batch_labels = [], []

            if cur_ids:
                pad_len = max_len - len(cur_ids)
                cur_ids.extend([pad_id] * pad_len)
                cur_labels.extend([ignore_idx] * pad_len)
                batch_ids.append(cur_ids)
                batch_labels.append(cur_labels)
            if batch_ids:
                table = pa.table({"input_ids": batch_ids, "labels": batch_labels})
                pq.write_table(table, os.path.join(out_dir, f"part_{part_num:04d}.parquet"))
                total_chunks += len(batch_ids)
            return total_chunks

        def load_parquet_dir(parquet_dir):
            parts = sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
            ds = load_dataset("parquet", data_files=parts, split="train")
            for f in parts: os.remove(f)
            os.rmdir(parquet_dir)
            return ds

        train_parquet = os.path.join(OUTPUT_DIR, "_tmp_train_parquet")
        for epoch in range(CFG["epochs"]):
            epoch_indices = list(range(len(train_samples)))
            np.random.seed(42 + epoch)
            np.random.shuffle(epoch_indices)
            shuffled = train_samples.select(epoch_indices)
            n = pack_to_disk(shuffled, MAX_SEQ, PAD_ID, IGNORE_INDEX, EOS_ID, train_parquet, desc=f"Train epoch {epoch+1}")
            print(f"  Epoch {epoch + 1}: {n:,} chunks")

        val_parquet = os.path.join(OUTPUT_DIR, "_tmp_val_parquet")
        val_count = pack_to_disk(val_samples, MAX_SEQ, PAD_ID, IGNORE_INDEX, EOS_ID, val_parquet, desc="Val")
        print(f"  Val: {val_count:,} chunks")

        train_dataset = load_parquet_dir(train_parquet)
        val_dataset = load_parquet_dir(val_parquet)
        train_dataset = train_dataset.shuffle(seed=42)

        os.makedirs(PACK_CACHE_DIR, exist_ok=True)
        train_dataset.save_to_disk(train_cache)
        val_dataset.save_to_disk(val_cache)
        print(f"  Cache saved. Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    from trl import SFTTrainer, SFTConfig

    DS_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_zero2.json")
    use_deepspeed = WORLD_SIZE > 1 and os.path.exists(DS_CONFIG)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=CFG["batch_size"],
        per_device_eval_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=scaled_grad_accum,
        learning_rate=CFG["lr"],
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=CFG["warmup_ratio"],
        optim="adamw_8bit",
        bf16=True, fp16=False, tf32=True,
        max_seq_length=MAX_SEQ,
        packing=False,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=42,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=5,
        remove_unused_columns=False,
        deepspeed=DS_CONFIG if use_deepspeed else None,
        ddp_find_unused_parameters=False if WORLD_SIZE > 1 else None,
        local_rank=LOCAL_RANK if WORLD_SIZE > 1 else -1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    total_steps = len(train_dataset) // effective_batch
    if IS_MAIN:
        print(f"\n  Effective batch: {effective_batch}")
        print(f"  Total steps:     ~{total_steps:,}")

    checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = sorted(
            [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"\n  Resuming from: {checkpoint}")

    print(f"\n{'=' * 60}\n  Starting training...\n{'=' * 60}")
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        if checkpoint and "checkpoint" in str(e).lower():
            print(f"\n  Checkpoint incompatible, starting fresh...")
            trainer_stats = trainer.train()
        else:
            raise

    if IS_MAIN:
        print(f"\n  TRAINING COMPLETE — {CFG['name']}")
        print(f"  Runtime:    {trainer_stats.metrics['train_runtime'] / 3600:.1f}h")
        print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")
        lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)
        print(f"  Saved: {lora_path}")


if __name__ == "__main__":
    train()
