"""Trainer optimized for maximum convergence within 2,000 steps on CPU:
- AdamW optimizer with weight decay (excluding 1D bias/norm tensors)
- Cosine Learning Rate Schedule with Linear Warmup (100 steps warmup to max_lr, decay to 10%)
- Gradient clipping (max_norm=1.0)
- Gradient Accumulation for custom effective batch size
"""
import argparse
import time
import math

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return 0.1 * max_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 0.1 * max_lr + coeff * 0.9 * max_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2, help="Effective batch size (number of sequences)")
    ap.add_argument("--micro_batch", type=int, default=1, help="Micro-batch size (number of sequences)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()
    
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    assert args.batch % args.micro_batch == 0, "batch size must be divisible by micro_batch size"
    
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.dropout = args.dropout
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # Configure weight decay groups
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    opt = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

    grad_accum_steps = args.batch // args.micro_batch
    print(f"Training parameters: effective batch {args.batch} (accumulating over {grad_accum_steps} steps of micro-batch {args.micro_batch})")

    model.train()
    t0 = time.time()
    losses = []
    
    for step in range(1, args.steps + 1):
        # Update learning rate according to scheduler
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
            
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = get_batch(ids, cfg.block_size, args.micro_batch, device)
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.item() * grad_accum_steps
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        losses.append(loss_accum)
        
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  lr {lr:.6f}  loss {avg:.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    # Save checkpoint
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
