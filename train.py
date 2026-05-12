"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset

try:
    from bleu import list_bleu
except ImportError:
    list_bleu = None


#   PYTORCH DATASET WRAPPER  

class TranslationDataset(Dataset):
    """Wraps the processed data for use with PyTorch DataLoader."""
    
    def __init__(self, data_list):
        """
        Args:
            data_list : List of (src_indices, tgt_indices) tuples
        """
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_indices, tgt_indices = self.data[idx]
        return torch.LongTensor(src_indices), torch.LongTensor(tgt_indices)


def collate_batch(batch):
    """Pad sequences to same length in batch."""
    srcs, tgts = zip(*batch)
    
    # Pad source sequences
    src_lens = [len(s) for s in srcs]
    max_src_len = max(src_lens)
    src_padded = torch.zeros(len(batch), max_src_len, dtype=torch.long)
    for i, src in enumerate(srcs):
        src_padded[i, :len(src)] = src
    
    # Pad target sequences
    tgt_lens = [len(t) for t in tgts]
    max_tgt_len = max(tgt_lens)
    tgt_padded = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    for i, tgt in enumerate(tgts):
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded



# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.smooth_value = smoothing / (vocab_size - 2)  # -2 because we exclude pad and one correct class

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            # Start with smooth distribution everywhere
            smooth_targets = torch.full_like(log_probs, self.smooth_value)
            # Set correct class to high confidence
            smooth_targets.scatter_(1, target.unsqueeze(1), self.confidence)
            # Set pad indices to 0 probability
            smooth_targets[:, self.pad_idx] = 0.0
        
        # KL-divergence between predicted and smoothed target distributions
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Ignore loss for padding tokens
        mask = (target != self.pad_idx).float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss





# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> tuple:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        Tuple[float, float]: (avg_loss, accuracy) where accuracy is token-level accuracy on validation.
                             During training, accuracy is 0.0 (not computed for efficiency).

    """
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0  # Track correct predictions for accuracy
    
    model.train() if is_train else model.eval()
    
    with torch.set_grad_enabled(is_train):
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create masks
            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt).to(device)
            
            # Forward pass (decoder input is tgt[:-1], target is tgt[1:])
            logits = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            
            # Reshape for loss computation
            logits_reshaped = logits.reshape(-1, logits.size(-1))
            tgt_reshaped = tgt[:, 1:].reshape(-1)
            
            # Compute loss
            loss = loss_fn(logits_reshaped, tgt_reshaped)
            
            # Compute token-level accuracy (non-pad tokens only)
            if not is_train:
                predictions = logits_reshaped.argmax(dim=-1)
                # Only count non-pad tokens
                non_pad_mask = (tgt_reshaped != 0)
                if non_pad_mask.sum() > 0:
                    correct_tokens += (predictions[non_pad_mask] == tgt_reshaped[non_pad_mask]).sum().item()
            
            if is_train:
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
            
            # Track loss
            batch_size = src.size(0)
            tgt_len = (tgt[:, 1:] != 0).sum().item()  # count non-pad tokens
            total_loss += loss.item() * tgt_len
            total_tokens += tgt_len
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    
    # Start with <sos>
    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).to(device)
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            # Prepare target mask for current sequence length
            tgt_mask = make_tgt_mask(ys).to(device)
            
            # Decode
            logits = model.decode(
                model.encode(src, src_mask),
                src_mask,
                ys,
                tgt_mask
            )
            
            # Get prediction for next token (greedy: argmax)
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)  # [1, 1]
            
            # Append next token
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if we generate <eos>
            if next_token.item() == end_symbol:
                break
    
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0-100).

    """
    model.eval()
    
    # Get special token indices
    sos_idx = tgt_vocab['<sos>']
    eos_idx = tgt_vocab['<eos>']
    pad_idx = tgt_vocab['<pad>']
    idx_to_token = {v: k for k, v in tgt_vocab.items()}

    def indices_to_tokens(indices):
        return [idx_to_token[idx] for idx in indices if idx in idx_to_token]
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create source mask
            src_mask = make_src_mask(src).to(device)
            
            # Generate predictions
            for i in range(src.size(0)):
                src_i = src[i:i+1]
                src_mask_i = src_mask[i:i+1]
                
                # Greedy decode
                pred_indices = greedy_decode(
                    model, src_i, src_mask_i, max_len,
                    sos_idx, eos_idx, device
                ).squeeze(0).cpu().tolist()
                
                # Convert indices to tokens
                pred_tokens = indices_to_tokens(pred_indices[1:])  # Skip <sos>
                if '<eos>' in pred_tokens:
                    pred_tokens = pred_tokens[:pred_tokens.index('<eos>')]
                
                # Reference
                ref_indices = tgt[i, 1:].cpu().tolist()  # Skip <sos>
                if eos_idx in ref_indices:
                    ref_indices = ref_indices[:ref_indices.index(eos_idx)]
                ref_tokens = indices_to_tokens(ref_indices)
                ref_tokens = [tok for tok in ref_tokens if tok != '<pad>']
                
                # Only add non-empty predictions and references
                pred_sent = ' '.join(pred_tokens)
                ref_sent = ' '.join(ref_tokens)
                
                # Skip if both are empty
                if pred_sent.strip() or ref_sent.strip():
                    predictions.append(pred_sent)
                    references.append([ref_sent])

    # Verify lengths match
    if len(predictions) != len(references):
        print(f"Warning: Predictions ({len(predictions)}) and references ({len(references)}) have different lengths!")
        # Trim to match lengths
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Use bleu library to compute BLEU score
    if len(predictions) == 0:
        print("Warning: No valid predictions generated!")
        return 0.0
    
    bleu_score = list_bleu(references, predictions)
    return float(bleu_score)


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': {
            'last_epoch': scheduler.last_epoch,
        },
        'model_config': model.config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.last_epoch = checkpoint['scheduler_state_dict']['last_epoch']
    
    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import wandb
    
    # Hyperparameters
    d_model = 512
    num_heads = 8
    d_ff = 2048
    N = 6
    num_epochs = 15
    batch_size = 32
    dropout = 0.1
    warmup_steps = 4000
    lr_smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize W&B
    wandb.init(
        project="da6401-a3",
        config={
            'd_model': d_model,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'N': N,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'dropout': dropout,
            'warmup_steps': warmup_steps,
            'label_smoothing': lr_smoothing,
        }
    )
    
    print("Loading dataset...")
    # Load and process dataset
    dataset = Multi30kDataset()
    dataset.build_vocab()
    train_data, val_data, test_data = dataset.process_data()
    
    src_vocab_size = len(dataset.de_vocab)
    tgt_vocab_size = len(dataset.en_vocab)
    pad_idx = dataset.en_vocab['<pad>']
    
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    
    # Create DataLoaders
    train_dataset = TranslationDataset(train_data)
    val_dataset = TranslationDataset(val_data)
    test_dataset = TranslationDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    
    print("Creating model...")
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        N=N,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    
    # Create scheduler
    scheduler = NoamScheduler(optimizer, d_model, warmup_steps)
    
    # Create loss function
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, pad_idx, lr_smoothing)
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss, _ = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch, is_train=True, device=device
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        val_loss, val_accuracy = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch, is_train=False, device=device
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': scheduler.get_last_lr()[0],
        })
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pt")
            print("Saved best checkpoint")
    
    # Load best model and evaluate on test set
    print("\nLoading best model and evaluating on test set...")
    load_checkpoint("best_checkpoint.pt", model, optimizer, scheduler)
    
    test_bleu = evaluate_bleu(model, test_loader, dataset.en_vocab, device, max_len=100)
    print(f"Test BLEU: {test_bleu:.2f}")
    
    wandb.log({'test_bleu': test_bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
