import argparse
import json
import math
import tqdm
import pickle
import random
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Stormbound Constants
EMB_DIM = 128
DECK_SIZE = 12  # Changed from 60
TIMESTEPS = 1000

class StormboundDeckDataset(Dataset):
    def __init__(self, deck_dir: str, embeddings_path: str, cards_json_path: str):
        """
        Dataset for Stormbound decks
        Args:
            deck_dir: Directory containing deck JSON files
            embeddings_path: Path to card embeddings pickle file
            cards_json_path: Path to cards.json with card data
        """
        # Load card embeddings
        with open(embeddings_path, "rb") as f:
            raw = pickle.load(f)
        self.card2vec = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}

        # Load card data
        with open(cards_json_path, "r", encoding="utf-8") as f:
            cards_data = json.load(f)

        # Create card ID to index mapping
        self.card_to_idx = {card["id"]: idx for idx, card in enumerate(cards_data)}
        self.idx_to_card = {idx: card["id"] for idx, card in enumerate(cards_data)}

        # Create faction mappings
        self.faction_map = {
            "중립": 0,  # Neutral
            "겨울": 1,  # Winter  
            "아이언클래드": 2,  # Ironclad
            "섀도우펜": 3,  # Shadowfen
            "스웜": 4   # Swarm
        }

        self.decks_embeds = []
        self.decks_indices = []
        self.deck_factions = []  # Track faction for each deck

        card_counts = {}

        # Process deck files
        for path in Path(deck_dir).glob("*.json"):
            try:
                deck_data = json.load(open(path, encoding="utf-8"))

                # Extract deck cards - expecting format: {"cards": [{"id": "u001", "level": 1}, ...]}
                deck_cards = deck_data.get("cards", [])

                if len(deck_cards) != DECK_SIZE:
                    print(f"Warning: Deck {path.name} has {len(deck_cards)} cards, expected {DECK_SIZE}. Skipping.")
                    continue

                deck_embeds = []
                deck_indices = []
                deck_faction_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Count cards per faction

                for card_entry in deck_cards:
                    card_id = card_entry["id"]

                    # Get embedding
                    if card_id not in self.card2vec:
                        print(f"Warning: Card {card_id} not found in embeddings. Skipping deck {path.name}")
                        break

                    # Get card index
                    if card_id not in self.card_to_idx:
                        print(f"Warning: Card {card_id} not found in card mapping. Skipping deck {path.name}")
                        break

                    # Find card faction from cards data
                    card_data = next((c for c in cards_data if c["id"] == card_id), None)
                    if not card_data:
                        print(f"Warning: Card {card_id} data not found. Skipping deck {path.name}")
                        break

                    faction = self.faction_map.get(card_data["kingdom"], 0)
                    deck_faction_counts[faction] += 1

                    deck_embeds.append(self.card2vec[card_id])
                    deck_indices.append(self.card_to_idx[card_id])

                    # Track card popularity
                    card_counts[card_id] = card_counts.get(card_id, 0) + 1

                if len(deck_embeds) != DECK_SIZE:
                    continue  # Skip incomplete decks

                # Validate faction constraint (Neutral + at most one other faction)
                non_neutral_factions = sum(1 for f, count in deck_faction_counts.items() if f != 0 and count > 0)
                if non_neutral_factions > 1:
                    print(f"Warning: Deck {path.name} has cards from multiple non-neutral factions. Skipping.")
                    continue

                # Determine primary faction (most common non-neutral, or neutral if all neutral)
                primary_faction = 0  # Default to neutral
                max_non_neutral = 0
                for faction, count in deck_faction_counts.items():
                    if faction != 0 and count > max_non_neutral:
                        max_non_neutral = count
                        primary_faction = faction

                self.decks_embeds.append(torch.stack(deck_embeds))
                self.decks_indices.append(torch.tensor(deck_indices, dtype=torch.long))
                self.deck_factions.append(primary_faction)

            except Exception as e:
                print(f"Error processing deck {path.name}: {e}")

        # Calculate card popularity
        if card_counts:
            highest_count = max(card_counts.values())
            self.card_popularity = {
                self.card_to_idx[card_id]: 1.0 - ((count - 1) / highest_count)
                for card_id, count in card_counts.items()
                if card_id in self.card_to_idx
            }
        else:
            self.card_popularity = {}

        print(f"Loaded {len(self.decks_embeds)} valid Stormbound decks")
        print(f"Calculated popularity for {len(self.card_popularity)} unique cards")

    def __len__(self):
        return len(self.decks_embeds)

    def __getitem__(self, idx):
        return (
            self.decks_embeds[idx],
            self.decks_indices[idx],
            self.deck_factions[idx]
        )

def sinusoidal_embedding(t: torch.Tensor, dim: int = EMB_DIM):
    """Sinusoidal time embedding"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * freqs[None]
    emb = torch.cat((args.sin(), args.cos()), dim=-1)
    if dim % 2:
        emb = nn.functional.pad(emb, (0, 1))
    return emb

class StormboundDiffusionModel(nn.Module):
    """Simplified diffusion model for Stormbound decks (no sideboard)"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_dim = cfg["model_dim"]
        nhead = cfg["heads"]
        dim_feedforward = cfg["dim_feedforward"]
        num_layers = cfg["layers"]
        activation = "gelu"
        batch_first = True

        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(EMB_DIM, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, EMB_DIM),
        )

        # Mask MLP
        self.mask_mlp = nn.Sequential(
            nn.Linear(1, EMB_DIM),
            nn.SiLU(),
            nn.Linear(EMB_DIM, EMB_DIM),
        )

        # Faction embedding (for conditioning on faction)
        self.faction_embedding = nn.Embedding(5, EMB_DIM)  # 5 factions

        # Transformer Encoder for deck processing
        self.input_proj = nn.Linear(EMB_DIM, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(model_dim, EMB_DIM)

    def forward(self, x_t, t, mask, faction):
        """
        Args:
            x_t: Noisy deck embeddings [batch, 12, 128]
            t: Timestep [batch]
            mask: Card visibility mask [batch, 12, 1]
            faction: Deck faction [batch]
        """
        batch_size = x_t.shape[0]

        # Time embedding
        sin_emb = sinusoidal_embedding(t, EMB_DIM)
        time_emb = self.time_mlp(sin_emb)
        time_emb = time_emb[:, None, :].expand(-1, DECK_SIZE, -1)  # [batch, 12, 128]

        # Mask embedding
        mask_emb = self.mask_mlp(mask)  # [batch, 12, 128]

        # Faction embedding
        faction_emb = self.faction_embedding(faction)  # [batch, 128]
        faction_emb = faction_emb[:, None, :].expand(-1, DECK_SIZE, -1)  # [batch, 12, 128]

        # Combine embeddings
        h = x_t + time_emb + mask_emb + faction_emb  # [batch, 12, 128]

        # Project and transform
        h_proj = self.input_proj(h)  # [batch, 12, model_dim]
        encoded = self.transformer_encoder(h_proj)  # [batch, 12, model_dim]
        noise_pred = self.output_proj(encoded)  # [batch, 12, 128]

        return noise_pred

def cosine_beta_schedule(T, s=0.008):
    """Cosine variance schedule for diffusion"""
    steps = torch.linspace(0, T, T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0, 0.999).float()

class StormboundDiffusionTrainer:
    def __init__(self, model, device, lr, weight_decay, total_epochs, card_popularity, T=TIMESTEPS, masks_per_deck=1):
        self.model = model
        self.device = device
        self.total_epochs = total_epochs
        self.card_popularity = card_popularity
        self.T = T
        self.masks_per_deck = masks_per_deck

        # Setup diffusion schedule
        self.register("betas", cosine_beta_schedule(T))
        self.register("alphas", 1 - self.betas)
        self.register("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss function
        self.loss_fn = nn.MSELoss()

    def register(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    @torch.no_grad()
    def _extract(self, a, t, shape, target_device):
        """Extract values from tensor a at indices t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t).to(target_device)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, mask, noise=None):
        """Add noise to clean data according to diffusion schedule"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape, x0.device)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape, x0.device)

        # Only add noise to visible cards (where mask == 1)
        noisy = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x0 * (1 - mask) + noisy * mask

    def _create_mask_row(self, k_target, deck_size, current_deck_indices, device):
        """Create mask for a single deck"""
        if k_target <= 0:
            return torch.zeros(deck_size, 1, device=device)

        if k_target >= deck_size:
            return torch.ones(deck_size, 1, device=device)

        # Popularity-based sampling
        card_popularities = torch.tensor([
            self.card_popularity.get(idx.item(), 0.5) for idx in current_deck_indices
        ], device=device)

        # Sample cards to mask based on popularity (higher popularity = more likely to be masked)
        probs = card_popularities / card_popularities.sum()
        indices = torch.multinomial(probs, k_target, replacement=False)

        mask = torch.zeros(deck_size, 1, device=device)
        mask[indices] = 1.0
        return mask

    def _generate_mask_and_k(self, B, N, deck_size, x0_indices_repeated, device):
        """Generate masks and k values for batch"""
        # Sample k values (number of cards to mask)
        k_values = torch.randint(1, deck_size, (B * N,), device=device).float()

        masks = []
        for i in range(B * N):
            k = int(k_values[i].item())
            current_deck_indices = x0_indices_repeated[i]
            mask = self._create_mask_row(k, deck_size, current_deck_indices, device)
            masks.append(mask)

        masks = torch.stack(masks)  # [B*N, deck_size, 1]
        return masks, k_values

    def p_losses(self, x0_embeddings, x0_indices, factions):
        """Calculate training loss"""
        B = x0_embeddings.shape[0]
        N = self.masks_per_deck
        device = x0_embeddings.device

        # Repeat data for multiple masks per deck
        x0_repeated = x0_embeddings.repeat_interleave(N, dim=0)  # [B*N, 12, 128]
        x0_indices_repeated = x0_indices.repeat_interleave(N, dim=0)  # [B*N, 12]
        factions_repeated = factions.repeat_interleave(N, dim=0)  # [B*N]

        # Generate masks
        masks, k_values = self._generate_mask_and_k(B, N, DECK_SIZE, x0_indices_repeated, device)

        # Sample timesteps
        t = torch.randint(0, self.T, (B * N,), device=device).long()

        # Sample noise
        noise = torch.randn_like(x0_repeated)

        # Create noisy input
        x_t = self.q_sample(x0_repeated, t, masks, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t, masks, factions_repeated)

        # Calculate loss only on masked positions
        loss = self.loss_fn(predicted_noise * masks, noise * masks)

        return loss

    def train(self, train_loader, epochs, start_epoch, save_path):
        """Training loop"""
        self.model.train()

        for epoch in range(start_epoch, epochs):
            total_loss = 0
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (x0_embeddings, x0_indices, factions) in enumerate(progress_bar):
                x0_embeddings = x0_embeddings.to(self.device)
                x0_indices = x0_indices.to(self.device)
                factions = factions.to(self.device)

                self.optimizer.zero_grad()
                loss = self.p_losses(x0_embeddings, x0_indices, factions)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"{save_path}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': self.model.cfg
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck_dir", required=True, help="Directory containing Stormbound deck JSON files")
    parser.add_argument("--embeddings_path", required=True, help="Path to card embeddings pickle file")
    parser.add_argument("--cards_json", required=True, help="Path to cards.json")
    parser.add_argument("--save_path", default="stormbound_diffusion_model", help="Save path for model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Feedforward dimension")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    cfg = {
        "model_dim": args.model_dim,
        "heads": args.heads,
        "layers": args.layers,
        "dim_feedforward": args.dim_feedforward,
    }

    # Create dataset
    dataset = StormboundDeckDataset(args.deck_dir, args.embeddings_path, args.cards_json)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Create model
    model = StormboundDiffusionModel(cfg).to(device)

    # Create trainer
    trainer = StormboundDiffusionTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_epochs=args.epochs,
        card_popularity=dataset.card_popularity
    )

    # Train model
    trainer.train(train_loader, args.epochs, 0, args.save_path)

if __name__ == "__main__":
    main() 