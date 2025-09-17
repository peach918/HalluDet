import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import random
import logging
import pandas as pd
import time
from openai import OpenAI
import numpy as np
from sklearn.model_selection import StratifiedKFold
import csv
from sklearn.metrics import roc_auc_score
import Multi-Path_generation

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):  # Alpha can now be a tensor
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss



class TrajectoryGatedClassifier_Transformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads=8, dropout_rate=0.4, num_transformer_layers=2, max_cot_len=64):
        super().__init__()
        self.main_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.main_layernorm = nn.LayerNorm(hidden_dim)
        self.main_segment_embed = nn.Embedding(3, hidden_dim)
        self.main_position_embed = nn.Parameter(torch.zeros(3, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.cot_transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        self.cot_position_embed = nn.Parameter(torch.zeros(max_cot_len, hidden_dim))
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_dict):
        E1, E2_traj, E3, E4 = x_dict['E1'], x_dict['E2_trajectory'], x_dict['E3'], x_dict['E4']
        main_features = torch.stack([E1, E3, E4], dim=1)
        B, N, D = main_features.size()
        seg = self.main_segment_embed(torch.arange(N, device=device)).unsqueeze(0).expand(B, -1, -1)
        pos = self.main_position_embed.unsqueeze(0).expand(B, -1, -1)
        main_features = self.main_layernorm(main_features + seg + pos)
        main_attn_out, _ = self.main_attention(main_features, main_features, main_features)
        main_representation = main_attn_out.reshape(B, -1)

        B_traj, S_traj, D_traj = E2_traj.size()

        pos_encoding = self.cot_position_embed[:S_traj, :].unsqueeze(0)
        E2_traj_with_pos = E2_traj + pos_encoding

        transformer_output = self.cot_transformer_encoder(E2_traj_with_pos)

        trajectory_rep = transformer_output.mean(dim=1)
        gate_value = self.gate_network(main_representation)
        gated_cot_rep = gate_value * trajectory_rep
        final_features = torch.cat([main_representation, gated_cot_rep], dim=1)

        return self.final_classifier(final_features)


# --- Training ---
def train():
    model_path = "MODEL_PATH"
    SAVED_DATASET_PATH = 'DATASET_PATH'
    CSV_OUTPUT_PATH = 'OUTPUT_PATH'

    if os.path.exists(SAVED_DATASET_PATH):
        print(f"Found saved dataset at '{SAVED_DATASET_PATH}'. Loading directly...")
        dataset = torch.load(SAVED_DATASET_PATH)
        try:
            config = AutoConfig.from_pretrained(model_path)
            hidden_size = config.hidden_size
        except Exception as e:
            sample_features, _ = dataset[0]
            hidden_size = sample_features['E1'].shape[-1]
    else:
        print(f"No saved dataset found. Starting new data generation...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Fatal: OpenAI API key not found.")
            return
        openai_client = OpenAI(api_key=api_key, base_url="https://yunwu.ai/v1")
        print(f"Loading base model and tokenizer from local path: {model_path}...")
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                             device_map="auto").eval()
            llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            if llm_tokenizer.pad_token is None: llm_tokenizer.pad_token = llm_tokenizer.eos_token
            hidden_size = llm_model.config.hidden_size
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}");
            return

        csv_path = "CSV_PATH"
        print(f"Loading questions from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
            prompts = df['Question'].tolist()[:800]
        except Exception as e:
            print(f"Failed to load CSV: {e}");
            return

        with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Question', 'AnswerA', 'AnswerB', 'Label'])
            print(f"Creating dataset and writing text data to {CSV_OUTPUT_PATH}...")
            dataset = HallucinationDataset(prompts, llm_model, llm_tokenizer, openai_client, csv_writer)

        if len(dataset) > 0:
            print(f"Data generation complete. Saving dataset to '{SAVED_DATASET_PATH}'...")
            torch.save(dataset, SAVED_DATASET_PATH)
            print("Dataset saved successfully.")

        del llm_model
        torch.cuda.empty_cache()

    if len(dataset) < 5:
        print("Dataset is too small for 5-fold cross-validation. Exiting.")
        return

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_labels = [label for _, label in dataset]
    fold_acc_results, fold_auroc_results = [], []

    config = {
        'learning_rate': 2e-5,
        'dropout_rate': 0.4,
        'focal_loss_gamma': 2.0,
        'batch_size': 8
    }
    print(f"Using config: {config}")

    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(dataset)), all_labels)):
        print(f"\n{'=' * 20} FOLD {fold + 1}/{k_folds} {'=' * 20}")

        train_labels = [all_labels[i] for i in train_ids]

        if len(np.unique(train_labels)) < 2:
            print("Warning: Training fold contains only one class. Skipping this fold.")
            continue

        class_counts = np.bincount(train_labels, minlength=2)

        total_samples = len(train_labels)
        num_classes = 2
        class_weights = total_samples / (num_classes * class_counts)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Calculated Class Weights for this fold: {class_weights_tensor}")

        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True,
                                  collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False,
                                collate_fn=custom_collate_fn)

        classifier_model = TrajectoryGatedClassifier_Transformer(
            hidden_dim=hidden_size,
            dropout_rate=config['dropout_rate'],
            num_heads=8,
            num_transformer_layers=2,
            max_cot_len=64
        ).to(device)
        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

        loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=config['focal_loss_gamma'])

        noise_level = 0.01
        accumulation_steps = 2
        for epoch in range(15):
            classifier_model.train()
            optimizer.zero_grad()
            for i, (x_dict, y) in enumerate(train_loader):
                for k in x_dict: x_dict[k] = x_dict[k].to(device, dtype=torch.float32)
                y = y.to(device)

                minority_mask = (y == 1)
                if minority_mask.any():
                    for k in x_dict:
                        minority_features = x_dict[k][minority_mask]
                        noise = torch.randn_like(minority_features) * noise_level
                        if k == 'E2_trajectory':
                            x_dict[k][minority_mask, :minority_features.shape[1], :] = x_dict[k][minority_mask,
                                                                                       :minority_features.shape[1],
                                                                                       :] + noise
                        else:
                            x_dict[k][minority_mask] = x_dict[k][minority_mask] + noise

                logits = classifier_model(x_dict)
                loss = loss_fn(logits, y)
                loss = loss / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        classifier_model.eval()
        all_labels_fold, all_scores_fold, total_val_correct = [], [], 0
        with torch.no_grad():
            for x_dict, y in val_loader:
                for k in x_dict: x_dict[k] = x_dict[k].to(device, dtype=torch.float32)
                y = y.to(device)
                logits = classifier_model(x_dict)
                probs = torch.softmax(logits, dim=1)
                scores = probs[:, 1]
                all_labels_fold.extend(y.cpu().numpy())
                all_scores_fold.extend(scores.cpu().numpy())
                preds = logits.argmax(dim=1)
                total_val_correct += (preds == y).sum().item()
        fold_accuracy = total_val_correct / len(val_subset)
        try:
            fold_auroc = roc_auc_score(all_labels_fold, all_scores_fold)
        except ValueError:
            fold_auroc = 0.5
        fold_acc_results.append(fold_accuracy);
        fold_auroc_results.append(fold_auroc)
        print(f"Fold {fold + 1} Validation -> Accuracy: {fold_accuracy:.4f}, AUROC: {fold_auroc:.4f}")

    print(f"\n{'=' * 20} Cross-Validation Summary {'=' * 20}")
    print(f"Average Accuracy: {np.mean(fold_acc_results):.4f} ± {np.std(fold_acc_results):.4f}")
    print(f"Average AUROC: {np.mean(fold_auroc_results):.4f} ± {np.std(fold_auroc_results):.4f}")


if __name__ == "__main__":
    train()