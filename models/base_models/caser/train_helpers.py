import torch
import torch.optim as optim
import numpy as np
from utils import minibatch
from caser import Caser
from tqdm import tqdm, trange
from utils import shuffle

class Recommender:
    def __init__(self,
                 n_iter,
                 batch_size,
                 l2,
                 neg_samples,
                 learning_rate,
                 use_cuda,
                 model_args):

        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self.model_args = model_args

        self._net = None
        self._optimizer = None
        self._candidate = dict()
        self.test_sequence = None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users
        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users, self._num_items, self.model_args).to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(), weight_decay=self._l2, lr=self._learning_rate)

    def fit(self, train, test=None):
        if self._net is None:
            self._initialize(train)

        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        for epoch in trange(self._n_iter, desc="Training Epochs"):
            users_np, sequences_np, targets_np = shuffle(users_np, sequences_np, targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, self._neg_samples)

            users, sequences, targets, negatives = (
                torch.from_numpy(users_np).long().to(self._device),
                torch.from_numpy(sequences_np).long().to(self._device),
                torch.from_numpy(targets_np).long().to(self._device),
                torch.from_numpy(negatives_np).long().to(self._device)
            )

            self._net.train()
            epoch_loss = 0.0
            num_batches = len(users) // self._batch_size + int(len(users) % self._batch_size != 0)

            for minibatch_num, (u, s, t, n) in enumerate(
                tqdm(minibatch(users, sequences, targets, negatives, batch_size=self._batch_size),
                total=num_batches,
                desc=f"Batches (Epoch {epoch+1})")
            ):
                items = torch.cat([t, n], dim=1)
                scores = self._net(s, u, items)

                pos_scores, neg_scores = torch.split(scores, [t.size(1), n.size(1)], dim=1)

                labels = torch.cat([
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores)
                ], dim=1).view(-1)
                preds = torch.cat([pos_scores, neg_scores], dim=1).view(-1)

                loss = torch.nn.BCEWithLogitsLoss()(preds, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self._n_iter} | Avg Loss: {epoch_loss / (minibatch_num + 1):.4f}")

    def _generate_negative_samples(self, users, interactions, n):
        users_ = users.squeeze()
        neg_samples = np.zeros((users_.shape[0], n), dtype=np.int64)

        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1
            train_csr = interactions.tocsr()
            for u in range(interactions.num_users):
                self._candidate[u] = list(set(all_items) - set(train_csr[u].indices))

        for i, u in enumerate(users_):
            choices = self._candidate[u]
            neg_samples[i] = np.random.choice(choices, size=n)

        return neg_samples

    def predict(self, user_id, item_ids=None):
        if self.test_sequence is None:
            raise ValueError("Test sequences not initialized")

        self._net.eval()
        with torch.no_grad():
            seq = self.test_sequence.sequences[user_id, :].reshape(1, -1)
            seq_tensor = torch.from_numpy(seq).long().to(self._device)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            item_tensor = torch.from_numpy(item_ids).long().to(self._device)
            user_tensor = torch.tensor([[user_id]]).long().to(self._device)

            scores = self._net(seq_tensor, user_tensor, item_tensor, for_pred=True)

        return scores.cpu().numpy().flatten()

    def save(self, path):
        torch.save(self._net.state_dict(), path)

    def load(self, path, num_users=None, num_items=None):
        if self._net is None:
            if num_users is None or num_items is None:
                raise ValueError("Model not initialized and num_users/items not provided.")
            self._net = Caser(num_users, num_items, self.model_args).to(self._device)
            self._optimizer = optim.Adam(self._net.parameters(), weight_decay=self._l2, lr=self._learning_rate)

        checkpoint = torch.load(path, map_location=self._device)
        model_dict = self._net.state_dict()

        checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}

        skipped_keys = [k for k in checkpoint if k not in checkpoint_dict]
        if skipped_keys:
            print(f"Skipped loading {len(skipped_keys)} parameter(s) due to shape mismatch:")
            for k in skipped_keys:
                print(f"   - {k}: {checkpoint[k].shape} → expected {model_dict.get(k, 'N/A')}")

        model_dict.update(checkpoint_dict)
        self._net.load_state_dict(model_dict)
        print(f"Loaded weights from {path} with partial matching")
