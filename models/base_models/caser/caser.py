import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False



class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # Conv layers
        self.conv_v = nn.Conv2d(1, self.n_v, kernel_size=(min(3, L), 1))
        self.conv_h = nn.ModuleList([
            nn.Conv2d(1, self.n_h, kernel_size=(h, dims), padding=(0, 0))
            for h in range(1, L + 1)
        ])

        # Dynamically determine FC input dim
        with torch.no_grad():
            dummy_seq = torch.zeros(1, L, dtype=torch.long)
            dummy_embs = self.item_embeddings(dummy_seq).unsqueeze(1)
            v_out = self.conv_v(dummy_embs).reshape(1, -1) if self.n_v else None

            h_outs = []
            if self.n_h:
                for conv in self.conv_h:
                    c = conv(dummy_embs).squeeze(3)
                    p = F.max_pool1d(c, c.size(2)).squeeze(2)
                    h_outs.append(p)
                h_out = torch.cat(h_outs, dim=1)
            else:
                h_out = None

            concat_out = torch.cat([x for x in [v_out, h_out] if x is not None], dim=1)
            fc1_input_dim = concat_out.size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_dim, dims)
        self.W2 = nn.Embedding(num_items, dims + dims)
        self.b2 = nn.Embedding(num_items, 1)

        # Dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # Weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / dims)
        self.item_embeddings.weight.data.normal_(0, 1.0 / dims)
        self.W2.weight.data.normal_(0, 1.0 / (dims + dims))
        self.b2.weight.data.zero_()

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        item_embs = self.item_embeddings(seq_var).unsqueeze(1).contiguous()  # [B, 1, L, d]
        user_emb = self.user_embeddings(user_var).squeeze(1 if user_var.dim() == 2 else 0)  # [B, d]

        # Conv layers
        out_v = self.conv_v(item_embs).reshape(seq_var.size(0), -1) if self.n_v else None

        out_hs = []
        if self.n_h:
            for conv in self.conv_h:
                c = self.ac_conv(conv(item_embs).squeeze(3))
                p = F.max_pool1d(c, c.size(2)).squeeze(2)
                out_hs.append(p)
            out_h = torch.cat(out_hs, dim=1)
        else:
            out_h = None

        # Combine and apply FC
        out = torch.cat([x for x in [out_v, out_h] if x is not None], dim=1)
        out = self.dropout(out)
        z = self.ac_fc(self.fc1(out))  # [B, d]
        x = torch.cat([z, user_emb], dim=1)  # [B, d + d]

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        # Evaluation: item_var is 2D [B, K] → w2: [B, K, d+d], x: [B, 1, d+d]
        if x.dim() == 2 and w2.dim() == 3:
            x = x.unsqueeze(1)                     # [B, 1, d+d]
            scores = (x * w2).sum(dim=2) + b2.squeeze(2)  # [B, K]
        else:
            scores = (x * w2).sum(dim=1) + b2.squeeze(1)  # [B]

        return scores
