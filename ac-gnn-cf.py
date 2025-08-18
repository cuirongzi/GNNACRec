# -*- coding: utf-8 -*-
"""
Attribute-Completion GNN for Collaborative Filtering (AC-GNN-CF)
=================================================================
Implements:
- Attribute completion via attention over item neighbors
  (feature-similarity & random-walk structural neighbors)
- LightGCN-style propagation on the user–item bipartite graph
- BPR loss optimization
- Recall@K and NDCG@K evaluation

Supported datasets:
- Douban Book  : ratings.csv (userId,bookId,rating?) ; books.csv (author,publisher,tags optional)
- FilmTrust    : ratings.csv (userId,movieId,rating?) ; movies.csv (genres optional)
- Last.fm      : user_artists.csv (userId,artistId,weight) ; artist_tags.csv (optional)
- Yelp         : yelp_reviews.csv (user_id,business_id,stars) ; yelp_businesses.csv (business_id,categories)
                 (Also supports Yelp JSON lines: review.json, business.json)

Usage examples
--------------
python ac_gnn_cf.py --dataset yelp --data_dir /path/to/yelp --epochs 10 --dim 64 --layers 2 --topk 50
python ac_gnn_cf.py --dataset filmtrust --data_dir /path/to/filmtrust
python ac_gnn_cf.py --dataset douban --data_dir /path/to/douban
python ac_gnn_cf.py --dataset lastfm --data_dir /path/to/lastfm
"""

import os
import argparse
from typing import Dict, Tuple, List, Optional
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_sparse_adj(adj: sparse.spmatrix) -> sparse.spmatrix:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def train_test_split_interactions(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """User-wise holdout split to preserve per-user distributions."""
    train_rows, test_rows = [], []
    for _, grp in interactions.groupby(user_col):
        grp = grp.sample(frac=1.0, random_state=42)
        n_train = int(len(grp) * train_ratio)
        train_rows.append(grp.iloc[:n_train])
        test_rows.append(grp.iloc[n_train:])
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def recall_at_k(ranked_items: List[int], ground_truth: set, k: int) -> float:
    topk = ranked_items[:k]
    hit = sum(1 for it in topk if it in ground_truth)
    return hit / max(1, len(ground_truth))


def ndcg_at_k(ranked_items: List[int], ground_truth: List[int], k: int) -> float:
    rel = [1.0 if it in ground_truth else 0.0 for it in ranked_items[:k]]
    dcg = 0.0
    for i, r in enumerate(rel, start=1):
        dcg += (2**r - 1) / np.log2(i + 1)
    ideal_rel = sorted(rel, reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal_rel, start=1):
        idcg += (2**r - 1) / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------
# Dataset Adapters
# ---------------------------

def _encode_ids(df: pd.DataFrame, user_col: str, item_col: str) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    u_enc, i_enc = LabelEncoder(), LabelEncoder()
    df["user_idx"] = u_enc.fit_transform(df[user_col].astype(str))
    df["item_idx"] = i_enc.fit_transform(df[item_col].astype(str))
    return df, u_enc, i_enc


def _multi_hot_from_list_strings(series: pd.Series, sep: str = "|") -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Convert 'a|b|c' style lists to bag-of-words via CountVectorizer.
    If your separator is comma, pass sep=',' (we normalize to space).
    """
    text = series.fillna("").astype(str).str.replace(sep, " ")
    cv = CountVectorizer(token_pattern=r"[^ ]+")
    X = cv.fit_transform(text.values)
    return X.tocsr(), list(cv.get_feature_names_out())


def load_douban_book(data_dir: str, rating_threshold: float = 0.0, max_items: Optional[int] = None):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    books_path = os.path.join(data_dir, "books.csv")
    books = pd.read_csv(books_path) if os.path.exists(books_path) else None

    if "rating" in ratings.columns:
        ratings = ratings[ratings["rating"] >= rating_threshold]
        ratings["implicit"] = 1
    else:
        ratings["implicit"] = 1

    df = ratings.rename(columns={"userId": "user", "bookId": "item"})
    if max_items is not None:
        keep_items = df["item"].drop_duplicates().sample(n=max_items, random_state=42)
        df = df[df["item"].isin(keep_items)]

    df, u_enc, i_enc = _encode_ids(df, "user", "item")

    n_items = df["item_idx"].nunique()
    if books is not None:
        books = books.rename(columns={"bookId": "item"})
        if max_items is not None:
            books = books[books["item"].isin(keep_items)]
        books_enc = books.copy()
        cols = [c for c in ["author", "publisher", "tags"] if c in books_enc.columns]
        if not cols:
            features = sparse.csr_matrix((n_items, 0))
            feat_names = []
        else:
            combo = books_enc[cols].fillna("").astype(str).agg(" ".join, axis=1)
            books_enc["item_idx"] = i_enc.transform(books_enc["item"].astype(str))
            text = pd.Series("", index=np.arange(n_items))
            for idx, s in zip(books_enc["item_idx"].values, combo.values):
                text.iloc[idx] = s
            cv = CountVectorizer(token_pattern=r"[^ ]+")
            X = cv.fit_transform(text.values)
            features = X.tocsr()
            feat_names = list(cv.get_feature_names_out())
    else:
        features = sparse.csr_matrix((n_items, 0))
        feat_names = []

    return df, features, feat_names, u_enc, i_enc, "user", "item"


def load_filmtrust(data_dir: str, rating_threshold: float = 0.0, max_items: Optional[int] = None):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies_path = os.path.join(data_dir, "movies.csv")
    movies = pd.read_csv(movies_path) if os.path.exists(movies_path) else None

    if "rating" in ratings.columns:
        ratings = ratings[ratings["rating"] >= rating_threshold]
        ratings["implicit"] = 1
    else:
        ratings["implicit"] = 1

    df = ratings.rename(columns={"userId": "user", "movieId": "item"})
    if max_items is not None:
        keep_items = df["item"].drop_duplicates().sample(n=max_items, random_state=42)
        df = df[df["item"].isin(keep_items)]
    df, u_enc, i_enc = _encode_ids(df, "user", "item")

    n_items = df["item_idx"].nunique()
    if movies is not None and "genres" in movies.columns:
        movies = movies.rename(columns={"movieId": "item"})
        if max_items is not None:
            movies = movies[movies["item"].isin(keep_items)]
        movies["item_idx"] = i_enc.transform(movies["item"].astype(str))
        X, feat_names = _multi_hot_from_list_strings(movies.set_index("item_idx")["genres"])
        features = sparse.csr_matrix((n_items, X.shape[1]))
        common_idx = movies["item_idx"].values
        features[common_idx] = X[movies.index]
    else:
        features = sparse.csr_matrix((n_items, 0))
        feat_names = []

    return df, features, feat_names, u_enc, i_enc, "user", "item"


def load_lastfm(data_dir: str, max_items: Optional[int] = None):
    ua = pd.read_csv(os.path.join(data_dir, "user_artists.csv"))
    ua = ua.rename(columns={"userId": "user", "artistId": "item"})
    ua["implicit"] = 1

    if max_items is not None:
        keep_items = ua["item"].drop_duplicates().sample(n=max_items, random_state=42)
        ua = ua[ua["item"].isin(keep_items)]
    df, u_enc, i_enc = _encode_ids(ua, "user", "item")

    tags_path = os.path.join(data_dir, "artist_tags.csv")
    n_items = df["item_idx"].nunique()
    if os.path.exists(tags_path):
        tags = pd.read_csv(tags_path).rename(columns={"artistId": "item"})
        if max_items is not None:
            tags = tags[tags["item"].isin(keep_items)]
        tags["item_idx"] = i_enc.transform(tags["item"].astype(str))
        tag_str = tags.groupby("item_idx")["tag"].apply(lambda s: " ".join(map(str, s))).reindex(np.arange(n_items)).fillna("")
        cv = CountVectorizer(token_pattern=r"[^ ]+")
        X = cv.fit_transform(tag_str.values)
        features = X.tocsr()
        feat_names = list(cv.get_feature_names_out())
    else:
        features = sparse.csr_matrix((n_items, 0))
        feat_names = []

    return df, features, feat_names, u_enc, i_enc, "user", "item"


def load_yelp(data_dir: str, rating_threshold: float = 4.0, max_items: Optional[int] = None):
    """
    Prefer CSVs:
      - yelp_reviews.csv    (user_id,business_id,stars)
      - yelp_businesses.csv (business_id,categories)
    Fallback (large): Yelp JSON lines review.json, business.json
    """
    reviews_csv = os.path.join(data_dir, "yelp_reviews.csv")
    business_csv = os.path.join(data_dir, "yelp_businesses.csv")
    reviews, businesses = None, None

    if os.path.exists(reviews_csv):
        reviews = pd.read_csv(reviews_csv)
    else:
        review_json = os.path.join(data_dir, "review.json")
        if os.path.exists(review_json):
            reviews = pd.read_json(review_json, lines=True)
        else:
            raise FileNotFoundError("Provide yelp_reviews.csv or review.json")

    if os.path.exists(business_csv):
        businesses = pd.read_csv(business_csv)
    else:
        business_json = os.path.join(data_dir, "business.json")
        if os.path.exists(business_json):
            businesses = pd.read_json(business_json, lines=True)
        else:
            businesses = None  # no features available

    # Normalize columns
    if "user_id" not in reviews.columns or "business_id" not in reviews.columns:
        for ucol in ["user", "userId", "uid"]:
            if ucol in reviews.columns:
                reviews.rename(columns={ucol: "user_id"}, inplace=True)
                break
        for icol in ["business", "item", "businessId", "bid"]:
            if icol in reviews.columns:
                reviews.rename(columns={icol: "business_id"}, inplace=True)
                break
    if "stars" not in reviews.columns:
        if "rating" in reviews.columns:
            reviews.rename(columns={"rating": "stars"}, inplace=True)
        else:
            reviews["stars"] = 5.0

    # Threshold to implicit
    reviews = reviews[reviews["stars"] >= rating_threshold].copy()
    reviews["implicit"] = 1

    df = reviews.rename(columns={"user_id": "user", "business_id": "item"})
    if max_items is not None:
        keep_items = df["item"].drop_duplicates().sample(n=max_items, random_state=42)
        df = df[df["item"].isin(keep_items)]

    df, u_enc, i_enc = _encode_ids(df, "user", "item")
    n_items = df["item_idx"].nunique()

    # Build features from categories if available
    if businesses is not None:
        if "business_id" not in businesses.columns:
            for icol in ["business", "item", "businessId", "bid"]:
                if icol in businesses.columns:
                    businesses.rename(columns={icol: "business_id"}, inplace=True)
                    break
        if "categories" not in businesses.columns:
            businesses["categories"] = ""

        # Align to encoded item indices
        try:
            businesses["item_idx"] = i_enc.transform(businesses["business_id"].astype(str))
        except Exception:
            common = set(businesses["business_id"].astype(str)).intersection(set(i_enc.classes_))
            sub = businesses[businesses["business_id"].astype(str).isin(common)].copy()
            sub["item_idx"] = i_enc.transform(sub["business_id"].astype(str))
            businesses = sub

        cat_series = businesses.set_index("item_idx")["categories"].fillna("").astype(str)
        cat_series = cat_series.str.replace(",", "|").str.replace(";", "|")
        X, feat_names = _multi_hot_from_list_strings(cat_series, sep="|")

        features = sparse.csr_matrix((n_items, X.shape[1]))
        common_idx = cat_series.index.values
        features[common_idx] = X[np.arange(len(common_idx))]
    else:
        features = sparse.csr_matrix((n_items, 0))
        feat_names = []

    return df, features, feat_names, u_enc, i_enc, "user", "item"


# ---------------------------
# Graph & Samplers
# ---------------------------

def build_bipartite_adj(n_users: int, n_items: int, ui_edges: np.ndarray) -> sparse.csr_matrix:
    rows = np.concatenate([ui_edges[:, 0], ui_edges[:, 1] + n_users])
    cols = np.concatenate([ui_edges[:, 1] + n_users, ui_edges[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_users + n_items, n_users + n_items))
    return adj


def build_interaction_matrix(n_users: int, n_items: int, ui_edges: np.ndarray) -> sparse.csr_matrix:
    data = np.ones(len(ui_edges), dtype=np.float32)
    R = sparse.csr_matrix((data, (ui_edges[:, 0], ui_edges[:, 1])), shape=(n_users, n_items))
    return R.tocsr()


def compute_item_neighbors(R: sparse.csr_matrix, item_features: sparse.csr_matrix, topk: int = 50,
                           alpha_feat: float = 0.5) -> List[np.ndarray]:
    """
    Compute top-k neighbors per item by combining:
    - co-occurrence similarity S_co = normalize(R^T R) row-wise
    - feature cosine similarity S_feat = cosine(item_features)
    S = alpha_feat * S_feat + (1-alpha_feat) * S_co
    """
    n_items = R.shape[1]
    RtR = (R.T @ R).tocsr()
    RtR.setdiag(0)
    row_sums = np.sqrt(RtR.power(2).sum(axis=1)).A.ravel()
    row_sums[row_sums == 0] = 1.0
    inv = sparse.diags(1.0 / row_sums)
    S_co = inv @ RtR

    if item_features.shape[1] > 0:
        norms = np.sqrt(item_features.power(2).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        Xn = item_features.multiply(1.0 / norms[:, None])
        S_feat = (Xn @ Xn.T).tocsr()
        S_feat.setdiag(0)
    else:
        S_feat = sparse.csr_matrix((n_items, n_items))

    S = S_feat.multiply(alpha_feat) + S_co.multiply(1 - alpha_feat)
    neighbors = []
    for i in range(n_items):
        row = S.getrow(i)
        if row.nnz == 0:
            neighbors.append(np.array([], dtype=np.int64))
            continue
        idx = row.indices
        vals = row.data
        if len(idx) > topk:
            top = np.argpartition(vals, -topk)[-topk:]
            top_idx = idx[top]
            order = np.argsort(vals[top])[::-1]
            neighbors.append(top_idx[order])
        else:
            order = np.argsort(vals)[::-1]
            neighbors.append(idx[order])
    return neighbors


def random_walk_item_neighbors(adj: sparse.spmatrix, n_users: int, walk_length: int = 4, num_walks: int = 10,
                               topk: int = 50) -> List[np.ndarray]:
    """
    Simple random walks starting from each item node to collect item co-occurrences through meta-paths.
    """
    n_total = adj.shape[0]
    n_items = n_total - n_users
    indptr, indices = adj.indptr, adj.indices

    def neighbors_of(v):
        return indices[indptr[v]:indptr[v+1]]

    item_neighbors = []
    for item in range(n_items):
        start = n_users + item
        visits = defaultdict(int)
        for _ in range(num_walks):
            v = start
            for _ in range(walk_length):
                neigh = neighbors_of(v)
                if len(neigh) == 0:
                    break
                v = int(np.random.choice(neigh))
                if v >= n_users:
                    j = v - n_users
                    if j != item:
                        visits[j] += 1
        if visits:
            js, cnts = zip(*visits.items())
            js = np.array(js)
            cnts = np.array(cnts, dtype=np.float32)
            if len(js) > topk:
                top = np.argpartition(cnts, -topk)[-topk:]
                top_js = js[top]
                order = np.argsort(cnts[top])[::-1]
                item_neighbors.append(top_js[order])
            else:
                order = np.argsort(cnts)[::-1]
                item_neighbors.append(js[order])
        else:
            item_neighbors.append(np.array([], dtype=np.int64))
    return item_neighbors


# ---------------------------
# Model Components
# ---------------------------

class AttributeCompletion(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        proj_in = max(1, in_dim)
        self.W_f = nn.Linear(proj_in, dim, bias=False)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)

    def forward(self, X_feat: torch.Tensor,
                neigh_feat_idx: List[np.ndarray],
                neigh_struct_idx: List[np.ndarray]) -> torch.Tensor:
        n_items = X_feat.shape[0]
        H0 = self.W_f(X_feat)

        device = H0.device
        dim = H0.shape[1]
        out = torch.zeros_like(H0)

        Q = self.W_q(H0)
        K = self.W_k(H0)
        V = self.W_v(H0)

        sqrt_d = np.sqrt(dim)

        for i in range(n_items):
            neigh = set()
            if len(neigh_feat_idx[i]) > 0:
                neigh.update(neigh_feat_idx[i].tolist())
            if len(neigh_struct_idx[i]) > 0:
                neigh.update(neigh_struct_idx[i].tolist())
            if not neigh:
                out[i] = H0[i]
                continue
            neigh = list(neigh)
            neigh_t = torch.tensor(neigh, dtype=torch.long, device=device)

            q = Q[i].unsqueeze(0)
            k = K[neigh_t]
            v = V[neigh_t]

            att = torch.softmax((q @ k.t()) / sqrt_d, dim=1)
            agg = att @ v
            out[i] = H0[i] + agg.squeeze(0)

        return self.W_out(out)


class LightGCNProp(nn.Module):
    def __init__(self, norm_adj: sparse.spmatrix, device: torch.device):
        super().__init__()
        coo = norm_adj.tocoo()
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        self.A = torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce().to(device)

    def propagate(self, E0: torch.Tensor, n_layers: int) -> torch.Tensor:
        all_layers = [E0]
        x = E0
        for _ in range(n_layers):
            x = torch.sparse.mm(self.A, x)
            all_layers.append(x)
        return torch.stack(all_layers, dim=0).mean(dim=0)


class ACGNNCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, norm_adj: sparse.spmatrix,
                 X_item_feat: sparse.spmatrix,
                 neigh_feat_idx: List[np.ndarray],
                 neigh_struct_idx: List[np.ndarray],
                 device: torch.device):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.dim = dim
        self.device = device

        self.user_emb = nn.Embedding(n_users, dim)
        in_dim = X_item_feat.shape[1] if X_item_feat is not None else 0
        self.ac = AttributeCompletion(in_dim=in_dim, dim=dim)

        if in_dim == 0:
            X = torch.zeros((n_items, 1), dtype=torch.float32, device=device)
        else:
            X = torch.tensor(X_item_feat.todense(), dtype=torch.float32, device=device)
        self.register_buffer("X_item", X)

        self.neigh_feat_idx = neigh_feat_idx
        self.neigh_struct_idx = neigh_struct_idx

        self.prop = LightGCNProp(norm_adj, device=device)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)

    def forward(self, n_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item_init = self.ac(self.X_item, self.neigh_feat_idx, self.neigh_struct_idx)
        E0 = torch.cat([self.user_emb.weight, item_init], dim=0)
        E = self.prop.propagate(E0, n_layers=n_layers)
        Eu = E[:self.n_users]
        Ei = E[self.n_users:]
        return Eu, Ei

    def predict(self, users: torch.Tensor, items: torch.Tensor, n_layers: int) -> torch.Tensor:
        Eu, Ei = self.forward(n_layers=n_layers)
        return (Eu[users] * Ei[items]).sum(dim=1)


# ---------------------------
# Training & Evaluation
# ---------------------------

class BPRSampler(Dataset):
    def __init__(self, R: sparse.csr_matrix):
        super().__init__()
        self.R = R
        self.user_pos = [set(R.getrow(u).indices.tolist()) for u in range(R.shape[0])]
        self.n_users, self.n_items = R.shape

        self.user_list = []
        self.pos_item_list = []
        for u in range(self.n_users):
            for i in self.user_pos[u]:
                self.user_list.append(u)
                self.pos_item_list.append(i)

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        u = self.user_list[idx]
        i = self.pos_item_list[idx]
        while True:
            j = np.random.randint(0, self.n_items)
            if j not in self.user_pos[u]:
                break
        return u, i, j


def evaluate(model: ACGNNCF, R_train: sparse.csr_matrix, test_df: pd.DataFrame,
             user_col: str, item_col: str, K_list: List[int], n_layers: int) -> Dict[int, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        Eu, Ei = model.forward(n_layers=n_layers)
        R_train_csr = R_train.tocsr()

        results = {K: {"recall": [], "ndcg": []} for K in K_list}

        gt = defaultdict(set)
        for _, row in test_df.iterrows():
            gt[int(row["user_idx"])].add(int(row["item_idx"]))

        n_users = R_train.shape[0]

        for u in range(n_users):
            train_items = set(R_train_csr.getrow(u).indices.tolist())
            if u not in gt or len(gt[u]) == 0:
                continue
            scores = (Eu[u].unsqueeze(0) @ Ei.t()).squeeze(0)
            scores_np = scores.cpu().numpy()
            scores_np[list(train_items)] = -np.inf
            ranked = np.argsort(scores_np)[::-1].tolist()

            for K in K_list:
                r = recall_at_k(ranked, gt[u], K)
                n = ndcg_at_k(ranked, list(gt[u]), K)
                results[K]["recall"].append(r)
                results[K]["ndcg"].append(n)

        for K in K_list:
            results[K]["recall"] = float(np.mean(results[K]["recall"])) if results[K]["recall"] else 0.0
            results[K]["ndcg"] = float(np.mean(results[K]["ndcg"])) if results[K]["ndcg"] else 0.0

    return results


def train_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_users: int,
    n_items: int,
    features: sparse.csr_matrix,
    user_col: str,
    item_col: str,
    layers: int = 2,
    dim: int = 64,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    reg: float = 1e-4,
    topk_neighbors: int = 50,
    alpha_feat: float = 0.5,
    device: Optional[torch.device] = None,
    eval_K: List[int] = [10, 20]
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ui_edges = df_train[["user_idx", "item_idx"]].values.astype(np.int64)
    adj = build_bipartite_adj(n_users, n_items, ui_edges)
    norm_adj = normalize_sparse_adj(adj)
    R_train = build_interaction_matrix(n_users, n_items, ui_edges)

    item_neighbors_feat = compute_item_neighbors(R_train, features, topk=topk_neighbors, alpha_feat=alpha_feat)
    item_neighbors_struct = random_walk_item_neighbors(adj, n_users, walk_length=4, num_walks=10, topk=topk_neighbors)

    model = ACGNNCF(
        n_users=n_users,
        n_items=n_items,
        dim=dim,
        norm_adj=norm_adj,
        X_item_feat=features,
        neigh_feat_idx=item_neighbors_feat,
        neigh_struct_idx=item_neighbors_struct,
        device=device
    ).to(device)

    sampler = BPRSampler(R_train)
    loader = DataLoader(sampler, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for u, i, j in loader:
            u = u.to(device)
            i = i.to(device)
            j = j.to(device)

            Eu, Ei = model.forward(n_layers=layers)
            xuij = (Eu[u] * (Ei[i] - Ei[j])).sum(dim=1)
            loss_bpr = -F.logsigmoid(xuij).mean()

            l2 = 0.0
            for p in model.parameters():
                l2 = l2 + p.pow(2).sum()
            loss = loss_bpr + reg * l2

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        metrics = evaluate(model, R_train, df_test, user_col, item_col, K_list=eval_K, n_layers=layers)
        msg = f"Epoch {epoch:03d} | Loss {total_loss/len(loader):.4f} | "
        msg += " | ".join([f"K={K}: Recall={metrics[K]['recall']:.4f} NDCG={metrics[K]['ndcg']:.4f}" for K in eval_K])
        print(msg)

    return model


def main():
    parser = argparse.ArgumentParser(description="Attribute Completion GNN-CF")
    parser.add_argument("--dataset", type=str, required=True, choices=["douban", "filmtrust", "lastfm", "yelp"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=50, help="neighbors per item for completion")
    parser.add_argument("--alpha_feat", type=float, default=0.5, help="blend of feature vs co-occurrence similarity")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rating_threshold", type=float, default=4.0, help="for explicit datasets")
    parser.add_argument("--max_items", type=int, default=None, help="limit items for quick runs (e.g., 20000 for Yelp)")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.dataset == "douban":
        df, features, feat_names, u_enc, i_enc, user_col, item_col = load_douban_book(args.data_dir, rating_threshold=0.0, max_items=args.max_items)
    elif args.dataset == "filmtrust":
        df, features, feat_names, u_enc, i_enc, user_col, item_col = load_filmtrust(args.data_dir, rating_threshold=0.0, max_items=args.max_items)
    elif args.dataset == "lastfm":
        df, features, feat_names, u_enc, i_enc, user_col, item_col = load_lastfm(args.data_dir, max_items=args.max_items)
    elif args.dataset == "yelp":
        df, features, feat_names, u_enc, i_enc, user_col, item_col = load_yelp(args.data_dir, rating_threshold=args.rating_threshold, max_items=args.max_items)
    else:
        raise ValueError("Unsupported dataset")

    train_df, test_df = train_test_split_interactions(df, "user_idx", "item_idx", train_ratio=args.train_ratio)

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    print(f"Dataset: {args.dataset} | Users={n_users} Items={n_items} Interactions={len(df)} Features={features.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = train_model(
        df_train=train_df,
        df_test=test_df,
        n_users=n_users,
        n_items=n_items,
        features=features,
        user_col=user_col,
        item_col=item_col,
        layers=args.layers,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        topk_neighbors=args.topk,
        alpha_feat=args.alpha_feat,
        device=device,
        eval_K=[10, 20]
    )


if __name__ == "__main__":
    main()
