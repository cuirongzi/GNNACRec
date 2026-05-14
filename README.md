# Attribute Completion GNN for Collaborative Filtering (AC-GNN-CF)

This repository provides a PyTorch implementation of **Attribute Completion–based Graph Neural Network for Collaborative Filtering**.  
The model addresses **data sparsity** and **missing item attributes** by combining **graph neural network propagation**, **attribute completion with attention**, and **random walk–based neighbor sampling**.

---

## 🔑 Key Features
- **Attribute Completion**: Missing item attributes are estimated using attention over feature similarity and random-walk neighbors.  
- **Graph Neural Network**: LightGCN-style propagation captures higher-order user–item interactions.  
- **Attention Mechanism**: Weighs the importance of different neighbors for robust representation learning.  
- **BPR Optimization**: Bayesian Personalized Ranking loss for implicit recommendation tasks.  
- **Evaluation Metrics**: Recall@K and NDCG@K.  

---

## 📊 Supported Datasets
The implementation currently supports four popular recommendation datasets:

1. **Douban Book**  
   - `ratings.csv` → (userId, bookId, rating)  
   - `books.csv` → (bookId, author, publisher, tags [optional])  

2. **FilmTrust**  
   - `ratings.csv` → (userId, movieId, rating)  
   - `movies.csv` → (movieId, title, genres [optional])  

3. **Last.fm**  
   - `user_artists.csv` → (userId, artistId, weight)  
   - `artist_tags.csv` → (artistId, tag [optional])  

4. **Yelp**  
   - `yelp_reviews.csv` → (user_id, business_id, stars)  
   - `yelp_businesses.csv` → (business_id, categories)  
   - *(Large JSON files `review.json` and `business.json` also supported)*  

---

## ⚙️ Installation
Clone this repository and install the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/ac-gnn-cf.git
cd ac-gnn-cf

pip install -r requirements.txt
```

---

## 📚 Usage

### Quick Start

```bash
# Train on Yelp dataset
python ac-gnn-cf.py --dataset yelp --data_dir /path/to/yelp --epochs 10 --dim 64 --layers 2 --topk 50

# Train on FilmTrust dataset
python ac-gnn-cf.py --dataset filmtrust --data_dir /path/to/filmtrust --epochs 20 --dim 128

# Train on Douban Book dataset
python ac-gnn-cf.py --dataset douban --data_dir /path/to/douban --epochs 15

# Train on Last.fm dataset
python ac-gnn-cf.py --dataset lastfm --data_dir /path/to/lastfm --epochs 10
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | - | Dataset choice: `douban`, `filmtrust`, `lastfm`, `yelp` |
| `--data_dir` | - | Path to dataset directory |
| `--epochs` | 10 | Number of training epochs |
| `--dim` | 64 | Embedding dimension |
| `--layers` | 2 | Number of GCN propagation layers |
| `--batch_size` | 2048 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--reg` | 1e-6 | L2 regularization coefficient |
| `--topk` | 50 | Top-k neighbors for attribute completion |
| `--alpha_feat` | 0.5 | Weight for feature-based vs co-occurrence similarity (0–1) |
| `--train_ratio` | 0.7 | Train/test split ratio |
| `--seed` | 42 | Random seed |
| `--rating_threshold` | 4.0 | Rating threshold for explicit feedback datasets |
| `--max_items` | None | Maximum items to load (e.g., 20000 for quick runs) |

---

## 🧠 Model Architecture

The AC-GNN-CF model consists of three main components:

### 1. Attribute Completion Module
- Estimates missing item attributes using multi-head attention
- Aggregates information from both feature-similarity and random-walk neighbors
- Leverages transformer-based self-attention mechanism

### 2. Graph Neural Network (LightGCN Propagation)
- Symmetric normalization of the user–item bipartite graph
- Multi-layer propagation to capture higher-order interactions
- Efficient sparse matrix operations on GPU

### 3. BPR Loss Training
- Bayesian Personalized Ranking for implicit feedback
- Differentiates positive and negative item pairs
- L2 regularization to prevent overfitting

---

## 📈 Results

On four benchmark datasets, AC-GNN-CF achieves:

- **NDCG@10**: Average improvement of **6.4%** over baselines
- **Recall@10**: Average improvement of **5.3%** over baselines

Detailed results across different sparsity levels available in the paper.

---

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@article{gnnacrec2024,
  title={Heterogeneous Graph Neural Network for Collaborative Filtering with Attribute Completion},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📧 Contact

For questions or inquiries, please reach out via GitHub issues.

---

## 🔗 References

- **LightGCN**: [He et al., 2020](https://arxiv.org/abs/2002.02126)
- **Graph Neural Networks for Recommendation**: [Survey](https://arxiv.org/abs/2004.11718)
- **Attention Mechanisms**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
