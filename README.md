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
