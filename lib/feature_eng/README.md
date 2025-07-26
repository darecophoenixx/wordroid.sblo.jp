jp [wiki page](https://github.com/darecophoenixx/wordroid.sblo.jp/wiki/WordAndDoc2vec)

# WordAndDoc2vec

**Efficient and Interpretable Large-Scale Data Analysis for Marketing**

---

*Watch how data intelligently organizes itself from a random state into meaningful clusters.*
![Training Process GIF](https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/images/weights_evolution005.gif?raw=true)

## Overview

WordAndDoc2vec is a novel embedding learning method designed to efficiently extract essential **nonlinear structures** from **large-scale, sparse data**, such as customer purchase histories and website browsing logs, in an interpretable manner.

This technique aims to deepen customer understanding and analyze market structures by intuitively visualizing and analyzing hidden insights within the data.

### ✨ Key Features

- **🚀 Exceptional Computational Efficiency and Scalability**: Avoids complex network architectures, enabling the model to complete training on datasets with millions of entries in a practical timeframe.
- **🧠 High Interpretability and Intuitive Understanding**: The distance between learned vectors directly corresponds to their similarity, making the model's behavior easy to understand.
- **📊 Excellent Affinity for Clustering**: Learned vectors naturally form **spherical clusters**, making it easy to apply standard algorithms like K-means for high-precision customer segmentation.
- **🎯 Unique Negative Sampling Strategy**: By introducing a unique repulsive force that pushes "rows away from other rows" and "columns away from other columns," this method prevents overfitting and promotes the formation of clear, beautiful cluster structures.

## Problem Statement

Modern marketing data is both massive and sparse. Extracting nonlinear structures—such as true customer needs and complex product relationships—is crucial. However, existing methods face several challenges:

- **High Computational Cost**: Deep learning models often require extensive time and resources.
- **Lack of Interpretability**: "Black box" models are difficult to translate into actionable business decisions.
- **Barriers to Further Analysis**: The resulting feature vectors are often not well-suited for subsequent analyses like clustering.

WordAndDoc2vec is designed to overcome these challenges.

## How It Works

The core of this method lies in a learning mechanism that balances **"attraction"** and **"repulsion"**.

1.  **Attraction (Positive Sampling)**: Based on actual data (e.g., customer A bought product X), the vectors for the related row (customer A) and column (product X) are **pulled closer** together in the vector space.
2.  **Repulsion (Negative Sampling)**: In a unique approach, randomly selected pairs of row vectors (e.g., customer A and customer B) and column vectors (e.g., product X and product Y) are **pushed apart**.

This simple dynamic allows the vectors to form meaningful structures (clusters) autonomously without overfitting. The method uses an **RBF kernel** based on Euclidean distance, rather than dot products, to calculate similarity, which creates its high affinity for clustering.

## 🛠️ Installation

```bash
pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp
```

## 🚀 Quick Start

You can see the [Kaggle notebook](https://www.kaggle.com/code/wordroid/sample023-k-2-gif) that created the GIF above.

```python
from feature_eng.neg_smpl25 import WordAndDoc2vec, calc_gsim
from scipy.sparse import csr_matrix

# 1. Prepare your sparse matrix data (e.g., a user-item matrix)
data = ...
sparse_matrix = csr_matrix(data)

# 2. Create a WordAndDoc2vec instance
wd2v = WordAndDoc2vec(wtsmart_csr_prob=wtsmart_col_csc_T_csr_prob, word_dic=word_dic, doc_dic=doc_dic,
                      idfs=None)

# 3. Initialize the model
num_features = 2
loss_wgt_neg = 0.05
models = wd2v.make_model(num_features=num_features,
                         num_neg=3, stack_size=100,
                         embeddings_val=0.1,
                         loss_wgt_neg=loss_wgt_neg)

# 4. Train the model
wd2v.train(epochs=64, verbose=1,
           batch_size=128,
           use_multiprocessing=True,
           workers=8)

# 5. Get the learned feature vectors
row_vectors = wd2v.wgt_row  # Vectors for rows (e.g., users)
col_vectors = wd2v.wgt_col # Vectors for columns (e.g., items)

# Now you can use these vectors for visualization (with PCA/t-SNE)
# or for clustering (with K-means).
```

## 📚 Examples & Tutorials

For more practical examples, please refer to the following Jupyter Notebooks:

- **[WordAndDoc2vec: Efficient and Interpretable Large-Scale Data Analysis for Marketing](https://www.kaggle.com/code/wordroid/wordanddoc2vec-e9v3lz)**: A detailed explanation of the method.
- **[Topic Analysis and Clustering on Reuters News Data (Under Construction)](examples/02_reuters_topic_modeling.ipynb)**: A walkthrough of extracting and interpreting topic structures from real text data.

## 🆚 Comparison with Other Methods

WordAndDoc2vec holds a distinct advantage over traditional methods like SVD and complex deep learning models, especially in terms of **computational cost, interpretability, and affinity for clustering**. For a detailed comparison, please see this [Kaggle notebook](https://www.kaggle.com/code/wordroid/wordanddoc2vec-gyh).
