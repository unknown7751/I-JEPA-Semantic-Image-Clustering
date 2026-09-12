# I-JEPA Semantic Image Clustering

A controlled evaluation of pretrained I-JEPA visual embeddings for semantic image organization and unsupervised clustering.

## Overview

This project investigates whether embeddings produced by a **pretrained, frozen** I-JEPA vision encoder contain enough category-level structure to support unsupervised, label-free image organization. The evaluation is built around a manually curated, category-balanced dataset and a fully unsupervised pipeline: image embeddings are projected with UMAP and clustered with HDBSCAN, and the known category labels are used **only after clustering**, purely to score the result.

The project does not train or fine-tune any model. It evaluates the out-of-the-box representational quality of I-JEPA embeddings for retrieval and clustering.

## Experimental Pipeline

```text
Image
  |
I-JEPA (facebook/ijepa_vith14_1k)
  |
Patch-token representations
  |
Mean pooling
  |
1280-D embedding
  |
L2 normalization
  |
 ┌──────────────┴──────────────┐
 |                              |
Retrieval sanity check      UMAP → HDBSCAN
(cosine similarity)         (unsupervised clustering)
                                  |
                          ARI / NMI / purity
                          (evaluated post-hoc against
                           known category labels)
```

## Dataset

The experiment uses a manually curated, category-balanced dataset of **480 images across 8 categories, with 60 images per category**:

| Category  | Images |
|-----------|--------|
| mountains | 60 |
| boats     | 60 |
| flowers   | 60 |
| cars      | 60 |
| shark     | 60 |
| snake     | 60 |
| cat       | 60 |
| dog       | 60 |
| **Total** | **480** |

Category labels come from the folder structure and are used **only for post-hoc evaluation**. They are never provided to I-JEPA, UMAP, HDBSCAN, or the retrieval system.

All 480 images passed dataset validation (existence, decodability, and expected per-category counts), so the full dataset was used with no images excluded.

## Model

The notebook uses the pretrained checkpoint:

```text
facebook/ijepa_vith14_1k
```

I-JEPA produces patch-level visual representations for each image. The embedding pipeline is:

1. Run the frozen I-JEPA encoder on each image to obtain patch-token representations.
2. Mean-pool the patch tokens into a single fixed-size vector per image (hidden dimension 1280).
3. L2-normalize each embedding so that cosine similarity between images reduces to a dot product.

For the 480-image dataset, this produces an embedding matrix of shape `(480, 1280)`, which was verified against the expected shape and checked for `NaN`/`Inf` values before use.

## Methodology

1. **Dataset discovery and validation** — categories and images are discovered from the folder structure (not a hard-coded list), and every image is checked for existence, decodability, extension, and expected per-category counts before inference.
2. **Image preprocessing** — images are loaded, converted to RGB, and passed through the model's associated processor.
3. **I-JEPA inference** — the frozen encoder is run in evaluation mode (`torch.inference_mode()`) to produce patch-token representations for every image.
4. **Patch-token mean pooling** — patch tokens are averaged into a single 1280-D vector per image.
5. **L2 normalization** — each embedding is scaled to unit length.
6. **Similarity retrieval** — a nearest-neighbor sanity check is run using cosine similarity on the normalized embeddings.
7. **UMAP dimensionality reduction** — normalized embeddings are projected into a lower-dimensional space (a 2D projection for visualization, a 10D projection for clustering), using cosine distance.
8. **HDBSCAN clustering** — density-based clustering is applied to the UMAP projection, with no constraint on the number of clusters and no access to category labels.
9. **ARI / NMI / purity evaluation** — discovered clusters are compared against the known category labels after clustering.
10. **Contingency analysis** — a cluster × category contingency table and heatmap show which categories are recovered cleanly, merged, or split.
11. **Parameter sensitivity analysis** — a grid sweep over UMAP `n_neighbors` and HDBSCAN `min_cluster_size` / `min_samples` measures how much clustering quality depends on configuration.
12. **Silhouette analysis** — a silhouette score is computed directly on the raw normalized embeddings using the true labels and cosine distance, independent of UMAP/HDBSCAN.

## Retrieval Evaluation

Before clustering, a simple image-to-image retrieval sanity check was run: one query image was sampled from each of the 8 categories, and its top-5 nearest neighbors (by cosine similarity on the L2-normalized embeddings) were retrieved.

| Query category | Top-5 same-category rate |
|-----------------|---------------------------|
| boats     | 100% |
| cars      | 100% |
| cat       | 100% |
| dog       | 100% |
| flowers   | 100% |
| mountains | 100% |
| shark     | 100% |
| snake     | 100% |

All 8 tested queries retrieved same-category images among their top-5 nearest neighbors. This is a **sanity check on 8 sampled queries**, not an exhaustive retrieval benchmark, and should not be read as a claim of perfect general-purpose retrieval performance across the full dataset.

## Unsupervised Clustering

I-JEPA embeddings are clustered without ever exposing the category labels to the clustering pipeline:

```text
I-JEPA embeddings (L2-normalized)
  → UMAP (cosine metric, 10D projection for clustering)
  → HDBSCAN (density-based, no fixed cluster count)
```

HDBSCAN was **not constrained to discover exactly 8 clusters**. It was free to find any number of clusters, or to leave points unassigned as noise (`-1`). The known category labels were used only afterward, to evaluate how well the discovered clusters align with the manually curated categories — they played no role in producing the embeddings, the UMAP projection, or the HDBSCAN assignment itself.

### Baseline clustering result

Baseline configuration: UMAP `n_neighbors=15`, HDBSCAN `min_cluster_size=10`, `min_samples=5`.

| Metric | Value |
|--------|-------|
| Discovered clusters | 9 (vs. 8 known categories) |
| Noise points | 1.5% |
| Adjusted Rand Index (ARI) | 0.7118 |
| Normalized Mutual Information (NMI) | 0.8239 |
| Mean cluster purity | 0.914 |

A cluster × category contingency table (and corresponding heatmap) in the notebook shows, per discovered cluster, which known category dominates it — useful for seeing which categories were recovered cleanly versus merged or split.

## Results

| Property | Value |
|---|---|
| Dataset size | 480 images (8 categories × 60 images) |
| Embedding dimension | 1280 (mean-pooled, L2-normalized) |
| Retrieval sanity check | 8/8 tested queries — 100% same-category in top-5 |
| Baseline clustering (ARI / NMI / purity) | 0.7118 / 0.8239 / 0.914 |
| Best tested clustering ARI (parameter sweep) | 0.7656 |
| Silhouette score (raw embeddings, true labels, cosine) | 0.1903 |

## Parameter Sensitivity

A grid sweep over UMAP `n_neighbors` (10, 15, 30) and HDBSCAN `min_cluster_size` (5, 10, 15, 25) × `min_samples` (3, 5, 10) was run to measure how sensitive the clustering result is to configuration. The baseline configuration was kept fixed as the primary reference point and was **not replaced** just because another configuration scored higher.

| Configuration | ARI |
|---|---|
| Baseline (`n_neighbors=15`, `min_cluster_size=10`, `min_samples=5`) | 0.7118 |
| Best tested (`n_neighbors=15`, `min_cluster_size=10`, `min_samples=3`) | 0.7656 |
| **Improvement** | **+0.0538** |

This best-tested configuration is the strongest result found **within the specific parameter grid searched in the notebook** — it is not presented as a universally optimal configuration, and the sweep is intended to characterize sensitivity rather than to cherry-pick a maximal score.

## Silhouette Analysis

The silhouette score was computed directly on the raw, L2-normalized I-JEPA embeddings using the true category labels and cosine distance — independent of the UMAP projection and HDBSCAN assignment.

**Silhouette score: 0.1903**

On a [-1, 1] scale, this is a modest value. It indicates that the 8 categories are not tightly or cleanly separated in the raw embedding space, even though the UMAP + HDBSCAN pipeline was able to recover substantially more structure downstream. This score should not be read as evidence of strong class separation; it is a useful counterweight to the stronger clustering metrics above, obtained through the nonlinear UMAP projection.

## Observations

- I-JEPA embeddings preserve useful local visual similarity: nearest-neighbor retrieval consistently returned same-category images for every tested query.
- The retrieval results indicate meaningful local structure in the embedding space, though this was checked on a small, sampled set of queries.
- UMAP + HDBSCAN can recover substantial category-level structure from I-JEPA embeddings without ever being given the category labels.
- Clustering quality is sensitive to UMAP and HDBSCAN parameters — the tested grid produced a meaningful but not dramatic ARI improvement over the baseline.
- The raw embedding space is not perfectly separated: the modest silhouette score (0.1903) shows category overlap exists even where downstream clustering metrics are strong.
- This controlled, visually distinct, category-balanced dataset is easier to cluster than an arbitrary real-world personal photo collection would likely be.

## Limitations

- **Small dataset**: 480 images across 8 categories is sufficient for a controlled pipeline evaluation, but far smaller than a typical real-world photo library.
- **Visually distinct categories**: the 8 categories (mountains, boats, flowers, cars, shark, snake, cat, dog) were chosen to be visually dissimilar from one another, which likely makes clustering easier than on more ambiguous or overlapping real-world content.
- **Manually curated category structure**: labels reflect a single manual labeling scheme and may contain borderline examples; this structure does not necessarily match how a real photo library would need to be organized.
- **Single checkpoint evaluated**: only `facebook/ijepa_vith14_1k` was tested; results may not generalize to other I-JEPA checkpoints or model scales.
- **Dependence on UMAP/HDBSCAN**: results are specific to this dimensionality-reduction and clustering combination, not to I-JEPA embeddings in isolation.
- **Parameter sensitivity**: clustering quality varies meaningfully with UMAP/HDBSCAN configuration, so a single baseline result should not be treated as a fixed, absolute measure of representation quality.
- **Controlled dataset ≠ arbitrary personal photo library**: strong performance on this controlled experiment does **not** demonstrate that arbitrary personal photo collections — with overlapping scenes, multiple objects per photo, and continuous visual variation — will cluster equally well.

## Reproducibility

### Environment

The notebook relies on the following third-party packages (installed automatically by the notebook's setup cell if missing): `pandas`, `Pillow`, `torch`, `transformers`, `numpy`, `umap-learn`, `hdbscan`, `scikit-learn`, `matplotlib`, `tqdm`, `seaborn`.

The notebook does not pin specific package versions; a `requirements.txt` is included in the repository for convenience, but exact versions should be filled in based on your working environment, as the notebook itself does not fix them.

A CUDA-capable GPU was used when the notebook was executed (device selection falls back to CPU automatically via `torch.cuda.is_available()`), but no specific hardware is required to run the pipeline.

### Dataset layout

Place the controlled dataset under `controlled_dataset/`, with one subfolder per category:

```text
controlled_dataset/
├── mountains/
├── boats/
├── flowers/
├── cars/
├── shark/
├── snake/
├── cat/
└── dog/
```

The notebook expects **8 categories × 60 images = 480 images total** and validates this automatically, reporting any missing, unreadable, or miscounted categories before proceeding.

### Running the notebook

1. Install the required dependencies (see Environment above).
2. Populate `controlled_dataset/` as described above.
3. Open and run `notebooks/ijepa_semantic_image_clustering.ipynb` top to bottom.

### Execution flow

Dataset discovery and validation → I-JEPA embedding extraction (1280-D per image, expected shape `(480, 1280)`) → L2 normalization → retrieval sanity check → UMAP projection → HDBSCAN clustering → ARI/NMI/purity evaluation → contingency analysis → parameter sweep → silhouette analysis → results summary.

Embeddings are cached locally to disk (keyed by a hash of the dataset's image IDs) so that re-running the notebook does not require recomputing I-JEPA inference unless the dataset changes.

## Dataset Sources

The 480-image controlled dataset is a **manually curated subset** assembled from images sourced from the following public datasets. These source datasets do not themselves contain this project's specific 8-category, 60-images-per-category structure — that structure was created by manually selecting and organizing a subset of images from these sources.

1. Hugging Face — Car Images
   https://huggingface.co/datasets/HumynLabs/car-images/tree/main
2. Kaggle — Mountains and Beaches Dataset
   https://www.kaggle.com/datasets/erennik/mountains-and-beaches-dataset
3. Kaggle — Dockship Boat Type Classification
   https://www.kaggle.com/datasets/imsparsh/dockship-boat-type-classification?select=Train
4. Kaggle — Flowers Dataset
   https://www.kaggle.com/datasets/imsparsh/flowers-dataset
5. Kaggle — Animal Image Dataset (90 Different Animals)
   https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

Refer to each original dataset page for its current license and attribution requirements before reuse.

## Repository Structure

```text
semantic-photo-intelligence/
├── controlled_dataset/
│   ├── boats/
│   ├── cars/
│   ├── cat/
│   ├── dog/
│   ├── flowers/
│   ├── mountains/
│   ├── shark/
│   └── snake/
├── notebooks/
│   └── ijepa_semantic_image_clustering.ipynb
├── data/
│   └── metadata/
│       └── controlled_manual_manifest.csv
├── README.md
├── requirements.txt
└── .gitignore
```

- **`controlled_dataset/`** — the manually curated, category-labeled image set used for the experiment (one folder per category).
- **`notebooks/`** — the executable notebook containing the full pipeline: embedding extraction, retrieval, clustering, evaluation, and parameter sweep.
- **`data/metadata/`** — dataset manifests and evaluation artifacts (e.g., the parameter sweep results). Cached embeddings and other intermediate files can be regenerated by re-running the notebook and are not required to be checked into the repository.
- **`requirements.txt`** — Python dependencies needed to run the notebook.

## Future Work

The following are potential future directions and are **not implemented** in this project:

- Evaluation on larger, real-world photo libraries
- Semantic image search
- Duplicate / near-duplicate detection
- Automatic album discovery
- Scalable FAISS-based similarity search
- FastAPI backend
- SQLite-based metadata storage
- Streamlit-based photo gallery interface

These represent possible architectural extensions beyond the current controlled evaluation, not existing functionality.

## Conclusion

Under controlled, category-balanced conditions, pretrained I-JEPA embeddings show useful local visual similarity (100% same-category retrieval on the 8 tested queries) and carry enough category-level structure for a fully unsupervised UMAP + HDBSCAN pipeline to recover it, achieving a baseline ARI of 0.7118, NMI of 0.8239, and mean cluster purity of 0.914 without ever seeing the category labels. A parameter sweep improved the best tested ARI to 0.7656, showing that clustering quality is sensitive to configuration, and a modest silhouette score (0.1903) on the raw embedding space indicates the underlying category separation is real but imperfect.

These results demonstrate that I-JEPA embeddings *can* carry recoverable class-level structure under favorable, controlled conditions. They do not establish that I-JEPA is broadly "good" for semantic photo organization in general, nor do they demonstrate that arbitrary, uncontrolled personal photo libraries — with overlapping scenes, multiple objects, and continuous visual variation — will cluster equally well. Evaluating this pipeline on less controlled, real-world data is left to future work.
