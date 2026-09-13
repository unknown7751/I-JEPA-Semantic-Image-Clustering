# Semantic Photo Intelligence Engine — Research Validation

A research and validation project investigating whether pretrained I-JEPA visual embeddings provide useful structure for similarity retrieval and unsupervised photo organization, as groundwork for a future local-first smart photo gallery.

## Project Purpose

**Core research question:**

> Do pretrained I-JEPA embeddings provide useful visual structure for similarity retrieval and unsupervised photo organization, and what kinds of visual structure — semantic, viewpoint, or otherwise — are actually represented in the embedding space?

**Long-term application:** the *Semantic Photo Intelligence Engine* — a local-first smart photo gallery that uses pretrained visual embeddings for similarity search and automatic visual organization.

**Current status:** this repository is a research and validation project, not a product. No smart gallery, backend, or user-facing application has been built yet. The experiments below evaluate whether the underlying representation (pretrained, frozen I-JEPA) is suitable groundwork for that application, and — just as importantly — what its structure implies about how such an application should be designed.

The project does not train or fine-tune any model at any stage. Every experiment evaluates the out-of-the-box representational quality of a frozen I-JEPA encoder.

## Research Story

The research validation proceeds in three stages, each building directly on the last:

```text
Experiment 1: Controlled Semantic Clustering
        ↓
Experiment 2: Viewpoint Sensitivity Analysis
        ↓
Experiment 2.1: Near-Duplicate-Controlled Validation
        ↓
Research Validation Complete
        ↓
Phase 1: Build Semantic Photo Intelligence Engine
```

1. **Experiment 1** asks whether I-JEPA embeddings carry recoverable category-level semantic structure, using a manually curated, category-balanced dataset.
2. **Experiment 2** asks a question raised directly by Experiment 1's own results (see below): whether some of that embedding-space structure is organized by *viewpoint* rather than, or in addition to, semantic category.
3. **Experiment 2.1** is a final, small validation experiment checking that Experiment 2's viewpoint result isn't primarily an artifact of near-duplicate images in the dataset.

Together, these three experiments constitute the research validation phase. Their combined result determines what the eventual Smart Gallery architecture needs to account for — see [Research Insight for the Future Smart Gallery](#research-insight-for-the-future-smart-gallery) below.

---

## Experiment 1 — Controlled Semantic Clustering

**Goal:** determine whether embeddings from a pretrained, frozen I-JEPA encoder contain enough category-level structure to support unsupervised, label-free image organization.

The evaluation is built around a manually curated, category-balanced dataset and a fully unsupervised pipeline: image embeddings are projected with UMAP and clustered with HDBSCAN, and the known category labels are used **only after clustering**, purely to score the result.

### Pipeline

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

### Dataset

A manually curated, category-balanced dataset of **480 images across 8 categories, with 60 images per category**:

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

### Model

```text
facebook/ijepa_vith14_1k
```

The embedding pipeline:

1. Run the frozen I-JEPA encoder on each image to obtain patch-token representations.
2. Mean-pool the patch tokens into a single fixed-size vector per image (hidden dimension 1280).
3. L2-normalize each embedding so that cosine similarity between images reduces to a dot product.

For the 480-image dataset, this produces an embedding matrix of shape `(480, 1280)`, verified against the expected shape and checked for `NaN`/`Inf` values before use.

This exact checkpoint, preprocessing, pooling, and normalization scheme is reused unchanged in Experiment 2 and Experiment 2.1.

### Methodology

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

### Retrieval Evaluation

One query image was sampled from each of the 8 categories, and its top-5 nearest neighbors (by cosine similarity on the L2-normalized embeddings) were retrieved.

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

### Unsupervised Clustering

```text
I-JEPA embeddings (L2-normalized)
  → UMAP (cosine metric, 10D projection for clustering)
  → HDBSCAN (density-based, no fixed cluster count)
```

HDBSCAN was **not constrained to discover exactly 8 clusters**. It was free to find any number of clusters, or to leave points unassigned as noise (`-1`). The known category labels were used only afterward, to evaluate how well the discovered clusters align with the manually curated categories.

**Baseline configuration:** UMAP `n_neighbors=15`, HDBSCAN `min_cluster_size=10`, `min_samples=5`.

| Metric | Value |
|--------|-------|
| Discovered clusters | 9 (vs. 8 known categories) |
| Noise points | 1.5% |
| Adjusted Rand Index (ARI) | 0.7118 |
| Normalized Mutual Information (NMI) | 0.8239 |
| Mean cluster purity | 0.914 |

A cluster × category contingency table (and corresponding heatmap) in the notebook shows, per discovered cluster, which known category dominates it. **This is where the motivation for Experiment 2 originates:** the `cars` category was observed to sometimes split across more than one discovered cluster, and visual inspection of those clusters suggested the split tracked camera viewpoint (front-facing vs. rear-facing shots) rather than a new semantic category. Experiment 1 had no viewpoint ground truth, so this was only a qualitative observation at this stage — it is the reason Experiment 2 exists.

### Parameter Sensitivity

A grid sweep over UMAP `n_neighbors` (10, 15, 30) and HDBSCAN `min_cluster_size` (5, 10, 15, 25) × `min_samples` (3, 5, 10) measured how sensitive the clustering result is to configuration. The baseline configuration was kept fixed as the primary reference point and was **not replaced** just because another configuration scored higher.

| Configuration | ARI |
|---|---|
| Baseline (`n_neighbors=15`, `min_cluster_size=10`, `min_samples=5`) | 0.7118 |
| Best tested (`n_neighbors=15`, `min_cluster_size=10`, `min_samples=3`) | 0.7656 |
| **Improvement** | **+0.0538** |

This best-tested configuration is the strongest result found **within the specific parameter grid searched** — it is not presented as a universally optimal configuration.

### Silhouette Analysis

Computed directly on the raw, L2-normalized I-JEPA embeddings using the true category labels and cosine distance — independent of the UMAP projection and HDBSCAN assignment.

**Silhouette score: 0.1903**

On a [-1, 1] scale, this is a modest value. It indicates that the 8 categories are not tightly or cleanly separated in the raw embedding space, even though the UMAP + HDBSCAN pipeline was able to recover substantially more structure downstream. This score should not be read as evidence of strong class separation; it is a useful counterweight to the stronger clustering metrics above, obtained through the nonlinear UMAP projection.

### Experiment 1 Results Summary

| Property | Value |
|---|---|
| Dataset size | 480 images (8 categories × 60 images) |
| Embedding dimension | 1280 (mean-pooled, L2-normalized) |
| Retrieval sanity check | 8/8 tested queries — 100% same-category in top-5 |
| Baseline clustering (ARI / NMI / purity) | 0.7118 / 0.8239 / 0.914 |
| Best tested clustering ARI (parameter sweep) | 0.7656 |
| Silhouette score (raw embeddings, true labels, cosine) | 0.1903 |

### Experiment 1 Caveats

- The retrieval check used only **8 sampled queries** (one per category), not an exhaustive benchmark.
- The 8 categories were chosen to be visually distinct from one another, which likely makes clustering easier than on more ambiguous or overlapping real-world content.
- The silhouette score (0.1903) is modest — raw embedding-space category separation is real but far from perfect.
- Clustering performance depends meaningfully on UMAP/HDBSCAN configuration (Parameter Sensitivity above).
- Strong results on this controlled, category-balanced dataset do **not** imply equally strong clustering on arbitrary, uncontrolled personal photo libraries, which contain overlapping scenes, multiple objects per photo, and continuous visual variation.

---

## Experiment 2 — Viewpoint Sensitivity Analysis

**Motivation:** Experiment 1's `cars` category was sometimes split by HDBSCAN into more than one cluster, with visual inspection suggesting the split tracked camera viewpoint (front vs. rear views) rather than a new semantic category. Experiment 1 had no angle ground truth to test this. Experiment 2 investigates it directly.

**Research question:**

> Does pretrained I-JEPA encode car viewpoint strongly enough that different viewing angles occupy different regions of the embedding space?

### Dataset

**Car Angle Classification Dataset** (Kaggle):
https://www.kaggle.com/datasets/amarcodes/car-angle-classification-dataset

The dataset contains angle-labeled folders. Angle labels are used **only for post-hoc evaluation** — never for image preprocessing, embedding extraction, normalization, FAISS indexing, UMAP fitting, or HDBSCAN clustering.

### Important scope notes

- This is **not** a supervised angle classifier. No model was trained or fine-tuned.
- The same I-JEPA checkpoint (`facebook/ijepa_vith14_1k`) and the same embedding methodology (mean-pooled patch tokens → 1280-D → L2 normalization) as Experiment 1 were used, unchanged.
- Angle labels are evaluation metadata only.

### Original Viewpoint Retrieval

Cosine-similarity nearest-neighbor retrieval (FAISS `IndexFlatIP` on L2-normalized embeddings), evaluated against the known angle labels:

| Metric | Value |
|---|---|
| Same-angle@5 | 93.34% |
| Same-angle@10 | 92.74% |
| Same-angle@20 | 92.38% |
| 1-NN angle agreement | 94.43% |
| 5-NN angle agreement | 95.37% |
| 10-NN angle agreement | 94.83% |

### Original 1280-D Embedding Space Analysis

| Metric | Value |
|---|---|
| Same-angle mean cosine similarity | 0.8113 |
| Different-angle mean cosine similarity | 0.5504 |
| Silhouette score (raw embeddings, angle labels, cosine) | 0.3626 |

The gap between same-angle and different-angle similarity, together with the >90% nearest-neighbor agreement rates above, indicates that viewpoint is a strong local organizing signal in the raw embedding space — well before any dimensionality reduction is applied.

### Clustering Comparison: Direct HDBSCAN vs. UMAP + HDBSCAN

To determine whether this viewpoint structure originates in the raw I-JEPA embedding space or is introduced/amplified by UMAP, HDBSCAN was run twice: once directly on the 1280-D normalized embeddings, and once on a UMAP projection of them.

**Direct HDBSCAN on original 1280-D embeddings:**

| Metric | Value |
|---|---|
| Discovered clusters | 9 |
| Noise | 14.5% |
| ARI | 0.4608 |
| NMI | 0.5825 |
| Weighted purity | 0.610 |

**UMAP + HDBSCAN:**

| Metric | Value |
|---|---|
| Discovered clusters | 29 |
| Noise | 6.7% |
| ARI | 0.6986 |
| NMI | 0.7007 |
| Mean purity | 0.884 |
| Weighted purity | 0.952 |

### Parameter Sensitivity

A grid sweep over UMAP `n_neighbors` (10, 15, 30) and HDBSCAN `min_cluster_size` (5, 10, 15, 25) × `min_samples` (3, 5, 10) was run on the UMAP + HDBSCAN pipeline, in the same spirit as Experiment 1's parameter sweep. The baseline configuration was kept fixed as the primary reference point and was **not replaced** just because another configuration scored higher.

| Configuration | ARI |
|---|---|
| Baseline (`n_neighbors=15`, `min_cluster_size=10`, `min_samples=5`) | 0.6986 |
| Best tested (`n_neighbors=15`, `min_cluster_size=25`, `min_samples=10`) | 0.8122 |
| **Improvement** | **+0.1136** |

This best-tested configuration is the strongest result found **within the specific parameter grid searched** — it is not presented as a universally optimal configuration.

### Interpretation

1. **Viewpoint structure already exists in the original I-JEPA embedding space.** Direct HDBSCAN on the raw 1280-D embeddings recovers non-trivial angle-aligned clusters (ARI 0.4608) without ever seeing UMAP.
2. **The very strong nearest-neighbor agreement (93–95%) demonstrates strong local viewpoint-sensitive structure**, independent of any clustering algorithm's specific behavior.
3. **UMAP substantially changes how that structure is exposed to HDBSCAN.** The UMAP + HDBSCAN pipeline finds many more, smaller, purer clusters (29 clusters, weighted purity 0.952) than direct clustering on the raw embeddings (9 clusters, weighted purity 0.610). UMAP does not appear to invent the viewpoint signal from nothing — it is present natively — but it reorganizes the density structure in a way that makes HDBSCAN much more effective at isolating it.
4. **Therefore, the front/rear car cluster separation observed in Experiment 1 is not simply random clustering behavior.** It is consistent with a real, measurable viewpoint-sensitive property of the I-JEPA embedding space, not an artifact of that one clustering run.
5. **A visually coherent cluster should not automatically be assumed to equal a semantic album.** A cluster that looks clean and internally consistent may be organized around viewpoint, lighting, framing, or another visual property rather than the human-meaningful category a downstream application would want.

**On wording:** the appropriate claim here is that *"the pretrained I-JEPA embedding exhibits strong viewpoint-sensitive structure on the car-angle dataset."* This is not the same as saying "I-JEPA understands car viewpoint" — the evidence is statistical association between embedding-space geometry and viewpoint labels, not evidence of concept understanding.

Viewpoint is also not established as the *only* factor driving the observed similarity structure. Car geometry, image framing, lighting, background, and overall composition may all contribute alongside viewpoint, and this experimental design cannot fully separate their individual contributions.

---

## Experiment 2.1 — Near-Duplicate-Controlled Viewpoint Retrieval Validation

**Goal:** determine whether the strong viewpoint-retrieval result in Experiment 2 was caused primarily by near-duplicate images in the dataset (resized, recompressed, or lightly cropped copies of the same photo, which would be trivial, non-informative retrieval hits).

Near-duplicate detection used perceptual hashing (pHash), completely independent of the angle labels. Detected duplicate groups were collapsed to one representative image per group (deterministic selection, not based on angle label), and all retrieval metrics were recomputed on the deduplicated set.

### Deduplication Summary

| Metric | Value |
|---|---|
| Images before deduplication | 3,000 |
| Near-duplicate groups found | 104 |
| Images involved in duplicate groups | 450 |
| Images removed/collapsed | 346 |
| Images remaining after deduplication | 2,654 |

### Results After Deduplication

| Metric | Value |
|---|---|
| Same-angle@5 | 92.99% |
| Same-angle@10 | 92.57% |
| Same-angle@20 | 92.53% |
| 1-NN angle agreement | 93.63% |
| 5-NN angle agreement | 95.06% |
| 10-NN angle agreement | 94.76% |
| Chance-level same-label rate | ≈ 20.22% |

### Comparison: Experiment 2 vs. Experiment 2.1

| Metric | Experiment 2 | Experiment 2.1 | Change |
|---|---|---|---|
| Same-angle@5 | 93.34% | 92.99% | −0.35 pp |
| Same-angle@10 | 92.74% | 92.57% | −0.17 pp |
| Same-angle@20 | 92.38% | 92.53% | +0.15 pp |
| 1-NN | 94.43% | 93.63% | −0.80 pp |
| 5-NN | 95.37% | 95.06% | −0.31 pp |
| 10-NN | 94.83% | 94.76% | −0.07 pp |

The viewpoint retrieval signal remained very strong after removing/collapsing near-duplicate images. The largest reduction across all six headline metrics was only about **0.8 percentage points**, on 1-NN agreement. Every deduplicated metric remains far above the chance-level same-label rate of ≈20.22%.

**Conclusion:** the viewpoint-sensitive structure observed in Experiment 2 is not primarily explained by near-duplicate leakage. It persists on a dataset from which near-duplicates have been identified and collapsed using a detection method that never had access to the angle labels.

**Experiment 2.1 closes the research validation phase for viewpoint sensitivity.** No further viewpoint experiments, additional datasets, alternative embedding models, or supervised classification studies are planned as part of this line of investigation.

---

## Research Conclusion

Taken together, Experiments 1, 2, and 2.1 support the following account of pretrained I-JEPA embeddings for this project's purposes:

1. **Pretrained I-JEPA embeddings provide useful local visual similarity** on a controlled semantic dataset (Experiment 1: 100% same-category retrieval on the 8 tested queries).
2. **UMAP + HDBSCAN can recover meaningful category-level structure** from those embeddings without category labels ever being given to the model or the clustering algorithm (Experiment 1: baseline ARI 0.7118, NMI 0.8239, mean purity 0.914).
3. **Raw embeddings are not perfectly globally separated** — the modest silhouette score (0.1903 in Experiment 1) shows category overlap exists in the raw embedding space even where downstream clustering metrics are strong.
4. **The car-angle experiment demonstrates that viewpoint is a strong organizing factor** in I-JEPA's embedding space (Experiment 2: 93–95% same-angle retrieval and nearest-neighbor agreement, well above chance).
5. **This viewpoint structure exists before UMAP** — direct HDBSCAN on the raw 1280-D embeddings already recovers angle-aligned clusters (ARI 0.4608), so the effect is not purely a UMAP artifact.
6. **UMAP can substantially improve the ability of HDBSCAN to expose density-separated viewpoint groups** — the UMAP + HDBSCAN pipeline found more, purer, angle-aligned clusters (29 clusters, weighted purity 0.952) than direct clustering did.
7. **The viewpoint result survives near-duplicate control** (Experiment 2.1) — deduplication changed the headline metrics by at most 0.8 percentage points, ruling out near-duplicate leakage as the primary explanation.
8. **Therefore, similarity retrieval and semantic album discovery should be treated as distinct workloads** in the eventual Smart Gallery, not assumed to be the same problem solved by the same pipeline.
9. **A discovered visual cluster should not automatically be interpreted as a human-defined semantic album.** It may instead reflect viewpoint, lighting, framing, or another visual property that happens to produce a dense, internally consistent group.
10. **These experiments validate the representation sufficiently to move from research notebooks to application engineering.** The research validation phase for the underlying representation is considered complete; remaining open questions (see Limitations, and Future Work below) are engineering and architecture questions for Phase 1 onward, not open questions about whether to proceed.

---

## Research Insight for the Future Smart Gallery

A central, recurring finding across Experiments 1 and 2 is that **retrieval and clustering are distinct workloads**, even though both are built on the same underlying embeddings:

- I-JEPA provides **strong local similarity** — nearest neighbors in embedding space are reliably visually/semantically related (Experiment 1's retrieval check) and reliably viewpoint-related within a single object category (Experiment 2's retrieval check).
- The **global embedding space is not perfectly clusterable** — Experiment 1's modest silhouette score (0.1903) and Experiment 2's much lower direct-HDBSCAN ARI (0.4608) compared to its UMAP+HDBSCAN ARI (0.6986) both show that global cluster structure is weaker, and more dependent on the specific clustering pipeline used, than local nearest-neighbor structure is.

This has a direct architectural consequence:

```text
I-JEPA
├── FAISS  → similarity retrieval        (strong, direct use of local structure)
└── UMAP + HDBSCAN → candidate visual groups   (global structure, pipeline-dependent)
```

**The future application should not assume that one discovered cluster equals one semantic album.** A cluster is a *candidate* visual grouping — it may correspond to a meaningful album, or it may correspond to a viewpoint, lighting condition, or framing pattern that a human would not consider a distinct album. Any automatic organization feature built on top of UMAP + HDBSCAN should treat cluster output as a starting point for further processing or user confirmation, not as a finished album boundary.

---

## Repository Structure

```text
semantic-photo-intelligence/
│
├── controlled_dataset/
│   ├── mountains/
│   ├── boats/
│   ├── flowers/
│   ├── cars/
│   ├── shark/
│   ├── snake/
│   ├── cat/
│   └── dog/
│
├── notebooks/
│   ├── ijepa_semantic_image_clustering.ipynb            # Experiment 1
│   ├── notebook03_ijepa_viewpoint_sensitivity_analysis.ipynb   # Experiment 2
│   └── notebook03_1_near_duplicate_controlled_retrieval.ipynb  # Experiment 2.1
│
├── data/
│   ├── metadata/
│   ├── embeddings/
│   └── index/
│
├── README.md
├── requirements.txt
└── .gitignore
```

- **`controlled_dataset/`** — the manually curated, category-labeled image set used in Experiment 1 (one folder per category).
- **`notebooks/`** — the executable notebooks for each experiment: embedding extraction, retrieval, clustering, evaluation, parameter sweeps, near-duplicate detection, and validation.
- **`data/metadata/`** — dataset manifests, duplicate-group listings, retrieval metrics, and evaluation artifacts for each experiment.
- **`data/embeddings/`** — cached I-JEPA embeddings, keyed by a hash of each experiment's image IDs, so re-running a notebook does not require recomputing inference unless the underlying dataset changes.
- **`data/index/`** — FAISS index artifacts used for similarity retrieval evaluation.
- **`requirements.txt`** — Python dependencies needed to run the notebooks.

> Note: the Car Angle Classification Dataset (Experiment 2/2.1) is not included in this repository; it is downloaded separately from Kaggle and placed locally per the instructions in `notebook03_ijepa_viewpoint_sensitivity_analysis.ipynb`.

## Dataset Sources

**Experiment 1** uses a manually curated 480-image subset assembled from images sourced from the following public datasets. These source datasets do not themselves contain this project's specific 8-category, 60-images-per-category structure — that structure was created by manually selecting and organizing a subset of images from these sources.

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

**Experiments 2 and 2.1** use the Car Angle Classification Dataset from Kaggle:
https://www.kaggle.com/datasets/amarcodes/car-angle-classification-dataset

Refer to each original dataset page for its current license and attribution requirements before reuse.

## Reproducibility

### Environment

The notebooks rely on the following third-party packages (installed automatically by each notebook's setup cell if missing): `pandas`, `Pillow`, `torch`, `transformers`, `numpy`, `umap-learn`, `hdbscan`, `scikit-learn`, `matplotlib`, `tqdm`, `seaborn`, `faiss-cpu`, `imagehash`.

The notebooks do not pin specific package versions; a `requirements.txt` is included in the repository for convenience, but exact versions should be filled in based on your working environment.

A CUDA-capable GPU was used when the notebooks were executed (device selection falls back to CPU automatically via `torch.cuda.is_available()`), but no specific hardware is required to run Experiment 1. Experiments 2 and 2.1 use the larger `facebook/ijepa_vith14_1k` model over more images and are substantially slower on CPU; a GPU is strongly recommended for those.

### Dataset layout

**Experiment 1** — place the controlled dataset under `controlled_dataset/`, with one subfolder per category (see Repository Structure above). The notebook expects **8 categories × 60 images = 480 images total** and validates this automatically.

**Experiments 2 / 2.1** — download the Car Angle Classification Dataset from Kaggle and place it locally; the notebook's configuration cell has a path variable to point at it, and dataset discovery is done dynamically (no folder structure is hard-coded or assumed).

### Running the notebooks

1. Install the required dependencies (see Environment above).
2. Populate `controlled_dataset/` for Experiment 1, and download the Car Angle Classification Dataset locally for Experiments 2/2.1.
3. Run the notebooks in order: Experiment 1 → Experiment 2 → Experiment 2.1. Experiment 2.1 will reuse Experiment 2's cached embeddings automatically if they are present and verified to match; otherwise it recomputes them using the identical pipeline.

### Execution flow

**Experiment 1:** dataset discovery and validation → I-JEPA embedding extraction (1280-D per image) → L2 normalization → retrieval sanity check → UMAP projection → HDBSCAN clustering → ARI/NMI/purity evaluation → contingency analysis → parameter sweep → silhouette analysis → results summary.

**Experiment 2:** dataset discovery and validation → visual sanity check → I-JEPA embedding extraction → L2 normalization → FAISS retrieval analysis (same-angle@k, k-NN agreement) → raw-embedding structure analysis → UMAP visualization → direct HDBSCAN vs. UMAP+HDBSCAN ablation → contingency and qualitative analysis → results summary.

**Experiment 2.1:** load/verify Experiment 2's dataset and embeddings → perceptual-hash near-duplicate detection → deterministic duplicate-group collapsing → rebuild FAISS index on deduplicated embeddings → recompute core retrieval metrics → compare against Experiment 2 → final decision.

Embeddings are cached locally to disk (keyed by a hash of each experiment's image IDs) so that re-running a notebook does not require recomputing I-JEPA inference unless the dataset changes.

## Limitations

- **Small datasets relative to real photo libraries**: 480 images (Experiment 1) and up to a few thousand (Experiments 2/2.1) are sufficient for controlled pipeline evaluation, but far smaller than a typical real-world photo library.
- **Visually distinct categories (Experiment 1)**: the 8 categories were chosen to be visually dissimilar from one another, which likely makes clustering easier than on more ambiguous or overlapping real-world content.
- **Single object class (Experiments 2/2.1)**: the car-angle dataset contains only cars, so it can measure whether viewpoint organizes the embedding space *within* a semantic category, but cannot by itself test whether viewpoint similarity ever outranks cross-category semantic similarity.
- **Manually curated category structure (Experiment 1)**: labels reflect a single manual labeling scheme and may contain borderline examples.
- **Single checkpoint evaluated**: only `facebook/ijepa_vith14_1k` was tested across all experiments; results may not generalize to other I-JEPA checkpoints or model scales.
- **Dependence on UMAP/HDBSCAN**: clustering results are specific to this dimensionality-reduction and clustering combination, not to I-JEPA embeddings in isolation.
- **Parameter sensitivity**: clustering quality varies meaningfully with UMAP/HDBSCAN configuration in both Experiment 1 and Experiment 2, so any single baseline result should not be treated as a fixed, absolute measure of representation quality.
- **Correlational, not causal**: the viewpoint-sensitivity findings in Experiment 2/2.1 are statistical associations between embedding geometry and angle labels; they do not isolate viewpoint from other correlated visual properties (car geometry, framing, lighting, background, composition).
- **Controlled datasets ≠ arbitrary personal photo library**: strong performance on these controlled experiments does **not** demonstrate that arbitrary personal photo collections — with overlapping scenes, multiple objects per photo, mixed subjects, and continuous visual variation — will retrieve or cluster equally well.

## Future Work

The research validation phase is complete. The following describes **planned future development**, not implemented functionality — no part of the Smart Gallery application exists yet in this repository.

### Phase 1 — Core Photo Engine

- Image scanning
- Image validation
- Exact duplicate detection
- I-JEPA embedding extraction
- Embedding cache

### Phase 2 — Persistent Indexing

- SQLite metadata storage
- FAISS similarity index
- Similarity search

### Phase 3 — Automatic Visual Organization

- UMAP
- HDBSCAN
- Cluster exploration
- Noise / unique-photo handling

### Phase 4 — Application

- Streamlit interface
- FastAPI backend
- GPU batch processing
- Incremental ingestion

**Explicitly postponed:** automatic album naming and VLM-based captioning/naming are not part of the immediate next phase. They remain future ideas, deferred until the core engine (Phases 1–3) is built and the cluster-vs-album distinction discussed above has a concrete architectural answer.
