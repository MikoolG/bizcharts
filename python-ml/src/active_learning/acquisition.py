"""Acquisition functions for active learning sample selection."""

import numpy as np
from sklearn.cluster import KMeans


def entropy(probs: np.ndarray) -> np.ndarray:
    """Compute entropy of probability distributions.

    Args:
        probs: Array of shape (n_samples, n_classes) with probabilities

    Returns:
        Array of shape (n_samples,) with entropy values
    """
    # Add small epsilon to avoid log(0)
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def uncertainty_sampling(
    probs: np.ndarray,
    n_select: int,
    method: str = "entropy",
) -> list[int]:
    """Select samples based on prediction uncertainty.

    Args:
        probs: Probability distributions of shape (n_samples, n_classes)
        n_select: Number of samples to select
        method: Uncertainty method ('entropy', 'margin', 'least_confident')

    Returns:
        List of indices of selected samples (highest uncertainty first)
    """
    if method == "entropy":
        # Higher entropy = more uncertain
        uncertainty = entropy(probs)
    elif method == "margin":
        # Smaller margin between top 2 classes = more uncertain
        sorted_probs = np.sort(probs, axis=1)
        uncertainty = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
    elif method == "least_confident":
        # Lower max probability = more uncertain
        uncertainty = 1 - np.max(probs, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get indices sorted by uncertainty (descending)
    sorted_indices = np.argsort(uncertainty)[::-1]

    return sorted_indices[:n_select].tolist()


def diversity_sampling(
    embeddings: np.ndarray,
    n_select: int,
) -> list[int]:
    """Select diverse samples using k-means clustering.

    Selects one sample closest to each cluster centroid to ensure
    coverage of the embedding space.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        n_select: Number of samples to select

    Returns:
        List of indices of selected samples
    """
    n_samples = len(embeddings)
    n_clusters = min(n_select, n_samples)

    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    # Select sample closest to each centroid
    selected = []
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = int(np.argmin(distances))
        if closest_idx not in selected:
            selected.append(closest_idx)

    return selected[:n_select]


def hybrid_acquisition(
    probs: np.ndarray,
    embeddings: np.ndarray,
    n_select: int = 50,
    uncertainty_weight: float = 0.6,
) -> list[int]:
    """Hybrid uncertainty-diversity acquisition function.

    Combines entropy-based uncertainty with k-means clustering for diversity.
    This ensures we select both informative (uncertain) and diverse samples,
    avoiding redundant similar examples.

    Pure uncertainty sampling can select many similar examples from the same
    region of the input space. Adding diversity ensures broader coverage.

    Args:
        probs: Probability distributions of shape (n_samples, n_classes)
        embeddings: Embedding vectors of shape (n_samples, embedding_dim)
        n_select: Number of samples to select
        uncertainty_weight: Weight for uncertainty vs diversity (0-1)

    Returns:
        List of indices of selected samples
    """
    n_samples = len(probs)
    if n_samples <= n_select:
        return list(range(n_samples))

    # Compute uncertainty scores (entropy)
    uncertainty = entropy(probs)

    # Normalize to [0, 1]
    if uncertainty.max() > uncertainty.min():
        uncertainty_norm = (uncertainty - uncertainty.min()) / (
            uncertainty.max() - uncertainty.min()
        )
    else:
        uncertainty_norm = np.zeros_like(uncertainty)

    # Cluster embeddings for diversity
    n_clusters = min(n_select, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute diversity scores (distance to cluster centroid)
    diversity = np.zeros(n_samples)
    for i in range(n_samples):
        centroid = kmeans.cluster_centers_[cluster_labels[i]]
        diversity[i] = np.linalg.norm(embeddings[i] - centroid)

    # Normalize diversity scores
    if diversity.max() > diversity.min():
        diversity_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min())
    else:
        diversity_norm = np.zeros_like(diversity)

    # Select highest uncertainty from each cluster
    selected = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Combined score within cluster
        cluster_uncertainties = uncertainty_norm[cluster_mask]
        cluster_diversities = diversity_norm[cluster_mask]

        combined_scores = (
            uncertainty_weight * cluster_uncertainties
            + (1 - uncertainty_weight) * cluster_diversities
        )

        # Select best in cluster
        best_in_cluster_idx = np.argmax(combined_scores)
        best_global_idx = cluster_indices[best_in_cluster_idx]

        if best_global_idx not in selected:
            selected.append(int(best_global_idx))

    # If we need more samples, add by pure uncertainty
    if len(selected) < n_select:
        remaining_indices = [i for i in range(n_samples) if i not in selected]
        remaining_uncertainties = uncertainty[remaining_indices]
        sorted_remaining = np.argsort(remaining_uncertainties)[::-1]

        for idx in sorted_remaining:
            if len(selected) >= n_select:
                break
            selected.append(remaining_indices[idx])

    return selected[:n_select]


def batch_acquisition(
    model,
    texts: list[str],
    n_select: int = 50,
    batch_size: int = 32,
) -> list[int]:
    """Run hybrid acquisition on a batch of texts.

    Convenience function that handles batching for large datasets.

    Args:
        model: Model with predict_proba() and get_embeddings() methods
        texts: List of input texts
        n_select: Number of samples to select
        batch_size: Batch size for model inference

    Returns:
        List of indices of selected samples
    """
    # Get predictions in batches
    all_probs = []
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        probs = model.predict_proba(batch_texts)
        embeddings = model.get_embeddings(batch_texts)
        all_probs.append(probs)
        all_embeddings.append(embeddings)

    probs = np.vstack(all_probs)
    embeddings = np.vstack(all_embeddings)

    return hybrid_acquisition(probs, embeddings, n_select)
