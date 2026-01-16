"""
Clustering: Separación de nanoporos individuales
Soporta HDBSCAN, KMeans, MeanShift y Agglomerative Clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Intentar importar HDBSCAN (opcional)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from ..utils.constants import DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES


class ClusteringEngine:
    """
    Motor de clustering unificado para múltiples algoritmos

    Soporta:
    - HDBSCAN: clustering jerárquico basado en densidad (detecta ruido)
    - KMeans: particionamiento en K clusters
    - MeanShift: basado en densidad con estimación automática de clusters
    - Agglomerative: clustering jerárquico aglomerativo
    """

    def __init__(self, positions: Optional[np.ndarray] = None):
        """
        Args:
            positions: array Nx3 de coordenadas [x, y, z] (opcional)
        """
        self.positions = positions
        self.labels = None
        self.metrics = {}
        self.n_clusters = 0

    def set_positions(self, positions: np.ndarray) -> None:
        """
        Establece las posiciones para clustering

        Args:
            positions: array Nx3 de coordenadas
        """
        self.positions = positions

    def apply_kmeans(self, n_clusters: int = 5) -> int:
        """
        Aplica KMeans clustering

        Args:
            n_clusters: número de clusters deseado

        Returns:
            n_clusters_found: número de clusters detectados
        """
        if self.positions is None:
            raise ValueError("No hay posiciones cargadas. Usa set_positions() primero.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.positions)
        self.n_clusters = len(np.unique(self.labels))

        # Calcular métricas
        if self.n_clusters > 1:
            self.metrics = {
                'silhouette': silhouette_score(self.positions, self.labels),
                'davies_bouldin': davies_bouldin_score(self.positions, self.labels),
                'calinski_harabasz': calinski_harabasz_score(self.positions, self.labels)
            }
        else:
            self.metrics = {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0}

        return self.n_clusters

    def apply_meanshift(self, quantile: float = 0.2) -> int:
        """
        Aplica MeanShift clustering (estimación automática de clusters)

        Args:
            quantile: percentil para estimación de bandwidth (default: 0.2)

        Returns:
            n_clusters_found: número de clusters detectados
        """
        if self.positions is None:
            raise ValueError("No hay posiciones cargadas. Usa set_positions() primero.")

        bandwidth = estimate_bandwidth(
            self.positions,
            quantile=quantile,
            n_samples=min(500, len(self.positions))
        )

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        self.labels = ms.fit_predict(self.positions)
        self.n_clusters = len(np.unique(self.labels))

        # Calcular métricas
        if self.n_clusters > 1:
            self.metrics = {
                'silhouette': silhouette_score(self.positions, self.labels),
                'davies_bouldin': davies_bouldin_score(self.positions, self.labels),
                'calinski_harabasz': calinski_harabasz_score(self.positions, self.labels),
                'bandwidth': bandwidth
            }
        else:
            self.metrics = {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0, 'bandwidth': bandwidth}

        return self.n_clusters

    def apply_agglomerative(self, n_clusters: int = 5, linkage_method: str = 'ward') -> int:
        """
        Aplica clustering aglomerativo jerárquico

        Args:
            n_clusters: número de clusters deseado
            linkage_method: método de enlace ('ward', 'complete', 'average', 'single')

        Returns:
            n_clusters_found: número de clusters detectados
        """
        if self.positions is None:
            raise ValueError("No hay posiciones cargadas. Usa set_positions() primero.")

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        self.labels = agg.fit_predict(self.positions)
        self.n_clusters = len(np.unique(self.labels))

        # Calcular métricas
        if self.n_clusters > 1:
            self.metrics = {
                'silhouette': silhouette_score(self.positions, self.labels),
                'davies_bouldin': davies_bouldin_score(self.positions, self.labels),
                'calinski_harabasz': calinski_harabasz_score(self.positions, self.labels),
                'linkage_method': linkage_method
            }
        else:
            self.metrics = {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0, 'linkage_method': linkage_method}

        return self.n_clusters

    def apply_hdbscan(self,
                     min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                     min_samples: Optional[int] = DEFAULT_MIN_SAMPLES) -> int:
        """
        Aplica HDBSCAN clustering (jerárquico basado en densidad)

        Requiere: pip install hdbscan

        Args:
            min_cluster_size: tamaño mínimo de cluster
            min_samples: mínimo de muestras (None = usa min_cluster_size)

        Returns:
            n_clusters_found: número de clusters detectados (excluyendo ruido -1)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN no está instalado. Instala con: pip install hdbscan"
            )

        if self.positions is None:
            raise ValueError("No hay posiciones cargadas. Usa set_positions() primero.")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        self.labels = clusterer.fit_predict(self.positions)

        # Número de clusters (excluyendo ruido -1)
        unique_labels = np.unique(self.labels)
        self.n_clusters = len(unique_labels[unique_labels != -1])

        # Calcular métricas (excluyendo ruido)
        mask = self.labels != -1
        n_noise = (self.labels == -1).sum()

        if mask.sum() > 0 and self.n_clusters > 1:
            self.metrics = {
                'silhouette': silhouette_score(self.positions[mask], self.labels[mask]),
                'davies_bouldin': davies_bouldin_score(self.positions[mask], self.labels[mask]),
                'calinski_harabasz': calinski_harabasz_score(self.positions[mask], self.labels[mask]),
                'noise_points': int(n_noise),
                'noise_fraction': n_noise / len(self.labels)
            }
        else:
            self.metrics = {
                'silhouette': 0,
                'davies_bouldin': 0,
                'calinski_harabasz': 0,
                'noise_points': int(n_noise),
                'noise_fraction': n_noise / len(self.labels)
            }

        return self.n_clusters

    def get_labels(self) -> np.ndarray:
        """
        Obtener etiquetas de cluster

        Returns:
            labels: array de etiquetas (-1 = ruido en HDBSCAN)
        """
        if self.labels is None:
            raise ValueError("No se ha ejecutado clustering. Usa apply_*() primero.")
        return self.labels

    def get_metrics(self) -> Dict[str, float]:
        """
        Obtener métricas de calidad del clustering

        Returns:
            metrics: diccionario con métricas
                - silhouette: [-1, 1], mayor es mejor
                - davies_bouldin: [0, inf), menor es mejor
                - calinski_harabasz: [0, inf), mayor es mejor
                - noise_points (HDBSCAN): número de puntos de ruido
        """
        return self.metrics

    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Obtener tamaños de cada cluster

        Returns:
            cluster_sizes: dict {cluster_id: n_atoms}
        """
        if self.labels is None:
            raise ValueError("No se ha ejecutado clustering. Usa apply_*() primero.")

        unique_labels = np.unique(self.labels)
        return {int(label): int(np.sum(self.labels == label)) for label in unique_labels}

    def split_by_clusters(self, positions: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Divide posiciones por cluster

        Args:
            positions: array Nx3 de posiciones (debe coincidir con labels)

        Returns:
            clusters_dict: {cluster_id: positions_subset}
        """
        if self.labels is None:
            raise ValueError("No se ha ejecutado clustering. Usa apply_*() primero.")

        if len(positions) != len(self.labels):
            raise ValueError(f"Número de posiciones ({len(positions)}) no coincide con labels ({len(self.labels)})")

        clusters_dict = {}
        for cluster_id in np.unique(self.labels):
            mask = self.labels == cluster_id
            clusters_dict[int(cluster_id)] = positions[mask]

        return clusters_dict

    def summary(self) -> str:
        """
        Genera resumen del clustering

        Returns:
            summary_text: string con resumen
        """
        if self.labels is None:
            return "No se ha ejecutado clustering."

        summary = [
            f"Clustering Summary",
            f"=" * 50,
            f"Total points: {len(self.labels)}",
            f"Clusters found: {self.n_clusters}",
            f"",
            f"Cluster sizes:"
        ]

        cluster_sizes = self.get_cluster_sizes()
        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            summary.append(f"  {label}: {size} atoms")

        summary.append("")
        summary.append("Quality metrics:")
        for metric, value in self.metrics.items():
            summary.append(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

        return "\n".join(summary)
