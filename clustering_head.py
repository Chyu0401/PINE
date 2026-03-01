import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment


class KMeansClusteringHead:
    """
    A lightweight clustering evaluation head for learned embeddings.

    This head is designed to be used in the same way as a classification head:
    - The encoder is frozen
    - Clustering is performed on embeddings
    - Metrics are computed for evaluation only (no backprop)

    Supports KMeans and MiniBatchKMeans.
    """

    def __init__(
        self,
        num_clusters: int,
        use_minibatch: bool = False,
        batch_size: int = 4096,
        n_init: int = 20,
        random_state: int = 42,
    ):
        self.num_clusters = num_clusters
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size
        self.n_init = n_init
        self.random_state = random_state

    def _clustering_accuracy(self, y_true, y_pred):
        """
        Clustering ACC with Hungarian matching.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(len(y_true)):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return float(w[row_ind, col_ind].sum() / len(y_true))

    @torch.no_grad()
    def evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor = None,
        n_runs: int = 5,
    ):
        """
        Args:
            embeddings: Tensor of shape [N, D]
            labels: Optional ground-truth labels [N], used only for evaluation
            n_runs: Number of repeated KMeans runs with different random seeds

        Returns:
            dict with clustering metrics (mean and std)
        """
        assert embeddings.dim() == 2, "embeddings must be [N, D]"

        X = embeddings.detach().cpu().numpy()

        nmi_list, ari_list, f1_list, acc_list, inertia_list = [], [], [], [], []

        for run in range(n_runs):
            random_state = self.random_state + run

            if self.use_minibatch:
                clusterer = MiniBatchKMeans(
                    n_clusters=self.num_clusters,
                    batch_size=self.batch_size,
                    n_init=self.n_init,
                    random_state=random_state,
                )
            else:
                clusterer = KMeans(
                    n_clusters=self.num_clusters,
                    n_init=self.n_init,
                    random_state=random_state,
                )

            pred = clusterer.fit_predict(X)
            inertia_list.append(float(clusterer.inertia_))

            if labels is not None:
                y = labels.detach().cpu().numpy()
                nmi_list.append(normalized_mutual_info_score(y, pred))
                ari_list.append(adjusted_rand_score(y, pred))
                f1_list.append(f1_score(y, pred, average='macro'))
                acc_list.append(self._clustering_accuracy(y, pred))

        results = {
            "num_clusters": self.num_clusters,
            "inertia_mean": float(np.mean(inertia_list)),
            "inertia_std": float(np.std(inertia_list)),
        }

        if labels is not None:
            results.update({
                "NMI_mean": float(np.mean(nmi_list)),
                "NMI_std": float(np.std(nmi_list)),
                "ARI_mean": float(np.mean(ari_list)),
                "ARI_std": float(np.std(ari_list)),
                "F1_mean": float(np.mean(f1_list)),
                "F1_std": float(np.std(f1_list)),
                "ACC_mean": float(np.mean(acc_list)),
                "ACC_std": float(np.std(acc_list)),
            })

        return results
