import numpy as np
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split


class KDE:
    def __init__(self, features):
        """
        features: list of column names representing the summary statistics.
        """
        self.features = features
        self.kdes = {}

    def fit(self, df_train, subsample=None):
        """
        Fits an independent 1D KDE for each summary statistic using the training set.

        For computational efficiency, the training data can be subsampled.
        `subsample` specifies the maximum number of samples to use; if the training
        set is larger, a random subset will be used to fit the KDEs.
        """
        if subsample is not None and len(df_train) > subsample:
            print(f"Subsampling training data from {len(df_train)} to {subsample} samples for KDE fitting.")
            df_train = df_train.sample(n=subsample, random_state=42)

        for feature in self.features:
            train_data = df_train[feature].values
            self.kdes[feature] = gaussian_kde(train_data)

    def score(self, df_test):
        """
        The score is the sum of the log-probabilities across all independent KDEs.
        """

        total_log_prob = np.zeros(len(df_test))
        for feature in self.features:
            test_data = df_test[feature].values
            feature_log_prob = self.kdes[feature].logpdf(test_data)
            total_log_prob += feature_log_prob

        return total_log_prob


class MaxQuantile:
    def __init__(self, feature_configs):
        """
        feature_configs: A dictionary mapping column names to higher_is_ood booleans.
        Example: {"kolmogorov_smirnov_statistic": True, "spectral_flatness": False}
        """
        self.feature_configs = feature_configs
        self.reference_data = {}

    def fit(self, df_ref):
        for feature in self.feature_configs:
            self.reference_data[feature] = np.sort(df_ref[feature].values)

    def score(self, df_test):
        """
        Computes the max percentile rank (considering OOD direction) across all features.
        """
        all_quantiles = []

        for feature, higher_is_ood in self.feature_configs.items():
            sorted_ref = self.reference_data[feature]
            test_vals = df_test[feature].values

            # Compute percentile rank [0, 1]
            q = np.searchsorted(sorted_ref, test_vals, side="right") / len(sorted_ref)

            # Flip if lower is OOD so that 1.0 always means "most OOD-like"
            if not higher_is_ood:
                q = 1.0 - q

            all_quantiles.append(q)

        # Return the element-wise maximum across all metrics
        return np.max(all_quantiles, axis=0)


class CalibratedMaxQuantile:
    """
    Two-stage calibrated version of MaxQuantile.

    The validation / reference data is split in half:
      - Half 1 fits a MaxQuantile (per-feature quantile transforms + max).
      - Half 2 is scored by that MaxQuantile to obtain held-out in-distribution
        max-quantile values, which form a second reference distribution.

    At score time the raw max-quantile value is converted to a calibrated
    quantile rank against the half-2 reference distribution.  A calibrated
    score of q means that only a fraction (1 - q) of true in-distribution
    samples produced a higher score, giving direct type-I error control:
    the threshold at calibrated score alpha rejects in-distribution samples
    at rate <= 1 - alpha.

    Parameters
    ----------
    feature_configs :
        Dict mapping column names to ``higher_is_ood`` booleans.
    random_state :
        Seed used for validation data splitting.
    """

    def __init__(self, feature_configs, random_state=42):
        self.feature_configs = feature_configs
        self.random_state = random_state
        self._mq = MaxQuantile(feature_configs)
        self._calibration_scores = None

    def fit(self, df_ref):
        """
        Split ``df_ref`` in half, fit MaxQuantile on the first half, then score
        the second half to build the calibration distribution.
        """
        idx = np.arange(len(df_ref))
        idx_half1, idx_half2 = train_test_split(idx, test_size=0.5, random_state=self.random_state)

        df_half1 = df_ref.iloc[idx_half1]
        df_half2 = df_ref.iloc[idx_half2]

        # Stage 1: fit per-feature quantile transforms on half 1
        self._mq.fit(df_half1)

        # Stage 2: build calibration reference from held-out half 2 scores
        raw_scores = self._mq.score(df_half2)
        self._calibration_scores = np.sort(raw_scores)

    def score(self, df_test):
        """
        Returns calibrated scores in [0, 1].  A score of *q* means that only
        a fraction (1 - q) of held-out in-distribution samples had a higher
        max-quantile value, so thresholding at *q* controls type-I error at
        level (1 - q).
        """
        if self._calibration_scores is None:
            raise RuntimeError("Call fit() before score().")

        raw = self._mq.score(df_test)
        calibrated = np.searchsorted(self._calibration_scores, raw, side="right") / len(self._calibration_scores)
        return calibrated