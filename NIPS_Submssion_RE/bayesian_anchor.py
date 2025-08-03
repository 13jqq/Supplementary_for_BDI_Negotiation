import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.mixture import GaussianMixture


def _dynamic_anchor_bayesian_regression(
        X_train, y_train, X_test,
        relative_time=None, max_preds=20, n_samples=20, is_opponent_view=False):
    """
    Dynamic Anchor Bayesian Regression

    This function performs Bayesian regression with dynamic anchor adjustment,
    incorporating median smoothing, first-order difference, time-step features,
    and an optional end-phase offset to enforce concession.

    Parameters:
        X_train (np.ndarray): Training input features.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Test input features for prediction.
        relative_time (float, optional): Current relative time (0~1) in negotiation, for adjusting weights or offsets.
        max_preds (int): Maximum number of predictions to return.
        n_samples (int): Number of samples to draw from posterior distribution.
        is_opponent_view (bool): Whether the model is viewed from the opponent's perspective (affects smoothing behavior).

    Returns:
        np.ndarray: A (n_samples × len(X_test)) matrix of predicted utility values.
    """
    print("_dynamic_anchor_bayesian_regression_v6 start")
    print(is_opponent_view)

    import pandas as pd

    try:
        if len(y_train) < 3:
            base_val = y_train[-1] if len(y_train) > 0 else 0.5
            return np.full((n_samples, len(X_test)), base_val)

        # Median smoothing of y_train
        y_train = pd.Series(y_train).rolling(window=3, center=True, min_periods=1).median().values

        # Exponential smoothing if in opponent view
        if is_opponent_view:
            alpha = 0.7
            for i in range(1, len(y_train)):
                y_train[i] = alpha * y_train[i - 1] + (1 - alpha) * y_train[i]

        # === Feature Construction ===
        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)

        # Add first-order difference of target values
        y_diff = np.diff(y_train, prepend=y_train[0])
        X_train_2d = np.column_stack((X_train_2d, y_diff.reshape(-1, 1)))

        # Add normalized time-step feature
        time_steps = np.arange(len(y_train)) / len(y_train)
        X_train_2d = np.column_stack((X_train_2d, time_steps.reshape(-1, 1)))

        # === Test Feature Construction ===
        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)
        t_test = np.linspace(len(y_train)/len(y_train), 1.0, len(X_test_2d))
        X_test_2d = np.column_stack((X_test_2d, y_diff[-1:].repeat(len(X_test_2d)).reshape(-1, 1)))
        X_test_2d = np.column_stack((X_test_2d, t_test.reshape(-1, 1)))

        if X_test_2d.shape[0] > max_preds:
            X_test_2d = X_test_2d[:max_preds]

        # === Bayesian Regression Training ===
        model = BayesianRidge(fit_intercept=False)
        model.fit(X_train_2d, y_train)
        mu = model.coef_
        Sigma = model.sigma_

        # === Anchor Construction ===
        n_anchors = min(5, max(2, len(y_train) // 4))
        sorted_y = sorted(y_train)
        splits = np.array_split(sorted_y, n_anchors)
        anchors = np.array([np.mean(split) for split in splits if len(split) > 0])
        anchors = anchors[(anchors >= 0.0) & (anchors <= 1.0)]
        if len(anchors) < 1:
            return np.full((n_samples, len(X_test_2d)), np.mean(y_train))

        results = []

        for x_i in X_test_2d:
            x_i = x_i.reshape(1, -1)
            thetas = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)
            trend_preds = thetas @ x_i.T
            trend_preds = trend_preds.flatten()

            # Smooth predicted values
            trend_preds = 0.8 * trend_preds + 0.2 * np.mean(trend_preds)

            # Anchor fusion
            trend_matrix = trend_preds.reshape(-1, 1)
            anchors_matrix = anchors.reshape(1, -1)
            distances = np.abs(trend_matrix - anchors_matrix)
            closest_indices = np.argmin(distances, axis=1)
            closest_anchors = anchors[closest_indices]

            w_anchor = min(0.8, 0.3 + 0.5 * (relative_time or 0.0))
            w_trend = 1.0 - w_anchor
            fused_preds = w_trend * trend_preds + w_anchor * closest_anchors

            # End-phase concession offset
            if relative_time and relative_time > 0.9:
                offset = np.clip(0.05 * (relative_time - 0.9) / 0.1, 0.0, 0.05)
                fused_preds += offset

            results.append(np.clip(fused_preds, 0.0, 1.0))

        return np.column_stack(results)

    except Exception as e:
        print(f"{e}")
        base_val = y_train[-1] if len(y_train) > 0 else 0.5
        return np.full((n_samples, len(X_test)), base_val)
