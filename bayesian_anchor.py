import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.mixture import GaussianMixture


def _dynamic_anchor_bayesian_regression(X_train, y_train, X_test, n_anchors=3):
    """
    动态锚点贝叶斯回归：结合趋势与锚定效应，提高对效用值的预测准确性。
    参数：
        X_train: np.ndarray, 训练特征（通常为时间步或差异特征）
        y_train: list[float], 训练标签（效用值）
        X_test: np.ndarray, 测试特征（单个或多个）
        n_anchors: int, 锚点个数（用于GMM提取）
    返回：
        float, 预测出的效用值，范围在[0.0, 1.0]内
    """
    try:
        if len(y_train) < 3:
            return y_train[-1]  # 数据不足，返回最后一个已知值

        # 格式标准化
        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)
        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)

        # 趋势建模
        trend_model = BayesianRidge()
        trend_model.fit(X_train_2d, y_train)
        trend_pred = trend_model.predict(X_test_2d)[0]

        # 锚点提取
        actual_n_anchors = min(n_anchors, len(y_train) - 1)
        if actual_n_anchors < 2:
            return max(min(trend_pred, 1.0), 0.0)

        try:
            gmm = GaussianMixture(n_components=actual_n_anchors, random_state=0)
            y_train_reshaped = np.array(y_train).reshape(-1, 1)
            gmm.fit(y_train_reshaped)
            anchors = gmm.means_.flatten()

            # 选择最接近的锚点
            closest_anchor_idx = np.argmin(np.abs(anchors - trend_pred))
            closest_anchor = anchors[closest_anchor_idx]

            # 加权融合
            final_prediction = 0.7 * trend_pred + 0.3 * closest_anchor
        except Exception as e:
            print(f"GMM拟合失败: {e}，使用趋势预测")
            final_prediction = trend_pred

        return max(min(final_prediction, 1.0), 0.0)

    except Exception as e:
        print(f"动态锚点贝叶斯回归失败: {e}")
        return y_train[-1] if len(y_train) > 0 else 0.5
def _dynamic_anchor_bayesian_regression_v2(X_train, y_train, X_test, relative_time=None, max_preds=20):
    """
    动态锚点贝叶斯回归 v2：支持多点预测、动态锚点数、锚点权重随时间调整、异常锚点过滤。
    参数：
        X_train: np.ndarray, 训练特征
        y_train: list[float], 训练标签
        X_test: np.ndarray, 测试特征（支持多个测试点）
        relative_time: float in [0,1], 谈判进行到的相对时间，用于调整锚点权重
        max_preds: int, 最多返回多少个预测点
    返回：
        np.ndarray, 预测出的效用值，范围在[0.0, 1.0]
    """
    try:
        if len(y_train) < 3:
            return np.array([y_train[-1]]) if len(y_train) > 0 else np.array([0.5])

        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)
        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)
        if X_test_2d.shape[0] > max_preds:
            X_test_2d = X_test_2d[:max_preds]

        # 趋势模型拟合
        trend_model = BayesianRidge()
        trend_model.fit(X_train_2d, y_train)
        trend_preds = trend_model.predict(X_test_2d)

        # 锚点提取
        raw_anchors = np.array(y_train).reshape(-1, 1)
        n_anchors = min(5, max(2, len(y_train) // 4))

        try:
            gmm = GaussianMixture(n_components=n_anchors, random_state=0)
            gmm.fit(raw_anchors)
            anchors = gmm.means_.flatten()

            # 过滤锚点：去除重复值、极端值
            anchors = np.unique(anchors)
            anchors = anchors[(anchors >= 0.0) & (anchors <= 1.0)]
            if len(anchors) < 1:
                return np.clip(trend_preds, 0.0, 1.0)

            # 为每个预测点选最近锚点
            closest_anchors = np.array([
                anchors[np.argmin(np.abs(anchors - pred))] for pred in trend_preds
            ])

            # 动态权重计算（越靠近尾局，anchor 权重越大）
            if relative_time is None:
                w_anchor = 0.3
            else:
                # 平滑函数：0.3 到 0.8 的权重变化
                w_anchor = min(0.8, 0.3 + 0.5 * relative_time)

            w_trend = 1.0 - w_anchor
            final_preds = w_trend * trend_preds + w_anchor * closest_anchors

        except Exception as e:
            print(f"[GMM失败] {e}，使用纯趋势预测")
            final_preds = trend_preds

        return np.clip(final_preds, 0.0, 1.0)

    except Exception as e:
        print(f"[动态锚点回归错误] {e}")
        return np.array([y_train[-1]]) if len(y_train) > 0 else np.array([0.5])


def _dynamic_anchor_bayesian_regression_v3(X_train, y_train, X_test, relative_time=None, max_preds=20):
    """
    快速版：动态锚点贝叶斯回归。
    替代 GMM 提取锚点的方法为：等分区间+局部均值；加速锚点匹配为向量化操作。
    """
    try:
        if len(y_train) < 3:
            return np.array([y_train[-1]]) if len(y_train) > 0 else np.array([0.5])

        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)
        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)
        if X_test_2d.shape[0] > max_preds:
            X_test_2d = X_test_2d[:max_preds]

        # 趋势模型拟合
        trend_model = BayesianRidge()
        trend_model.fit(X_train_2d, y_train)
        trend_preds = trend_model.predict(X_test_2d)

        # 锚点提取（使用等分区间+局部均值）
        n_anchors = min(5, max(2, len(y_train) // 4))
        sorted_y = sorted(y_train)
        splits = np.array_split(sorted_y, n_anchors)
        anchors = np.array([np.mean(split) for split in splits if len(split) > 0])

        # 限定合法值范围
        anchors = anchors[(anchors >= 0.0) & (anchors <= 1.0)]
        if len(anchors) < 1:
            return np.clip(trend_preds, 0.0, 1.0)

        # 加速锚点匹配（向量化）
        anchors_matrix = anchors.reshape(1, -1)                         # shape: (1, m)
        trend_matrix = trend_preds.reshape(-1, 1)                       # shape: (n, 1)
        distances = np.abs(trend_matrix - anchors_matrix)              # shape: (n, m)
        closest_indices = np.argmin(distances, axis=1)                 # shape: (n,)
        closest_anchors = anchors[closest_indices]                     # shape: (n,)

        # 动态权重
        if relative_time is None:
            w_anchor = 0.3
        else:
            w_anchor = min(0.8, 0.3 + 0.5 * relative_time)
        w_trend = 1.0 - w_anchor

        final_preds = w_trend * trend_preds + w_anchor * closest_anchors
        return np.clip(final_preds, 0.0, 1.0)

    except Exception as e:
        print(f"[动态锚点回归错误 - Fast版本] {e}")
        return np.array([y_train[-1]]) if len(y_train) > 0 else np.array([0.5])



def _dynamic_anchor_bayesian_regression_v4(X_train, y_train, X_test, relative_time=None, max_preds=20, n_samples=20):
    """
    v4：v3的完整结构 + 共享theta采样方式，返回每个测试点n_samples个候选效用预测
    输出 shape: (n_samples, len(X_test))，每行是一次采样下的全部预测
    """
    import numpy as np
    from sklearn.linear_model import BayesianRidge

    try:
        if len(y_train) < 3:
            base_val = y_train[-1] if len(y_train) > 0 else 0.5
            return np.full((n_samples, len(X_test)), base_val)

        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)
        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)
        if X_test_2d.shape[0] > max_preds:
            X_test_2d = X_test_2d[:max_preds]

        # 1. 拟合贝叶斯回归模型
        model = BayesianRidge(fit_intercept=False)
        model.fit(X_train_2d, y_train)
        mu = model.coef_
        Sigma = model.sigma_

        # 2. 多次采样 theta 生成多个趋势预测
        thetas = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)  # shape: (n_samples, d)
        trend_preds_samples = thetas @ X_test_2d.T  # shape: (n_samples, n_test)

        # 3. 构造锚点
        n_anchors = min(5, max(2, len(y_train) // 4))
        sorted_y = sorted(y_train)
        splits = np.array_split(sorted_y, n_anchors)
        anchors = np.array([np.mean(split) for split in splits if len(split) > 0])
        anchors = anchors[(anchors >= 0.0) & (anchors <= 1.0)]
        if len(anchors) < 1:
            return np.clip(trend_preds_samples, 0.0, 1.0)

        # 4. 对每个样本的每个预测点，匹配最近锚点并进行融合
        results = []
        for i in range(n_samples):
            trend_preds = trend_preds_samples[i]  # shape: (n_test,)
            trend_matrix = trend_preds.reshape(-1, 1)                   # (n_test, 1)
            anchors_matrix = anchors.reshape(1, -1)                    # (1, m)
            distances = np.abs(trend_matrix - anchors_matrix)          # (n_test, m)
            closest_indices = np.argmin(distances, axis=1)             # (n_test,)
            closest_anchors = anchors[closest_indices]                 # (n_test,)

            # 5. 动态权重融合
            if relative_time is None:
                w_anchor = 0.3
            else:
                w_anchor = min(0.8, 0.3 + 0.5 * relative_time)
            w_trend = 1.0 - w_anchor

            fused_preds = w_trend * trend_preds + w_anchor * closest_anchors
            results.append(np.clip(fused_preds, 0.0, 1.0))

        return np.array(results)  # shape: (n_samples, n_test)

    except Exception as e:
        print(f"[动态锚点回归错误 - v4版本] {e}")
        base_val = y_train[-1] if len(y_train) > 0 else 0.5
        return np.full((n_samples, len(X_test)), base_val)



def _dynamic_anchor_bayesian_regression_v5(X_train, y_train, X_test, relative_time=None, max_preds=20, n_samples=20, is_opponent_view=False):
    """
    v5：v3结构 + 每个测试点独立采样theta生成多个预测值（更细致建模）
    输出 shape: (n_samples, len(X_test))，每列是该测试点的多个候选预测值
    """
    import numpy as np
    from sklearn.linear_model import BayesianRidge
    if is_opponent_view:
        y_train = np.array(y_train)
        alpha = 0.7  # 趋势平滑系数，可调
        for i in range(1, len(y_train)):
            y_train[i] = alpha * y_train[i-1] + (1 - alpha) * y_train[i]

    try:
        if len(y_train) < 3:
            base_val = y_train[-1] if len(y_train) > 0 else 0.5
            return np.full((n_samples, len(X_test)), base_val)

        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)


        # ✅ 增强输入特征（加入一阶差分）
        y_diff = np.diff(y_train, prepend=y_train[0])
        X_train_2d = np.column_stack((X_train_2d, y_diff.reshape(-1, 1)))


        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)
        if X_test_2d.shape[0] > max_preds:
            X_test_2d = X_test_2d[:max_preds]

        model = BayesianRidge(fit_intercept=False)
        model.fit(X_train_2d, y_train)
        mu = model.coef_
        Sigma = model.sigma_

        # 锚点构造（与 v4 相同）
        n_anchors = min(5, max(2, len(y_train) // 4))
        sorted_y = sorted(y_train)
        splits = np.array_split(sorted_y, n_anchors)
        anchors = np.array([np.mean(split) for split in splits if len(split) > 0])
        anchors = anchors[(anchors >= 0.0) & (anchors <= 1.0)]
        if len(anchors) < 1:
            return np.full((n_samples, len(X_test_2d)), np.mean(y_train))

        results = []

        for x_i in X_test_2d:
            x_i = x_i.reshape(1, -1)  # shape: (1, d)
            thetas = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)  # shape: (n_samples, d)
            trend_preds = thetas @ x_i.T  # shape: (n_samples, 1)
            trend_preds = trend_preds.flatten()

            # ✅ 平滑预测值（强化趋势一致性，抑制尖峰）
            trend_preds = 0.8 * trend_preds + 0.2 * np.mean(trend_preds)

            trend_matrix = trend_preds.reshape(-1, 1)               # (n_samples, 1)
            anchors_matrix = anchors.reshape(1, -1)                 # (1, m)
            distances = np.abs(trend_matrix - anchors_matrix)      # (n_samples, m)
            closest_indices = np.argmin(distances, axis=1)
            closest_anchors = anchors[closest_indices]             # (n_samples,)

            if relative_time is None:
                w_anchor = 0.3
            else:
                w_anchor = min(0.8, 0.3 + 0.5 * relative_time)
            w_trend = 1.0 - w_anchor

            fused_preds = w_trend * trend_preds + w_anchor * closest_anchors
            results.append(np.clip(fused_preds, 0.0, 1.0))  # list of (n_samples,)

        return np.column_stack(results)  # shape: (n_samples, n_test)

    except Exception as e:
        print(f"[动态锚点回归错误 - v5版本] {e}")
        base_val = y_train[-1] if len(y_train) > 0 else 0.5
        return np.full((n_samples, len(X_test)), base_val)


def _dynamic_anchor_bayesian_regression_v6(
        X_train, y_train, X_test,
        relative_time=None, max_preds=20, n_samples=20, is_opponent_view=False):
    """
    v5 加强版：中值平滑 + 时间特征 + 尾局偏移
    """
    print("_dynamic_anchor_bayesian_regression_v6开始")
    # input()
    print(is_opponent_view)
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import BayesianRidge

    try:
        if len(y_train) < 3:
            base_val = y_train[-1] if len(y_train) > 0 else 0.5
            return np.full((n_samples, len(X_test)), base_val)

        # ✅ 中值平滑
        y_train = pd.Series(y_train).rolling(window=3, center=True, min_periods=1).median().values

        # 对手视角下平滑（进一步稳定）
        if is_opponent_view:
            alpha = 0.7
            for i in range(1, len(y_train)):
                y_train[i] = alpha * y_train[i - 1] + (1 - alpha) * y_train[i]

        # === 构建特征 ===
        X_train_2d = X_train if X_train.ndim > 1 else X_train.reshape(-1, 1)

        # ✅ 增加一阶差分
        y_diff = np.diff(y_train, prepend=y_train[0])
        X_train_2d = np.column_stack((X_train_2d, y_diff.reshape(-1, 1)))

        # ✅ 显式加入时间步特征
        time_steps = np.arange(len(y_train)) / len(y_train)
        X_train_2d = np.column_stack((X_train_2d, time_steps.reshape(-1, 1)))

        # 测试点特征
        X_test_2d = X_test if X_test.ndim > 1 else X_test.reshape(-1, 1)
        t_test = np.linspace(len(y_train)/len(y_train), 1.0, len(X_test_2d))
        X_test_2d = np.column_stack((X_test_2d, y_diff[-1:].repeat(len(X_test_2d)).reshape(-1, 1)))  # diff特征
        X_test_2d = np.column_stack((X_test_2d, t_test.reshape(-1, 1)))  # 时间特征

        if X_test_2d.shape[0] > max_preds:
            X_test_2d = X_test_2d[:max_preds]

        # 模型训练
        model = BayesianRidge(fit_intercept=False)
        model.fit(X_train_2d, y_train)
        mu = model.coef_
        Sigma = model.sigma_

        # 锚点构造
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

            # ✅ 平滑预测值
            trend_preds = 0.8 * trend_preds + 0.2 * np.mean(trend_preds)

            # 锚点融合
            trend_matrix = trend_preds.reshape(-1, 1)
            anchors_matrix = anchors.reshape(1, -1)
            distances = np.abs(trend_matrix - anchors_matrix)
            closest_indices = np.argmin(distances, axis=1)
            closest_anchors = anchors[closest_indices]

            w_anchor = min(0.8, 0.3 + 0.5 * (relative_time or 0.0))
            w_trend = 1.0 - w_anchor
            fused_preds = w_trend * trend_preds + w_anchor * closest_anchors

            # ✅ 尾局偏移机制（最后阶段人工强化让步）
            if relative_time and relative_time > 0.9:
                offset = np.clip(0.05 * (relative_time - 0.9) / 0.1, 0.0, 0.05)
                fused_preds += offset

            results.append(np.clip(fused_preds, 0.0, 1.0))

        return np.column_stack(results)

    except Exception as e:
        print(f"[动态锚点回归错误 - v5版本] {e}")
        base_val = y_train[-1] if len(y_train) > 0 else 0.5
        return np.full((n_samples, len(X_test)), base_val)
    
    print("_dynamic_anchor_bayesian_regression_v6结束")
