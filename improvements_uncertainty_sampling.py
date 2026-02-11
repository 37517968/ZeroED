"""
不确定性采样方法改进模块
实现多种不确定性度量方法，提升主动学习的样本选择质量
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy
from collections import Counter


class UncertaintySamplingEnsemble:
    """
    集成多种不确定性采样方法
    """
    
    def __init__(self, methods=None, weights=None, adaptive=True):
        """
        Args:
            methods: 使用的方法列表，默认使用所有方法
            weights: 各方法的权重，默认均等
            adaptive: 是否使用自适应权重调整
        """
        if methods is None:
            methods = ['least_confidence', 'margin', 'entropy', 'feature_diversity']
        
        self.methods = methods
        
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        self.weights = np.array(weights)
        
        self.adaptive = adaptive
        self.performance_history = []
    
    def least_confidence(self, probas):
        """
        最小置信度采样
        uncertainty = 1 - max(P(y|x))
        
        Args:
            probas: 预测概率数组 (n_samples, n_classes)
        
        Returns:
            uncertainty_scores: 不确定性分数数组
        """
        if len(probas.shape) == 1:
            # 二分类，只有一个概率值
            max_proba = np.maximum(probas, 1 - probas)
        else:
            max_proba = np.max(probas, axis=1)
        
        return 1 - max_proba
    
    def margin_sampling(self, probas):
        """
        边界采样
        uncertainty = 1 - (P(y1|x) - P(y2|x))
        其中y1, y2是概率最高的两个类别
        
        Args:
            probas: 预测概率数组 (n_samples, n_classes)
        
        Returns:
            uncertainty_scores: 不确定性分数数组
        """
        if len(probas.shape) == 1:
            # 二分类
            margin = np.abs(probas - 0.5) * 2
            return 1 - margin
        else:
            # 多分类
            sorted_probas = np.sort(probas, axis=1)
            margin = sorted_probas[:, -1] - sorted_probas[:, -2]
            return 1 - margin
    
    def entropy_sampling(self, probas):
        """
        熵采样
        uncertainty = -Σ P(yi|x) * log(P(yi|x))
        
        Args:
            probas: 预测概率数组 (n_samples, n_classes)
        
        Returns:
            uncertainty_scores: 不确定性分数数组（归一化到0-1）
        """
        if len(probas.shape) == 1:
            # 二分类
            probas_2d = np.column_stack([1 - probas, probas])
        else:
            probas_2d = probas
        
        # 计算熵
        entropies = entropy(probas_2d.T)
        
        # 归一化到0-1
        max_entropy = np.log(probas_2d.shape[1])
        if max_entropy > 0:
            entropies = entropies / max_entropy
        
        return entropies
    
    def feature_diversity_sampling(self, features, probas, n_clusters=10):
        """
        基于特征多样性的采样
        结合预测不确定性和特征空间的多样性
        
        Args:
            features: 特征数组 (n_samples, n_features)
            probas: 预测概率数组 (n_samples, n_classes)
            n_clusters: 聚类数量
        
        Returns:
            uncertainty_scores: 不确定性分数数组
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances
        
        # 基础不确定性（使用熵）
        base_uncertainty = self.entropy_sampling(probas)
        
        # 特征多样性分数
        if len(features) <= n_clusters:
            diversity_scores = np.ones(len(features))
        else:
            # 使用KMeans聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # 计算每个样本到其聚类中心的距离
            distances = pairwise_distances(features, kmeans.cluster_centers_)
            min_distances = np.min(distances, axis=1)
            
            # 归一化距离
            if np.max(min_distances) > 0:
                diversity_scores = min_distances / np.max(min_distances)
            else:
                diversity_scores = np.ones(len(features))
        
        # 组合不确定性和多样性
        combined_scores = 0.7 * base_uncertainty + 0.3 * diversity_scores
        
        return combined_scores
    
    def query_by_committee(self, X, models):
        """
        Query-by-Committee (QBC)
        训练多个模型，计算预测的分歧度
        
        Args:
            X: 特征数组
            models: 已训练的模型列表
        
        Returns:
            uncertainty_scores: 不确定性分数数组
        """
        if len(models) < 2:
            raise ValueError("QBC需要至少2个模型")
        
        # 收集所有模型的预测
        all_predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if len(pred.shape) == 1:
                    pred = np.column_stack([1 - pred, pred])
                all_predictions.append(pred)
            else:
                pred = model.predict(X)
                all_predictions.append(pred)
        
        # 计算投票熵
        n_samples = len(X)
        vote_entropies = np.zeros(n_samples)
        
        for i in range(n_samples):
            # 收集所有模型对样本i的预测
            votes = []
            for pred in all_predictions:
                if len(pred.shape) == 2:
                    # 概率预测，取最大概率的类别
                    votes.append(np.argmax(pred[i]))
                else:
                    # 直接预测
                    votes.append(pred[i])
            
            # 计算投票分布的熵
            vote_counts = Counter(votes)
            vote_probs = np.array([vote_counts.get(c, 0) for c in range(2)]) / len(votes)
            vote_entropies[i] = entropy(vote_probs + 1e-10)
        
        # 归一化
        max_entropy = np.log(2)
        if max_entropy > 0:
            vote_entropies = vote_entropies / max_entropy
        
        return vote_entropies
    
    def compute_uncertainty(self, probas, features=None, models=None):
        """
        计算综合不确定性分数
        
        Args:
            probas: 预测概率数组
            features: 特征数组（用于feature_diversity方法）
            models: 模型列表（用于QBC方法）
        
        Returns:
            uncertainty_scores: 综合不确定性分数
        """
        all_scores = []
        
        for method in self.methods:
            if method == 'least_confidence':
                scores = self.least_confidence(probas)
            elif method == 'margin':
                scores = self.margin_sampling(probas)
            elif method == 'entropy':
                scores = self.entropy_sampling(probas)
            elif method == 'feature_diversity':
                if features is not None:
                    scores = self.feature_diversity_sampling(features, probas)
                else:
                    scores = self.entropy_sampling(probas)
            elif method == 'qbc':
                if models is not None and len(models) >= 2:
                    scores = self.query_by_committee(features, models)
                else:
                    scores = self.entropy_sampling(probas)
            else:
                raise ValueError(f"未知的采样方法: {method}")
            
            all_scores.append(scores)
        
        # 加权组合
        all_scores = np.array(all_scores)
        combined_scores = np.dot(self.weights, all_scores)
        
        return combined_scores
    
    def update_weights(self, method_performances):
        """
        根据各方法的性能更新权重（自适应）
        
        Args:
            method_performances: 各方法的性能字典 {method: f1_score}
        """
        if not self.adaptive:
            return
        
        # 根据性能调整权重
        performances = np.array([method_performances.get(m, 0.5) for m in self.methods])
        
        # Softmax归一化
        exp_perf = np.exp(performances * 2)  # 放大差异
        new_weights = exp_perf / np.sum(exp_perf)
        
        # 平滑更新（避免剧烈变化）
        self.weights = 0.7 * self.weights + 0.3 * new_weights
        
        # 归一化
        self.weights = self.weights / np.sum(self.weights)


def stratified_uncertainty_sampling(uncertainty_scores, n_select, ratios=[0.5, 0.3, 0.2]):
    """
    分层不确定性采样
    
    Args:
        uncertainty_scores: 不确定性分数数组
        n_select: 要选择的样本数
        ratios: 高/中/低不确定性层的采样比例
    
    Returns:
        selected_indices: 选中的样本索引
    """
    # 按不确定性分数排序
    sorted_indices = np.argsort(uncertainty_scores)[::-1]
    
    # 分层
    n_samples = len(uncertainty_scores)
    high_end = int(n_samples * 0.33)
    mid_end = int(n_samples * 0.67)
    
    high_uncertainty_indices = sorted_indices[:high_end]
    mid_uncertainty_indices = sorted_indices[high_end:mid_end]
    low_uncertainty_indices = sorted_indices[mid_end:]
    
    # 按比例采样
    n_high = int(n_select * ratios[0])
    n_mid = int(n_select * ratios[1])
    n_low = n_select - n_high - n_mid
    
    # 确保不超过各层的样本数
    n_high = min(n_high, len(high_uncertainty_indices))
    n_mid = min(n_mid, len(mid_uncertainty_indices))
    n_low = min(n_low, len(low_uncertainty_indices))
    
    # 随机采样（在各层内）
    selected = []
    if n_high > 0:
        selected.extend(np.random.choice(high_uncertainty_indices, n_high, replace=False))
    if n_mid > 0:
        selected.extend(np.random.choice(mid_uncertainty_indices, n_mid, replace=False))
    if n_low > 0:
        selected.extend(np.random.choice(low_uncertainty_indices, n_low, replace=False))
    
    return np.array(selected)


def adaptive_sampling_strategy(iteration, total_iterations, base_method='entropy'):
    """
    自适应采样策略：根据迭代阶段选择合适的采样方法
    
    Args:
        iteration: 当前迭代次数
        total_iterations: 总迭代次数
        base_method: 基础方法
    
    Returns:
        method_weights: 各方法的权重字典
    """
    progress = iteration / total_iterations
    
    if progress < 0.3:
        # 初期：强调多样性
        weights = {
            'least_confidence': 0.1,
            'margin': 0.1,
            'entropy': 0.2,
            'feature_diversity': 0.6
        }
    elif progress < 0.7:
        # 中期：强调边界样本
        weights = {
            'least_confidence': 0.2,
            'margin': 0.5,
            'entropy': 0.2,
            'feature_diversity': 0.1
        }
    else:
        # 后期：强调困难样本
        weights = {
            'least_confidence': 0.3,
            'margin': 0.2,
            'entropy': 0.5,
            'feature_diversity': 0.0
        }
    
    return weights

