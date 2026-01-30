"""
ZeroED方法改进实现
解决三个核心问题：
1. 错误模式函数过于严格
2. 召回率低
3. 方法不够成熟
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from collections import defaultdict
import re


# ==================== 问题1：多canonical模式识别 ====================

def identify_multiple_canonical_patterns(cluster_values, cluster_descriptions, 
                                        canonical_scores, threshold=0.65,
                                        min_cluster_size=10):
    """
    识别多个canonical模式，而不是只选择一个
    
    改进点：
    - 允许多个聚类都是正确的
    - 只有明显错误的才标记为error
    """
    canonical_patterns = []
    
    for idx, (values, score) in enumerate(zip(cluster_values, canonical_scores)):
        if len(values) < min_cluster_size:
            continue
            
        # 放宽阈值，允许多个canonical
        if score >= threshold:
            pattern = {
                'cluster_id': idx,
                'pattern_description': cluster_descriptions.get(idx, ''),
                'example_values': values[:10],
                'canonical_score': score,
                'cluster_size': len(values)
            }
            canonical_patterns.append(pattern)
    
    return canonical_patterns


def identify_strict_error_patterns(cluster_values, cluster_descriptions,
                                   canonical_patterns, incompatibility_threshold=0.9):
    """
    严格识别error模式，只标记明显的错误
    
    改进点：
    - 提高不兼容性阈值（0.9而不是0.8）
    - 只标记明显错误的模式（如nan、格式混乱）
    - 引入"uncertain pattern"概念
    """
    error_patterns = []
    uncertain_patterns = []
    
    for idx, values in enumerate(cluster_values):
        # 跳过已经是canonical的
        if any(p['cluster_id'] == idx for p in canonical_patterns):
            continue
        
        if len(values) < 10:
            continue
        
        # 计算与所有canonical的不兼容性
        max_incompatibility = 0
        for canonical in canonical_patterns:
            incompatibility = calculate_incompatibility(values, canonical['example_values'])
            max_incompatibility = max(max_incompatibility, incompatibility)
        
        # 明显错误：不兼容性很高
        if max_incompatibility >= incompatibility_threshold:
            error_patterns.append({
                'cluster_id': idx,
                'pattern_description': cluster_descriptions.get(idx, ''),
                'example_values': values[:10],
                'incompatibility_score': max_incompatibility,
                'cluster_size': len(values)
            })
        # 不确定：不兼容性中等
        elif max_incompatibility >= 0.6:
            uncertain_patterns.append({
                'cluster_id': idx,
                'pattern_description': cluster_descriptions.get(idx, ''),
                'example_values': values[:10],
                'incompatibility_score': max_incompatibility,
                'cluster_size': len(values)
            })
    
    return error_patterns, uncertain_patterns


def calculate_incompatibility(values1, values2):
    """计算两组值的不兼容性"""
    # 简化实现：基于字符串相似度
    from difflib import SequenceMatcher
    
    similarities = []
    for v1 in values1[:5]:
        for v2 in values2[:5]:
            sim = SequenceMatcher(None, str(v1), str(v2)).ratio()
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities) if similarities else 0
    return 1 - avg_similarity


# ==================== 问题2：提高召回率 ====================

class ActiveLearningSampler:
    """
    主动学习采样器
    选择最有价值的样本进行标注
    """
    
    def __init__(self, model=None):
        self.model = model
    
    def select_samples(self, unlabeled_data, features, n_samples=100, strategy='combined'):
        """
        选择样本进行标注
        
        策略：
        1. uncertainty: 模型不确定的样本
        2. diversity: 多样化的样本
        3. representative: 代表性样本
        4. combined: 组合策略
        """
        if strategy == 'uncertainty':
            return self.uncertainty_sampling(unlabeled_data, features, n_samples)
        elif strategy == 'diversity':
            return self.diversity_sampling(unlabeled_data, features, n_samples)
        elif strategy == 'representative':
            return self.representative_sampling(unlabeled_data, features, n_samples)
        else:
            return self.combined_sampling(unlabeled_data, features, n_samples)
    
    def uncertainty_sampling(self, unlabeled_data, features, n_samples):
        """选择模型最不确定的样本"""
        if self.model is None:
            # 如果没有模型，随机选择
            indices = np.random.choice(len(unlabeled_data), n_samples, replace=False)
            return [unlabeled_data[i] for i in indices]
        
        # 获取预测概率
        probs = self.model.predict_proba(features)
        
        # 计算不确定性（熵）
        uncertainties = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        
        # 选择最不确定的样本
        top_indices = np.argsort(uncertainties)[-n_samples:]
        return [unlabeled_data[i] for i in top_indices]
    
    def diversity_sampling(self, unlabeled_data, features, n_samples):
        """选择多样化的样本"""
        from sklearn.cluster import KMeans
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(features)
        
        # 选择每个聚类的中心点
        selected_indices = []
        for i in range(n_samples):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) > 0:
                # 选择离聚类中心最近的点
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(features[cluster_indices] - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return [unlabeled_data[i] for i in selected_indices]
    
    def representative_sampling(self, unlabeled_data, features, n_samples):
        """选择代表性样本（密度高的区域）"""
        from sklearn.neighbors import NearestNeighbors
        
        # 计算每个点的密度（k近邻距离的倒数）
        nbrs = NearestNeighbors(n_neighbors=10).fit(features)
        distances, _ = nbrs.kneighbors(features)
        densities = 1 / (np.mean(distances, axis=1) + 1e-10)
        
        # 选择密度最高的样本
        top_indices = np.argsort(densities)[-n_samples:]
        return [unlabeled_data[i] for i in top_indices]
    
    def combined_sampling(self, unlabeled_data, features, n_samples):
        """组合多种策略"""
        n_per_strategy = n_samples // 3
        
        samples = []
        samples.extend(self.uncertainty_sampling(unlabeled_data, features, n_per_strategy))
        samples.extend(self.diversity_sampling(unlabeled_data, features, n_per_strategy))
        samples.extend(self.representative_sampling(unlabeled_data, features, n_samples - 2*n_per_strategy))
        
        # 去重
        seen = set()
        unique_samples = []
        for sample in samples:
            sample_id = id(sample)
            if sample_id not in seen:
                seen.add(sample_id)
                unique_samples.append(sample)
        
        return unique_samples


class AnomalyFeatureExtractor:
    """
    异常检测特征提取器
    增加异常检测特征以提高召回率
    """
    
    def __init__(self):
        self.isolation_forest = None
        self.lof = None
    
    def fit(self, data):
        """训练异常检测器"""
        # Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(data)
        
        # Local Outlier Factor
        self.lof = LocalOutlierFactor(novelty=True, contamination=0.1)
        self.lof.fit(data)
    
    def extract_features(self, value, column_data):
        """提取异常检测特征"""
        features = []
        
        # 特征1：Isolation Forest异常分数
        if self.isolation_forest is not None:
            iso_score = self.isolation_forest.score_samples([value])[0]
            features.append(iso_score)
        else:
            features.append(0)
        
        # 特征2：Local Outlier Factor
        if self.lof is not None:
            lof_score = self.lof.score_samples([value])[0]
            features.append(lof_score)
        else:
            features.append(0)
        
        # 特征3：与列均值的距离
        if len(column_data) > 0:
            mean_distance = np.abs(value - np.mean(column_data))
            features.append(mean_distance)
        else:
            features.append(0)
        
        # 特征4：与列中位数的距离
        if len(column_data) > 0:
            median_distance = np.abs(value - np.median(column_data))
            features.append(median_distance)
        else:
            features.append(0)
        
        return features


# ==================== 问题3：引入最新技术 ====================

class LLMBasedDetector:
    """
    基于LLM的Few-shot检测器
    使用in-context learning进行错误检测
    """
    
    def __init__(self, query_llm_func):
        self.query_llm = query_llm_func
        self.few_shot_examples = {'correct': [], 'error': []}
    
    def set_examples(self, correct_examples, error_examples):
        """设置few-shot示例"""
        self.few_shot_examples['correct'] = correct_examples
        self.few_shot_examples['error'] = error_examples
    
    def detect(self, value, column_name, context=None):
        """
        使用LLM检测错误
        
        返回：(is_error, confidence)
        """
        prompt = self._build_prompt(value, column_name, context)
        response = self.query_llm(prompt)
        
        # 解析响应
        is_error, confidence = self._parse_response(response)
        return is_error, confidence
    
    def _build_prompt(self, value, column_name, context):
        """构建few-shot prompt"""
        prompt = f"""You are a data quality expert. Your task is to determine if a value in a database column is correct or contains an error.

Column name: {column_name}

Correct examples from this column:
"""
        for i, ex in enumerate(self.few_shot_examples['correct'][:5], 1):
            prompt += f"{i}. {ex}\n"
        
        prompt += "\nError examples from this column:\n"
        for i, ex in enumerate(self.few_shot_examples['error'][:5], 1):
            prompt += f"{i}. {ex}\n"
        
        if context:
            prompt += f"\nContext (related columns):\n{context}\n"
        
        prompt += f"""
Now, analyze this value:
Value: {value}

Is this value correct or does it contain an error?
Provide your answer in the following format:
Answer: [correct/error]
Confidence: [0.0-1.0]
Reason: [brief explanation]
"""
        return prompt
    
    def _parse_response(self, response):
        """解析LLM响应"""
        # 提取答案
        answer_match = re.search(r'Answer:\s*(correct|error)', response, re.IGNORECASE)
        is_error = answer_match.group(1).lower() == 'error' if answer_match else False
        
        # 提取置信度
        conf_match = re.search(r'Confidence:\s*([0-9.]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        return is_error, confidence


class EnsembleDetector:
    """
    集成检测器
    结合多种检测方法
    """
    
    def __init__(self):
        self.detectors = []
        self.weights = []
    
    def add_detector(self, detector, weight=1.0):
        """添加检测器"""
        self.detectors.append(detector)
        self.weights.append(weight)
    
    def detect(self, value, column_name, context=None):
        """
        集成检测
        
        返回：(is_error, confidence)
        """
        votes = []
        confidences = []
        
        for detector, weight in zip(self.detectors, self.weights):
            try:
                is_error, conf = detector.detect(value, column_name, context)
                votes.append(1 if is_error else 0)
                confidences.append(conf * weight)
            except Exception as e:
                # 如果某个检测器失败，跳过
                continue
        
        if not votes:
            return False, 0.0
        
        # 加权投票
        weighted_vote = np.average(votes, weights=confidences)
        avg_confidence = np.mean(confidences)
        
        is_error = weighted_vote > 0.5
        return is_error, avg_confidence


class SelfTrainingDetector:
    """
    自训练检测器
    使用伪标签扩充训练集
    """
    
    def __init__(self, base_model, confidence_threshold=0.9):
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.pseudo_labeled_data = []
    
    def fit(self, X_labeled, y_labeled, X_unlabeled, max_iterations=5):
        """
        自训练过程
        
        参数：
        - X_labeled: 已标注数据的特征
        - y_labeled: 已标注数据的标签
        - X_unlabeled: 未标注数据的特征
        - max_iterations: 最大迭代次数
        """
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()
        
        for iteration in range(max_iterations):
            # 训练模型
            self.base_model.fit(X_train, y_train)
            
            # 预测未标注数据
            if len(X_pool) == 0:
                break
            
            probs = self.base_model.predict_proba(X_pool)
            predictions = self.base_model.predict(X_pool)
            
            # 选择高置信度的预测作为伪标签
            max_probs = np.max(probs, axis=1)
            high_conf_mask = max_probs >= self.confidence_threshold
            
            if not np.any(high_conf_mask):
                break
            
            # 添加伪标签数据到训练集
            X_pseudo = X_pool[high_conf_mask]
            y_pseudo = predictions[high_conf_mask]
            
            X_train = np.vstack([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])
            
            # 从池中移除已标注的数据
            X_pool = X_pool[~high_conf_mask]
            
            print(f"Iteration {iteration+1}: Added {len(X_pseudo)} pseudo-labeled samples")
        
        # 最终训练
        self.base_model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """预测"""
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        return self.base_model.predict_proba(X)


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    
    # 1. 多canonical模式识别
    print("=" * 60)
    print("示例1：多canonical模式识别")
    print("=" * 60)
    
    cluster_values = [
        ['2023-01-01', '2023-02-15', '2023-03-20'],  # 聚类0: ISO格式
        ['01/01/2023', '02/15/2023', '03/20/2023'],  # 聚类1: 美国格式
        ['nan', 'null', 'N/A'],                       # 聚类2: 缺失值
    ]
    cluster_descriptions = {
        0: 'ISO date format (YYYY-MM-DD)',
        1: 'US date format (MM/DD/YYYY)',
        2: 'Missing values'
    }
    canonical_scores = [0.95, 0.90, 0.10]
    
    canonical_patterns = identify_multiple_canonical_patterns(
        cluster_values, cluster_descriptions, canonical_scores, threshold=0.65
    )
    
    print(f"识别出 {len(canonical_patterns)} 个canonical模式：")
    for p in canonical_patterns:
        print(f"  - 聚类{p['cluster_id']}: {p['pattern_description']} (分数={p['canonical_score']:.2f})")
    
    # 2. 主动学习采样
    print("\n" + "=" * 60)
    print("示例2：主动学习采样")
    print("=" * 60)
    
    sampler = ActiveLearningSampler()
    unlabeled_data = list(range(1000))
    features = np.random.randn(1000, 10)
    
    selected = sampler.select_samples(unlabeled_data, features, n_samples=50, strategy='combined')
    print(f"选择了 {len(selected)} 个样本进行标注")
    
    # 3. LLM检测器（需要实际的LLM函数）
    print("\n" + "=" * 60)
    print("示例3：LLM检测器")
    print("=" * 60)
    
    def mock_llm(prompt):
        return "Answer: error\nConfidence: 0.85\nReason: Invalid date format"
    
    llm_detector = LLMBasedDetector(mock_llm)
    llm_detector.set_examples(
        correct_examples=['2023-01-01', '2023-02-15'],
        error_examples=['2023-13-01', 'invalid']
    )
    
    is_error, conf = llm_detector.detect('2023-99-99', 'date_column')
    print(f"检测结果: {'错误' if is_error else '正确'}, 置信度={conf:.2f}")


if __name__ == '__main__':
    example_usage()

