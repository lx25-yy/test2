import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

# Function to load CSV files using pandas
def load_csv_pandas(file_name):
    try:
        return pd.read_csv(file_name, index_col=0, header=0)
    except UnicodeDecodeError:
        # 尝试常见中文编码
        for encoding in ['gbk', 'gb18030', 'big5', 'utf-8', 'latin1']:
            try:
                return pd.read_csv(file_name, index_col=0, header=0, encoding=encoding)
            except:
                continue
        # 最终尝试忽略错误
        return pd.read_csv(file_name, index_col=0, header=0, encoding='utf-8', errors='ignore')

# T-Test p-value based feature ranking
def perform_ttest(X_train, y_train):
    p_values = []
    for feature in X_train.columns:
        mdr_group = X_train[y_train == 'MDR'][feature]
        nomdr_group = X_train[y_train == 'NoMDR'][feature]
        _, p_value = ttest_ind(mdr_group, nomdr_group, equal_var=False)
        p_values.append((feature, p_value))
    sorted_p_values = sorted(p_values, key=lambda x: x[1])
    return sorted_p_values

# Recursive Feature Elimination (RFE) for feature selection
def rfe_feature_selection(X, y, estimator, n_features=5):
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

# LASSO Feature Selection with L1 regularization
def lasso_feature_selection(X, y, n_features=5):
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 训练LASSO回归模型
    lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    lasso.fit(X_scaled, y)
    # 获取特征重要性(系数绝对值)
    feature_importance = np.abs(lasso.coef_[0])
    # 选择最重要的特征
    sorted_indices = np.argsort(feature_importance)[::-1]
    selected_features = X.columns[sorted_indices[:n_features]].tolist()
    return selected_features

# 计算敏感性和特异性
def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# 自定义评分函数：平衡准确率
def balanced_accuracy_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (sensitivity + specificity) / 2

# 优化模型超参数
def optimize_model_params(model, param_grid, X, y):
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_scorer),
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy'
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=scoring,
        refit='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

# 优化的交叉验证函数
def cross_validate_classifiers(X, y, selected_features, method_name):
    # 基础分类器定义
    base_classifiers = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Naive Bayes': GaussianNB()
    }
    
    # 参数网格用于优化
    param_grids = {
        'SVM': [
            {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]},
            {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        ],
        'Random Forest': [
            {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
            {'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        ],
        'Naive Bayes': [{}]  # Naive Bayes没有需要调整的主要参数
    }
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    cv_results = {}
    all_roc_data = {}  # 存储所有分类器的ROC数据
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nEvaluating feature set using {method_name} ({len(selected_features)} features)")
    
    # 标准化整个特征集
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[selected_features])
    
    # 优化每个分类器
    optimized_classifiers = {}
    for name, clf in base_classifiers.items():
        print(f"Optimizing hyperparameters for {name} classifier...")
        best_clf, best_params = optimize_model_params(clf, param_grids[name], X_scaled, y_encoded)
        optimized_classifiers[name] = best_clf
        print(f"{name} best parameters: {best_params}")
    
    for name, clf in optimized_classifiers.items():
        print(f"Training {name} classifier...")
        
        # 用于存储每个fold的指标
        fold_accuracy = []
        fold_sensitivity = []
        fold_specificity = []
        fold_auc = []
        fold_balanced_acc = []
        
        # 用于整个数据集的预测概率（用于ROC曲线）
        y_probas = np.zeros(len(y_encoded))
        
        for train_index, test_index in skf.split(X_scaled, y_encoded):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            
            # 使用优化后的分类器
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # 存储当前fold的预测概率
            y_probas[test_index] = y_proba
            
            # 计算指标
            accuracy = np.mean(y_pred == y_test)
            sensitivity, specificity = calculate_sensitivity_specificity(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)
            balanced_acc = (sensitivity + specificity) / 2
            
            fold_accuracy.append(accuracy)
            fold_sensitivity.append(sensitivity)
            fold_specificity.append(specificity)
            fold_auc.append(auc_score)
            fold_balanced_acc.append(balanced_acc)
        
        # 存储结果
        cv_results[name] = {
            'Accuracy': fold_accuracy,
            'Sensitivity': fold_sensitivity,
            'Specificity': fold_specificity,
            'AUC': fold_auc,
            'Balanced_Accuracy': fold_balanced_acc
        }
        
        # 为ROC曲线存储数据
        fpr, tpr, _ = roc_curve(y_encoded, y_probas)
        roc_auc = auc(fpr, tpr)
        all_roc_data[name] = (fpr, tpr, roc_auc)
    
    # 绘制ROC曲线（所有分类器在一张图上）
    plot_combined_roc(all_roc_data, method_name)
    
    # 打印性能摘要
    print(f"\nEvaluation completed for {method_name} feature set:")
    for name in optimized_classifiers:
        acc_mean = np.mean(cv_results[name]['Accuracy'])
        acc_std = np.std(cv_results[name]['Accuracy'])
        acc_ci = 1.96 * acc_std / np.sqrt(len(cv_results[name]['Accuracy']))
        
        sens_mean = np.mean(cv_results[name]['Sensitivity'])
        sens_std = np.std(cv_results[name]['Sensitivity'])
        sens_ci = 1.96 * sens_std / np.sqrt(len(cv_results[name]['Sensitivity']))
        
        spec_mean = np.mean(cv_results[name]['Specificity'])
        spec_std = np.std(cv_results[name]['Specificity'])
        spec_ci = 1.96 * spec_std / np.sqrt(len(cv_results[name]['Specificity']))
        
        auc_mean = np.mean(cv_results[name]['AUC'])
        auc_std = np.std(cv_results[name]['AUC'])
        auc_ci = 1.96 * auc_std / np.sqrt(len(cv_results[name]['AUC']))
        
        bal_acc_mean = np.mean(cv_results[name]['Balanced_Accuracy'])
        bal_acc_std = np.std(cv_results[name]['Balanced_Accuracy'])
        bal_acc_ci = 1.96 * bal_acc_std / np.sqrt(len(cv_results[name]['Balanced_Accuracy']))
        
        print(f"{name} performance:")
        print(f"  Accuracy: {acc_mean:.4f} ± {acc_ci:.4f} (95% CI: {acc_mean-acc_ci:.4f}-{acc_mean+acc_ci:.4f})")
        print(f"  Sensitivity: {sens_mean:.4f} ± {sens_ci:.4f} (95% CI: {sens_mean-sens_ci:.4f}-{sens_mean+sens_ci:.4f})")
        print(f"  Specificity: {spec_mean:.4f} ± {spec_ci:.4f} (95% CI: {spec_mean-spec_ci:.4f}-{spec_mean+spec_ci:.4f})")
        print(f"  Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_ci:.4f} (95% CI: {bal_acc_mean-bal_acc_ci:.4f}-{bal_acc_mean+bal_acc_ci:.4f})")
        print(f"  AUC: {auc_mean:.4f} ± {auc_ci:.4f} (95% CI: {auc_mean-auc_ci:.4f}-{auc_mean+auc_ci:.4f})")
    
    return cv_results

# 绘制组合ROC曲线（所有分类器在一张图上）
def plot_combined_roc(roc_data, method_name):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red']
    linestyles = ['-', '--', ':']
    
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i], lw=2,
                 label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    
    # 修改标题为指定的格式 - 替换为ResNet50
    plt.title(f'ROC Curves - ResNet50 features + Clinical/Lab ({method_name.split("(")[-1].replace(")", "")})', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # 修改文件名 - 替换为ResNet50
    plt.savefig(f"ResNet50_features_Clinical_Lab_{method_name.split('(')[-1].replace(')', '').replace(' ', '_')}_combined_roc.png", dpi=300)
    plt.close()
    print(f"Saved combined ROC curve: ResNet50_features_Clinical_Lab_{method_name.split('(')[-1].replace(')', '').replace(' ', '_')}_combined_roc.png")

# 绘制箱式图（准确率、敏感性、特异性、平衡准确率）
def plot_cv_results(cv_results, method_name):
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Balanced_Accuracy']
    plt.figure(figsize=(18, 4))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 4, i)
        metric_data = []
        labels = []
        for clf_name, results in cv_results.items():
            metric_data.append(results[metric])
            labels.append(clf_name)
        
        # 创建箱线图
        box = plt.boxplot(metric_data, patch_artist=True, widths=0.6)
        
        # 设置不同颜色
        colors = ['lightblue', 'lightgreen', 'salmon']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加均值点
        means = [np.mean(data) for data in metric_data]
        for j, mean in enumerate(means):
            plt.plot(j+1, mean, 'ro', markersize=8)
        
        plt.xticks(range(1, len(labels) + 1), labels, fontsize=10)
        
        # 修改标题为指定的格式 - 替换为ResNet50
        plt.title(f"{metric} - ResNet50 features + Clinical/Lab ({method_name.split('(')[-1].replace(')', '')})", fontsize=12)
        plt.ylabel(metric, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 添加y轴标签
        plt.ylim(0.0, 1.0)
    
    plt.tight_layout()
    
    # 修改保存文件名 - 替换为ResNet50
    method_short = method_name.split('(')[-1].replace(')', '').replace(' ', '_')
    plt.savefig(f"ResNet50_features_Clinical_Lab_{method_short}_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved boxplot: ResNet50_features_Clinical_Lab_{method_short}_boxplot.png")

if __name__ == "__main__":
    print("Starting program execution...")
    
    # 数据加载
    data_matrix_file = 'DL特征图片归一化.csv'
    class_label_file = '分类.csv'
    clinical_feature_file = '临床特征.csv'  # 临床特征文件
    
    print("Loading data matrix...")
    data_matrix = load_csv_pandas(data_matrix_file)
    print("Loading class labels...")
    class_labels = load_csv_pandas(class_label_file)
    print("Loading clinical features...")
    clinical_features = load_csv_pandas(clinical_feature_file)
    
    # 获取样本名称和特征名称
    sample_names = data_matrix.columns.values
    feature_names = data_matrix.index.values
    
    # 转置数据矩阵
    data_matrix = data_matrix.T
    
    # ====== 优化：更高效的缺失值处理 ======
    print("\n===== Handling Missing Values =====")
    # 直接指定铁相关特征
    iron_features = ["铁", "铁蛋白", "总铁结合力"]
    
    # 检查这些特征是否存在于临床特征中
    existing_iron_features = [col for col in iron_features if col in clinical_features.columns]
    
    if existing_iron_features:
        print(f"Iron-related features requiring imputation: {existing_iron_features}")
        
        # 检查缺失情况
        missing_count = clinical_features[existing_iron_features].isnull().sum().sum()
        print(f"Total missing values in iron-related features: {missing_count}")
        
        if missing_count > 0:
            print("Attempting fast multiple imputation...")
            try:
                # 使用轻量级模型和更少迭代
                imputer = IterativeImputer(
                    estimator=ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=42),
                    max_iter=5,  # 减少迭代次数
                    random_state=42
                )
                
                # 只对铁相关特征进行插补
                clinical_features[existing_iron_features] = imputer.fit_transform(clinical_features[existing_iron_features])
                print("Fast multiple imputation completed!")
                
                # 打印插补后的统计信息
                print("\nStatistics after imputation:")
                for feature in existing_iron_features:
                    print(f"{feature}: Mean={clinical_features[feature].mean():.2f}, Median={clinical_features[feature].median():.2f}")
            except Exception as e:
                print(f"Multiple imputation failed: {e}, switching to median imputation")
                # 降级到简单插补
                imputer = SimpleImputer(strategy='median')
                clinical_features[existing_iron_features] = imputer.fit_transform(clinical_features[existing_iron_features])
                print("Median imputation completed!")
        else:
            print("No missing values in iron-related features, skipping imputation")
    else:
        print("No iron-related features detected, skipping imputation")
    
    # ====== 合并数据 ======
    print("\n===== Data Merging =====")
    # 创建样本-类别映射
    sample_class_map = {idx: row['Class'] for idx, row in class_labels.iterrows()}
    class_labels_list = [sample_class_map.get(name, None) for name in sample_names]
    
    # 创建合并数据
    merged_data = pd.DataFrame(data_matrix.values, index=sample_names, columns=feature_names)
    merged_data.insert(0, 'Class', class_labels_list)
    
    # 特征选择和分析部分
    X = merged_data.iloc[:, 1:]  # 所有特征(不包括Class列)
    y = merged_data.iloc[:, 0]   # Class列
    
    # 通用函数：从临床特征中选择5个最重要的特征
    def select_top_clinical_features(clinical_df, y, n_features=5):
        """
        从临床特征中选择最重要的5个特征
        使用T-Test方法进行特征选择
        """
        # 确保有足够的特征可选
        n_available = min(n_features, clinical_df.shape[1])
        
        # 如果特征数不足5个，则全部选择
        if n_available < n_features:
            print(f"Warning: Only {n_available} clinical features available, selecting all")
            return clinical_df.columns.tolist()
        
        # 使用T-Test选择最重要的特征
        sorted_clinical = perform_ttest(clinical_df, y)
        top_clinical_features = [f[0] for f in sorted_clinical[:n_features]]
        return top_clinical_features
    
    print("\n===== T-Test Feature Selection =====")
    # 从DL特征中选择5个
    sorted_features = perform_ttest(X, y)
    dl_top_features = [f[0] for f in sorted_features[:5]]
    print(f"Top 5 DL features selected by T-Test: {dl_top_features}")
    
    # 从临床特征中选择5个
    clinical_top_features = select_top_clinical_features(clinical_features, y)
    print(f"Top 5 clinical features selected by T-Test: {clinical_top_features}")
    
    # 合并筛选特征
    combined_features = pd.concat([X[dl_top_features], clinical_features[clinical_top_features]], axis=1)
    print(f"Combined features (T-Test selection): {list(combined_features.columns)}")
    print(f"Total features: {len(combined_features.columns)}")
    
    # 修改评估名称 - 保持原始名称不变
    cv_results_ttest = cross_validate_classifiers(combined_features, y, combined_features.columns, "ResNet50 features + Clinical/Lab (T-Test)")
    plot_cv_results(cv_results_ttest, "ResNet50 features + Clinical/Lab (T-Test)")
    
    print("\n===== RFE Feature Selection =====")
    X1 = X.iloc[:, :2000]  # 前2000个DL特征
    
    # 从DL特征中选择5个
    selected_dl_features_rfe = rfe_feature_selection(X1, y, SVC(kernel='linear', random_state=42, class_weight='balanced'))
    print(f"DL features selected by RFE: {selected_dl_features_rfe}")
    
    # 从临床特征中选择5个
    selected_clinical_features_rfe = rfe_feature_selection(
        clinical_features, y, 
        SVC(kernel='linear', random_state=42, class_weight='balanced'),
        n_features=min(5, clinical_features.shape[1])
    )
    print(f"Clinical features selected by RFE: {selected_clinical_features_rfe}")
    
    # 合并筛选特征
    combined_features_rfe = pd.concat([X1[selected_dl_features_rfe], clinical_features[selected_clinical_features_rfe]], axis=1)
    print(f"Combined features (RFE selection): {list(combined_features_rfe.columns)}")
    print(f"Total features: {len(combined_features_rfe.columns)}")
    
    # 修改评估名称 - 保持原始名称不变
    cv_results_rfe_svm = cross_validate_classifiers(combined_features_rfe, y, combined_features_rfe.columns, "ResNet50 features + Clinical/Lab (RFE)")
    plot_cv_results(cv_results_rfe_svm, "ResNet50 features + Clinical/Lab (RFE)")
    
    print("\n===== LASSO Feature Selection =====")
    # 从DL特征中选择5个
    selected_dl_features_lasso = lasso_feature_selection(X1, y)
    print(f"DL features selected by LASSO: {selected_dl_features_lasso}")
    
    # 从临床特征中选择5个
    selected_clinical_features_lasso = lasso_feature_selection(clinical_features, y, n_features=min(5, clinical_features.shape[1]))
    print(f"Clinical features selected by LASSO: {selected_clinical_features_lasso}")
    
    # 合并筛选特征
    combined_features_lasso = pd.concat([X1[selected_dl_features_lasso], clinical_features[selected_clinical_features_lasso]], axis=1)
    print(f"Combined features (LASSO selection): {list(combined_features_lasso.columns)}")
    print(f"Total features: {len(combined_features_lasso.columns)}")
    
    # 修改评估名称 - 保持原始名称不变
    cv_results_lasso = cross_validate_classifiers(combined_features_lasso, y, combined_features_lasso.columns, "ResNet50 features + Clinical/Lab (LASSO)")
    plot_cv_results(cv_results_lasso, "ResNet50 features + Clinical/Lab (LASSO)")
    
    print("\nAll evaluations completed successfully!")
