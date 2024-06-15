import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class CustomerSegmentation:
    def __init__(self, data, target_column, n_clusters=3, save_dir='models'):
        self.data = data
        self.target_column = target_column
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.best_metric = float('-inf')
        self.best_classifier = None
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def preprocess_data(self):
        features = self.data.drop(columns=[self.target_column])
        scaled_features = self.scaler.fit_transform(features)
        self.data[features.columns] = scaled_features

    def choose_clustering(self, method):
        if method == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters)
        elif method == 'kmedoid':
            self.model = KMedoids(n_clusters=self.n_clusters)

    def perform_clustering(self):
        features = self.data.drop(columns=[self.target_column])
        self.cluster_labels = self.model.fit_predict(features)
        self.data['customer_type'] = self.cluster_labels

    def choose_classifier(self):
        classifiers = {
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'CatBoost': CatBoostClassifier(verbose=0),
            'LightGBM': LGBMClassifier()
        }
        return classifiers

    def train_classifiers(self):
        X = self.data.drop(columns=[self.target_column, 'customer_type'])
        y = self.data['customer_type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifiers = self.choose_classifier()
        for name, classifier in tqdm(classifiers.items(), desc="Training Classifiers"):
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            f1 = f1_score(y_test, predictions, average='macro')
            if f1 > self.best_metric:
                self.best_metric = f1
                self.best_classifier = (name, classifier)

    def save_results(self):
        if self.best_classifier:
            model_path = os.path.join(self.save_dir, f'{self.best_classifier[0]}_best_model.joblib')
            joblib.dump(self.best_classifier[1], model_path)
            print(f'Best model ({self.best_classifier[0]}) saved with F1 Score: {self.best_metric:.4f}')
        else:
            print("No classifier was trained successfully.")
            
    def plot_feature_importance(self, threshold_ratio=0.1):
        if self.best_classifier and hasattr(self.best_classifier[1], 'feature_importances_'):
            importances = self.best_classifier[1].feature_importances_
            features = self.data.columns.drop([self.target_column, 'customer_type'])
            indices = np.argsort(importances)[::-1]

            # Only keep features whose importance is greater than the threshold
            threshold = threshold_ratio * max(importances)
            significant_indices = [i for i in indices if importances[i] > threshold]
            
            # Selecting significant features and their importances
            significant_features = [features[i] for i in significant_indices]
            significant_importances = [importances[i] for i in significant_indices]

            plt.figure(figsize=(12, 6))
            plt.title('Significant Feature Importances', fontsize=16)
            bars = plt.bar(significant_features, significant_importances, color='steelblue')

            # Adding text annotations on bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

            plt.xticks(rotation=45, ha='right', fontsize=12)  # ha: horizontal alignment
            plt.yticks(fontsize=12)
            plt.xlabel('Features', fontsize=14)
            plt.ylabel('Importance', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        else:
            print("The best model does not support feature importance or is not set.")
            
    def create_customer_types_for_data(self, new_data):
            # Load the best model
        if self.best_classifier:
            model = self.best_classifier[1]
        else:
            raise ValueError("No classifier has been trained.")

        # Predict customer types for new data
        predictions = self.predict_customer_types(new_data)
        new_data['customer_type'] = predictions
        return new_data
    
    def predict_customer_types(self, new_data):
        new_features = new_data.drop(columns=[self.target_column])
        scaled_new_features = self.scaler.transform(new_features)
        predictions = self.best_classifier[1].predict(scaled_new_features)
        return predictions

    
    def load_model(self, model_name):
        model_path = os.path.join(self.save_dir, model_name)
        return joblib.load(model_path)

    def run_pipeline(self, clustering_method):
        self.preprocess_data()
        self.choose_clustering(clustering_method)
        self.perform_clustering()
        self.train_classifiers()
        self.save_results()
        self.plot_feature_importance()
    
    
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class SegmentationViz:
    def __init__(self, data, target_column, n_clusters=3):
        self.data = data
        self.target_column = target_column
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.pca = PCA(n_components=3)

    def preprocess_data(self):
        # Handle categorical columns by adding a new category for NaNs
        for column in self.data.select_dtypes(include=['category']).columns:
            if self.data[column].isnull().any():
                self.data[column] = self.data[column].cat.add_categories(['Missing']).fillna('Missing')

        # Handle numerical columns
        for column in self.data.select_dtypes(include=[np.number]).columns:
            self.data[column] = self.data[column].fillna(0)

        # Standardize the numerical data except the target and non-standardizable features
        features = self.data.columns.difference([self.target_column, 'county', 'installed_capacity'])
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data[features])

    def perform_clustering(self):
        self.cluster_labels = self.model.fit_predict(self.scaled_data)
        # Calculate silhouette score
        self.silhouette = silhouette_score(self.scaled_data, self.cluster_labels)

    def reduce_dimensions(self):
        self.pca_results = self.pca.fit_transform(self.scaled_data)

    def plot_clusters(self):
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))

        # 3D PCA Plot
        ax3d = fig.add_subplot(221, projection='3d')
        scatter = ax3d.scatter(self.pca_results[:, 0], self.pca_results[:, 1], self.pca_results[:, 2],
                               c=self.cluster_labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
        ax3d.set_title(f'3D PCA of Customer Data - Silhouette: {self.silhouette:.2f}')
        ax3d.set_xlabel('Principal Component 1')
        ax3d.set_ylabel('Principal Component 2')
        ax3d.set_zlabel('Principal Component 3')
        fig.colorbar(scatter, ax=ax3d, label='Cluster Label')

        # County distribution in clusters
        self.data['Cluster'] = self.cluster_labels
        sns.countplot(data=self.data, x='county', hue='Cluster', ax=ax[0, 1])
        ax[0, 1].set_title('Distribution of Counties in Clusters')
        ax[0, 1].set_xlabel('County')
        ax[0, 1].set_ylabel('Count')
        ax[0, 1].legend(title='Cluster', loc='upper right')

        # Mean and positive standard deviation of target per cluster
        cluster_target_stats = self.data.groupby('Cluster')[self.target_column].agg(['mean', 'std'])
        error_bars = [(0, std) for std in cluster_target_stats['std']]  # Only upper error bars
        ax[1, 0].bar(cluster_target_stats.index, cluster_target_stats['mean'], yerr=np.array(error_bars).T,
                     alpha=0.6, color='orange', capsize=5)
        ax[1, 0].set_title('Target Mean and Positive STD per Cluster')
        ax[1, 0].set_xlabel('Cluster')
        ax[1, 0].set_ylabel('Target Value')

        # Mean Installed Capacity per Cluster
        cluster_means = self.data.groupby('Cluster')['installed_capacity'].mean()
        ax[1, 1].bar(cluster_means.index, cluster_means.values, alpha=0.6, color='green')
        ax[1, 1].set_title('Mean Installed Capacity per Cluster')
        ax[1, 1].set_xlabel('Cluster')
        ax[1, 1].set_ylabel('Installed Capacity Mean')

        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        self.preprocess_data()
        self.perform_clustering()
        self.reduce_dimensions()
        self.plot_clusters()
