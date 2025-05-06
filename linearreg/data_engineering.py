import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import norm
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

class DatasetNormalizer:
    def __init__(self, dataframe):
        """
        Initialize with a pandas DataFrame.
        """
        self.original_df = dataframe.copy()
        self.df = dataframe.copy()
        self.target_name = dataframe.columns[-1]
        self._scalers = {
            'minmax': None,
            'zscore': None,
            'robust': None
        }
    
    def reset(self):
        """Reset to original unnormalized data."""
        self.df = self.original_df.copy()
        return self.df

    def minmax(self, feature_range=(0, 1)):
        scaler = MinMaxScaler(feature_range=feature_range)
        self.df[:] = scaler.fit_transform(self.df)
        self._scalers['minmax'] = scaler
        return self.df

    def inverse_minmax(self):
        scaler = self._scalers.get('minmax')
        if scaler:
            self.df[:] = scaler.inverse_transform(self.df)
        else:
            raise ValueError("MinMax scaler not fitted. Call minmax() first.")
        return self.df

    def zscore(self):
        scaler = StandardScaler()
        self.df[:] = scaler.fit_transform(self.df)
        self._scalers['zscore'] = scaler
        return self.df

    def inverse_zscore(self):
        scaler = self._scalers.get('zscore')
        if scaler:
            self.df[:] = scaler.inverse_transform(self.df)
        else:
            raise ValueError("Z-score scaler not fitted. Call zscore() first.")
        return self.df

    def robust(self):
        scaler = RobustScaler()
        self.df[:] = scaler.fit_transform(self.df)
        self._scalers['robust'] = scaler
        return self.df

    def inverse_robust(self):
        scaler = self._scalers.get('robust')
        if scaler:
            self.df[:] = scaler.inverse_transform(self.df)
        else:
            raise ValueError("Robust scaler not fitted. Call robust() first.")
        return self.df

    def log(self, base=np.e):
        self.df[:] = np.log(self.df + 1e-9) / np.log(base)
        return self.df

    def exp(self, base=np.e):
        self.df[:] = base ** self.df
        return self.df

    def exponential_normalize(self, axis=0):
        max_vals = self.df.max(axis=axis)
        shifted = self.df.subtract(max_vals, axis=1 if axis == 0 else 0)
        exp_vals = np.exp(shifted)
        self.df[:] = exp_vals.div(exp_vals.sum(axis=axis), axis=1 if axis == 0 else 0)
        return self.df

    def rank_based_zscore(self, dropoutliers=0.005):
        """
        Apply rank-based z-score transformation using percentile rank and inverse normal CDF.
        Useful for reducing the impact of outliers.
        """
        ds = self.df.copy()
        percrank = np.ceil(ds.rank(ascending=False)/(ds.count()) * 100)
        zscore = (norm.ppf(1 - percrank/100 + dropoutliers).round(4))
        self.df[:] = zscore
        return self.df
    
    def features_corr_grouped(self, threshold=0.9):
        X = self.df.drop(self.target_name, axis=1)
        y = self.df[self.target_name]
        corr_abs = X.corr().abs()
        used_features = set()
        selected_features = []
        for col in corr_abs.columns:
            if col in used_features:
                continue
            group = corr_abs.index[corr_abs[col] > threshold].tolist()
            
            group = [g for g in group if g not in used_features]
            
            if not group:
                continue
            
            best_feature = max(group, key=lambda f: abs(np.corrcoef(df[f], y)[0, 1]))
            
            selected_features.append(best_feature)
            used_features.update(group)
        return self.df[selected_features]
    
    def features_pca(self, n_components):
        features = self.df.copy().drop(self.target_name, axis=1)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(features)
        return pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    def features_f_regression(self, n_features):
        X = self.df.copy().drop(self.target_name, axis=1)
        y = self.df[self.target_name]
        selector = SelectKBest(f_regression, k=n_features)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return self.df[selected_features]
    
    def features_random_forest(self, n_features, random_state=42, threshold="median"):
        X = self.df.copy().drop(self.target_name, axis=1)
        y = self.df[self.target_name]
        rf = RandomForestRegressor(n_estimators=n_features, random_state=random_state)
        rf.fit(X, y)
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        X_selected = selector.transform(X)
        mask = selector.get_support()
        selected_features = X.columns[mask]
        return self.df[selected_features]

    def get(self):
        return self.df

    def summary(self):
        return self.df.describe()



def normalize_and_split(df, target_col="Ride Count", method='zscore', test_size=0.2, random_state=42, dropoutliers=0.01, use_log=True, feature_selection={"None": 0.9}):
    normalizer = DatasetNormalizer(df)

    method = method.lower()
    if method == 'minmax':
        normalizer.minmax()
    elif method == 'zscore':
        normalizer.zscore()
    elif method == 'robust':
        normalizer.robust()
    elif method == 'log':
        normalizer.log()
    elif method == 'exp':
        normalizer.exp()
    elif method == 'exponential':
        normalizer.exponential_normalize()
    elif method == 'rankzscore':
        normalizer.rank_based_zscore(dropoutliers=dropoutliers)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    if list(feature_selection.keys())[0] == "Correlation_grouped":
        X = normalizer.features_corr_grouped(threshold=np.float(list(feature_selection.values())[0]))
    elif list(feature_selection.keys())[0] == "PCA":
        X = normalizer.features_pca(n_components=np.float(list(feature_selection.values())[0]))
    elif list(feature_selection.keys())[0] == "f_regression":
        X = normalizer.features_f_regression(n_features=np.float(list(feature_selection.values())[0]))
    elif list(feature_selection.keys())[0] == "Random_forest":
        X = normalizer.features_random_forest(n_features=np.float(list(feature_selection.values())[0]))
    else:
        X = normalizer.get().drop(target_col, axis=1)

    if use_log:
        y = np.log1p(normalizer.get()[target_col])
    else:
        y = normalizer.get()[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_regression_model(X_train, X_test, y_train, y_test, 
                           model_type='linear', use_log=False, 
                           alpha=1.0, l1_ratio=0.5):

    model_type = model_type.lower()
    if model_type == 'linear':
        model = LinearRegression(fit_intercept=True)
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha, fit_intercept=True)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'linear', 'ridge', 'elasticnet'.")

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train': {
            'r2': r2_score(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train)
        },
        'test': {
            'r2': r2_score(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test)
        }
    }

    if use_log:
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
        y_pred_train_orig = np.expm1(y_pred_train)
        y_pred_test_orig = np.expm1(y_pred_test)

        metrics['train_original_scale'] = {
            'mae': mean_absolute_error(y_train_orig, y_pred_train_orig),
            'mse': mean_squared_error(y_train_orig, y_pred_train_orig)
        }
        metrics['test_original_scale'] = {
            'mae': mean_absolute_error(y_test_orig, y_pred_test_orig),
            'mse': mean_squared_error(y_test_orig, y_pred_test_orig)
        }

    return model, metrics


def visualize_metrics(metrics, use_log=True):
    """Visualize comparison between train and test metrics"""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Log-scale metrics
    if use_log:
        ax[0].bar(['Train', 'Test'], [metrics['train']['mae'], metrics['test']['mae']], color=['blue', 'orange'])
        ax[0].set_title('MAE (log scale)')
        ax[1].bar(['Train', 'Test'], [metrics['train']['r2'], metrics['test']['r2']], color=['blue', 'orange'])
        ax[1].set_title('R² Score (log scale)')
    
    # Original-scale metrics
    if 'train_original_scale' in metrics:
        fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
        ax2[0].bar(['Train', 'Test'], 
                  [metrics['train_original_scale']['mae'], 
                   metrics['test_original_scale']['mae']], 
                  color=['blue', 'orange'])
        ax2[0].set_title('MAE (original scale)')
        ax2[1].bar(['Train', 'Test'], 
                  [metrics['train_original_scale']['mse'], 
                   metrics['test_original_scale']['mse']], 
                  color=['blue', 'orange'])
        ax2[1].set_title('MSE (original scale)')
        
    plt.tight_layout()
    plt.show()
    plt.savefig('linear_regression_metrics_comparison.png')
    print("Metrics visualized and saved as linear_regression_metrics_comparison.png")
    
    
def main():
    df = pd.read_csv('nyc_data_cleaned_numeric.csv')  # Replace with your dataset path

    X_train, X_test, y_train, y_test = normalize_and_split(df, method='zscore', test_size=0.2, random_state=42, dropoutliers=0.005, feature_selection={"Correlation_grouped": 0.9})

    lr_model, metrics = train_regression_model(X_train, X_test, y_train, y_test, model_type='linear', use_log=True, alpha=1.0, l1_ratio=0.5)

    visualize_metrics(metrics, use_log=False)