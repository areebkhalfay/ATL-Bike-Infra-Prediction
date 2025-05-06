import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

class BikeDemandPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, nyc_data_path, atlanta_data_path=None):
        """
        Load the NYC and Atlanta datasets
        
        Parameters:
        -----------
        nyc_data_path : str
            Path to the cleaned NYC Citibike dataset
        atlanta_data_path : str, optional
            Path to the cleaned Atlanta dataset 
            
        Returns:
        --------
        nyc_df : pandas.DataFrame
            NYC Citibike data
        atlanta_df : pandas.DataFrame or None
            Atlanta data if path provided, otherwise None
        """
        nyc_df = pd.read_csv(nyc_data_path)
        
        # Display basic info about the NYC dataset
        print("NYC Dataset Info:")
        print(f"Shape: {nyc_df.shape}")
        print(f"Columns: {nyc_df.columns.tolist()}")
        
        # Load Atlanta data if provided
        atlanta_df = None
        if atlanta_data_path:
            atlanta_df = pd.read_csv(atlanta_data_path)
            print("\nAtlanta Dataset Info:")
            print(f"Shape: {atlanta_df.shape}")
            print(f"Columns: {atlanta_df.columns.tolist()}")
            
        return nyc_df, atlanta_df
    
    def preprocess_data(self, nyc_df, feature_columns=None, target_column='trip_count', test_size=0.2, random_state=42):
        """
        Preprocess the data for SVM model
        
        Parameters:
        -----------
        nyc_df : pandas.DataFrame
            NYC Citibike data
        feature_columns : list, optional
            List of feature column names to use
        target_column : str, default='trip_count'
            Target column name (bike demand)
        test_size : float, default=0.2
            Test set proportion
        random_state : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : numpy arrays
            Training and testing data splits
        """
        # Store target column name
        self.target_column = target_column
        
        # If feature columns not specified, use all columns except target
        if feature_columns is None:
            self.feature_columns = [col for col in nyc_df.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
        
        print(f"Using features: {self.feature_columns}")
        print(f"Target column: {self.target_column}")
        
        # Extract features and target
        X = nyc_df[self.feature_columns].values
        y = nyc_df[self.target_column].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, cv=5, tune_hyperparams=True):
        """
        Train an SVM regression model using the training data
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target values
        cv : int, default=5
            Number of cross-validation folds
        tune_hyperparams : bool, default=True
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        self : returns an instance of self
        """
        # Create pipeline with scaling and SVR
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            
            # Define hyperparameter grid
            param_grid = {
                'svr__kernel': ['linear', 'rbf', 'poly'], #Kernel types
                'svr__C': [0.1, 1, 10, 100], #Regularization parameter 
                'svr__gamma': ['scale', 'auto', 0.1, 0.01, 0.001], # Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’ 
                'svr__epsilon': [0.1, 0.2, 0.5, 0.01] #Tolerance for margin of error
            }
            
            # Create grid search
            self.model = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1  # Use all available cores
            )
            
            # Train with grid search
            self.model.fit(X_train, y_train)
            
            # Print best parameters
            print(f"Best parameters: {self.model.best_params_}")
            print(f"Best CV score: {-self.model.best_score_:.4f} MSE")
            
        else:
            # Train with default parameters
            self.model = pipeline
            self.model.fit(X_train, y_train)
            
        return self
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Testing features
        y_test : numpy.ndarray
            Testing target values
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        # Print results
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Bike Demand')
        plt.tight_layout()
        plt.savefig('svm_prediction_results.png')
        plt.show()
            
        return metrics
    
    def predict_atlanta_demand(self, atlanta_df):
        """
        Predict bike demand for Atlanta using the trained model
        
        Parameters:
        -----------
        atlanta_df : pandas.DataFrame
            Atlanta data with the same feature columns as NYC training data
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted bike demand values
        """
        if not self.model:
            raise ValueError("Model must be trained before making predictions")
            
        # Ensure Atlanta data has the same features as training data
        atlanta_features = atlanta_df[self.feature_columns].values
        
        # Make predictions
        predictions = self.model.predict(atlanta_features)
        
        # Add predictions to the dataframe
        atlanta_df['predicted_demand'] = predictions
        
        print(f"Predicted {len(predictions)} demand values for Atlanta")
        print(f"Average predicted demand: {predictions.mean():.2f}")
        print(f"Min predicted demand: {predictions.min():.2f}")
        print(f"Max predicted demand: {predictions.max():.2f}")
        
        return predictions
    
    def save_model(self, model_path='bike_demand_svm_model.pkl'):
        """
        Save the trained model to disk
        
        Parameters:
        -----------
        model_path : str, default='bike_demand_svm_model.pkl'
            Path to save the model
        """
        if not self.model:
            raise ValueError("No model to save")
            
        # Save model, feature_columns, and target_column
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
    def load_model(self, model_path='bike_demand_svm_model.pkl'):
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        model_path : str, default='bike_demand_svm_model.pkl'
            Path to the saved model
            
        Returns:
        --------
        self : returns an instance of self
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        
        print(f"Model loaded from {model_path}")
        print(f"Feature columns: {self.feature_columns}")
        print(f"Target column: {self.target_column}")
        
        return self


############# e.g:

def main():
    # Initialize the predictor
    predictor = BikeDemandPredictor()
    
    # Load data (update paths with our actual data files)
    nyc_df, atlanta_df = predictor.load_data(
        nyc_data_path='nyc_citibike_cleaned.csv',
        atlanta_data_path='atlanta_features.csv'
    )
    
    # some example features to consider
    # Modify these based on our actual data columns
    example_features = [
        'temperature','day_of_week',
        'hour_of_day', 'is_weekend', 'is_holiday', 'population_density'
    ]
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = predictor.preprocess_data(
        nyc_df, 
        feature_columns=example_features,
        target_column='trip_count'  # Update with our actual target column
    )
    
    # Train the model with hyperparameter tuning
    predictor.train_model(X_train, y_train, tune_hyperparams=True)
    
    # Evaluate the model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Save the model
    predictor.save_model('bike_demand_svm_model.pkl')
    
    # If Atlanta data is available, predict demand
    if atlanta_df is not None:
        predictions = predictor.predict_atlanta_demand(atlanta_df)
        
        # Visualize predictions for Atlanta
        plt.figure(figsize=(12, 6))
        plt.plot(atlanta_df.index, predictions, 'b-')
        plt.title('Predicted Bike Demand for Atlanta')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Demand')
        plt.tight_layout()
        plt.savefig('atlanta_predictions.png')
        plt.show()
        
        # Save predictions to CSV
        atlanta_df.to_csv('atlanta_with_predictions.csv', index=False)
        print("Saved Atlanta predictions to 'atlanta_with_predictions.csv'")


if __name__ == "__main__":
    main()
