import unittest
import io
import numpy as np
from contextlib import redirect_stdout
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import logging
import random
import sqlite3  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipelineFinal:
    def __init__(self, dataset_url,target,name = "Placeholder"):
        """
        Initializes the DataPipelineFinal class.

        Parameters:
        - dataset_url (str): The URL or path to the dataset.
        - target (str): The target column in the dataset.
        - name (str, optional): The name of the dataset. Defaults to "Placeholder".
        """
        self.dataset = dataset_url
        self.name = name
        self.preprocessor = None
        self.dtc_model = None
        self.target = target
        self.predictors = None
        self.name_final = name
        if self.name.endswith('.csv'):
            self.name_final = self.name[:-4] 
        
        logger.info("DataPipelineFinal initialized with dataset URL: %s and target column: %s", dataset_url, target)
        
        
    def clean_and_process_data(self):
        """
        Cleans and processes the dataset.

        Returns:
        - cleaned_df (DataFrame): The cleaned and processed DataFrame.
        """
        temp = self.dataset
        if 'id' in temp.columns:
            temp=temp.drop(columns=['id'])
        temp = temp.drop(columns=[self.target])
        y = self.dataset[self.target]
        numeric_features = temp.select_dtypes(include=['number']).columns
        categorical_features = temp.select_dtypes(include=['object']).columns  # Add categorical feature column names here
        
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', IterativeImputer(max_iter=10, random_state=0)),  # Multiple imputation
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
            ('onehot', OrdinalEncoder())])
        
        # Preprocess data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        self.preprocessor = preprocessor
        
        cleaned_data = preprocessor.fit_transform(self.dataset)
        
        temp_array = []
        for x in numeric_features.values:
            temp_array.append(x)
        for x in categorical_features.values:
            temp_array.append(x)
        # Convert preprocessed data back to DataFrame with column names
        cleaned_df = pd.DataFrame(cleaned_data, columns=temp_array)
        cleaned_df[self.target] = y
        cleaned_df.to_csv(self.name_final + " Cleaned.csv", index=False)
        logger.info("Cleaned data written to 'cleaned_output_file.csv'.")
        return cleaned_df
    
    def visualize_data(self):
        """
        Visualizes numeric and categorical features using appropriate plots.
        """
        temp = self.dataset
        if 'id' in temp.columns:
            temp = temp.drop(columns=['id'])
        temp_y = temp[self.target]
        temp = temp.drop(columns=[self.target])
        temp[self.target] = temp_y

        # Select numeric and categorical features
        numeric_features = temp.select_dtypes(include=['int', 'float']).columns
        categorical_features = temp.select_dtypes(include=['object']).columns
        
        # Visualize numeric features
        selected_data_numeric = temp[numeric_features]
        if not selected_data_numeric.empty:
            selected_features_numeric = selected_data_numeric.columns.values
            num_features_numeric = len(selected_features_numeric) - 1  # Exclude the target column
            num_rows_numeric = (num_features_numeric + 4) // 5
            num_cols_numeric = min(num_features_numeric, 5)

            plt.figure(figsize=(num_cols_numeric * 4, num_rows_numeric * 3))
            for i, feature in enumerate(selected_features_numeric[:-1], start=1):
                plt.subplot(num_rows_numeric, num_cols_numeric, i)
                sns.boxplot(x=self.target, y=feature, data=temp)
                plt.title(feature)
            plt.tight_layout()
           
            plt.show()# Save the plot as a PNG file
            plt.close()  
            logger.info("Numeric data visualized.")
        else:
            print("No numeric features to visualize.")
            logger.warning("No numeric features to visualize.")

        # Visualize categorical features
        for feature in categorical_features:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=feature, data=temp,  hue=self.target)
            plt.title(f'{feature} Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()# Save the plot as a PNG file
            plt.close()  
            logger.info(f"{feature} visualized.")



    def get_target(self):
        """
        Returns the target column name.

        Returns:
        - target (str): The target column name.
        """
        logger.info("Returning target column: %s", self.target)
        return self.target
    def train_eval_model(self,model=DecisionTreeClassifier()):
        """
        Trains and evaluates a machine learning model.

        Parameters:
        - model (object): The machine learning model to train. Defaults to DecisionTreeClassifier().
        """

        self.model = model
        temp = self.clean_and_process_data()
        
        X = temp.drop(columns=[self.target])
   
        y = temp[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=89)
        if str(self.model) != 'SVC()':
            
            Dtc_pipeline = Pipeline(steps=[
                                       ('classifier', self.model)])
        else:
            Dtc_pipeline = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
    
        Dtc_pipeline.fit(X_train, y_train)
        # Predictions
        y_pred = Dtc_pipeline.predict(X_test)
        # Store the trained model
        
        self.dtc_model = Dtc_pipeline
        # Evaluate the model
        
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy = accuracy
        print(f"Accuracy: {accuracy:.2f} of " + str(self.model))
        # Classification report
        print("\n Classification report  of " + str(self.model) + "\n")
        print(classification_report(y_test, y_pred))
        print('\n Confusion matrix \n of ' + str(self.model) + "\n")
        print(confusion_matrix(y_test, y_pred))
        logger.info("Accuracy: %.2f", accuracy)
        
        logger.info("\n Classification report \n")
        logger.info(classification_report(y_test, y_pred))
        logger.info("\n Confusion matrix \n")
        logger.info(confusion_matrix(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        
    def make_prediction(self,features):
        """
        Makes a prediction using the trained model.

        Parameters:
        - features (list): The list of features for prediction.

        Returns:
        - prediction (str): The predicted label.
        """
        numeric_values = [x for x in features if isinstance(x, (int, float, np.int64, np.float64))]
        string_values = [x for x in features if isinstance(x, str)]
        
# Combine numeric values and string values
        reordered_data = numeric_values + string_values
        temp_df = pd.DataFrame(columns=self.clean_and_process_data().columns)
        row_data = {col: val for col, val in zip(temp_df.columns, reordered_data)}
        
        df = pd.DataFrame(row_data, index=[0])
#         if self.target != None:
#             df = df.drop(columns=[self.target])

        scaled_data = (self.preprocessor.transform(df))
        
        temp2 = {col: val for col, val in zip(temp_df.columns, scaled_data[0])}
        df2 = pd.DataFrame(temp2, index=[0])
        print('\n Chosen Features \n')
        print(df2)
        self.predictors = df
        print('\n Prediction of ' +(str(self.model)) + ': \n')
        self.prediction = (str(self.dtc_model.predict(df2))) 
        return (str(self.dtc_model.predict(df2)))  
        
    def write_model_evaluation_to_database(self):
         
        # Connect to SQLite database
        features_str = ",".join(str(feature) for feature in self.predictors)
        conn = sqlite3.connect('model_evaluation2.db')

        # Create a cursor object
        cursor = conn.cursor()

        # Define SQL command to create a table if not exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            features TEXT NOT NULL,
            accuracy REAL NOT NULL,
            prediction TEXT NOT NULL,
            dataset TEXT NOT NULL
        );
        """
        cursor.execute(create_table_sql)

        # Define SQL command to insert data into the table
        insert_data_sql = "INSERT INTO model_evaluations (model_name, features, accuracy, prediction,dataset) VALUES (?, ?, ?, ?, ?)"

        # Execute the SQL command to insert data into the table
        cursor.execute(insert_data_sql, (str(self.model), features_str, self.accuracy, self.prediction,self.name_final))

        # Commit the changes
        conn.commit()

        # Close the connection
        conn.close()
        logger.info("\n Information pushed to database \n")
    def run_pipeline(self):
        """
        Runs the data pipeline.
        """
        self.clean_and_process_data()
        self.visualize_data()
        logger.info("Pipeline executed.")
        

# Example usage:
if __name__ == "__main__":
    database = 'MaternalDataset.csv'
    pipeline = DataPipelineFinal((pd.read_csv(database)),'RiskLevel',database)
    pipeline.run_pipeline()
    pipeline.train_eval_model()
#     df = df_features
    df = pd.read_csv(database)
    row = df.iloc[random_number]
    row_df = row.to_frame().T
    
    pipeline.make_prediction(row_df.values.ravel())
    pipeline.write_model_evaluation_to_database()

    pipeline.train_eval_model(LogisticRegression())
    print(pipeline.make_prediction(row_df.values.ravel()))
    pipeline.write_model_evaluation_to_database()
    pipeline.train_eval_model(SVC())
    print(pipeline.make_prediction(row_df.values.ravel()))
    pipeline.write_model_evaluation_to_database()
    pipeline.train_eval_model(RandomForestClassifier())
    print(pipeline.make_prediction(row_df.values.ravel()))
    pipeline.write_model_evaluation_to_database()
    
    
