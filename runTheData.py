import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

def clean_product_names(df):
    # Define a mapping for product names
    product_mapping = {
        'Rest Go': 'Rest/Rest+/Restore/Rest Go',
        'Rest 2nd Gen': 'Rest/Rest+/Restore/Rest Go',
        'Rest PLUS 2nd Gen': 'Rest/Rest+/Restore/Rest Go',
        'Grow': 'Grow',
        'Restore': 'Rest/Rest+/Restore/Rest Go'
    }

    # Define keywords to identify products
    product_keywords = {
        'Rest Go': 'Rest Go',
        'Rest 2nd Gen': 'Rest 2nd Gen',
        'Rest PLUS 2nd Gen': 'Rest PLUS 2nd Gen',
        'Grow': 'Grow',
        'Restore': 'Restore'
    }

    # Function to map product names based on keywords
    def map_product_name(name):
        for key, keyword in product_keywords.items():
            if keyword.lower() in name.lower():
                return product_mapping[key]
        return None

    # Apply the mapping function to the Product Names column
    df['Product Names'] = df['Product Names'].apply(map_product_name)

    # Filter out rows with non-matching product names
    df = df[df['Product Names'].notna()]

    return df

def clean_destination_names(df):
    # Function to clean destination names based on the given rules
    def map_destination_name(row):
        name_lower = row['Destination Name'].lower()
        discharge_port = row['Discharge Port']

        west_coast_ports = [
            'Long Beach, California', 'Los Angeles, CA', 'Oakland, CA',
            'San Francisco, CA', 'Seattle, WA', 'Tacoma'
        ]
        east_coast_ports = [
            'New York, NY / Newark, NJ', 'Norfolk, VA', 'Savannah, GA',
            'Charleston, SC', 'Miami, FL', 'Jacksonville, FL',
            'Houston, TX', 'Chicago, IL', 'New York, NY', 'Newark, NJ', 'Norfolk, VA'
        ]

        if any(keyword in name_lower for keyword in ['reserve', 'deliverr']):
            return 'D-Reserve'
        if any(keyword in name_lower for keyword in ['jd logistics', 'baby']):
            return 'Babylist'
        if 'premier' in name_lower:
            return 'Premier Canada'
        if row['Destination Name'][:4].isupper() or any(keyword in name_lower for keyword in ['fba', 'amazon']):
            if discharge_port in west_coast_ports:
                return 'FBA- Amazon West Coast'
            elif discharge_port in east_coast_ports:
                return 'FBA- Amazon East Coast'
            elif any(keyword in discharge_port for keyword in ['Sarnia', 'Port Huron', 'Vancouver']):
                return 'Amazon Canada'
        return row['Destination Name']

    # Apply the mapping function to the Destination Name column
    df['Destination Name'] = df.apply(map_destination_name, axis=1)

    return df

def clean_data(df):
    # Convert date columns to datetime
    date_columns = [
        'Origin Actual Cargo Ready Date',  'Destination Actual Arrival Date'
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Calculate the lead time
    df['lead_time'] = (df['Destination Actual Arrival Date'] - df['Origin Actual Cargo Ready Date']).dt.days

    # Drop rows with missing lead times
    df = df.dropna(subset=['lead_time'])

    # Remove entries where departure or arrival ports are airports (denoted by the first 3 characters being all caps)
    df = df[~df['Departure Port'].str[:3].str.isupper()]
    df = df[~df['Arrival Port'].str[:3].str.isupper()]

    # Remove entries where the departure port is 'Sarnia'



    # Clean the product names
    df = clean_product_names(df)

    # Clean the destination names
    df = clean_destination_names(df)


    # Create additional time-based features
    df['Cargo Ready Month'] = df['Origin Actual Cargo Ready Date'].dt.month


    return df

def main():
    # Load the cleaned data
    df = pd.read_csv('/Users/catherinereller/Desktop/Hatch/uncleanedData.csv')

    # Clean the data and add new features
    df = clean_data(df)

    df.to_csv('/Users/catherinereller/Desktop/Hatch/cleanedData.csv', index=False)

    # Select relevant features and target
    features = ['Product Names', 'Departure Port', 'Destination Name', 'Cargo Ready Month']
    target = 'lead_time'

    # Split the data into training and testing sets
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the features using OneHotEncoder for categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Product Names', 'Departure Port', 'Destination Name']),
            ('num', 'passthrough', ['Cargo Ready Month'])
        ])

    # Create a pipeline that includes preprocessing and model training
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save the best model to a file
    joblib.dump(best_model, 'pipeline.pkl')
    joblib.dump((X_test, y_test), 'test_data.pkl')


if __name__ == "__main__":
    main()