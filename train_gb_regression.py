import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load the encoders fitted in the notebooks
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Preprocess the data, same steps as salaryregression.ipynb
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data['Gender'] = label_encoder_gender.transform(data['Gender'])
geo_encoded = onehot_encoder_geo.transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

X = data.drop('EstimatedSalary', axis=1)
y = data['EstimatedSalary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler for the ANN regression model, saved separately so it doesn't overwrite the churn scaler.pkl
scaler = StandardScaler()
scaler.fit(X_train)
with open('regression_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Gradient boosting regressor; trees don't need scaled features
# absolute_error loss to match the ANN's mean_absolute_error
gb_model = GradientBoostingRegressor(
    loss='absolute_error',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)

with open('gb_regression_model.pkl', 'wb') as file:
    pickle.dump(gb_model, file)

# Compare against the ANN and a predict-the-median baseline
ann_model = tf.keras.models.load_model('regression_model.h5')
ann_mae = mean_absolute_error(y_test, ann_model.predict(scaler.transform(X_test), verbose=0).ravel())
gb_mae = mean_absolute_error(y_test, gb_model.predict(X_test))
baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.median()))

print(f'Baseline (median) MAE : {baseline_mae:.2f}')
print(f'ANN MAE               : {ann_mae:.2f}')
print(f'Gradient Boosting MAE : {gb_mae:.2f}')
