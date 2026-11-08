import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score

# Load the encoders and scaler fitted in experiments.ipynb
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Preprocess the data, same steps as experiments.ipynb
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data['Gender'] = label_encoder_gender.transform(data['Gender'])
geo_encoded = onehot_encoder_geo.transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

X = data.drop('Exited', axis=1)
y = data['Exited']

# Same split as the ANN and gradient boosting so the test sets match
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune with 5-fold cross-validation on the training set only
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [None, 10],
    'min_samples_leaf': [1, 5],
    'class_weight': [None, 'balanced_subsample'],
}
search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, scoring='roc_auc', cv=5, n_jobs=-1
)
search.fit(X_train, y_train)
rf_model = search.best_estimator_
print(f'Best params : {search.best_params_}')
print(f'CV ROC-AUC  : {search.best_score_:.4f}')


with open('rf_churn_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Compare against the ANN and gradient boosting on the test set
ann_model = tf.keras.models.load_model('model.h5')
with open('gb_churn_model.pkl', 'rb') as file:
    gb_model = pickle.load(file)

probs = {
    'ANN': ann_model.predict(scaler.transform(X_test), verbose=0).ravel(),
    'Gradient Boosting': gb_model.predict_proba(X_test)[:, 1],
    'Random Forest': rf_model.predict_proba(X_test)[:, 1],
}

print('\nTest set ROC-AUC')
for name, p in probs.items():
    print(f'  {name:18}: {roc_auc_score(y_test, p):.4f}')

print('\nChurn class metrics by threshold')
for name, p in probs.items():
    print(name)
    for t in [0.5, 0.4, 0.3, 0.25]:
        pred = p > t
        print(f'  t={t:<4}  accuracy={accuracy_score(y_test, pred):.3f}  recall={recall_score(y_test, pred):.3f}  '
              f'precision={precision_score(y_test, pred):.3f}  f1={f1_score(y_test, pred):.3f}  flagged={pred.sum()}')

print('\nFeature importance (Random Forest)')
print(pd.Series(rf_model.feature_importances_, X.columns).sort_values(ascending=False).round(3).to_string())
