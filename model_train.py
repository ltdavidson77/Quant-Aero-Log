# ==========================
# model_train.py
# ==========================
# Trains classification model using labeled signals and angular features.

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from multi_timeframe import compute_multi_timeframe_signals
from label_generator import generate_signal_labels

# ----------------------------------------------
# Predictive Model Training
# ----------------------------------------------
def train_predictive_model(df, column='price', horizon=15, threshold=0.3):
    signals = compute_multi_timeframe_signals(df, column=column)
    labels = generate_signal_labels(df, column=column, horizon=horizon, threshold=threshold)

    features = signals.dropna().iloc[:-horizon]
    target = labels.iloc[:-horizon]
    aligned_idx = features.index.intersection(target.index)

    X = features.loc[aligned_idx]
    y = target.loc[aligned_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=False)
    cm = confusion_matrix(y_test, preds)

    return model, report, cm, X_test, y_test, preds
