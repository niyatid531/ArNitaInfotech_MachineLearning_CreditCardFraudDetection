import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/fraudTrain.csv')
test  = pd.read_csv('data/fraudTest.csv')

print('Train shape:', train.shape)
print('Test shape: ', test.shape)
print('Fraud % in train:', round(train['is_fraud'].mean() * 100, 2), '%')

def engineer_features(df):
    df = df.copy()

    # Extract time features from transaction datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour']      = df['trans_date_trans_time'].dt.hour
    df['day']       = df['trans_date_trans_time'].dt.day
    df['month']     = df['trans_date_trans_time'].dt.month
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

    # Extract age from date of birth
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (pd.Timestamp('today') - df['dob']).dt.days // 365

    # Distance between cardholder and merchant
    df['distance'] = np.sqrt(
        (df['lat'] - df['merch_lat'])**2 +
        (df['long'] - df['merch_long'])**2
    )

    # Encode categorical columns
    cat_cols = ['merchant', 'category', 'gender', 'state', 'job']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

train = engineer_features(train)
test  = engineer_features(test)


features = ['amt', 'hour', 'day', 'month', 'dayofweek',
            'age', 'distance', 'city_pop',
            'merchant', 'category', 'gender', 'state']

X_train = train[features]
y_train = train['is_fraud']
X_test  = test[features]
y_test  = test['is_fraud']

print('Features used:', len(features))
print('Train fraud count:', y_train.sum())

smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print('Before SMOTE:', y_train.value_counts().to_dict())
print('After SMOTE: ', pd.Series(y_train_bal).value_counts().to_dict())

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced'),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, class_weight='balanced', random_state=42),

    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1),
}

scores = {}
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train_bal, y_train_bal)
    preds  = model.predict(X_test)
    acc    = accuracy_score(y_test, preds)
    roc    = roc_auc_score(y_test, preds)
    scores[name] = {'accuracy': acc, 'roc_auc': roc}
    print(f'{name} — Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}')
    print(classification_report(y_test, preds,
          target_names=['Legit', 'Fraud'], zero_division=0))
    print('-' * 60)

best_name  = max(scores, key=lambda x: scores[x]['roc_auc'])
best_model = models[best_name]
print(f'Best model: {best_name}')

os.makedirs('outputs', exist_ok=True)

joblib.dump(best_model, 'outputs/fraud_model.pkl')

final_preds = best_model.predict(X_test)
test_out = test[['trans_num', 'amt']].copy() if 'trans_num' in test.columns \
           else test[['amt']].copy()
test_out['actual_fraud']    = y_test.values
test_out['predicted_fraud'] = final_preds
test_out.to_csv('outputs/predictions.csv', index=False)

print('Saved: outputs/fraud_model.pkl')
print('Saved: outputs/predictions.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Fraud vs Legit distribution
counts = y_train.value_counts()
axes[0].bar(['Legit', 'Fraud'], counts.values, color=['#2455A4','#C0392B'])
axes[0].set_title('Fraud vs Legit (Train)', fontweight='bold')
axes[0].set_ylabel('Count')

# Plot 2: Transaction amount — fraud vs legit
test_plot = test.copy()
test_plot['is_fraud'] = y_test.values
legit_amt = test_plot[test_plot['is_fraud']==0]['amt'].clip(upper=500)
fraud_amt = test_plot[test_plot['is_fraud']==1]['amt'].clip(upper=500)
axes[1].hist(legit_amt, bins=50, alpha=0.6, label='Legit', color='#2455A4')
axes[1].hist(fraud_amt, bins=50, alpha=0.6, label='Fraud', color='#C0392B')
axes[1].set_title('Transaction Amount Distribution', fontweight='bold')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()

# Plot 3: Model ROC-AUC comparison
model_names = list(scores.keys())
roc_scores  = [scores[m]['roc_auc'] for m in model_names]
axes[2].bar(model_names, roc_scores, color=['#2455A4','#27AE60','#E67E22'])
axes[2].set_title('Model ROC-AUC Comparison', fontweight='bold')
axes[2].set_ylabel('ROC-AUC Score')
axes[2].set_ylim(0.5, 1.0)
for i, v in enumerate(roc_scores):
    axes[2].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/plots.png', dpi=150)
plt.close()
print('Plot saved to outputs/plots.png')

def predict_transaction(amt, hour, category_encoded,
                        distance, age, city_pop,
                        day=15, month=6, dayofweek=2,
                        merchant=0, gender=0, state=0):
    features_input = [[amt, hour, day, month, dayofweek,
                       age, distance, city_pop,
                       merchant, category_encoded, gender, state]]
    pred = best_model.predict(features_input)[0]
    return 'FRAUD' if pred == 1 else 'LEGIT'


if __name__ == '__main__':
    # High amount at 3AM — likely fraud
    print(predict_transaction(amt=1200, hour=3,  category_encoded=5,
                              distance=800, age=34, city_pop=50000))
    # Small grocery purchase at noon — likely legit
    print(predict_transaction(amt=45,   hour=12, category_encoded=2,
                              distance=2,   age=45, city_pop=200000))
