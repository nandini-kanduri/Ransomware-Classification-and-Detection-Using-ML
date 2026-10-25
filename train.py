"""Train ransomware/benign classifiers on a labeled, numeric-feature CSV.

Example: python train.py --data dataset.csv --target label
The target must contain exactly two classes. No dataset is bundled.
"""
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to a labeled CSV')
    p.add_argument('--target', default='label', help='Binary target column name')
    args = p.parse_args()
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f'Target column {args.target!r} not found. Available: {list(df.columns)}')
    df = df.dropna(subset=[args.target]).drop_duplicates()
    X, y = df.drop(columns=[args.target]), df[args.target]
    if y.nunique() != 2:
        raise ValueError('The target must have exactly two classes (ransomware and benign).')
    if len(X) < 10 or y.value_counts().min() < 2:
        raise ValueError('Need at least 10 samples and at least 2 examples per class.')
    # Drop common identifier fields to reduce obvious leakage; inspect dataset for other leakage.
    X = X.drop(columns=[c for c in X if c.lower() in {'id', 'sha256', 'sha1', 'md5', 'filename', 'file_name'}])
    if X.empty:
        raise ValueError('No usable features remain.')
    nums = X.select_dtypes(include='number').columns.tolist()
    cats = X.select_dtypes(exclude='number').columns.tolist()
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    results = []
    for name, model in models.items():
        pipe = Pipeline([('preprocess', pre), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {'model': name, 'accuracy': accuracy_score(y_test, pred),
               'precision_macro': precision_score(y_test, pred, average='macro', zero_division=0),
               'recall_macro': recall_score(y_test, pred, average='macro', zero_division=0),
               'f1_macro': f1_score(y_test, pred, average='macro', zero_division=0)}
        results.append(row)
        print(f'\n{name}\n{classification_report(y_test, pred, zero_division=0)}')
        print('Confusion matrix:\n', confusion_matrix(y_test, pred))
        joblib.dump(pipe, Path('models') / (name.lower().replace(' ', '_') + '.joblib'))
    result_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    result_df.to_csv('outputs/model_comparison.csv', index=False)
    ax = result_df.plot.bar(x='model', y='accuracy', legend=False, ylim=(0, 1), figsize=(9, 5))
    ax.set_ylabel('Test accuracy')
    ax.set_xlabel('Model')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150)
    print('\nModel comparison:\n', result_df.to_string(index=False))
    print('\nSaved models/ and outputs/. Results depend on your dataset and split.')


if __name__ == '__main__':
    main()
