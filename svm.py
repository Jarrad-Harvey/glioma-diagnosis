import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def generate_dataset():
    labels = ['HGG', 'LGG']
    df = pd.read_csv('name_mapping.csv')

    # Step 2: Extract Grade and BRATS 2020 Subject ID
    df_subset = df[['Grade', 'BraTS_2020_subject_ID']]

    # Step 3: Extract Volume Number
    def extract_volume_number(subject_id):
        match = re.search(r'\d+$', subject_id)
        if match:
            return f'Volume_{int(match.group())}'
        else:
            return None

    df_subset = df_subset.copy()
    df_subset['Volume'] = df_subset['BraTS_2020_subject_ID'].apply(extract_volume_number)

    df_subset['Volume_Number'] = df_subset['Volume'].apply(lambda x: int(x.split('_')[1]))
    df_sorted = df_subset.sort_values(by='Volume_Number')
    df_final = df_sorted[['Volume', 'Grade']]

    # Step 4: Combine Data

    df_radiomic = pd.read_csv('radiomic_features.csv')
    df_radiomic = df_radiomic.copy()

    df_radiomic['Volume_Number'] = df_radiomic['Volume'].apply(lambda x: int(x.split('_')[1]))
    df_radiomic_sorted = df_radiomic.sort_values(by='Volume_Number')
    df_radiomic_final = df_radiomic_sorted.drop(columns=['Volume_Number'])

    df_merged = pd.merge(df_final, df_radiomic_final, on='Volume', suffixes=('', '_radiomic'))

    df_merged.to_csv('volume_feature.csv', index=False)

    df_final_csv = df_merged.drop('Volume', axis=1)

    df_final_csv.to_csv('final_feature.csv', index=False)

    # Step 5: Save to CSV
    df_final.to_csv('output.csv', index=False)


def train_SVM():
    features_df = pd.read_csv('final_feature.csv')
    class_names = ['HGG', 'LGG']
    filtered_samples, hidden_test_samples, class_mapping = filter_samples(features_df, 'Grade', class_names)

    X = filtered_samples.drop('Grade', axis=1)  # Features
    y = filtered_samples['Grade']

    # Split training and validation sets with K Fold
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    i = 1
    clf = GridSearchCV(SVC(gamma='auto', kernel='rbf'), param_grid, cv=10, return_train_score=False)
    for train_index, test_index in cv_strategy.split(X, y):
        print('\n{} of kfold {}'.format(i, cv_strategy.n_splits))
        X_train = X.iloc[train_index, :]
        X_val = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_val = y.iloc[test_index]

        clf.fit(X_train, y_train)
        print(clf.best_params_)
        print(clf.best_score_)
        pred = clf.predict(X_val)
        print('accuracy_score', accuracy_score(y_val, pred))
        i += 1

    print("results")
    print(clf.cv_results_)
    print(clf.best_score_)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    best_model.fit(X, y)

    hidden_test_X = hidden_test_samples.drop('Grade', axis=1)
    hidden_test_y = hidden_test_samples['Grade']
    pred_y = best_model.predict(hidden_test_X)
    test_accuracy = accuracy_score(hidden_test_y, pred_y)
    print("Test Accuracy (hidden test set):", test_accuracy)


def filter_samples(df, label_column, class_names, n_hidden_test=10, random_state=42):
    label_encoder = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[label_column] = label_encoder.fit_transform(df_encoded[label_column])

    # Determine the mapping between original class names and encoded labels
    class_mapping = {label: name for label, name in enumerate(label_encoder.classes_)}

    # Filter samples based on the encoded labels corresponding to each class name
    filtered_samples = []
    hidden_test_samples = []
    for class_name in class_names:
        encoded_label = label_encoder.transform([class_name])[0]
        filtered_samples_class = df_encoded[df_encoded[label_column] == encoded_label]
        hidden_test_class = filtered_samples_class.sample(n=n_hidden_test, random_state=random_state)
        hidden_test_samples.append(hidden_test_class)
        filtered_samples.append(filtered_samples_class.drop(hidden_test_class.index))

    filtered_samples_combined = pd.concat(filtered_samples, axis=0)
    hidden_test_samples_combined = pd.concat(hidden_test_samples, axis=0)
    return filtered_samples_combined, hidden_test_samples_combined, class_mapping
