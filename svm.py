import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, cross_val_predict, \
    GridSearchCV
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
    # df_final = df_subset[['Volume', 'Grade']].iloc[:118]

    df_radiomic = pd.read_csv('radiomic_features.csv')
    df_radiomic = df_radiomic.copy()

    df_radiomic['Volume_Number'] = df_radiomic['Volume'].apply(lambda x: int(x.split('_')[1]))
    df_radiomic_sorted = df_radiomic.sort_values(by='Volume_Number')
    df_radiomic_final = df_radiomic_sorted.drop(columns=['Volume_Number'])

    # df_conventional = pd.read_csv('conventional_features.csv')
    # combined_features_df = pd.concat([df_radiomic, df_conventional], axis=1)
    # print(combined_features_df)
    # combined_features_df.to_csv('combined.csv')

    df_merged = pd.merge(df_final, df_radiomic_final, on='Volume', suffixes=('', '_radiomic'))

    # df_final_features = pd.concat([df_radiomic_final, df_final], axis=1)
    df_merged.to_csv('volume_feature.csv', index=False)

    df_final_csv = df_merged.drop('Volume', axis=1)

    df_final_csv.to_csv('final_feature.csv', index=False)

    # Step 5: Save to CSV
    df_final.to_csv('output.csv', index=False)

def train_SVM():
    features_df = pd.read_csv('final_feature.csv')
    class_names = ['HGG', 'LGG']
    filtered_samples, hidden_test_samples, class_mapping = filter_samples(features_df, 'Grade', class_names)
    # print(filtered_samples.value_counts())
    # print(filtered_samples,flush=True)

    # label_encoder = LabelEncoder()
    # features_df['Grade'] = label_encoder.fit_transform(features_df['Grade'])
    # class_mapping = {label: class_name for label, class_name in enumerate(label_encoder.classes_)}
    # print("Class Mapping:", class_mapping)
    # hgg_samples = features_df[features_df['Grade'] == 'HGG']
    # lgg_samples = features_df[features_df['Grade'] == 'LGG']
    #
    # hidden_test_hgg = hgg_samples.sample(n=10, random_state=42)
    # hidden_test_lgg = lgg_samples.sample(n=10, random_state=42)
    # hidden_test_set = pd.concat([hidden_test_hgg, hidden_test_lgg])
    # features_df = features_df.drop(hidden_test_set.index)
    # original_labels = features_df['Grade'].unique()
    #
    # # unique_encoded_labels = np.unique(features_df)
    # print("Unique encoded labels:", original_labels)
    # numerical_labels = label_encoder.classes_
    #
    # # Create a mapping dictionary
    # label_mapping = dict(zip(numerical_labels, original_labels))
    # print("numerical encoded labels:", original_labels)

    X = filtered_samples.drop('Grade', axis=1)  # Features
    # print("X")
    # print(X,flush=True)

    y = filtered_samples['Grade']
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X, y)
    # y_resampled_series = pd.Series(y_resampled)
    # class_counts_resampled = y_resampled_series.value_counts()
    #
    # # Print the value counts for each class
    # print("Class Counts in Resampled Data:")
    # print(class_counts_resampled)
    # print('lebgth')
    # print(len(X_resampled))
    # print(len(y_resampled))
    # print(X_resampled.shape)
    # print(y_resampled.shape)

    # hgg_samples = features_df[features_df['Grade'] == 'HGG']
    # lgg_samples = features_df[features_df['Grade'] == 'LGG']
    # print(hgg_samples.size)
    # print(lgg_samples.size)

    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,  # Features
        y,  # Labels
        test_size=0.2,  # Size of hidden test set
        stratify=y,  # Stratify by grade to ensure proportional split
        random_state=42
    )
    #
    # # Check the sizes of each set
    # print("Hidden Test Set:", hidden_test_samples.shape)
    # print("Training Set:", X_train.shape)
    # print("Validation Set:", X_val.shape)
    #
    # svm_model = SVC(kernel='rbf', C=0.1, gamma='auto', random_state=42)
    #
    # cv_strategy = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)
    # cv_strategy_hidden = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(svm_model, X_resampled, y_resampled, cv=cv_strategy, scoring='accuracy')

    clf = GridSearchCV(SVC(class_weight='balanced'),{
        'C':[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
        'kernel':['rbf', 'sigmoid'],
        'gamma':['scale', 'auto']
    },cv = 5,return_train_score = False, verbose=3)

    # print("X_train")
    # print(X_train)
    # print("y_train")
    # print(y_train)

    clf.fit(X_train,y_train)

    print("results")
    print(clf.cv_results_)
    print(clf.best_score_)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    best_model.fit(X_train,y_train)

    hidden_test_X = hidden_test_samples.drop('Grade', axis=1)
    hidden_test_y = hidden_test_samples['Grade']
    test_accuracy = best_model.score(hidden_test_X, hidden_test_y)
    print("Test Accuracy (hidden test set):", test_accuracy)



    # cv_strategies = [
    #     KFold(n_splits=5, shuffle=True, random_state=42),
    #     StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    #     # Add more cross-validation strategies as needed
    # ]
    # for cv_strategy in cv_strategies:
    #     scores = cross_val_score(svm_model, X_resampled, y_resampled, cv=cv_strategy, scoring='accuracy')
    #     print(f"Mean accuracy ({cv_strategy.__class__.__name__}): {scores.mean():.4f} (+/- {scores.std():.4f})")

    # print("Cross-validation scores:", cv_scores)
    #
    # # Step 5: Print mean and standard deviation of cross-validation scores
    # print("Mean CV accuracy:", np.mean(cv_scores))
    # print("Standard deviation of CV accuracy:", np.std(cv_scores))

    # Fit the SVM model to the training data
    # svm_model.fit(X_train, y_train)
    #
    # # Make predictions on the validation set
    # hidden_test_X = hidden_test_samples.drop('Grade', axis=1)
    # hidden_test_y = hidden_test_samples['Grade']
    # y_pred = cross_val_predict(svm_model, hidden_test_X, hidden_test_y, cv=cv_strategy_hidden)
    #
    # #
    # # # Evaluate the model
    # accuracy = accuracy_score(hidden_test_y, y_pred)
    # print("Hidden test Accuracy:", accuracy)

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
        # print(class_name)
        # print(class_mapping)
        # print(filtered_samples_class,flush=True)
    filtered_samples_combined = pd.concat(filtered_samples, axis=0)
    hidden_test_samples_combined = pd.concat(hidden_test_samples, axis=0)
    return filtered_samples_combined, hidden_test_samples_combined, class_mapping