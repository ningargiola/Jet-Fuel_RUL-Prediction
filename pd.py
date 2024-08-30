import math 
import os 

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from catboost import CatBoostClassifier



# Load data
path = "CMaps/train_FD001.txt"
eda_train1 = pd.read_csv(path, sep=" ", header=None)

print(eda_train1.head())



# Find missing data 
plt.figure(figsize=(10,6))
sns.heatmap(eda_train1.isna().transpose(),
            cmap = sns.diverging_palette(230, 20, as_cmap=True),
            cbar_kws={'label': 'Missing Data'})

eda_train1.isna().sum()

# Exclude missing data 
train1_copy = eda_train1.drop([26,27], axis=1)

# Organize data 
cols = ['engine_id', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3'] + [f's{i}' for i in range(1,22)]
train1_copy.columns = cols 

train1_copy.head()

train1_copy.describe().T

# Columns with low std and mean==min/max won't help in training
train1_copy = train1_copy.drop(['s19', 's18', 's16', 's10', 's5', 's1', 'op_setting3'], axis=1)

corr = train1_copy.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10,6))
cmap = sns.diverging_palette(230,20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink':.5})

# Columns with low correlation get removed 
train1_copy = train1_copy.drop(['op_setting1', 'op_setting2', 's6', 's14'], axis=1)

# Creating array of EOL of all Ids
EOL=[]

for i in train1_copy['engine_id']:
    EOL.append(((train1_copy[train1_copy['engine_id'] == i]['cycle']).values)[-1])

train1_copy['EOL'] = EOL

# Calculate learning rate
train1_copy['LR'] = train1_copy['cycle'].div(train1_copy['EOL'])

train1_copy['label'] = pd.cut(train1_copy['LR'], bins=[0, 0.6, 0.8, np.inf], labels=[0, 1, 2], right=False)

train1_copy.drop(columns=['engine_id', 'EOL', 'LR'], inplace=True)

print(train1_copy.head())

# Evaluation Methods
def score_func(y_true, y_pred):
    """
    model evaluation

    Args:
        y_true = true target RUL
        y_pred = predicted target RUL
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    score_list = [round(mae,2), round(mse, 2), round(r2, 2)]
    
    # Printing scores
    print("Classification Report:\n", report)
    print(f' Mean Absolute Error (MAE): {score_list[0]}')
    print(f' Root Mean Squared Error (RMSE): {score_list[1]}')
    print(f' R2 Score: {score_list[2]}')
    print('<)------------X------------(>)')

def plot_confmatrix(y_true, y_pred):
    confm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(confm, index=sorted(set(y_true)), columns=sorted(set(y_true)))

    return sns.heatmap(df_cm, cmap = sns.diverging_palette(230, 20, as_cmap=True), annot=True, fmt='d')

# Data Management 
def transform_df(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.drop([26,27], axis=1)
    cols = ['engine_id', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3'] + [f's{i}' for i in range(1,22)]

    df.columns = cols 

    # Drop specified columns
    drop_cols = ['s19', 's18', 's16', 's10', 's5', 's1', 's6', 's14', 'op_setting3', 'op_setting1', 'op_setting2']
    df = df.drop(columns= drop_cols, axis=1)

    # Making an array with EOL of all Ids
    EOL = []

    for i in df['engine_id']:

        EOL.append(((df[df['engine_id'] == i]['cycle']).values)[-1])

    df["EOL"] = EOL

    # Calculate Learning Rate
    df['LR'] = df['cycle'].div(df['EOL'])

    # Create label column
    bins = [0, 0.6, 0.8, np.inf]
    labels = [0, 1, 2]
    df['label'] = pd.cut(df['LR'], bins=bins, labels=labels, right=False)
    # Drop columns
    df.drop(columns=['engine_id', 'EOL', 'LR'], inplace=True)
    X_train = df.drop(['label'], axis=1).values
    y_train = df['label']

    return X_train, y_train

# Model
trainset_path = 'CMaps/train_FD001.txt'
X_train, y_train = transform_df(trainset_path)
print(f'Dimensions of feature matrix: {X_train.shape}\nDimension of target vector: {y_train.shape}')

testset_path = 'CMaps/test_FD001.txt'
X_test, y_test = transform_df(testset_path)
print(f'Dimensions of feature matrix: {X_test.shape}\nDimension of target vector: {y_test.shape}')

# Training
X_train_def, X_val_def, y_train_def, y_val_def = train_test_split(X_train, y_train, test_size=0.3, random_state=7)

model = CatBoostClassifier(iterations=100, loss_function="MultiClass", classes_count=3,verbose=False)

# Fit model on training data
model.fit(X_train_def, y_train_def)

y_pred_def1 = model.predict(X_val_def)

plot_confmatrix(y_val_def, y_pred_def1)
score_func(y_val_def, y_pred_def1)

