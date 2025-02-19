import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import sys
import json

def feature_engineering(data):
    data['Total_Costs_Log'] = np.log1p(data['Total Costs'])
    data['Length_of_Stay_Log'] = np.log1p(data['Length of Stay'])
    data['Age_Log'] = np.log1p(data['Age Group'])

    data['Age_TotalCost_Interaction'] = data['Age Group'] * data['Total Costs']

    data['Zip_Code_First_Digit'] = data['Zip Code - 3 digits'] // 10

    data = data.drop(columns=['Operating Certificate Number', 'Permanent Facility Id', 'Facility Name'])

    return data

def drop_zero_variance_columns(df):
    return df.loc[:, df.var() != 0]

def apply_one_hot_encoding(df, mapping_file):
    with open(mapping_file, 'r') as json_file:
        data_dict = json.load(json_file)

    for key in data_dict.keys():
        if key in df.columns:
            possible_values = data_dict[key]
            one_hot = pd.get_dummies(df[key])
            one_hot = one_hot.reindex(columns=sorted(possible_values), fill_value=False)
            one_hot = one_hot.iloc[:, 1:]  # Drop first column to avoid multicollinearity
            one_hot = one_hot.astype(int)
            one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]

            # Replace the original column with the new one-hot encoded columns
            column_index = df.columns.get_loc(key)
            df = df.drop(columns=[key])
            df = pd.concat([df.iloc[:, :column_index], one_hot, df.iloc[:, column_index:]], axis=1)

    return df

train_file = sys.argv[1]
created_file = sys.argv[2]
selected_file = sys.argv[3]
mapping_file = 'mapping.json'  # Specify the path to your mapping file

train_data = pd.read_csv(train_file)

# train_data = feature_engineering(train_data)
train_data = apply_one_hot_encoding(train_data, mapping_file)
train_data = drop_zero_variance_columns(train_data)
X_train = train_data.drop(columns=['Gender_1']).values
y_train = train_data['Gender_1'].values

# Feature Selection
selector = SelectKBest(f_classif, k=1000)
X_train_selected = selector.fit_transform(X_train, y_train)

# Save the created features to created.txt
created_features = train_data.drop(columns=['Gender_1']).columns.tolist()
with open(created_file, 'w') as f:
    for feature in created_features:
        f.write(f"{feature}\n")

# Save the selected features to selected.txt
selected_mask = selector.get_support().astype(int)
with open(selected_file, 'w') as f:
    for select in selected_mask:
        f.write(f"{select}\n")
