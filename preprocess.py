import pandas as pd
import numpy as np
import statistics as st
import random
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])

def missing_value_checker(df, name):
    chk_null = df.isnull().sum()
    chk_null_pct = chk_null / (df.index.max() + 1)
    chk_null_tbl = pd.concat([chk_null[chk_null > 0], chk_null_pct[chk_null_pct > 0]], axis=1)
    chk_null_tbl = chk_null_tbl.rename(columns={0: "欠損数",1: "欠損割合"})
    print(name)
    print(chk_null_tbl, end="\n\n")

def drop_columns(df):
    target_columns = [
        "default",
    ]

    return df.drop(target_columns,axis=1)

def categorical_encoding(df):
    target_columns = [
        'campaign',
        'previous',
    ]

    df['campaign_1'] = ((df['campaign'] == 1)).astype(int)
    df['campaign_2'] = ((df['campaign'] == 2)).astype(int)
    df['campaign_3'] = ((df['campaign'] == 3)).astype(int)
    df['campaign_4over'] = ((df['campaign'] >= 4)).astype(int)

    df['previous_0'] = ((df['previous'] == 0)).astype(int)
    df['previous_1'] = ((df['previous'] == 1)).astype(int)
    df['previous_2'] = ((df['previous'] == 2)).astype(int)
    df['previous_3over'] = ((df['previous'] >= 3)).astype(int)

    df = df.drop(target_columns,axis=1)

    return df

def calculate_std(x, y):
    return pd.concat([x, y], axis=1).std(axis=1, ddof=1)

def calculate_mean(x, y):
    return (x + y) / 2

def calculate_median(x, y):
    return pd.concat([x, y], axis=1).median(axis=1)

def calculate_q75(df,columns):
    return df[columns].quantile(0.75, axis=1)

def calculate_q25(df,columns):
    return df[columns].quantile(0.25, axis=1)

def calculate_zscore(x, mean, std):
    return (x - mean) / (std + 1e-3)

def process_outlier(df, columns):
    # std_values = calculate_std(df[col_x], df[col_y])
    # mean_values = calculate_mean(df[col_x], df[col_y])
    # median_values = calculate_median(df[col_x], df[col_y])
    if isinstance(columns, str):
        columns = [columns]
    q1 = df[columns].quantile(0.25)
    q3 = df[columns].quantile(0.75)
    iqr = q3 - q1 

    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr

    # 上限外の値を修正
    for col in columns:
        df[col] = df[col].apply(
            lambda x: max(lower_limit[col], min(x, upper_limit[col]))
        )
    
    return df

def conbination_columns(df):
    new_features = []
    feature_columns = df[['fico','revol.util']]
    for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = df[feature1], df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 + f2)
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))
    
            # # 新しい特徴量操作
            # new_features.append(calculate_mean(f1, f2))
            # new_features.append(calculate_median(f1, f2))
            # new_features.append(calculate_q75(f1, f2))
            # new_features.append(calculate_q25(f1, f2))
            # zscore_f1 = calculate_zscore(
            #     f1, calculate_mean(f1, f2), calculate_std(f1, f2))
            # new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                # for operation in ['multiplied_by', 'divided_by', 'q75', 'q25', 'zscore_f1']]
                                # for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'mean', 'median', 'q75', 'q25', 'zscore_f1']]
                                for operation in ['plus', 'minus', 'multiplied_by', 'divided_by']]
    
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def previous_poutcome(df):

    columns_to_encode = ['previous','poutcome']
    df['previous_poutcome_combined'] = df['previous'].astype(str) + "_" + df['poutcome'].astype(str)

    # ordinal_encoder = OrdinalEncoder()
    # df[columns_to_encode] = ordinal_encoder.fit_transform(df[columns_to_encode])
    # new_features = []
    # for (feature1, feature2) in combinations(columns_to_encode, 2):
    #         f1, f2 = df[feature1], df[feature2]

    #         # 既存の特徴量操作
    #         new_features.append(f1 + f2)
    #         new_features.append(f1 - f2)
    #         # new_features.append(f1 * f2)
    #         # new_features.append(f1 / (f2 + 1e-8))
    
    #         # # 新しい特徴量操作
    #         # new_features.append(calculate_mean(f1, f2))
    #         # new_features.append(calculate_median(f1, f2))
    #         # new_features.append(calculate_q75(f1, f2))
    #         # new_features.append(calculate_q25(f1, f2))
    #         # zscore_f1 = calculate_zscore(
    #         #     f1, calculate_mean(f1, f2), calculate_std(f1, f2))
    #         # new_features.append(zscore_f1)
    # new_features_df = pd.concat(new_features, axis=1)

    # new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
    #                             for (feature1, feature2) in combinations(columns_to_encode, 2)
    #                             # for operation in ['multiplied_by', 'divided_by', 'q75', 'q25', 'zscore_f1']]
    #                             # for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'mean', 'median', 'q75', 'q25', 'zscore_f1']]
    #                             # for operation in ['plus', 'minus', 'multiplied_by', 'divided_by']]
    #                             for operation in ['plus', 'minus']]
    new_features_df = df['previous_poutcome_combined']

    # result_df = pd.concat([df, new_features_df], axis=1)

    return df

def day_bin(df):
    target_columns = [
          'day'
     ]
    
    df['day_0-5'] = ((df['day'] >= 0) & (df['day'] < 5)).astype(int)
    df['day_5-10'] = ((df['day'] >= 5) & (df['day'] < 10)).astype(int)
    df['day_10-15'] = ((df['day'] >= 10) & (df['day'] < 15)).astype(int)
    df['day_15-20'] = ((df['day'] >= 15) & (df['day'] < 20)).astype(int)
    df['day_20-25'] = ((df['day'] >= 20) & (df['day'] < 25)).astype(int)
    df['day_25-30'] = ((df['day'] >= 25) & (df['day'] < 30)).astype(int)
    df['day_30over'] = ((df['day'] >=30)).astype(int)


    df = df.drop(target_columns,axis=1)

    return df

def pdays_bin(df):
    target_columns = [
          'pdays'
     ]
    
    df['pdays_0-200'] = ((df['pdays'] >= 0) & (df['pdays'] < 200)).astype(int)
    df['pdays_200-400'] = ((df['pdays'] >= 200) & (df['pdays'] < 400)).astype(int)
    df['pdays_400-600'] = ((df['pdays'] >= 400) & (df['pdays'] < 600)).astype(int)
    df['pdays_600-800'] = ((df['pdays'] >= 600) & (df['pdays'] < 800)).astype(int)
    df['pdays_800over'] = ((df['pdays'] >800)).astype(int)

    df = df.drop(target_columns,axis=1)

    return df
   
df = train_test

df.to_csv('datasets/concat_fix.csv', index=False)

print(df.info())

drop_columns(df)

# df = previous_poutcome(df)

categorical_encoding(df)
print(df.columns)
# df = process_outlier(df, "age")
# df = process_outlier(df, "duration")  # 正しいカラム名を指定

df = process_outlier(df,"pdays")

df = process_outlier(df,"balance")

# df = day_bin(df)

# df = pdays_bin(df)


print(df.info())

train_test = df

train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]

test = test.drop("y",axis=1)

# csvファイルの作成
train.to_csv('datasets/train_fix.csv', index=False)
test.to_csv('datasets/test_fix.csv', index=False)
