import pandas as pd
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_df_action(filepaths_csv, filepaths_meta, action2int=None, delimiter=";"):
    '''
        Data loading function

        Code provided through the courtesy of professor Francesco Ponzio
    '''


    # Load dataframes
    print("Loading data.")
    # Make dataframes
    # Some classes show the output boolean parameter as True rather than true. Fix here
    dfs_meta = list()
    for filepath in filepaths_meta:
        df_m = pd.read_csv(filepath, sep=delimiter)
        df_m.str_repr = df_m.str_repr.str.replace('True', 'true')
        df_m['filepath'] = filepath
        dfs_meta.append(df_m)

    df_meta = pd.concat(dfs_meta)
    df_meta.index = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['completed_timestamp'] = pd.to_datetime(df_meta.completed_timestamp.astype('datetime64[ms]'),
                                                    format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['init_timestamp'] = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'),
                                               format="%Y-%m-%dT%H:%M:%S.%f")

    # Eventually reduce number of classes
    # df_meta['str_repr'] = df_meta.str_repr.str.split('=', expand = True,n=1)[0]
    # df_meta['str_repr'] = df_meta.str_repr.str.split('(', expand=True, n=1)[0]

    actions = df_meta.str_repr.unique()
    dfs = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)

    # Sort columns by name !!!
    df = df.sort_index(axis=1)

    # Set timestamp as index
    df.index = pd.to_datetime(df.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    # Drop useless columns
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)
    signals = df.columns

    df_action = list()
    for action in actions:
        for index, row in df_meta[df_meta.str_repr == action].iterrows():
            start = row['init_timestamp']
            end = row['completed_timestamp']
            df_tmp = df.loc[start: end].copy()
            df_tmp['action'] = action
            # Duration as string (so is not considered a feature)
            df_tmp['duration'] = str((row['completed_timestamp'] - row['init_timestamp']).total_seconds())
            df_action.append(df_tmp)
    df_action = pd.concat(df_action, ignore_index=True)
    df_action.index = pd.to_datetime(df_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_action = df_action[~df_action.index.duplicated(keep='first')]

    # Drop NaN
    df = df.dropna(axis=0)
    df_action = df_action.dropna(axis=0)

    if action2int is None:
        action2int = dict()
        j = 1
        for label in df_action.action.unique():
            action2int[label] = j
            j += 1

    df_merged = df.merge(df_action[['action']], left_index=True, right_index=True, how="left")
    # print(f"df_merged len: {len(df_merged)}")
    # Where df_merged in NaN Kuka is in idle state
    df_idle = df_merged[df_merged['action'].isna()].copy()
    df_idle['action'] = 'idle'
    df_idle['duration'] = df_action.duration.values.astype(float).mean().astype(str)
    df_action = pd.concat([df_action, df_idle])

    # ile label must be 0 for debug mode
    action2int['idle'] = 0
    print(f"Found {len(set(df_action['action']))} different actions.")
    print("Loading data done.\n")

    return df_action, df, df_meta, action2int

def get_standardizer(X_train):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    return scaler

def standardization(X, standardizer):
    X_train_norm = pd.DataFrame(standardizer.transform(X), columns=X.columns)

    return X_train_norm

def get_variance_selector(X_train):
    selector_variance = VarianceThreshold()
    selector_variance.fit(X_train)

    return selector_variance

def remove_zero_variance_features(X, selector_variance):
    X_transformed = selector_variance.transform(X)
    
    return pd.DataFrame(X_transformed ,columns=X.columns.values[selector_variance.get_support()])

def get_lasso_selected_features(X_train, y_train):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
    lasso = SelectFromModel(lsvc, prefit=True)
    selected_features = X_train.columns.values[lasso.get_support()]
    #X_train = X_train[selected_features].copy()

    return selected_features

def get_pca_object(X_train, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    return pca

def apply_pca(X, pca_obj):
    X_transformed = pca_obj.transform(X)

    return pd.DataFrame(X_transformed)


def data_preprocessing(X_train, y_train, X_test, y_test, X_anomalies):
    '''
    Performs:
        - standardization (subtracting the mean and dividing by the standard deviation)
        - 
    '''

if __name__ == '__main__':
    ROOTDIR_DATASET_NORMAL = '~/politoCourses/ml-app/labs/lab01/normal'
    
    filepath_csv = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.csv") for r in [0, 2, 3, 4]]
    filepath_meta = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_0.1s.metadata") for r in [0, 2, 3, 4]]
    df_action, df, df_meta, action2int = get_df_action(filepath_csv, filepath_meta)

    print(f"Nr of columns of df: {len(df.columns)}")
    print(f"Nr of time steps of df: {len(df.index)}")
    print(f"Nr of columns of df: {len(df_action.columns)}")
    print(f"Nr of time steps of df_action: {len(df_action.index)}")

    df_train, df_test = train_test_split(df_action)

    print(df_train.head(3))
    print(df_test.head(3))
    print(f"Shape of df_train{df_train.shape}")
    print(f"Shape of df_test{df_test.shape}")

    df_train = df_train.drop(["action", "duration", "time"], axis=1)
    df_test = df_test.drop(["action", "duration", "time"], axis=1)

    print(df_train.columns)

    standardizer = get_standardizer(df_train)
    df_train_norm = standardization(df_train, standardizer)
    df_test_norm = standardization(df_test, standardizer)

    print(df_train_norm.head(3))
    print(df_test_norm.head(3))
    print(f"Shape of df_train_norm{df_train_norm.shape}")
    print(f"Shape of df_test_norm{df_test_norm.shape}")

    variance_selector = get_variance_selector(df_train_norm)
    df_train_featFiltered = remove_zero_variance_features(df_train_norm, variance_selector)
    df_test_featFiltered = remove_zero_variance_features(df_test_norm, variance_selector)

    print(df_train_featFiltered.head(3))
    print(df_test_featFiltered.head(3))
    print(f"Shape of df_train_featFiltered: {df_train_featFiltered.shape}")
    print(f"Shape of df_test_featFiltered: {df_test_featFiltered.shape}")
