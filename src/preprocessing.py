import pandas as pd
import numpy as np
import os
import gc

from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Libraries imported successfully.\n")

# Load raw datasets
df_paysim = pd.read_csv('./data/raw/paysim/PS_20174392719_1491204439457_log.csv')
print("PaySim Shape:", df_paysim.shape)

df_ieee_trans = pd.read_csv('./data/raw/ieee_fraud/train_transaction.csv')
df_ieee_id = pd.read_csv('./data/raw/ieee_fraud/train_identity.csv')
print("IEEE Transaction Shape:", df_ieee_trans.shape)
print("IEEE Identity Shape:", df_ieee_id.shape)

df_ieee_test_trans = pd.read_csv('./data/raw/ieee_fraud/test_transaction.csv')
df_ieee_test_id = pd.read_csv('./data/raw/ieee_fraud/test_identity.csv')
print("IEEE Test Transaction Shape:", df_ieee_test_trans.shape)
print("IEEE Test Identity Shape:", df_ieee_test_id.shape)

df_elliptic_features = pd.read_csv('./data/raw/elliptic/elliptic_txs_features.csv', header=None)
df_elliptic_edges = pd.read_csv('./data/raw/elliptic/elliptic_txs_edgelist.csv')
df_elliptic_classes = pd.read_csv('./data/raw/elliptic/elliptic_txs_classes.csv')
print("Elliptic Features Shape:", df_elliptic_features.shape)
print("Elliptic Edges Shape:", df_elliptic_edges.shape)
print("Elliptic Classes Shape:", df_elliptic_classes.shape, "\n")

# PaySim Preprocessing & Graph
df_paysim = df_paysim.copy()
df_paysim['log_amount'] = np.log1p(df_paysim['amount'])
df_paysim['type_encoded'] = LabelEncoder().fit_transform(df_paysim['type'])

features_p = ['log_amount','type_encoded','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step']
X_paysim = df_paysim[features_p]
y_paysim = df_paysim['isFraud']

scaler = StandardScaler()
X_paysim_scaled = scaler.fit_transform(X_paysim)

df_paysim['tx_id'] = df_paysim.index

edges_orig = (
    df_paysim[['tx_id','nameOrig']]
    .merge(df_paysim[['tx_id','nameOrig']], on='nameOrig')
    .query('tx_id_x != tx_id_y')[['tx_id_x','tx_id_y']]
)
edges_orig.columns = ['source','target']

edges_dest = (
    df_paysim[['tx_id','nameDest']]
    .merge(df_paysim[['tx_id','nameDest']], on='nameDest')
    .query('tx_id_x != tx_id_y')[['tx_id_x','tx_id_y']]
)
edges_dest.columns = ['source','target']

edges_paysim = pd.concat([edges_orig, edges_dest], ignore_index=True)

nodes_paysim = pd.DataFrame(X_paysim_scaled, columns=features_p)
nodes_paysim['tx_id'] = df_paysim['tx_id']
nodes_paysim['label'] = y_paysim.values

del df_paysim, X_paysim, X_paysim_scaled, edges_orig, edges_dest
gc.collect()
print(f"PaySim nodes: {nodes_paysim.shape}, edges: {edges_paysim.shape}\n")

# IEEE-CIS Preprocessing & Graph
def preprocess_ieee(df_trans, df_id, is_train=True):
    df_ieee = df_trans.merge(df_id, on='TransactionID', how='left')
    del df_trans, df_id
    gc.collect()

    num_cols = df_ieee.select_dtypes(include=np.number).columns
    for col in num_cols:
        df_ieee[col] = df_ieee[col].fillna(df_ieee[col].median())

    cat_cols = df_ieee.select_dtypes(include='object').columns
    for col in cat_cols:
        df_ieee[col] = df_ieee[col].fillna('unknown')
        df_ieee[col] = LabelEncoder().fit_transform(df_ieee[col].astype(str))

    if is_train:
        y = df_ieee['isFraud'].values
        df_ieee.drop(columns=['TransactionID', 'isFraud'], inplace=True)
    else:
        y = None
        df_ieee.drop(columns=['TransactionID'], inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ieee)

    df_ieee['tx_id'] = np.arange(len(df_ieee))

    edges_list = []
    for col in ['card1', 'P_emaildomain', 'DeviceType']:
        tmp = df_ieee[['tx_id', col]].sort_values(by=[col, 'tx_id'])
        tmp['next_tx_id'] = tmp['tx_id'].shift(-1)
        tmp['next_col'] = tmp[col].shift(-1)
        valid_edges = tmp[tmp[col] == tmp['next_col']]
        edge_df = valid_edges[['tx_id','next_tx_id']].rename(columns={'tx_id':'source','next_tx_id':'target'})
        edges_list.append(edge_df)

    edges = pd.concat(edges_list, ignore_index=True)
    edges['target'] = edges['target'].astype(int)

    nodes = pd.DataFrame(X_scaled, columns=df_ieee.columns.drop('tx_id'))
    nodes['tx_id'] = df_ieee['tx_id'].values
    if is_train:
        nodes['label'] = y

    del df_ieee, X_scaled, edges_list
    gc.collect()
    return nodes, edges

nodes_ieee, edges_ieee = preprocess_ieee(df_ieee_trans, df_ieee_id, is_train=True)
print(f"IEEE train nodes: {nodes_ieee.shape}, edges: {edges_ieee.shape}")

nodes_ieee_test, edges_ieee_test = preprocess_ieee(df_ieee_test_trans, df_ieee_test_id, is_train=False)
print(f"IEEE test nodes: {nodes_ieee_test.shape}, edges: {edges_ieee_test.shape}\n")

# Elliptic Preprocessing & Graph
num_cols = df_elliptic_features.shape[1]
feature_cols = [f'feature_{i}' for i in range(1,num_cols-1)]
df_elliptic_features.columns = ['txId','timestep'] + feature_cols

df_elliptic = df_elliptic_features.merge(df_elliptic_classes, on='txId')
df_elliptic = df_elliptic[df_elliptic['class'] != 'unknown']
df_elliptic['class'] = df_elliptic['class'].astype(int)

X_elliptic = df_elliptic.drop(columns=['txId','class'])
y_elliptic = df_elliptic['class']

scaler = StandardScaler()
X_elliptic_scaled = scaler.fit_transform(X_elliptic)

nodes_elliptic = pd.DataFrame(X_elliptic_scaled, columns=X_elliptic.columns)
nodes_elliptic['tx_id'] = df_elliptic['txId']
nodes_elliptic['label'] = y_elliptic.values

edges_elliptic = df_elliptic_edges.copy()
edges_elliptic.columns = ['source','target']

del df_elliptic_features, df_elliptic_classes, df_elliptic_edges, df_elliptic, X_elliptic, X_elliptic_scaled
gc.collect()
print(f"Elliptic nodes: {nodes_elliptic.shape}, edges: {edges_elliptic.shape}\n")

# Save processed CSVs
os.makedirs("./data/processed", exist_ok=True)

os.makedirs("./data/processed/paysim", exist_ok=True)
nodes_paysim.to_csv("./data/processed/paysim/paysim_nodes.csv", index=False)
edges_paysim.to_csv("./data/processed/paysim/paysim_edges.csv", index=False)
print("Saved PaySim nodes and edges CSVs.")

os.makedirs("./data/processed/ieee_fraud", exist_ok=True)
nodes_ieee.to_csv("./data/processed/ieee_fraud/ieee_train_nodes.csv", index=False)
edges_ieee.to_csv("./data/processed/ieee_fraud/ieee_train_edges.csv", index=False)
print("Saved IEEE train nodes and edges CSVs.")

nodes_ieee_test.to_csv("./data/processed/ieee_fraud/ieee_test_nodes.csv", index=False)
edges_ieee_test.to_csv("./data/processed/ieee_fraud/ieee_test_edges.csv", index=False)
print("Saved IEEE test nodes and edges CSVs.")

os.makedirs("./data/processed/elliptic", exist_ok=True)
nodes_elliptic.to_csv("./data/processed/elliptic/elliptic_nodes.csv", index=False)
edges_elliptic.to_csv("./data/processed/elliptic/elliptic_edges.csv", index=False)
print("Saved Elliptic nodes and edges CSVs.")

print("\nAll node and edge CSVs saved successfully for GNN training and inference.")