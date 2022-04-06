
from flask import Flask
from flask import request
app = Flask(__name__)

### Imports
import pandas as pd
import numpy as np
import sys
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.sparse import coo_matrix, csr_matrix
from numpy import bincount, log, sqrt
import itertools
import time
from pathlib import Path

path = 'C:/Users/sumai/Desktop/project/dataset/'
# data = pd.read_csv(path + "instacart.csv")

path = 'C:/Users/sumai/Desktop/project/dataset/'
orders_path = "orders.csv"
products_path = "products.csv"
test_data_path ='user_products__test.csv'
matrix_df_path = 'user_products__prior.csv'
matrix_path = "product_user_matrix.npz"
product_user_matrix_path= "product_user_matrix.npz"
product_factor_50_path="product_factor_50.npy"
user_factor_50_path= "user_factor_50.npy"
product_factor_100_path= "product_factor_100.npy"
user_factor_100_path= "user_factor_100.npy"

df_order_products_prior = pd.read_csv('order_products__prior.csv')
df_order_products_train = pd.read_csv('order_products__train.csv')
df_orders =  pd.read_csv(orders_path)

df_products = pd.read_csv(products_path)

def make_test_data(filepath, df_orders, df_order_products_train):
    """
    Generates the test dataset and saves it to disk at the given path
    """
    
    start = time.time()
    print("Creating test data ...")

    # Read train csv
    df_order_user_current = df_orders.loc[(df_orders.eval_set == "train")].reset_index()
    df_order_user_current = df_order_user_current[["order_id", "user_id"]]
    
    # Sanity check #1: `current_order_user_df` and `df_order_products_train` should have the same number of 
    # unique order ids
    assert len(df_order_user_current["order_id"].unique()) == len(df_order_products_train["order_id"].unique())

    # Convert train dataframe to a similar format
    df_order_products_test = df_order_products_train[["order_id", "product_id"]]
    df_order_products_test = df_order_products_test.groupby("order_id")["product_id"].apply(list).reset_index().rename(columns={"product_id": "products"})

    # Sanity check #2: `df_order_products_test` and `df_order_user_current` should have the same number of 
    # records before attempting to merge them
    assert df_order_products_test.size == df_order_user_current.size

    # Merge on order id
    df_user_products_test = pd.merge(df_order_user_current, df_order_products_test, on="order_id")
    df_user_products_test = df_user_products_test[["user_id", "products"]]

    # Write to disk
    df_user_products_test.to_csv(filepath, index_label=False)
    
    print("Completed in {:.2f}s".format(time.time() - start))


# Generate test data if it doesn't exist already
REBUILD_TEST_DATA = False
if REBUILD_TEST_DATA or not Path(test_data_path).is_file():
    make_test_data(test_data_path, df_orders, df_order_products_train)

df_user_products_test = pd.read_csv(test_data_path)

# Just making sure that the test data isn't corrupted
assert len(df_user_products_test) == 131209

def get_user_product_prior_df(filepath, df_orders, df_order_products_prior):
    
    """
    Generates a dataframe of users and their prior products purchases, and writes it to disk at the given path
    """
    
    start = time.time()
    print("Creating prior user product data frame ...")

    df_merged = pd.merge(df_orders, df_order_products_prior, on="order_id")
    df_user_product_prior = df_merged[["user_id", "product_id"]]
    df_user_product_prior = df_user_product_prior.groupby(["user_id", "product_id"]).size().reset_index().rename(columns={0:"quantity"})
    
    # Write to disk
    df_user_product_prior.to_csv(filepath, index_label=False)

    print("Completed in {:.2f}s".format(time.time() - start))


# Build dataframe of users, products and quantity bought using prior datasets
REBUILD_MATRIX_DF = False
if REBUILD_MATRIX_DF or not Path(matrix_df_path).is_file():
    get_user_product_prior_df(matrix_df_path, df_orders, df_order_products_prior)
df_user_product_prior = pd.read_csv(matrix_df_path)
# Making them as category for making dictonary of user and item ids later for easy 
# mapping from sparse Matrix representation
df_user_product_prior["user_id"] = df_user_product_prior["user_id"].astype("category")
df_user_product_prior["product_id"] = df_user_product_prior["product_id"].astype("category")

def build_product_user_matrix(matrix_path, df_user_product_prior):
    """
    Generates a utility matrix representing purchase history of users, and writes it to disk.
    Rows and Columns represent products and users respectively.
    """
    start = time.time()
    print("Creating product user matrix ...")

    product_user_matrix = sparse.coo_matrix((df_user_product_prior["quantity"],
                                            (df_user_product_prior["product_id"].cat.codes.copy(),
                                             df_user_product_prior["user_id"].cat.codes.copy())))    
    sparse.save_npz(matrix_path, product_user_matrix)
    
    print("Completed in {:.2f}s".format(time.time() - start))

# Build dataframe of users, products and quantity bought using prior datasets
REBUILD_USER_MATRIX_DF = False
if REBUILD_USER_MATRIX_DF or not Path(matrix_path).is_file():
    build_product_user_matrix(matrix_path, df_user_product_prior)    
product_user_matrix=sparse.load_npz(product_user_matrix_path).tocsr().astype(np.float32)

# Just making sure that the the generated matrix is accurates
# User=1 bought product=196 10 times
assert product_user_matrix[195, 0] == 10

# product_factors_50,user_factors_50 here denote 50 latent Factors considered
REBUILD_FACTORS= False
if REBUILD_FACTORS or not ((Path(product_factor_50_path)).is_file() 
                           and (Path(user_factor_50_path)).is_file()): 
    #Calculating the product and user factors
    product_factors_50, S, user_factors_50 = linalg.svds(product_user_matrix, 50)
    # changing to user* factor format
    user_factors_50=user_factors_50.T*S
    # saving the user and product factors
    np.save(product_factor_50_path, product_factors_50)
    np.save(user_factor_50_path, user_factors_50)
else:
    # Loading the user and product factors 
    product_factors_50=np.load(product_factor_50_path)
    user_factors_50=np.load(user_factor_50_path)   

# product_factors_100,user_factors_100 here denotes 100 latent Factors considered
REBUILD_FACTORS= False
if REBUILD_FACTORS or not ((Path(product_factor_100_path)).is_file() 
                           and (Path(user_factor_100_path)).is_file()): 
    #Calculating the product and user factors
    product_factors_100, S, user_factors_100 = linalg.svds(product_user_matrix, 100)
    # changing to user* factor format
    user_factors_100=user_factors_100.T*S
    # saving the user and product factors
    np.save(product_factor_100_path, product_factors_100)
    np.save(user_factor_100_path, user_factors_100)
else:
    # Loading the user and product factors 
    product_factors_100=np.load(product_factor_100_path)
    user_factors_100=np.load(user_factor_100_path)    

# Class To find the top recommended items given a user_id
class TopRecommended(object):
    def __init__(self, product_factors,user_factors,product_user_matrix):
        self.product_factors =product_factors
        self.user_factors =user_factors
        self.product_user_matrix=product_user_matrix
    def recommend(self, user_id, N=10):
        
        """
        Finds top K Recommendations
        """
        scores =  self.user_factors[user_id].dot(self.product_factors.T)
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

    def recommend_new(self, user_id, N=10):
        """
        Finds Top k new Recommendations
        """
        scores =  self.user_factors[user_id].dot(self.product_factors.T)
        bought_indices=product_user_matrix.T[user_id].nonzero()[1]
        count = N + len(bought_indices)
        ids = np.argpartition(scores, -count)[-count:]
        best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])        
        return list(itertools.islice((rec for rec in best if rec[0] not in bought_indices), N))  

# Since the utility matrix is 0-indexed, the below dict is required to convert between `ids` and `indices`.
# For example, `product_id` 1 in the dataset is represented by the `0`th row of the utility matrix.

# Maps user_id: user index
u_dict = {uid:i for i, uid in enumerate(df_user_product_prior["user_id"].cat.categories)}

# Maps product_index: product id
p_dict = dict(enumerate(df_user_product_prior["product_id"].cat.categories))

# Initializing class for factors 50 which returns top recommended items for a user_id
svd_recm=TopRecommended(product_factors_50,user_factors_50,product_user_matrix)

# Initializing class for factors 100 which returns top recommended items for a user_id
svd_recm_100=TopRecommended(product_factors_100,user_factors_100,product_user_matrix)

# Recommend items for a user 1
user_id = 10
print("User ID :",user_id)
# New Recommendations and Old Recommendations
recommendations_all = svd_recm.recommend(u_dict[user_id],N=10)
recommendations_new = svd_recm.recommend_new(u_dict[user_id],N=10)

@app.route("/home")
def home():
    return "Flask API get endpoint running"

@app.route('/api/predict', methods=['POST'])
def post__api_predict():
    #cuttle-environment-set-config CU22-api method=POST route=/api/predict response=output
    
    # file = 10 
    # file_string = file.read()
    
    # print(file_string)
    #string =
    uid = int(request.form['userId'])
    # New Recommendations and Old Recommendations
    recommendations_all = svd_recm.recommend(u_dict[uid],N=10)
    recommendations_new = svd_recm.recommend_new(u_dict[uid],N=10)
    
    row = df_user_products_test.loc[df_user_products_test.user_id == uid]
    actual = list(row["products"])
    
    all_recm_products=[]
    for recommend in recommendations_all:
        all_recm_products.extend((df_products.loc[df_products.product_id == p_dict[recommend[0]]].product_name).tolist())
        
    output = ','.join([elem for elem in all_recm_products]) 
    output 
    return output


if __name__ == '__main__':
    app.run()
