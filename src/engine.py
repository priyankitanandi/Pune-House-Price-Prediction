#importing required packages
import pandas as pd
from sklearn.model_selection import train_test_split

from ML_Pipeline.utils import *
from ML_Pipeline.data_cleaning import *
from ML_Pipeline.fea_engg import encode_categorical_variables,fea_eng_sa,onehot_end
from ML_Pipeline.feat_sel import feat_sel, feature_sa
from ML_Pipeline.models import final_model
from ML_Pipeline.mlp_model import *
from ML_Pipeline.mlp_model_evaluation import *

import configparser
config = configparser.RawConfigParser()
config.read('..\\input\\config.ini')
DATA_DIR = config.get('DATA','data_dir')
OUTPUT_DIR = config.get('DATA', 'output_dir')


dfr = read_data(DATA_DIR) # Read the initial dataset

#############################################################################################################################################################
 
                                      ### DATA CLEANING ###

dfr = rename_col(dfr,"Propert Type", "Property Type" ) # reanme the data column
dfr = drop_val(dfr,"Property Type","shop") # drop a row an property column
df_norm = normaliseProps(dfr) # Normalising the Propert Type and Property Area in Sq. Ft.

                                  ################################

x_prt = df_norm['Property Type'] # # Checking the outliers for Property type.  ## you can change the columns and check for the outliers for different features.
prt_up_lim = computeUpperFence(x_prt)
df_norm[ x_prt>prt_up_lim]
df_norm.drop(index=86, inplace=True) # drop row 86 - after viewing the data we decide to remove rows 86 as it looks an a potential outliers
df_norm.drop(index=df_norm[df_norm["Property Type"]==7].index, inplace=True) # dropping 7 bhk entry as they are potential ouliers that we can view with the help of scatter plot. 


# price selection
# There are two target variables  - price in lakhs and price in millions; with the help of the plot we can conclude that they are the same variables; hence wew drop one;

df_norm["Price in lakhs"] = df_norm["Price in lakhs"]\
                    .apply(lambda x: pd.to_numeric(x, errors='coerce') ) # Comparing Price in Millions with Price in lakhs
df_norm = drop_col(df_norm, ["Price in lakhs"] )

                            ################################

compute_fill_rate( df_norm ) ### Dealing with the NAN values
df_norm[["Sub-Area", "TownShip Name/ Society Name", "Total TownShip Area in Acres" ]]\
    .sort_values("Sub-Area").reset_index(drop=True)        # Total TownShip Area in Acres
df_norm = drop_empty_axis(df_norm, minFillRate=.5)  # Drop columns filled by less than 50%


                                 ################################


### Regularising the categorical columns ##
binary_cols = df_norm.iloc[:,-7:].columns.to_list()
df_norm = df_norm[df_norm["Price in Millions"]<80] # keep the target values less than 80
binary_cols = reg_catvar(df_norm, binary_cols) # convert to binary 

obj_cols = df_norm.select_dtypes(include="object").columns.to_list() ## Multi-categorical columns
multiCat_cols = list(set(obj_cols)^set(binary_cols))
multiCat_cols = reg_catvar(df_norm, multiCat_cols) # convert for multicategorical vars

df_norm = drop_col(df_norm, ["Location"]) # drop columns
df_norm.columns=[ "index", "sub_area", "n_bhk", "surface", "price", 
                                     "company_name", "township",
                                     "club_house", "school", "hospital", 
                                     "mall", "park", "pool", "gym"] # Renaming the columns

df_norm.to_csv("../input/resd_clean.csv", index=False)       # here we are saving the dataframe in csv      


#############################################################################################################################################################

                                                       ### DATA ANALYSIS ### 
df = read_data_csv('../input/resd_clean.csv') ## read the cleaned dataset
df = drop_col(df,["index", "company_name", "township"]) # drop the columns
df = df.drop_duplicates()

binary_cols = df.iloc[:, 4:].columns.to_list() # convert binary columns 
df = encode_categorical_variables(df, binary_cols)

## sub-area contribustion

# Contribution of different sub-areas on the dataset 
df_sa_count = df.groupby("sub_area")["price"].count().reset_index()\
                .rename(columns={"price":"count"})\
                .sort_values("count", ascending=False)\
                .reset_index(drop=True)
df_sa_count["sa_contribution"] = df_sa_count["count"]/len(df)

df = fea_eng_sa(df_sa_count, "sub_area",df, 7) # feature enggineering on sub area 
X_dummy = onehot_end(df,"sub_area")
X_dummy = X_dummy.astype("int64") # Type conversion

sa_cols_name = ["sa"+str(i+1) for i in range(X_dummy.shape[1])] # Adding the dummy columns to the dataset
df.loc[:,sa_cols_name] = X_dummy

df[["sub_area"]+sa_cols_name].drop_duplicates()\
            .sort_values("sub_area").reset_index(drop=True) # Sub_area and dummy columns relationship 

data = df.select_dtypes(exclude="object") # check only object datatype columns
float_cols = data.select_dtypes( include="float" ).columns.to_list()

# Price correlation with all other columns
corr_cols_list = float_cols+binary_cols # Sorted correlations
corr_list = feat_sel(data, corr_cols_list, "price", sa_cols_name)


df_corr = pd.DataFrame( data=zip(corr_cols_list, corr_list), 
                 columns=["col_name", "corr"] )\
            .sort_values("corr", ascending=False)\
            .reset_index(drop=True)

features = df_corr.loc[abs(df_corr["corr"])>.1, "col_name"].to_list() 

lst = feature_sa(df, "sub_area", "price", features)

### Data scaling ##########

sel_data = data[features+["price"]].copy() # Selection the final dataset
sel_data = data_scale(sel_data, "surface")
sel_data.to_csv("../input/resd_features.csv", index=False) # save the new data

#############################################################################################################################################################

                                        ########### MODELS #################

data = read_data_csv("../input/resd_features.csv") # # read the final csv data 
data = data.sort_values("surface").reset_index(drop=True)

X = data.iloc[:, :-1] # # Selecting the feature matrix and target vector
y = data["price"]

rs = 118 # # Random sate for data splitting
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=rs) 

# # ## Regresiion models - MODEL BUILDING  ###
model_reg= final_model('random',X,y,rs,X_train, X_test, y_train, y_test) # # run the required model    
print("Regression Model Executed")

                     ################   MLP with TensorFlow #################
rs=13
y_scaled = y_scaled_val(y)
X_train, X_test, y_train, y_test = train_test_split(X.values, y_scaled.values, #train test split
                                                    test_size=.3, 
                                                    random_state=rs)

mlp_tensorflow_model= mlp_tf_model(X_train, X_test, y_train, y_test)
mlp_tensorflow_model.summary() # check the model summary

plot_model_history(mlp_tensorflow_model.history)  ## plot the training history
plotResidueMLP(mlp_tensorflow_model, X, y_scaled, rs=rs)  #plot residual 
plot_real_pred(mlp_tensorflow_model, X, y_scaled, rs) # plot predictions
print("MLP Model Executed")

#############################################################################################################################################################