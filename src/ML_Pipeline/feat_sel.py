import numpy as np

# fucntion to perform feature selection
def feat_sel(data, corr_cols_list,target,col_name):
# Price correlation with all other columns
    corr_cols_list.remove(target)
    corr_cols_list.extend(col_name)
    corr_list = [] # to keep the , correlations with price
    for col in corr_cols_list:
        corr_list.append(round(data[target].corr(data[col]),2) )    
    return corr_list


# function for surface area count
def feature_sa(df, df_col, target,features):
# Keeping the sub areas' name, their mean price and frequency (count)
    sa_feature_list = [sa for sa in features if "sa" in sa]
    lst = []
    for col in sa_feature_list:
        sa_triger = df[col]==1
        sa = df.loc[sa_triger, df_col].to_list()[0]
        x = df.loc[sa_triger, target]
        lst.append( (sa, np.mean(x), df[col].sum()) )

    return lst