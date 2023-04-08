from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Function for performing label encoding for the categorical variables

def encode_categorical_variables(df, cat_vars):
# Transforming the yes/no to 1/0
    laben = LabelEncoder()
    for col in cat_vars:
        df[col] = laben.fit_transform(df[col])
    return (df)


# function for surface area column
def fea_eng_sa(df_count, df_col, df, n):
        sa_sel_col = df_count.loc[df_count["count"]>n, df_col].to_list()
        df[df_col] = df[df_col].where(df[df_col].isin(sa_sel_col), "other")
        return df


# function to perform one hot encoding
def onehot_end(df,col_name):
    # Dummy variable conversion
    hoten = OneHotEncoder(sparse=False)
    X_dummy = hoten.fit_transform(df[[col_name]] )
    return X_dummy