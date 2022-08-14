from nis import cat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import category_encoders
from category_encoders.one_hot import OneHotEncoder

def data_prep(data, ohe, isModeling: bool):
    #before standardization & modeling, one hot coding is necessary for categorical features.
    #get_dummies is cool for modeling, however in the prediction ohe is only applied to supplied categorical values.
    #ex: user does not entered '1' for cp, get_dummies won't append a new column called cp_1. 
    
    cat_columns = list(data.select_dtypes(include=['object']).columns)

    num_data = data.select_dtypes(exclude=['object'])

    if isModeling:
        cat_data = ohe.fit_transform(data[cat_columns])
    else:
        cat_data = ohe.transform(data[cat_columns])
    cat_data = pd.DataFrame(cat_data)
    new_data = pd.concat([num_data, cat_data], axis=1)
    return new_data    

def model_build():
    # import data
    url = 'data/heart.csv'
    data = pd.read_csv(url)

    # 5 important features are need to be kept
    # choices of those 5 features are explained in the end of notebook
    data = data[['age', 'sex', 'cp', 'chol', 'thalach', 'ca', 'target']]

    # one hot encoding of categorical feat.
    data['cp'] = data['cp'].astype(str)
    ohe = OneHotEncoder()
    new_data = data_prep(data, ohe, True)

    # train test split
    x = new_data.drop(columns="target")
    y = new_data.target

    #Standardization
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    x_std= pd.DataFrame(x_std, 
                       index=x.index,
                       columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = .33, random_state=35, stratify=y)

    xgb = XGBClassifier(colsample_bytree=0.5,
                        gamma=0,
                        learning_rate=0.15250000000000002,
                        max_depth=10,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        subsample=0.8,
                        random_state = 35)

    xgb.fit(x_train, y_train)

    modelfile = 'models/xgb_model.pickle'
    scalerfile = 'models/std_scaler.pickle'
    ohefile = 'models/ohe.pickle'
    pickle.dump(xgb, open(modelfile, 'wb'))
    pickle.dump(scaler, open(scalerfile, 'wb'))
    pickle.dump(ohe, open(ohefile, 'wb'))
    
    print('done!')


if __name__ == '__main__':
    model_build()