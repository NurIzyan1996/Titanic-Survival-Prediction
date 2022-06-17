import os
import pandas as pd
import numpy as np
from modules import ModelLoading
#%% PATHS
TEST_PATH = os.path.join(os.getcwd(),'datasets','test.csv')
SEX_LE_PATH = os.path.join(os.getcwd(),'saved_model','sex_le_scaler.pkl')
EMB_LE_PATH = os.path.join(os.getcwd(),'saved_model','emb_le_scaler.pkl')
KNN_IMP_PATH = os.path.join(os.getcwd(),'saved_model','knn_imp_scaler.pkl')
MMS_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'saved_model','best_model.pkl')
OUTCOME_SAVE_PATH = os.path.join(os.getcwd(),'datasets','new_outcome.csv')

#%% STEP 1: Model Loading
ml = ModelLoading()
model,sex_le,embarked_le,mms,knn_imp = ml.load_model(MODEL_PATH,SEX_LE_PATH,
                                                     EMB_LE_PATH,MMS_PATH,
                                                     KNN_IMP_PATH)

#%% STEP 2: Data Loading
df = pd.read_csv(TEST_PATH)
df2 = df.copy()
#%% STEP 3: Data Inspection
# a) information about the data
df.info()
df.describe()

# b) check null values
df.isnull().sum()
'''Observation: Age contains 86 null values, Cabin contains 327 null values 
and Fare contains 1 null values'''

#%% STEP 4: Data Cleaning
# a) Eliminate columns 
df2 = df2.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1)

# b) Scaling the 'Sex' and 'Embarked' using Label Encoder
df2['Sex'] = sex_le.transform(df2['Sex'])
df2['Embarked'] = embarked_le.transform(df2['Embarked'])

# c) imputer null values using KNN Imputer approach
df2.insert(0,'Survived',np.nan)
df2 = pd.DataFrame(knn_imp.transform(df2))
df2.columns = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

#%% STEP 5: Data Preprocessing
# a) scaling the data using MinMaxScaler
X= df2.drop(columns='Survived', axis=1)
X = mms.transform(X)

#%% STEP 5: Model Deployment
# a) deploy the model
outcome = pd.DataFrame(model.predict(X)).astype(int)
outcome.columns = ['Survived']

# b) combine columns PassengerId and Survived
passengerid= df['PassengerId']
final = pd.concat([passengerid,outcome],axis=1)

# c) save outcome as new csv file
final.to_csv(OUTCOME_SAVE_PATH,index=False)
