
import os
import pickle
import pandas as pd
import missingno as msno
from sklearn.model_selection import train_test_split
from modules import ExploratoryDataAnalysis,DataVisualization,DataPreprocessing
from modules import ModelCreation,ModelEvaluation
#%% PATHS
TRAIN_PATH = os.path.join(os.getcwd(),'datasets','train.csv')
SEX_LE_PATH = os.path.join(os.getcwd(),'saved_model','sex_le_scaler.pkl')
EMB_LE_PATH = os.path.join(os.getcwd(),'saved_model','emb_le_scaler.pkl')
KNN_IMP_PATH = os.path.join(os.getcwd(),'saved_model','knn_imp_scaler.pkl')
MMS_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','best_model.pkl')
#%% STEP 1: Data Loading
df = pd.read_csv(TRAIN_PATH)
#%% STEP 2: Data Inspection
# a) information about the data
df.info()
df.describe()

# b) check null values
df.isnull().sum()
'''Observation: Age contains 177 null values, Cabin contains 687 null values 
and Embarked contains 2 null values'''

# c) visualize data
msno.matrix(df) 
'''Observation: msno_plot.png show Cabin contains big amount of nan values'''

#%% STEP 3: Data Cleaning
# a) Eliminate columns 
df = df.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1)

# b) eliminate rows in Embarked as it contains only 2 null values
df = df[df['Embarked'].notnull()]

# c) Scaling the 'Sex' and 'Embarked' using Label Encoder
eda = ExploratoryDataAnalysis()
df['Sex'] = eda.label_encoder(df['Sex'], SEX_LE_PATH)
df['Embarked'] = eda.label_encoder(df['Embarked'], EMB_LE_PATH)

# d) imputer null values in Ages column using KNN Imputer approach
df = pd.DataFrame(eda.knn_imputer(df,KNN_IMP_PATH))

#%% STEP 4: Feature Selection
# a) visualize the data to see the trends
dv = DataVisualization()
df.columns = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
dv.plot_graph(df, hue='Survived')
'''Observation: higher survival rate when passengers from upper class, female,
 younger age, higher fare, smaller family size and departed from Cherbourg.
 the features show obvious trend hence no feature selection'''

#%% STEP 5: Data Preprocessing
# a) split features(X) and target(y) data
X= df.drop(columns='Survived', axis=1) 
y = df['Survived'] 

# b) scaling features using MinMaxScale
dp = DataPreprocessing()
X = dp.mms_scaler(X, MMS_PATH)

#%% STEP 6: Model Creation
# a) split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=123)
# b) build pipelines and train the model
mc = ModelCreation()
pipelines = mc.ml_pipeline_model(X_train,y_train)

#%% STEP 7: Model Evaluation
# a) model prediction
me = ModelEvaluation()
predictions, best_pipeline = me.predict_trained_model(pipelines,X_test,y_test)
'''Observation: SVM produces the highest accuracy score which is 80.15%. '''

# b) model performance
me.model_score(predictions,y_test)

# c) save the prediction ML model
pickle.dump(best_pipeline,open(MODEL_SAVE_PATH,'wb'))
