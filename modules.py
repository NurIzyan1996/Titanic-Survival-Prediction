
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def label_encoder(self,data,path):
        le = LabelEncoder()
        data = le.fit_transform(data)
        pickle.dump(le, open(path, 'wb'))
        return data
    
    def knn_imputer(self,data,path):
        imputer = KNNImputer()
        data= imputer.fit_transform(data)
        pickle.dump(imputer,open(path,'wb'))
        return data
    

class DataVisualization():
    def __init__(self):
        pass
    
    def plot_graph(self,data,hue):
        plt.figure()
        sns.pairplot(data=data, hue=hue)
        plt.show()
        
class DataPreprocessing():
    def __init__(self):
        pass
    
    def mms_scaler(self,data,path):
        mms = MinMaxScaler()
        data = mms.fit_transform(data)
        pickle.dump(mms,open(path,'wb'))
        return data

class ModelCreation():
    def __init__(self):
        pass
    
    def ml_pipeline_model(self,data_1,data_2):
        steps_nb = [('NB',GaussianNB())]
        steps_svm = [('SVM', SVC())]
        steps_knc = [('KNeighbors Classifier', KNeighborsClassifier())]
        steps_tree = [('Decision Tree Classifier',DecisionTreeClassifier())]
        steps_forest = [('Random Forest Classifier',RandomForestClassifier())]
        steps_log = [('Logistic Regression',LogisticRegression(solver='liblinear'))]
        
        pipeline_nb = Pipeline(steps_nb) 
        pipeline_svm = Pipeline(steps_svm) 
        pipeline_knc = Pipeline(steps_knc)
        pipeline_tree = Pipeline(steps_tree)
        pipeline_forest = Pipeline(steps_forest)
        pipeline_log = Pipeline(steps_log)
        
        pipelines= [pipeline_nb, pipeline_svm, pipeline_knc,pipeline_tree,
                    pipeline_forest,pipeline_log]
        
        for pipe in pipelines:
            pipe.fit(data_1,data_2)
        return pipelines
            
class ModelEvaluation():
    def __init__(self):
        pass
    
    def predict_trained_model(self,pipelines,data_1,data_2):
        predictions = [] # prediction of all models
        best_score = 0
        best_scaler = 0
        best_pipeline = ''
        pipe_dict = {0:'NB', 1:'SVM', 2:'KNeighbors Classifier', 
                     3:'Decision Tree Classifier',4:'Random Forest Classifier',
                     5: 'Logistic Regression'}
        for i,model in enumerate(pipelines):
            predictions.append(model.predict(data_1))
            print("{} Test Accuracy:{}".format(pipe_dict[i], 
                                               model.score(data_1,data_2)))
            if model.score(data_1, data_2) > best_score:
                best_score = model.score(data_1, data_2)
                best_scaler = i        
                best_pipeline = model
            
        print('Best Model is {} with accuracy of \
              {}%'.format(pipe_dict[best_scaler],round((best_score)*100,2)))
        return predictions, best_pipeline
    
    def model_score(self,predictions,data):
        best_pipeline_prediction = predictions[1]
        target_names = ['Died', 'Survived']
        print(classification_report(data, best_pipeline_prediction,
                                    target_names=target_names))
        print(confusion_matrix(data, best_pipeline_prediction))
        
class ModelLoading():
    def __init__(self):
        pass
    
    def load_model(self,path_1,path_2,path_3,path_4,path_5):
        model_1 = pickle.load(open(path_1,'rb'))
        model_2 = pickle.load(open(path_2,'rb'))
        model_3 = pickle.load(open(path_3,'rb'))
        model_4 = pickle.load(open(path_4,'rb'))
        model_5 = pickle.load(open(path_5,'rb'))
        return model_1,model_2,model_3,model_4,model_5
