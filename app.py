import os
import pickle
import numpy as np
import streamlit as st

#%% PATHS
MMS_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'saved_model','best_model.pkl')

#%% STEP 1: Model Loading
model = pickle.load(open(MODEL_PATH,'rb'))
mms = pickle.load(open(MMS_PATH,'rb'))

#%% build app using streamlit

# create the form
with st.form('Titanic survival'):
    st.write("Passenger's info")
    #features selected 
    Pclass = int(st.number_input('Pclass(Social-economic status):1-Upper,2-Middle,3-Lower'))
    Sex = int(st.number_input('Sex:0-Female,1-Male'))
    Age = st.number_input('Age')
    sibsp = int(st.number_input('Total no. of siblings,spouses abroad'))
    parch = int(st.number_input('Total no. of parents and children abroad'))
    Fare = st.number_input('Fare')
    Embarked = int(st.number_input('Embarked(Port of Embarkation):0-Cherbourg,1-Queenstown,2-Southampton'))
    
    submitted= st.form_submit_button('Submit')
    
    
    # to observe if the information appear if i click submit
    if submitted == True:
        
        passenger_info = np.array([Pclass,Sex,Age,sibsp,parch,Fare,Embarked])
        
        info_scaled = mms.transform(np.expand_dims(passenger_info, axis=0))
        
        outcome= model.predict(info_scaled)
        
        if outcome == 1:
            st.success('Survive')
        else:
            st.balloons() 
            st.warning('Not survive')
            
            