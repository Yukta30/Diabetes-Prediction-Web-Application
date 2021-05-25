import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
        DIABETES DETECTION
           DETECT DIABETES USING THE GIVEN DATA.
""")

image=Image.open('C:/Users/DELL/Downloads/archive/diabetes.png')
st.image(image,caption="DIABETES DETECTION USING MACHINE LEARNING",use_column_width=True)

df = pd.read_csv('C:/Users/DELL/Downloads/archive/diabetes.csv')

st.subheader('Data Information:')
st.dataframe(df)
st.write(df.describe())
chart = st.bar_chart(df)

X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure',0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness',0,999,3)
    insulin = st.sidebar.slider('insulin',0.0,846.0,30.5)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.375)
    age = st.sidebar.slider('Age',21,81,29)

    user_data = {'pregnancies': pregnancies,
                 'glucose':glucose,
                 'blood_pressure':blood_pressure,
                 'skin_thickness':skin_thickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'DPF':DPF,
                 'age':age
                 }
    feature=pd.DataFrame(user_data,index=[0])
    return feature

user_input = get_user_input()

st.subheader('User input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+ '%')

prediction=RandomForestClassifier.predict(user_input)

st.subheader('Classification:')
st.write(prediction)