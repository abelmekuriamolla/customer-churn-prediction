#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#load the model from disk
import joblib
model = joblib.load(r"./notebook/awashmodelnew.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Awash Bank Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in an Awash Bank use case.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('Awash.jpg')
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        #st.subheader("Demographic data")
        Sex = st.selectbox('Sex:', ('Male', 'Female'))
        Age = st.number_input('Age',min_value=1, max_value=100, value=1)
        CivilStatus = st.selectbox('Civil Status', ('Single', 'Married', 'Divorced'))
        ProductType = st.selectbox('Product Type', ('1320', '1328', '1335', '1336', '1347', '1348', '1349', '1356', '1425', '1427'))
        Location = st.selectbox('Location', ('HALABA', 'HADIYA', 'GAMO', 'OMO', 'WOLAITA', 'SILTE', 'KEMBATA', 'KONSO', 'GOFA', 'DAWRO'))
        Tenure = st.slider('Number of months the customer has stayed with the bank', min_value=0, max_value=200, value=0)
        ATM = st.selectbox('ATM Card:', ('Yes', 'No'))
        MobileBanking = st.selectbox('Mobile Banking:', ('Yes', 'No'))
        InternetBanking = st.selectbox('Internet Banking:', ('Yes', 'No'))
 

        data = {
                'Sex': Sex,
                'Age': Age,
                'CivilStatus':CivilStatus,
                'ProductType': ProductType,
                'Location': Location,
                'Tenure': Tenure,
                'ATM': ATM,
                'MobileBanking': MobileBanking,
                'InternetBanking': InternetBanking,
  
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)[0]
        probability = model.predict_proba(preprocess_df)[0][1]

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
                st.write('Probability of Churn: {:.2f}%'.format(probability*100))
            else:
                st.success('No, the customer is happy with Bank Services.')
        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data)
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.', 
                                                    0:'No, the customer is happy with Bank Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()




