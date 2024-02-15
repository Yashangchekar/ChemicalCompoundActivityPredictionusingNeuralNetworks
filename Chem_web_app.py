import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import streamlit as st
# Load the trained model
loaded_model = pickle.load(open('C:/Users/yash/Documents/Deeplearning/practise/Chemp/trained_model.sav', 'rb'))

# Load the StandardScaler used during training
ss = pickle.load(open('C:/Users/yash/Documents/Deeplearning/practise/Chemp/standard_scaler.sav', 'rb'))

def Pic50Prediction(input_data):
    # Convert the activity to a numerical value if needed
    # Ensure all inputs are converted to appropriate data types
    
    # Changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Assuming float data type

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the loaded StandardScaler
    std_data = ss.transform(input_data_reshaped)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(std_data)
    return prediction


def main():
    st.title("Chemical predictive ")
    
    standard_value = st.text_input("Enter the Standard_Value")
    class1 = st.text_input("Enter the class")
    MW = st.text_input("Enter hte molecular weight")
    LogP = st.text_input("Enter the log p value")
    NumHDonors = st.text_input("Enter the NumHDonars")
    NumHAcceptors=st.text_input("Enter the NumHAcceptors")

    Calculate_PiC50 = ''
    if st.button("Calculate the PiC50"):
        
        Calculate_PiC50 = Pic50Prediction([standard_value,class1,MW,LogP,NumHDonors,NumHAcceptors])

    st.success(Calculate_PiC50)

if __name__ == '__main__':
    main()