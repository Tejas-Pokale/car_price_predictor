import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

model = pkl.load(open('car_price_predictor.pkl','rb'))

st.header('Car Price Prediction Model')

cars_data = pd.read_csv('CAR DETAILS.csv')

def get_brand_name(car_model):
  return car_model.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

brand = st.selectbox('Select Car Brand',cars_data['name'].unique())
year = st.slider('Car Manufacture Year',1992,2024)
kms = st.number_input('No. of KMs Driven')
fuel = st.selectbox('Select Fuel Type',cars_data['fuel'].unique())
seller = st.selectbox('Select Seller Type',cars_data['seller_type'].unique())
transmission = st.selectbox('Select Transmission Type',cars_data['transmission'].unique())
owner = st.selectbox('Select Owner Type',cars_data['owner'].unique())

if st.button('Predict'):
  input_data_model = pd.DataFrame(
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner'],
    data=[[brand,year,kms,fuel,seller,transmission,owner]]
  )
  input_data_model['name'] = input_data_model['name'].replace(cars_data['name'].unique(),[i for i in range(1,len(cars_data['name'].unique())+1)])
  input_data_model['transmission']= input_data_model['transmission'].replace(cars_data['transmission'].unique(),[i for i in range(1,len(cars_data['transmission'].unique())+1)]).astype(int)
  input_data_model['seller_type']= input_data_model['seller_type'].replace(cars_data['seller_type'].unique(),[i for i in range(1,len(cars_data['seller_type'].unique())+1)]).astype(int)
  input_data_model['fuel']= input_data_model['fuel'].replace(cars_data['fuel'].unique(),[i for i in range(1,len(cars_data['fuel'].unique())+1)]).astype(int)
  input_data_model['owner']= input_data_model['owner'].replace(cars_data['owner'].unique(),[i for i in range(1,len(cars_data['owner'].unique())+1)]).astype(int)

  predict = model.predict(input_data_model)
  st.write('Car Price Should be {} rupees'.format(predict[0].round(2)))