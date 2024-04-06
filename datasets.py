import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_blobs

from sklearn.datasets import load_diabetes, fetch_california_housing


def Regression_Datasets():
    data = st.selectbox("Select Dataset", ['Diabetes','California Housing', "Diamonds", 'Tips'])

    if data=="Diabetes":
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target

    if data=="California Housing":
        california_housing = fetch_california_housing()
        X, y = california_housing.data, california_housing.target

    if data=="Diamonds":
        diamonds = sns.load_dataset("diamonds")
        diamonds = diamonds.dropna()  
        X = diamonds[['carat']] 
        y = diamonds['price']  

    if data=='Tips':
        tips = sns.load_dataset("tips")
        tips = tips.dropna()  
        X = tips[['total_bill']]  
        y = tips['tip'] 

    
    return X,y


def Classification_Datasets():
    c1,c2,c3=st.columns(3)
    with c1:
        n_sample = st.slider('Number of samples', min_value=1000, max_value=10000, value=1000, step=100)
    with c2:
        centers = st.number_input("Number of classes", min_value=2,  value=2, max_value=5)
    with c3:
        cluster_std = st.number_input("Select Standard Deviation", min_value=1, value=1,  max_value=10)
    
    X,y = make_blobs(n_samples=n_sample, centers=centers, cluster_std=cluster_std, random_state=42)

    return X,y