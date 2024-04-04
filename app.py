# Importing ML Libraries
import pickle
import pandas as pd
import streamlit as st
from io import StringIO
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions, plot_learning_curves
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder


# Importing ML Alorithm Implementation
from Logistics_regression import Logistics_regression_implementation
from Knn import Knn_classification_implementation, Knn_Regression_Implementation
from Svm import Svm_classification_implementation, Svm_regression_implementation
from XGBoost import XGBoost_Classifier_Implementation, XGBoost_Regressor_Implementation
from AdaBoost import AdaBoost_Classifier_Implementation, AdaBoost_Regressor_Implementation
from DecisionTree import Decision_Tree_Classifier_Implementation, Decision_Tree_Regressor_Implementation
from RandomForest import Random_Forest_Classifier_Implementation, Random_Forest_Regressor_Implementation
from GradientBoosting import Gradient_Boosting_Classifier_Implementation, Gradient_Boosting_Regressor_Implementation
from Linear_regression import Linear_Regression_Implementation, Ridge_Regression_Implementation, Lasso_Regression_Implementation


# Function to style streamlit page
def streamlit_style():
    st.markdown("""
    <style>
        .st-emotion-cache-16txtl3{
            padding:1rem 1.5rem;
        }

        .st-emotion-cache-1y4p8pa {
            padding:2rem 1rem 10rem;
            max-width: 65rem
        }   

        .st-emotion-cache-uf99v8 {
            background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
        }

        .st-emotion-cache-1avcm0n {
            background-color: rgba(255, 255, 255, 0);
        }

        /*Side Bar*/
        .st-emotion-cache-16txtl3{
            background-image: linear-gradient(to right top, #dd3bce, #b35ee3, #8574ed, #5482eb, #1f8be0);
            background-image: linear-gradient(to right top, #091b36, #081d3b, #081f3f, #062244, #052449);            background-repeat: no-repeat; /* Prevent image from repeating */
            background-position: center;
        }

        .st-emotion-cache-10y5sf6{
            color:white;
            font-weight:800;
        }

        .st-hf {
            background: linear-gradient(to right, rgb(255, 75, 75) 0%, rgb(255, 75, 75) 47.3684%, rgba(172, 177, 195, 0.25) 47.3684%, rgba(172, 177, 195, 0.25) 100%);
        }
        
        /*Cvs file uploader*/
        .st-emotion-cache-1erivf3,.st-da, .st-emotion-cache-1erivf3, .st-ek, .st-gf{
            background-image: linear-gradient(to right top, #1d3354, #25537b, #2576a2, #1e9bc8, #12c2eb);
        }

        /*fileUploadLabel*/
        .st-emotion-cache-19rxjzo.ef3psqc12{
            color:black;
        }


        .st-emotion-cache-19rxjzo{
            background-color:white;
        }  
        
    </style>
    """, unsafe_allow_html=True)

streamlit_style()


# Function to load data and display that data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Dataset Summary
    st.markdown("<h4>Dataset Overview - </h4>", unsafe_allow_html=True)
    tb1, tb2, tb3 = st.tabs(['Head', 'Tail', 'All'])
    with tb1:
        st.table(df.head(5))
    with tb2:
        st.table(df.tail(5))
    with tb3:
        st.write(df)

    # Display dataset shape, null values, and duplicates
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"Shape - {df.shape}")
    with c2:
        st.write(f"Null - {df.isnull().sum().sum()}")
    with c3:
        st.write(f"Duplicates - {df.duplicated().sum()}")

    return df


# Code for mapping correlation matrix
def correlation_matrix(df,numerical_features):
    corr_matrix = df[numerical_features].corr().round(2)
    st.markdown("<h4>Correlation Heatmap</h4>", unsafe_allow_html=True)

    color = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']

    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=corr_matrix.index.tolist(),
        y=corr_matrix.columns.tolist(),
        showscale=True
    )

    fig.update_layout(width=700, height=500,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',)
    st.plotly_chart(fig)


# Dependent and Independent Features
def dep_indep_features(df):
    st.markdown("<h4>Select Dependent & Independent Features - </h4>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        dependent_feature = st.selectbox('Select Dependent Feature', sorted(list(set(df.columns))))
    with c2:
        independent_features = st.multiselect('Select Independent Features', [i for i in df.columns if i != dependent_feature])
    return dependent_feature,independent_features


# Code to Save the Trained Model using Selected Parameters
def save_model(model):
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)


# Used to make prediction on trained model
def predictions(independent_features, numerical_features, categorical_features, dependent_feature,model, df):
    st.title("")

    with st.expander("Make Predictions Using Trained model"):
        st.header("Predictions")

        # Input values for independent features
        input_values = {}
        for feature in independent_features:
            if feature in numerical_features:
                input_values[feature] = st.number_input(f'Enter value for {feature}', value=0.0)
            elif feature in categorical_features:
                input_values[feature] = st.selectbox(f'Select value for {feature}', sorted(df[feature].unique()))

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([input_values])

        # Make prediction
        pred = st.button("Predict")
        if pred:
            prediction = model.predict(input_data)
            st.write(f"Predicted {dependent_feature}: {round(prediction[0])}")


    with st.expander("Save Trained Model"):
        save_model(model)
        
        with open('trained_model.pkl', 'rb') as f:
            model_bytes = f.read()
            st.download_button(
                label="Download Model",
                data=model_bytes,
                file_name='trained_model.pkl',
                mime='application/octet-stream'
            )





# main
def main():
    st.markdown("<h1><center>Machine Learning Fine Tuner</center></h1>",unsafe_allow_html=True)
    st.title("")
    uploaded_file = st.file_uploader("Upload Preprocessed CSV File", type=['csv'])

    if uploaded_file is not None:
        # loading data
        df = load_data(uploaded_file)

        # displaying options for user to select features    
        st.title(" ")
        dependent_feature,independent_features =  dep_indep_features(df)

        
        # Calculate the correlation matrix
        if dependent_feature and independent_features:
            features = independent_features + [dependent_feature]
            numerical_features = [feature for feature in features if df[feature].dtype != 'O']
            categorical_features = [feature for feature in features if df[feature].dtype == 'O']



            # displaying correlation matrix
            st.title("")
            correlation_matrix(df,numerical_features)


            # Selecting the test size and random state
            c1,c2=st.columns(2)
            with c1:
                Test_size = st.slider('Select Test Size (in percentage): ', 5, 50, value=20) / 100
            with c2:
                random_ = st.slider('Select Random State: ', 1, 100, value=42)

            X = df[independent_features]
            y = df[dependent_feature]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_size, random_state=random_)

            st.write("")

            # Displaying Training and Testing data Shape
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"X_train Shape - {X_train.shape}")
                st.write(f"X_test Shape - {X_test.shape}")
           
            with col2:
                st.write(f"y_train Shape - {y_train.shape}")
                st.write(f"y_test Shape - {y_test.shape}")



            st.title("")
            col1, col2 = st.columns(2)
            # Applying Scaling
            with col1:
                transform = st.selectbox("Select Transformation", ["Standard Scaler", "Min-Max Scaling"])
                if transform == "Standard Scaler":
                    numerical_transformer = StandardScaler()
                elif transform == "Min-Max Scaling":
                    numerical_transformer = MinMaxScaler()
            
            # Applying Encoder
            with col2:
                encoding = st.selectbox("Encoder", ["One Hot Encoding"])
                if encoding == "One Hot Encoding":
                    categorical_transformer = OneHotEncoder(drop='first', sparse=False)
                if encoding == "Label Encoding":
                    class ModifiedLabelEncoder(LabelEncoder):
                        def fit_transform(self, y, *args, **kwargs):
                            return super().fit_transform(y).reshape(-1, 1)

                        def transform(self, y, *args, **kwargs):
                            return super().transform(y).reshape(-1, 1)
                    
                    categorical_transformer = ModifiedLabelEncoder()

            

            numerical_features = [feature for feature in numerical_features if feature != dependent_feature]
            if numerical_transformer and categorical_transformer:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, numerical_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])


                st.title("")
                
                
                if len(y.unique())<20:
                    # Select Algorithm
                    model = st.selectbox("Select Machine Learning Algorithm", ['Logistics Regression','K-Nearest Neighbour Classifier','Support Vector Machine Classifier','Decision Tree Classifier', "Random Forest Classifier", "Gradient Boosting Classifier", "AdaBoost Classifier", "XGBoost Classifier"])
                    st.markdown(f"<br><h2><center>{model}</h2></center><br>",unsafe_allow_html=True)

                    if model == "Logistics Regression":
                        regressor = Logistics_regression_implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor,df)
                        st.sidebar.write("")
                        st.sidebar.write("")
                        st.sidebar.write("")
                        st.sidebar.write("")
                    
                    if model == "K-Nearest Neighbour Classifier":
                        regressor = Knn_classification_implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor,df)
                        st.sidebar.write("")
                        st.sidebar.write("")
                        st.sidebar.write("")
                        st.sidebar.write("")

                    if model == "Support Vector Machine Classifier":
                        regressor = Svm_classification_implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor,df)
                    
                    if model =='Decision Tree Classifier':
                        regressor = Decision_Tree_Classifier_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =="Random Forest Classifier":
                        regressor = Random_Forest_Classifier_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =="Gradient Boosting Classifier":
                        regressor = Gradient_Boosting_Classifier_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =='AdaBoost Classifier':
                        regressor = AdaBoost_Classifier_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =='XGBoost Classifier':
                        regressor = XGBoost_Classifier_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    


                else:
                    model = st.selectbox("Select Machine Learning Algorithm", ['Linear Regression','Ridge Regression','Lasso Regression','K-Nearest Neighbour Regressor', 'Support Vector Machine Regressor','Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'AdaBoost Regressor', "XGBoost Regressor"])
                    st.markdown(f"<h2><center>{model}</h2></center><br>",unsafe_allow_html=True)


                    if model == "Linear Regression":
                        regressor = Linear_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model == "Ridge Regression":
                        regressor = Ridge_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)
                        st.sidebar.title("")
                        st.sidebar.title("")
                        st.sidebar.title("")
                        st.sidebar.title("")


                    if model == "Lasso Regression":
                        regressor = Lasso_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)
                        st.sidebar.title("")
                        st.sidebar.title("")
                        st.sidebar.title("")
                        st.sidebar.title("")
                        st.sidebar.title("")
                        st.sidebar.title("")
                    
                    if model=="K-Nearest Neighbour Regressor":
                        regressor= Knn_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)
                        st.sidebar.title("")


                    if model== 'Support Vector Machine Regressor':
                        regressor= Svm_regression_implementation(preprocessor, X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =='Decision Tree Regressor':
                        regressor = Decision_Tree_Regressor_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =="Random Forest Regressor":
                        regressor = Random_Forest_Regressor_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =="Gradient Boosting Regressor":
                        regressor = Gradient_Boosting_Regressor_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =='AdaBoost Regressor':
                        regressor = AdaBoost_Regressor_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)

                    if model =='XGBoost Regressor':
                        regressor = XGBoost_Regressor_Implementation(preprocessor,X_train, y_train, X_test, y_test)
                        predictions(independent_features, numerical_features, categorical_features, dependent_feature, regressor, df)



if __name__=="__main__":
    try:
        main()

    except Exception as e:
        st.error(f"Error has occured - {e}")