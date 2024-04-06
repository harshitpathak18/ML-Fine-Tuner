import streamlit as st
from sklearn.pipeline import Pipeline
from metrics import classifier_metrics, regression_metrics, regression_visulizer
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor


# AdaBoost Regressor
def AdaBoost_Regressor_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    loss = st.sidebar.selectbox("Loss Function", ["linear", "square", "exponential"], index=0)
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, value=50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=1.0)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Number of Estimators:**
            This parameter determines the number of individual estimators (weak learners) to use in the boosting process. 
            Increasing the number of estimators can improve the model's performance but might also increase computation time.
            
            **Learning Rate:**
            Learning rate shrinks the contribution of each weak learner. 
            It's a crucial parameter to control overfitting. 
            Lower values generally require more estimators.
            
            **Loss Function:**
            The loss function to use when updating the weights after each boosting iteration. 
            It affects how AdaBoost models deal with errors. 
            - *linear*: Linear loss function. 
            - *square*: Quadratic loss function. 
            - *exponential*: Exponential loss function.
            """
        )

    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")

    return n_estimators, learning_rate, loss

def AdaBoost_Regressor_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_estimators, learning_rate, loss = AdaBoost_Regressor_Sidebar()

    # Create a pipeline with preprocessing and AdaBoost Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    regression_metrics(y_test, y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)


    return model




# AdaBoost Classifier
def AdaBoost_Classifier_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    algorithm = st.sidebar.selectbox("Algorithm", ["SAMME", "SAMME.R"], index=0)
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, value=50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=1.0)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Number of Estimators:**
            The number of boosting rounds.
            
            **Learning Rate:**
            Step size shrinkage used in updates to prevent overfitting.
            Lower values make the model more robust, but it needs more boosting rounds (n_estimators).
            
            **Algorithm:**
            The algorithm to use for updating the weights after each boosting round.
            - *SAMME*: Stagewise Additive Modeling using a Multiclass Exponential loss function.
            - *SAMME.R*: The SAMME.R algorithm is similar to SAMME, but it uses the predicted class probabilities rather than the predicted classes themselves.
            """
        )
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")

    return n_estimators, learning_rate, algorithm

def AdaBoost_Classifier_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_estimators, learning_rate, algorithm = AdaBoost_Classifier_Sidebar()

    # Create a pipeline with preprocessing and AdaBoost Classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    classifier_metrics(y_test, y_pred, y_pred_proba)

    return model
