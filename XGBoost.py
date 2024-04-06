import xgboost as xgb
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from metrics import classifier_metrics, regression_metrics, regression_visulizer


# XGBoost Regressor
def XGBoost_Regressor_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, value=3)
    min_child_weight = st.sidebar.slider("Min Child Weight", 1, 10, value=1)
    gamma = st.sidebar.slider("Gamma", 0.0, 1.0, value=0.0)

    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Number of Estimators:**
            The number of boosting rounds.
            
            **Learning Rate:**
            Step size shrinkage used in updates to prevent overfitting.
            Lower values make the model more robust, but it needs more boosting rounds (n_estimators).
            
            **Max Depth:**
            Maximum depth of a tree.
            Increasing this value will make the model more complex and more likely to overfit.
            
            **Min Child Weight:**
            Minimum sum of instance weight (hessian) needed in a child.
            It is used to control over-fitting.
            
            **Gamma:**
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
            """
        )
    
    return n_estimators, learning_rate, max_depth, min_child_weight, gamma

def XGBoost_Regressor_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_estimators, learning_rate, max_depth, min_child_weight, gamma = XGBoost_Regressor_Sidebar()

    # Create a pipeline with preprocessing and XGBoost Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    regression_metrics(y_test, y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)

    return model


# XGBoost Classifier
def XGBoost_Classifier_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, value=3)
    min_child_weight = st.sidebar.slider("Min Child Weight", 1, 10, value=1)
    gamma = st.sidebar.slider("Gamma", 0.0, 1.0, value=0.0)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Number of Estimators:**
            The number of boosting rounds.
            
            **Learning Rate:**
            Step size shrinkage used in updates to prevent overfitting.
            Lower values make the model more robust, but it needs more boosting rounds (n_estimators).
            
            **Max Depth:**
            Maximum depth of a tree.
            Increasing this value will make the model more complex and more likely to overfit.
            
            **Min Child Weight:**
            Minimum sum of instance weight (hessian) needed in a child.
            It is used to control over-fitting.
            
            **Gamma:**
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
            """
        )

    return n_estimators, learning_rate, max_depth, min_child_weight, gamma

def XGBoost_Classifier_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_estimators, learning_rate, max_depth, min_child_weight, gamma = XGBoost_Classifier_Sidebar()

    # Create a pipeline with preprocessing and XGBoost Classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    classifier_metrics(y_test, y_pred, y_pred_proba)

    return model

