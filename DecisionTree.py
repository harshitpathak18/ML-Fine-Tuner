import streamlit as st
from sklearn.pipeline import Pipeline
from metrics import classifier_metrics, regression_metrics
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Decision Tree Regressor
def Decision_Tree_Regressor_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    criterion = st.sidebar.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0)
    max_features = st.sidebar.selectbox("Max Features", [None, "sqrt", "log2"], index=0)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, value=5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, value=1)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Criterion:**
            The function to measure the quality of a split. 
            Supported criteria are "squared_error" for the mean squared error, "friedman_mse" for the mean squared error with Friedman's improvement score, "absolute_error" for the mean absolute error, and "poisson" for the Poisson deviance.
            
            **Max Depth:**
            The maximum depth of the tree. 
            Increasing this value may lead to overfitting. 
            
            **Min Samples Split:**
            The minimum number of samples required to split an internal node. 
            Increase this value to prevent overfitting.
            
            **Min Samples Leaf:**
            The minimum number of samples required to be at a leaf node. 
            Increase this value to prevent overfitting.
            
            **Max Features:**
            The number of features to consider when looking for the best split. 
            - *None*: All features are considered.
            - *"sqrt"*, *"log2"*: The square root or logarithm of the total number of features is considered.
            """
        )

    return criterion, max_depth, min_samples_split, min_samples_leaf, max_features

def Decision_Tree_Regressor_Implementation(preprocessor,X_train, y_train, X_test, y_test):
    criterion, max_depth, min_samples_split, min_samples_leaf, max_features = Decision_Tree_Regressor_Sidebar()

    # Create a pipeline with preprocessing and Decision Tree Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    regression_metrics(y_test, y_pred)

    return model




# Decision Tree Classifier
def Decision_Tree_Classifier_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"], index=0)
    max_features = st.sidebar.selectbox("Max Features", [None,"sqrt", "log2"], index=0)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, value=5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, value=1)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Criterion:**
            The function to measure the quality of a split.
            - *gini*: Gini impurity.
            - *entropy*: Information gain.
            
            **Max Depth:**
            The maximum depth of the tree. 
            Increasing this value may lead to overfitting.
            
            **Min Samples Split:**
            The minimum number of samples required to split an internal node. 
            Increase this value to prevent overfitting.
            
            **Min Samples Leaf:**
            The minimum number of samples required to be at a leaf node. 
            Increase this value to prevent overfitting.
            
            **Max Features:**
            The number of features to consider when looking for the best split. 
            - *None*: All features are considered.
            - *"sqrt"*, *"log2"*: The square root or logarithm of the total number of features is considered.
            """
        )

    return criterion, max_depth, min_samples_split, min_samples_leaf, max_features

def Decision_Tree_Classifier_Implementation(preprocessor,X_train, y_train, X_test, y_test):
    criterion, max_depth, min_samples_split, min_samples_leaf, max_features = Decision_Tree_Classifier_Sidebar()

    # Create a pipeline with preprocessing and Decision Tree Classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    classifier_metrics(y_test,y_pred,y_pred_proba)


    return model
