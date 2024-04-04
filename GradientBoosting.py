import streamlit as st
from sklearn.pipeline import Pipeline
from metrics import classifier_metrics, regression_metrics
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting Regressor
def Gradient_Boosting_Regressor_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    loss = st.sidebar.selectbox("Loss Function", ['absolute_error', 'squared_error', 'huber', 'quantile'], index=0)
    max_features = st.sidebar.selectbox("Max Features", [None, "sqrt", "log2"], index=0)
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, value=3)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, value=1)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Number of Estimators:**
            The number of boosting stages to be run. 
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
            
            **Learning Rate:**
            Shrinks the contribution of each tree. 
            There is a trade-off between learning rate and the number of estimators. 
            Lower learning rates usually require more trees.
            
            **Loss Function:**
            The loss function to be optimized. 
            Different loss functions yield different algorithms:
            - *absolute_error*: The absolute error loss (also known as L1 loss).
            - *squared_error*: The squared error loss (also known as L2 loss).
            - *huber*: Huber loss for robust regression.
            - *quantile*: Quantile loss allows quantile regression (use alpha to specify the quantile).
            
            **Max Depth:**
            The maximum depth of the individual regression estimators. 
            The maximum depth limits the number of nodes in the tree. 
            Tune this parameter for best performance; the best value depends on the interaction of the input variables.
            
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

    return n_estimators, learning_rate, loss, max_depth, min_samples_split, min_samples_leaf, max_features

def Gradient_Boosting_Regressor_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_estimators, learning_rate, loss, max_depth, min_samples_split, min_samples_leaf, max_features = Gradient_Boosting_Regressor_Sidebar()

    # Create a pipeline with preprocessing and Gradient Boosting Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
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



# Gradient Boosting Classifier
def Gradient_Boosting_Classifier_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    max_features = st.sidebar.selectbox("Max Features", [None, "sqrt", "log2"], index=0)
    criterion = st.sidebar.selectbox("Criterion", ["friedman_mse", 'squared_error'], index=0)
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, value=3)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, value=1)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Number of Estimators:**
            The number of boosting rounds.
            
            **Learning Rate:**
            Step size shrinkage used in updates to prevent overfitting.
            Lower values make the model more robust, but it needs more boosting rounds (n_estimators).
            
            **Criterion:**
            The function to measure the quality of a split.
            - *friedman_mse*: Friedman mean squared error.
            - *squared_error*: Mean squared error.
            
            **Max Depth:**
            Maximum depth of a tree.
            Increasing this value will make the model more complex and more likely to overfit.
            
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

    return n_estimators, learning_rate, criterion, max_depth, min_samples_split, min_samples_leaf, max_features

def Gradient_Boosting_Classifier_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_estimators, learning_rate, criterion, max_depth, min_samples_split, min_samples_leaf, max_features = Gradient_Boosting_Classifier_Sidebar()

    # Create a pipeline with preprocessing and Gradient Boosting Classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
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

    classifier_metrics(y_test, y_pred, y_pred_proba)

    return model
