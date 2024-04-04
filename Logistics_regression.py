import numpy as np
import streamlit as st
from sklearn.pipeline import Pipeline
from metrics import classifier_metrics
from sklearn.linear_model import LogisticRegressionCV


# Logistics Regression
def Logistics_sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    penalty = st.sidebar.selectbox(
        'Regularization',
        ('l2', 'l1','elasticnet')
    )


    if penalty=='l1':
        solver = st.sidebar.selectbox(
            'Solver',
            ('liblinear', 'saga')
        )
    
    elif penalty=='none':
        solver = st.sidebar.selectbox(
            'Solver',
            ('newton-cg', 'lbfgs', 'sag', 'saga')
        )

    elif penalty=='elasticnet':
        solver = st.sidebar.selectbox(
            'Solver',
            ('saga',)
        )
    else:
        solver = st.sidebar.selectbox(
            'Solver',
            ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
        )

    max_iter = int(st.sidebar.number_input('Max Iterations',value=100, step=10))

    cv = int(st.sidebar.number_input('Cross Validation',value=5, step=1, min_value=1,max_value=15))

    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Regularization:**
            Regularization technique used in logistic regression.
            - *l1*: L1 regularization.
            - *l2*: L2 regularization.
            - *elasticnet*: Elastic-Net regularization.
            
            **Solver:**
            Algorithm to use in the optimization problem.
            - *liblinear*: Library for large linear classification.
            - *saga*: Stochastic Average Gradient descent solver.
            - *newton-cg*: Newton-Conjugate Gradient algorithm.
            - *lbfgs*: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm.
            - *sag*: Stochastic Average Gradient descent solver.
            
            **Max Iterations:**
            Maximum number of iterations taken for the solvers to converge.
            
            **Cross Validation:**
            Number of folds in cross-validation.
            """
        )

    return penalty,solver,max_iter,cv

def Logistics_regression_implementation(preprocessor, X_train, y_train, X_test, y_test):
    # Create a pipeline with preprocessing and linear regression
    penalty,solver,max_iter,cv = Logistics_sidebar()
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LogisticRegressionCV(cv=cv,penalty=penalty,solver=solver,max_iter=max_iter,multi_class='auto',l1_ratios=[0.01,0.2,0.001,0.05]))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    classifier_metrics(y_test,y_pred,y_pred_proba)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return model
