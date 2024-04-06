import streamlit as st
from sklearn.pipeline import Pipeline
from metrics import regression_metrics, regression_visulizer
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, Ridge, Lasso

# Linear Regression
def Linear_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    # Create a pipeline with preprocessing and linear regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("")
    st.write("")
    regression_metrics(y_test,y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)
                        


    return model
    

# Ridge Regression
def Ridge_Regression_Sidebar():

    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    c1,c2,c3=st.columns(3)
    with c1:
        alpha = st.sidebar.slider("Alpha (Regularization strength)", 0.001, 1.0, step=0.001, value=1.0)
    with c2:
        solver = st.sidebar.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], index=6)
    with c3:
        max_iter = st.sidebar.slider("Max Iterations", 100, 20000, step=100, value=1000)
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Alpha (Regularization Strength):**
            Alpha controls the strength of the regularization. Larger values specify stronger regularization, which can help prevent overfitting.
            
            **Solver:**
            Solver refers to the algorithm used to compute the weights. The choice of solver can affect the speed and performance of the model.
            
            **Max Iterations:**
            Max Iterations determines the maximum number of iterations for the optimization algorithm to converge.
            """
        )
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")

    return alpha,solver,max_iter

def Ridge_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    alpha,solver,max_iter = Ridge_Regression_Sidebar()

    # Create a pipeline with preprocessing and linear regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(
            alpha=alpha,
            max_iter=max_iter,
            solver=solver
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    regression_metrics(y_test,y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)

    return model


# Lasso Regression
def Lasso_Regression_Sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")

    c1,c2=st.columns(2)
    with c1:
        alpha = st.sidebar.slider("Alpha (Regularization strength)", 0.001, 1.0, step=0.001, value=1.0)
    with c2:
        max_iter = st.sidebar.slider("Max Iterations", 1000, 20000, step=100, value=5000)

    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Alpha (Regularization Strength):**
            Alpha controls the strength of the regularization. Larger values specify stronger regularization, which can help prevent overfitting.
            
            **Max Iterations:**
            Max Iterations determines the maximum number of iterations for the optimization algorithm to converge.
            """
        )
    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.header('')

    return alpha,max_iter

def Lasso_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    alpha,max_iter = Lasso_Regression_Sidebar()

    # Create a pipeline with preprocessing and linear regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso(
            alpha=alpha,
            max_iter=max_iter,
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    regression_metrics(y_test,y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)


    return model

