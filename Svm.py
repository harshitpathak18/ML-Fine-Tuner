import numpy as np
import streamlit as st
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from metrics import classifier_metrics, regression_metrics, plot_confusion_matrix, plot_roc_curve, regression_visulizer


def hinge_loss(y_true, y_pred):
    loss = np.maximum(0, 1 - y_true * y_pred)
    avg_loss = np.mean(loss)
    return avg_loss



# SVM Regression
def Svm_regression_sidebar():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")
    C = st.sidebar.slider("C (Regularization parameter)", min_value=1, max_value=20, value=4, step=1)
    degree = st.sidebar.slider("Degree (Degree of polynomial kernel)", min_value=1, max_value=10, value=3, step=1)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])

    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Kernel:**
            Specifies the kernel type to be used in the algorithm.
            - *linear*: Linear kernel.
            - *poly*: Polynomial kernel.
            - *rbf*: Radial basis function (RBF) kernel.
            - *sigmoid*: Sigmoid kernel.
            
            **C (Regularization parameter):**
            Penalty parameter C of the error term.
            
            **Degree (Degree of polynomial kernel):**
            Degree of the polynomial kernel function ('poly').
            
           """
        )

    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")

    return kernel, C,  degree

def Svm_regression_implementation(preprocessor, X_train, y_train, X_test, y_test):
    kernel, C, degree = Svm_regression_sidebar()

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel=kernel, C=C,  degree=degree))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    regression_metrics(y_test, y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)

    loss = hinge_loss(y_test, y_pred)
    st.subheader(f"Hinge Loss - {round(loss, 2)}")


    
    return model





# Support Vector Machine Regression
def  Svm_sidebar_classifier():
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")
    C = st.sidebar.slider("C (Regularization parameter)", min_value=1, max_value=20, value=4, step=1)
    degree = st.sidebar.slider("Degree (Degree of polynomial kernel)", min_value=1, max_value=10, value=3, step=1)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **Kernel:**
            Specifies the kernel type to be used in the algorithm.
            - *linear*: Linear kernel.
            - *poly*: Polynomial kernel.
            - *rbf*: Radial basis function (RBF) kernel.
            - *sigmoid*: Sigmoid kernel.
            
            **C (Regularization parameter):**
            Penalty parameter C of the error term.
            
            **Degree (Degree of polynomial kernel):**
            Degree of the polynomial kernel function ('poly').
            
            """
        )

    st.sidebar.title("")
    st.sidebar.title("")

    return kernel, C, degree

def Svm_classification_implementation(preprocessor, X_train, y_train, X_test, y_test):
    # Create a pipeline with preprocessing and linear regression
    kernel, C,  degree= Svm_sidebar_classifier()

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVC(kernel=kernel, C=C, degree=degree))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Accuracy Score - {round(accuracy_score(y_test, y_pred)*100, 2)}%")

    with col2:
        loss = hinge_loss(y_test, y_pred)
        st.subheader(f"Hinge Loss - {round(loss, 2)}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_confusion_matrix(y_test, y_pred))
    with col2:
        outputs = len(y_test.value_counts().unique())
        if outputs==2:
            st.plotly_chart(plot_roc_curve(y_test, y_pred, prob=False))
    return model
