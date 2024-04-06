import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline
from metrics import classifier_metrics, regression_metrics, regression_visulizer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, log_loss

# K Nearest Neighbors Regresssion
def Knn_sidebar_regressor(y_train):
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")
    n_neighbour= st.sidebar.slider("N Neighbour", min_value=2, max_value=round(len(y_train)**0.5),  value=5,  step=1)
    weights = st.sidebar.selectbox("Weight function", ["uniform", "distance"])
    algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree"])
    metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])

    # Expandable section for parameter explanation
    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **N Neighbours:**
            Number of neighbors to use for kneighbors queries. 
            It should be less than or equal to the total number of samples.
            
            **Weight Function:**
            Function used to calculate the weights of neighbors.
            - *uniform*: All points in each neighborhood are weighted equally.
            - *distance*: Weight points by the inverse of their distance.
            
            **Algorithm:**
            Algorithm used to compute the nearest neighbors:
            - *auto*: Choose the most appropriate algorithm automatically.
            - *ball_tree*: Use BallTree for queries.
            - *kd_tree*: Use KDTree for queries.
            
            **Distance Metric:**
            The distance metric used for the tree. 
            The Minkowski metric with p=2 is equivalent to the standard Euclidean metric.
            - *euclidean*: Standard Euclidean distance.
            - *manhattan*: Manhattan distance.
            - *chebyshev*: Chebyshev distance.
            - *minkowski*: Minkowski distance with a parameter p.
            """
        )
    st.sidebar.header("")
    st.sidebar.header("")

    return  n_neighbour, weights, algorithm, metric

def Knn_plot_regressor(X_train,y_train,X_test,y_test, preprocessor, weights, algorithm, metric):
    # Initialize lists to store accuracy scores for different K values
    mae_scores = []
    r2_scores = []
    
    val_range = np.array([i for i in range(1,25)])
    # Iterate over the K values
    for k in val_range:
        # Create a pipeline with scaling and KNN regressor
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(
                n_neighbors=k,
                weights=weights,
                algorithm=algorithm,
                metric=metric
                ))
        ])
        
        # Fit the pipeline on the training data
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate mean absolute error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)
        
        # Calculate R-squared (R2) score
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    # Plot the scores
    
    col1,col2=st.columns(2) 
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val_range, y=r2_scores, mode='lines+markers', name='R-squared (R2) Score', line=dict(color='lightgreen'),marker=dict(color='darkgreen', size=8)))
        fig.update_layout(
            title='R2 Score at Different Values of K',
            xaxis_title='K',
            yaxis_title='Score',
            xaxis=dict(tickvals=val_range),
            width=500,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val_range, y=mae_scores, mode='lines+markers', name='Mean Absolute Error', line=dict(color='orange'),marker=dict(color='red', size=8)))
        fig.update_layout(
            title='MAE at Different Values of K',
            xaxis_title='K',
            yaxis_title='Score',
            xaxis=dict(tickvals=val_range),
            width=500,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
            )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Display the plot using Streamlit
        st.plotly_chart(fig)

def Knn_Regression_Implementation(preprocessor, X_train, y_train, X_test, y_test):
    n_neighbour, weights, algorithm, metric = Knn_sidebar_regressor(y_train)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(
            n_neighbors=n_neighbour,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    regression_metrics(y_test, y_pred)
    regression_visulizer(model,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None)

    Knn_plot_regressor(X_train, y_train, X_test, y_test, preprocessor, weights, algorithm, metric)

    return model







# K Nearest Neightbour Classification
def Knn_sidebar_classifier(y_train):
    st.sidebar.subheader("")
    st.sidebar.header("Parameter Tuning")
    st.sidebar.subheader("")
    n_neighbour= st.sidebar.slider("N Neighbour", min_value=2, max_value=round(len(y_train)**0.5),  value=5,  step=1)
    weights = st.sidebar.selectbox("Weight function", ["uniform", "distance"])
    algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree"])
    metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])

    with st.sidebar.expander("About Parameters", expanded=False):
        st.write(
            """
            **N Neighbours:**
            Number of neighbors to use for classification.
            It should be less than or equal to the total number of samples.
            
            **Weight Function:**
            Function used to calculate the weights of neighbors.
            - *uniform*: All points in each neighborhood are weighted equally.
            - *distance*: Weight points by the inverse of their distance.
            
            **Algorithm:**
            Algorithm used to compute the nearest neighbors:
            - *auto*: Choose the most appropriate algorithm automatically.
            - *ball_tree*: Use BallTree for queries.
            - *kd_tree*: Use KDTree for queries.
            
            **Distance Metric:**
            The distance metric used for the tree. 
            - *euclidean*: Standard Euclidean distance.
            - *manhattan*: Manhattan distance.
            - *chebyshev*: Chebyshev distance.
            - *minkowski*: Minkowski distance with a parameter p.
            """
        )
    st.sidebar.header("")
    st.sidebar.header("")

    return  n_neighbour, weights, algorithm, metric

def Knn_plot_classifier(X_train,y_train,X_test,y_test, preprocessor, weights, algorithm, metric):
    # Initialize lists to store accuracy scores for different K values
    accuracy_scores = []
    log_loss_values = []
    
    val_range = np.array([i for i in range(1,25)])
    # Iterate over the K values
    for k in val_range:
        # Create a pipeline with scaling and KNN regressor
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(
                n_neighbors=k,
                weights=weights,
                algorithm=algorithm,
                metric=metric
                ))
        ])
        
        # Fit the pipeline on the training data
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate mean absolute error (MAE)
        acc = accuracy_score(y_test, np.array([round(i) for i in y_pred]))
        accuracy_scores.append(acc)
        
        # Calculate R-squared (R2) score
        if y_test.nunique()<=2:
            lg_loss = log_loss(y_test, y_pred) 
            log_loss_values.append(lg_loss)

    # Plot the scores
    
    col1,col2=st.columns(2) 
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val_range, y=accuracy_scores, mode='lines+markers', name='Accuracy' ,line=dict(color='lightgreen'),marker=dict(color='darkgreen', size=8)))
        fig.update_layout(
            title='Accuracy at Different Values of K',
            xaxis_title='K',
            yaxis_title='Accuracy',
            xaxis=dict(tickvals=val_range),
            width=500,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
            )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

    with col2:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=val_range, y=log_loss_values, mode='lines+markers', name='LogLoss', line=dict(color='orange'),marker=dict(color='red', size=8)))
            fig.update_layout(
                title='Logloss at Different Values of K',
                xaxis_title='K',
                yaxis_title='Logloss',
                xaxis=dict(tickvals=val_range),
                width=500,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50),
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)


            # Display the plot using Streamlit
            st.plotly_chart(fig)
        except Exception as e:
            st.write("")
    
def Knn_classification_implementation(preprocessor, X_train, y_train, X_test, y_test):
    # Create a pipeline with preprocessing and linear regression
    n_neighbour, weights, algorithm, metric = Knn_sidebar_classifier(y_train)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsClassifier(n_neighbors=n_neighbour,weights=weights,algorithm=algorithm,metric=metric))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    classifier_metrics(y_test,y_pred,y_pred_proba)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    Knn_plot_classifier(X_train,y_train,X_test,y_test, preprocessor, weights, algorithm, metric)
    
    return model

