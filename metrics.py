import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,mean_absolute_error, r2_score, log_loss, roc_curve, auc, roc_auc_score


def regression_visulizer(regressor,X_test,y_test,X_train,y_train,scaler=None, X=None, y=None):

    # Predict the values for training and testing sets
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)


    # Create Plotly figures
    fig_train = go.Figure()
    fig_test = go.Figure()

    # Add the training data points
    fig_train.add_trace(go.Scatter(x=y_train, y=y_pred_train, mode='markers', name='Predicted vs True (Training)', marker=dict(color='#051937')))
    # Add the perfect fit line for training data
    fig_train.add_trace(go.Scatter(x=[min(y_train), max(y_train)], y=[min(y_train), max(y_train)], mode='lines', name='Perfect Fit (Training)', line=dict(color='#86BD0A')))

    # Add the testing data points
    fig_test.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='Predicted vs True (Testing)', marker=dict(color='#051937')))
    # Add the perfect fit line for testing data
    fig_test.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='Perfect Fit (Testing)', line=dict(color='#86BD0A')))

    # Update layout for both figures
    for fig in [fig_train, fig_test]:
        fig.update_layout(
            xaxis_title='True Progression',
            yaxis_title='Predicted Progression',
            legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)'),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            width=450
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    # Show plots in Streamlit
    st.markdown("""<center><h4>Model Performance</h4></center>""", unsafe_allow_html=True)

    # Display plots side by side using columns layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<center><h5>Training Data</h5></center>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c2:
            st.write(f"Mean Absolute Error - {round(mean_absolute_error(y_train, y_pred_train), 2)}")
        with c1:
            st.write(f"R2 Score - {round(r2_score(y_train, y_pred_train)*100, 2)}%")


        st.plotly_chart(fig_train)

    with col2:
        st.markdown("""<center><h5>Testing Data</h5></center>""", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c4:
            st.write(f"Mean Absolute Error - {round(mean_absolute_error(y_test, y_pred_test), 2)}")
        with c3:
            st.write(f"R2 Score - {round(r2_score(y_test, y_pred_test)*100, 2)}%")
        st.plotly_chart(fig_test)

    

    if scaler:
        Y_Pred = regressor.predict(scaler.fit_transform(X))
        chart_data = {"Actual": y, "Predicted": Y_Pred}
        fig = px.scatter(
            chart_data,
            x="Actual",
            y="Predicted",
            log_x=True,
            color_discrete_sequence=['#032D42']
        )

        fig.update_layout(
            title="Plotting All Data",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
            )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig, theme=None, use_container_width=True)


def classification_visualizer(clf1,X, y, X_test,y_test):
    y_pred = clf1.predict(X_test) 

    try:
        y_pred_proba =clf1.predict_proba(X_test)
        col1, col2 = st.columns(2)
        with col2:
            st.subheader(f"Log loss - {round(log_loss(y_test, y_pred_proba), 2)}")
        with col1:
            st.subheader(f"Accuracy Score - {round(accuracy_score(y_test, y_pred)*100, 2)}%")
    except Exception as e:
        pass

    
    col1,col2=st.columns(2)
    with col1:
        # Create mesh grid to cover the entire feature space
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision regions using Plotly
        fig = go.Figure()

        # Add contour plot for decision regions
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z,
                                colorscale='blues', showscale=False,
                                opacity=1))

        # Add scatter plot for data points
        colorsclale_options=['viridis', 'electric','edge']
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                                marker=dict(color=y, colorscale='electric', size=10),
                                showlegend=False))

        fig.update_layout(title="Classifier Decision Regions",
                        xaxis_title="Feature 1",
                        yaxis_title="Feature 2",
                        width=450,
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',)

        # Show plot using Streamlit
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)


    with col2:
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = ff.create_annotated_heatmap(z=cm,
                                        x=[f'Predicted ' + str(i) for i in range(len(cm))],
                                        y=[f'Actual ' + str(i) for i in range(len(cm))],
                                        colorscale='speed')

        fig.update_layout(title='Confusion Matrix',
                        xaxis_title=f'Predicted',
                        yaxis_title=f'Actual',
                        width=400,
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',)
        
        st.plotly_chart(fig)



def plot_roc_curve(y_test, y_prob, prob=False):
    # Compute ROC curve and ROC area for each class
    if prob:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    else:
        fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             mode='lines',
                             name='ROC curve (area = %0.2f)' % roc_auc))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines', line_dash='dash', name='Random'))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      height=400, 
                      width=450,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_confusion_matrix(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    fig = ff.create_annotated_heatmap(z=cm,
                                      x=[f'Predicted ' + str(i) for i in range(len(cm))],
                                      y=[f'Actual ' + str(i) for i in range(len(cm))],
                                      colorscale='speed')

    fig.update_layout(title='Confusion Matrix',
                      xaxis_title=f'Predicted {str.capitalize(y_true.name)}',
                      yaxis_title=f'Actual {str.capitalize(y_true.name)}',
                      width=400,
                      height=400,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',)
    
    return fig


def classifier_metrics(y_test, y_pred, y_pred_proba):
    col1, col2 = st.columns(2)
    with col2:
        st.subheader(f"Log loss - {round(log_loss(y_test, y_pred_proba), 2)}")
    with col1:
        st.subheader(f"Accuracy Score - {round(accuracy_score(y_test, y_pred)*100, 2)}%")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_confusion_matrix(y_test, y_pred))
    with col2:
        outputs = len(y_test.value_counts().unique())
        if outputs==2:
            st.plotly_chart(plot_roc_curve(y_test, y_pred_proba, prob=True))
        


def regression_metrics(y_test,y_pred):
    col1, col2 = st.columns(2)
    with col2:
        st.subheader(f"Mean Absolute Error - {round(mean_absolute_error(y_test, y_pred), 2)}")
    with col1:
        st.subheader(f"R2 Score - {round(r2_score(y_test, y_pred)*100, 2)}%")

    chart_data = {"Actual": y_test, "Predicted": y_pred}
    fig = px.scatter(
        chart_data,
        x="Actual",
        y="Predicted",
        log_x=True,
        color_discrete_sequence=['#032D42']
    )

    fig.update_layout(
        title="",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, theme=None, use_container_width=True)


