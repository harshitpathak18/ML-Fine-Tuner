import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,mean_absolute_error, r2_score, log_loss, roc_curve, auc, roc_auc_score


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
                      yaxis_title=f'True {str.capitalize(y_true.name)}',
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
