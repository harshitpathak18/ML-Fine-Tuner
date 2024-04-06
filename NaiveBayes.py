import numpy as np
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from metrics import plot_confusion_matrix, plot_roc_curve, accuracy_score,  confusion_matrix, log_loss


# Naive Bayes Classifier
def NaiveBayes_sidebar():
    nb_type = st.selectbox(
        'Naive Bayes Type',
        ('Gaussian', 'Multinomial', 'Bernoulli')
    )

    return nb_type

def NaiveBayes_implementation(preprocessor, X_train, y_train, X_test, y_test):
    # Create a pipeline with preprocessing and selected Naive Bayes classifier
    nb_type = NaiveBayes_sidebar()
    
    if nb_type == 'Gaussian':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ])
    elif nb_type == 'Multinomial':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultinomialNB())
        ])
    elif nb_type == 'Bernoulli':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', BernoulliNB())
        ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Accuracy Score - {round(accuracy_score(y_test, y_pred)*100, 2)}%")

    with col2:
        loss = log_loss(y_test, y_pred)
        st.subheader(f"Log Loss - {round(loss, 2)}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_confusion_matrix(y_test, y_pred))
    with col2:
        outputs = len(y_test.value_counts().unique())
        if outputs==2:
            st.plotly_chart(plot_roc_curve(y_test, y_pred, prob=False))

    return model
