import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import plotly.express as px

# Set page layout
st.set_page_config(
    page_title='Content-Based Phishing Detection',
    page_icon='🎣',
    layout='wide'
)

# Title and sidebar
st.title('Content-Based Phishing Detection')
st.sidebar.header('Navigation')

# Sidebar options
page_options = [
    'Home',
    'Dataset Overview',
    'Model Evaluation',
    'Phishing Detection'
]

# Selectable for navigation
selected_page = st.sidebar.selectbox('Go to', page_options)

# Home Page
if selected_page == 'Home':
    st.subheader('Welcome to Content-Based Phishing Detection!')
    st.write(
        'This system is designed to classify phishing and legitimate websites using content-based filtering.'
    )

# Dataset Overview Page
# Dataset Overview Page
elif selected_page == 'Dataset Overview':
    st.subheader('Dataset Overview')
    st.write(
        'I used [Phishtank](https://www.phishtank.com/) and [Tranco](https://tranco-list.eu/) as data sources.'
    )
    st.write(f'Legitimate Websites: {ml.legitimate_df.shape[0]}')
    st.write(f'Phishing Websites: {ml.phishing_df.shape[0]}')

    # Display Pie chart using Plotly
    labels = ['Phishing', 'Legitimate']
    sizes = [ml.phishing_df.shape[0], ml.legitimate_df.shape[0]]
    fig = px.pie(
        names=labels, values=sizes,
        title='Phishing vs Legitimate Websites',
        hole=0.4, color_discrete_sequence=['#FFFFFF', '#66B2FF']  # Specify your own colors here
    )
    st.plotly_chart(fig)

    # Dataset Details
    st.subheader('Dataset Details')
    st.dataframe(ml.df.head(10))

    # Download CSV button for the dataset
    st.download_button(
        label='Download Data as CSV',
        data=ml.df.to_csv().encode('utf-8'),
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv'
    )

# Model Evaluation Page
elif selected_page == 'Model Evaluation':
    st.subheader('Model Evaluation')

    # Bar chart for model comparison using Plotly
    fig = px.bar(
        ml.df_results, barmode='group',
        labels={'index': 'Model'},
        title='Model Evaluation',
        color_discrete_sequence=['#32CD32', '#0000FF', '#FFD700']
    )
    st.plotly_chart(fig)

# Phishing Detection Page
elif selected_page == 'Phishing Detection':
    st.subheader('Phishing Detection')
    st.sidebar.subheader('Select Model')

    # Model selection dropdown
    models = {
        'Support Vector Machine': ml.svm_model,
        'Decision Tree': ml.dt_model,
        'Random Forest': ml.rf_model,
        'AdaBoost': ml.ab_model,
        'Neural Network': ml.nn_model,
        'K-Nearest Neighbors': ml.kn_model
    }

    # Model selection dropdown
    chosen_model = st.sidebar.selectbox('Choose a Model', list(models.keys()))

    # Use the selected model for prediction
    model = models.get(chosen_model)

    # Check if the selected model is a classifier with a 'predict' method
    if model and hasattr(model, 'predict') and callable(getattr(model, 'predict')):
        # User input for URL
        url = st.text_input('Enter the URL to check for phishing')

        # Check URL button
        if st.button('Check for Phishing'):
            try:
                # Get response from the URL
                response = re.get(url, verify=False, timeout=4)
                if response.status_code != 200:
                    st.error('HTTP connection was not successful for the URL.')
                else:
                    # Parse HTML content
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract features from the URL
                    vector = [fe.create_vector(soup)]

                    # Use the selected model for prediction
                    result = model.predict(vector)
                    # Display result
                    if result[0] == 1:

                        st.success('This web page seems legitimate!')
                    else:
                        st.warning('Attention! This web page is a potential phishing site!')
            except re.exceptions.RequestException as e:
                st.error(f'Error: {e}')
    else:
        st.error('Selected model is not a valid classifier with a "predict" method.')

# Show footer
st.sidebar.markdown(
    '---\n'
    '**About:**\n'
    'This Streamlit app is designed for Content-Based Phishing Detection. '
    'It uses machine learning models and feature extraction techniques.'
)
