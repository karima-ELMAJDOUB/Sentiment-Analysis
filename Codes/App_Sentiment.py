import os
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import pickle
import base64
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import transformers


nltk.download('vader_lexicon')

# Load the pre-trained French sentiment analysis models
flair_classifier = TextClassifier.load('sentiment-fast')
flaubert_model_name = 'flaubert/flaubert_base_cased'
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(flaubert_model_name)
flaubert_model = FlaubertForSequenceClassification.from_pretrained(flaubert_model_name)

# Initialize SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Create a SessionState object to store data between user interactions
if 'state' not in st.session_state:
    st.session_state.state = {
        'comment_index': None,
        'updated_data': None,
        'model_trained': False
    }
previous_model = None
previous_model_path = 'C:/Users/HP/Desktop/data/pip/trained_model.pkl'
if os.path.exists(previous_model_path):
    with open(previous_model_path, 'rb') as model_file:
        previous_model = pickle.load(model_file)
        previous_model.eval()  # Make sure to set the model to evaluation model       

# Function to get the sentiment prediction from the models
def get_sentiment(commentaire):
    sentence_flair = Sentence(commentaire)
    flair_classifier.predict(sentence_flair)
    predicted_label_flair = sentence_flair.labels[0].value

    inputs = flaubert_tokenizer(commentaire, return_tensors='pt', truncation=True, max_length=500)
    with torch.no_grad():
        outputs = flaubert_model(**inputs)
    predicted_label_flaubert = 'positive' if outputs.logits[0][0] > 0 else 'negative'

    sentiment_score = analyzer.polarity_scores(commentaire)
    predicted_label_sia = 'positive' if sentiment_score['compound'] >= 0 else 'negative'

    predicted_labels = [predicted_label_flair, predicted_label_flaubert, predicted_label_sia]
    ensemble_predicted_label = max(set(predicted_labels), key=predicted_labels.count)

    return ensemble_predicted_label

# Load the dataset containing the "commentaire" column
file_path = 'C:/Users/HP/Desktop/data/pip/data_cleann.csv'
data = pd.read_csv(file_path)

# Drop rows with missing or NaN values in the 'commentaire' column
data = data.dropna(subset=['commentaire'])
st.title('French Sentiment Analysis')

commentaire = st.text_area('Enter your comment here:', max_chars=512)

if st.button('Predict Sentiment'):
    sentiment = get_sentiment(commentaire)
    st.write('Predicted Sentiment:', sentiment)
    
    # Enregistrer la prédiction dans la session
    st.session_state.state['predicted_sentiment'] = sentiment

    user_feedback = st.radio('Is the prediction correct?', ('Yes', 'No'))

    if user_feedback == 'No':
        # Utiliser la prédiction enregistrée pour l'inversion du sentiment
        predicted_sentiment = st.session_state.state['predicted_sentiment']
        corrected_sentiment = 'positive' if predicted_sentiment == 'negative' else 'negative'
        st.write('Corrected Sentiment:', corrected_sentiment)
        st.session_state.state['corrected_sentiment'] = corrected_sentiment
   # Ajouter la nouvelle ligne au DataFrame uniquement si le sentiment a été corrigé
        if corrected_sentiment != sentiment:
            new_row = pd.DataFrame({'commentaire': [commentaire], 'corrected_sentiment': [st.session_state.state['corrected_sentiment']], 'user_feedback': [user_feedback]})
            data = data.append(new_row, ignore_index=True)
        
            # Enregistrer le nouvel "cleaned_dataset" dans le fichier CSV
            data.to_csv(file_path, index=False, encoding='utf-8')

            # Afficher les commentaires mis à jour
            st.write('---')
            st.write('Updated Comments:')
            st.write(data[['commentaire', 'corrected_sentiment']])   
    else:
        st.session_state.state['corrected_sentiment'] = sentiment


   # Append new comment to the loaded "cleaned_dataset"
    new_row = pd.DataFrame({'commentaire': [commentaire], 'corrected_sentiment': [st.session_state.state['corrected_sentiment']], 'user_feedback': [user_feedback]})
    data = data.append(new_row, ignore_index=True)
    
      # Save the updated "cleaned_dataset" to the CSV file
    data.to_csv(file_path, index=False, encoding='utf-8')

    st.write('---')
    st.write('Previous Comments:')
    st.write(data[['commentaire', 'corrected_sentiment']])

    updated_training_data = pd.concat([data, new_row], ignore_index=True)
    updated_data_csv = updated_training_data.to_csv(index=False, encoding='utf-8')

    # Provide a download link for the updated dataset

# Utiliser la fonction st.download_button() pour permettre le téléchargement
    st.download_button(
            label="Download CSV",
            data=updated_data_csv.encode('utf-8'),
            file_name='updated_dataset.csv',
            mime='text/csv'
)

        
    st.write("Updated dataset has been saved.")

    st.write('---')
    st.write('Previous Comments:')
    st.write(updated_training_data[['commentaire', 'corrected_sentiment']])
    
    st.session_state.comment_index = len(updated_training_data) - 1

    
    #if st.button("Save Updated Dataset"):
        #updated_training_data.to_csv('C:/Users/HP/Desktop/data/pip/updated_dataset.csv', index=False)
       # st.write("Updated dataset has been saved.")
        #st.write('---')
        #st.write('Download Updated Dataset')
        #st.download_button(label="Download CSV", data=updated_training_data.to_csv(index=False), file_name='updated_dataset.csv', mime='text/csv')

    if st.button("Train Model"):
        train_data, val_data = train_test_split(updated_training_data, test_size=0.2, random_state=42)

        class SentimentClassifier(nn.Module):
            def init(self):
                super(SentimentClassifier, self).init()
                self.fc1 = nn.Linear(768, 128)
                self.fc2 = nn.Linear(128, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        model = SentimentClassifier()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10
        batch_size = 32

        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                input_ids = []
                attention_mask = []
                labels = []

                for _, row in batch.iterrows():
                    comment = row['commentaire']
                    label = row['corrected_sentiment']

                    inputs = flaubert_tokenizer(comment, return_tensors='pt', truncation=True, max_length=500)
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])

                    label_id = 1 if label == 'positive' else 0
                    labels.append(label_id)

                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                labels = torch.tensor(labels)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

       # torch.save(model.state_dict(), 'C:/Users/HP/Desktop/data/pip/trained_model.pth')
       # st.write("Model has been trained and saved.")
                model_state_dict = model.state_dict()
                model_path = 'C:/Users/HP/Desktop/data/pip/trained_model.pth'
                torch.save(model_state_dict, model_path)
                st.write("Model has been trained and saved.")

                # Provide a download link for the trained model
                def download_model(model_path):
                    with open(model_path, 'rb') as model_file:
                        model_data = model_file.read()
                    b64 = base64.b64encode(model_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="trained_model.pkl">Download Trained Model</a>'
                    st.markdown(href, unsafe_allow_html=True)

                download_model(model_path)