import pandas as pd
import streamlit as st

problem_categories = {
'Qualité du Service': ['service',' Pas de sourire ', 'sans respect','accueil','Ignorance des clients', 'personnel','Erreurs de commande','communication','qualite', 'incompetent', 'amabilite', 'attitude', 'professionnalisme', 'courtoisie', 'serviable', 'reception', 'clientele', 'serveur', 'accueillant', 'gestion', 'souriant', 'professionnel', 'soin', 'Service mediocre',
'Personnel impoli','Mauvaise nourriture','mal parlé', 'Service lent', 'Nourriture froide','Plats mal prepares','Personnel inattentif'],
    'Propreté et Hygiène': ['proprete', 'hygiene', 'sale', 'insalubre', 'nettoyage', 'desordre', 'odeur', 'toilettes', 'entretien', 'draps', 'serviettes', 'poussiere', 'linge', 'nettoyer', 'bain', 'hygienique', 'desinfecter', 'propre'],
    'Problèmes d\'Équipements et d\'Installations': ['équipement', 'installation', 'dysfonctionnement', 'reparation', 'panne', 'vetuste', 'maintenance','fuite', 'appareil', 'fonctionner', 'eclairage', 'climatisation', 'chauffage', 'douche', 'TV', 'wifi', 'interrupteur', 'clims',
  'ascenseur', 'installation', 'fonctionnelle', 'defectueux'],
    'Problèmes de Réservation': ['reservation', 'reserve', 'confirmation', 'annulation', 'erreur', 'disponibilite', 'réservation', 'confirmer', 'annuler', 'erreur', 'dates', 'réservée', 'confirmation', 'occupee', 'indisponible', 'reservable'],
    'Bruits': ['bruit', 'sonore', 'dormir','nuisance', 'insonorisation', 'tapage', 'vacarme', 'musique', 'travaux', 'bruyant', 'calme', 'isole', 'derangement', 'sommeil', 'nuisible', 'ambiance', 'intense', 'eclatant','gênant','genant'],
    'Problèmes de Prix': ['prix', 'facturation','Prix eleves','coût', 'cout','tarif', 'cher', 'abordable', 'budget', 'facture', 'economique', 'offre', 'promotion', 'coûteux','couteux','couteuse', 'exorbitant', 'tarification', 'abordabilite', 'prix eleve', 'payer', 'depense', 'abordable', 'valeurs', 'prix justifie'],
    'Retard': ['retard', 'attente', 'horaire', 'ponctualite', 'retarder', 'attendre', 'retardataire', 'retarde', 'arrivee', 'decalage', 'retenu', 'attendu', 'retardataires', 'prevu', 'heures', 'delai', 'retards', 'programme', 'heure', 'retardee'],
    # Add more keywords to each category as needed
}
# Function to predict problems using the dictionary
def predict_problems(comment):
    if isinstance(comment, str):
        predicted_problems = []
        for category, keywords in problem_categories.items():
            for keyword in keywords:
                if keyword in comment.lower():
                    predicted_problems.append(category)
                    break
        return predicted_problems
    else:
        return []


# Chargez le dataset avec les prédictions
data_with_predictions = pd.read_csv('C:/Users/HP/Desktop/data/pip/data_with_predictions.csv')


# Définissez votre application Streamlit
st.title('SeoSentiment System ')






st.title('Problem prediction')
nouveau_commentaire = st.text_input('Enter a new comment :')



if st.button('Predicted problems'):
    
    
    nouveau_predicted_problems = predict_problems(nouveau_commentaire)
    
    
    
    
    
    
    if nouveau_predicted_problems:
        #st.write("Problèmes prédits :")
        
                 # Créer un DataFrame pour afficher le tableau
        df = pd.DataFrame({'Predicted problems': nouveau_predicted_problems})
        
        # Créer un tableau HTML personnalisé avec des bordures et des styles CSS
        table_html = (
            "<table style='border-collapse: collapse; width: 100%; border: 2px solid #ddd;'>"
            "<thead><tr>"
            "<th style='border: 2px solid #ddd; padding: 12px; text-align: center; background-color: #f2f2f2; color: red ;'>Predicted problems</th>"
            "</tr></thead>"
            "<tbody>"
        )
        
        for index, row in df.iterrows():
            table_html += (
                "<tr>"
                f"<td style='border: 2px solid #ddd; padding: 12px; text-align: center;'>{row['Predicted problems']}</td>"
                "</tr>"
            )
        
        table_html += "</tbody></table>"
        
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.write("Aucun problème prédit.")
