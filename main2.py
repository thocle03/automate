#Ce projet vise à automatiser le traitement des emails entrants grâce à BERT. Nous mettons en place un workflow intelligent où les emails sont :
# 1) Classifiés selon leur catégorie 
# 2) Analysés pour en extraire les mots-clés 
# 3) Regroupés par sujets principaux 
# 4) Visualisés sous forme de flux


# Technologies utilisées
# Python pour le développement Hugging Face Transformers pour l'IA Pandas pour la gestion des données NetworkX & Matplotlib pour la visualisation
# Code qui marche sur collab directement sans rien a importer

# Le processus suit 4 étapes principales : 
# 1) Un email est reçu 
# 2) Il est classé avec un modèle BERT pré-entraîné 
# 3) Ses mots-clés sont extraits avec un modèle de reconnaissance d’entités nommées 
# 4) Un modèle de classification détermine le sujet principal

#Enfin, les résultats sont enregistrés dans un fichier CSV.


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline


emails = [
    "Bonjour Thomas, votre inscription au tournoi de tennis de Thoiry est confirmée. Merci d'envoyer votre chèque.",
    "Cher Thomas, voici les 5 compétences essentielles à maîtriser pour réussir en Data Science.",
    "Votre facture EDF de 78,45€ est disponible. Veuillez la régler avant le 15 avril 2025.",
    "Nous avons remarqué une activité inhabituelle sur votre compte bancaire. Contactez-nous immédiatement.",
    "Profitez de notre promotion exclusive : -30% sur tous les articles jusqu'à dimanche soir !",
    "Votre commande Amazon a été expédiée et arrivera d'ici 3 jours. Suivez votre colis ici.",
    "Félicitations ! Vous êtes qualifié pour la prochaine phase du championnat régional de tennis.",
    "Nous avons mis à jour notre politique de confidentialité. Consultez les nouveaux termes ici."
]


classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
classification_result = [classifier(email)[0]['label'] for email in emails]


keyword_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
keywords = [[entity['word'] for entity in keyword_extractor(email)] for email in emails]


topic_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
topics = ["Facture", "Tournoi sportif", "Sécurité", "Promotion", "Commande en ligne", "Confidentialité", "Carrière"]
topic_results = [topic_model(email, topics)["labels"][0] for email in emails]


G = nx.DiGraph()
G.add_edges_from([
    ("Emails entrants", "Classification des emails"),
    ("Classification des emails", "Extraction des mots-clés"),
    ("Extraction des mots-clés", "Modélisation des sujets"),
    ("Modélisation des sujets", "Génération de rapports"),
])


plt.figure(figsize=(8, 5))
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)
plt.title("Flux d'automatisation basé sur BERT")
plt.show()


df = pd.DataFrame({"Email": emails, "Catégorie": classification_result, "Mots-clés": keywords, "Sujet": topic_results})
df.to_csv("emails_classifiés.csv", index=False)

print("Processus terminé,  résultats dans 'emails_classifiés.csv'")
