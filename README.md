# Analyse de documents

Ce projet propose une interface pour charger des documents pour constituer une base de connaissance qui pourra être questionnée avec un grand modèle de langage (_LLM_).


# Analyse de Documents avec IA (Streamlit App)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Built with](https://img.shields.io/badge/Built_with-Streamlit-blue)
![Powered by](https://img.shields.io/badge/Powered_by-Azure_OpenAI-orange)

Cette application permet de poser des questions à partir de documents PDF téléversés et d'obtenir des réponses générées par un modèle de langage, avec système de feedback et historique intégré.

---

##  Aperçu

- 📁 Téléversement de PDF
- 🔍 Recherche contextuelle dans les documents
- 💬 Interface de chat multilingue (FR, EN, ES, DE)
- ⭐ Notation et commentaire des réponses
- 🕓 Historique des questions posées
- 🧠 Vectorisation via Azure OpenAI + FAISS
- 🗃️ Stockage des feedbacks/questions avec SQLite



## 🧪 Comment utiliser l'application

### 1. Cloner le projet

```bash
git clone https://github.com/votre-utilisateur/analyse-documents-ia.git
cd analyse-documents-ia
---

## 2. Créer un environnement virtuel
---
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
---

## 3. Installer les dépendances
---
pip install -r requirements.txt
---

## 4. Configurer les clés AzureOpenAI
---
[embedding]
azure_api_key = "VOTRE_CLE"
azure_endpoint = "https://votre-endpoint.openai.azure.com/"
azure_deployment = "deployment-name"
azure_api_version = "2024-03-01-preview"

[chat]
azure_api_key = "VOTRE_CLE"
azure_endpoint = "https://votre-endpoint.openai.azure.com/"
azure_deployment = "chat-deployment"
azure_api_version = "2024-03-01-preview"
---

## 5. Lancer l'application
---
streamlit run app.py
---
L’interface est accessible à l’adresse : https://rag-project-baronebiffe.streamlit.app/


#Structure du projet
.
├── app.py                    # Interface principale Streamlit
├── rag/
│   └── my_langchain.py      # Fonctions d'analyse contextuelle
├── feedback.db              # Base de données SQLite
├── requirements.txt
└── .streamlit/
    └── secrets.toml         # Clés API (non versionnées)


## Fonctionnalités principales

Fonction	Description
📄 Téléversement de documents	Support des fichiers PDF
🤖 Chat intelligent	Posez des questions en langage naturel
🧠 Vectorisation	FAISS + Azure OpenAI pour recherche sémantique
⭐ Feedback	Système de notation (1 à 5 étoiles) + commentaires
🕓 Historique	Visualisation des anciennes questions
🌐 Langues	Choix de langue pour la réponse (FR, EN, ES, DE)



 Exemple de flow utilisateur

* L'utilisateur téléverse un fichier PDF.
* Il pose une question en français.
* Le système vectorise le PDF et interroge Azure OpenAI.
* La réponse est affichée dans le chat.
* L'utilisateur peut évaluer la réponse et ajouter un commentaire.
* La question, la réponse et le feedback sont stockés.


## Personnalisation

Il est possible de modifier les paramètres du moteur dans rag/my_langchain.py, changer les langues proposées, ou ajouter d'autres types de documents (Word, TXT, etc.).


 Exigences

    Python 3.10+

    Azure OpenAI déployé avec :

        Un modèle d’embedding (text-embedding-ada-002)

        Un modèle de chat (gpt-35-turbo ou gpt-4)


## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou proposer une pull request :)

