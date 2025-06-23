# Analyse de Documents avec IA (Streamlit App)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Built with](https://img.shields.io/badge/Built_with-Streamlit-blue)
![Powered by](https://img.shields.io/badge/Powered_by-Azure_OpenAI-orange)

Cette application permet de poser des questions à partir de documents PDF téléversés et d'obtenir des réponses générées par un modèle de langage, avec système de feedback et historique intégré.

---

## Aperçu

- 📁 Téléversement de PDF
- 🔍 Recherche contextuelle dans les documents
- 💬 Interface de chat multilingue (FR, EN, ES, DE)
- ⭐ Notation et commentaire des réponses
- 🕓 Historique des questions posées
- 🧠 Vectorisation via Azure OpenAI + FAISS
- 💃️ Stockage des feedbacks/questions avec SQLite

---

## Comment utiliser l'application

### 1. Cloner le projet

```bash
git clone https://github.com/votre-utilisateur/analyse-documents-ia.git
cd analyse-documents-ia
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les clés AzureOpenAI

Créez un fichier `.streamlit/secrets.toml` et ajoutez :

```toml
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
```

### 5. Lancer l'application

```bash
streamlit run app.py
```

L’interface est accessible à l’adresse : [https://rag-project-baronebiffe.streamlit.app](https://rag-project-baronebiffe.streamlit.app)

---

## 🗂 Structure du projet

```
.
├── app.py                    # Interface principale Streamlit
├── rag/
│   └── my_langchain.py      # Fonctions d'analyse contextuelle
├── feedback.db              # Base de données SQLite
├── requirements.txt
└── .streamlit/
    └── secrets.toml         # Clés API (non versionnées)
```

---

## Fonctionnalités principales

| Fonction                   | Description                                        |
|---------------------------|----------------------------------------------------|
|  Téléversement          | Support des fichiers PDF                          |
|  Chat intelligent       | Posez des questions en langage naturel            |
|  Vectorisation          | FAISS + Azure OpenAI pour recherche sémantique    |
|  Feedback                | Système de notation (1 à 5 étoiles) + commentaires|
|  Historique              | Visualisation des anciennes questions             |
|  Langues                | Choix : FR, EN, ES, DE                            |

---

##  Exemple de flow utilisateur

1. L'utilisateur téléverse un fichier PDF.
2. Il pose une question en français.
3. Le système vectorise le PDF et interroge Azure OpenAI.
4. La réponse est affichée dans le chat.
5. L'utilisateur évalue la réponse et ajoute un commentaire.
6. Les données sont stockées dans la base SQLite.

---

##  Personnalisation

Tu peux modifier :
- les paramètres du moteur dans `rag/my_langchain.py`
- les langues proposées
- les types de fichiers acceptés (ajouter Word, TXT...)

---

##  Exigences

- Python 3.10+
- Azure OpenAI déployé avec :
  - `text-embedding-ada-002`
  - `gpt-35-turbo` ou `gpt-4`

---

##  Contribuer

Les contributions sont les bienvenues !  
N'hésitez pas à ouvrir une *issue* ou soumettre une *pull request* :)
