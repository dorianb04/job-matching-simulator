import agentops
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

#!/usr/bin/env python3
"""
Mini application utilisant AutoGen avec Groq et intégrant AgentOps.
Ce script initialise AgentOps, crée un agent assistant avec un modèle Groq
et démarre une session de chat simple.
"""

import os
import agentops
from autogen import ConversableAgent, UserProxyAgent

# Initialiser AgentOps en fournissant votre clé API (et des tags optionnels)
agentops.init(api_key=AGENTOPS_API_KEY, tags=["minie-app-autogen-groq"])

# Définir la configuration LLM pour utiliser le client Groq.
# Vous pouvez définir la clé Groq via la variable d'environnement GROQ_API_KEY
config_list = [
    {
        "model": "llama3-8b-8192",
        "api_key": GROQ_API_KEY,
        "api_type": "groq",
        # Vous pouvez ajouter d'autres paramètres Groq si nécessaire, par exemple :
        # "temperature": 0.7,
        # "max_tokens": 2048,
    }
]

# Création de l'agent assistant utilisant AutoGen avec la configuration Groq
assistant = ConversableAgent(
    name="GroqAssistant",
    llm_config={"config_list": config_list},
    system_message="Bonjour, je suis un assistant basé sur Groq et AutoGen. Comment puis-je vous aider aujourd'hui ?"
)

# Création de l'agent utilisateur (proxy)
user = UserProxyAgent(name="User", code_execution_config=False)

# Démarrer le chat. La session est automatiquement suivie par AgentOps.
assistant.initiate_chat(user, message="Bonjour, pouvez-vous me donner la météo à Paris ?")

# Clôturer la session AgentOps une fois le chat terminé.
agentops.end_session("Success")