# Préparation avec environnement virtuel si pas deja fait en place dans vscode 
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Entraînement des modèles
python models/train_dense.py
python models/train_cnn.py

# Lancement de TensorBoard
tensorboard --logdir=logs --port=6006

# Lancement de MLflow
mlflow ui

# Lancement de l'application Streamlit
streamlit run streamlit_app/app.py