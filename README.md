# ipssi-ai-tensorflow-image-classification

Mini-Projet fait avec Python(Streamlit) consiste à analyser, prétraiter et visualiser des images.\
Il permet de découvrir les notions du machine learning avec [TensorFlow](https://www.tensorflow.org/) et celles vu en cours.\
Dans ce projet nous analyserons des images de chiens et essayer de prédire, grâce au CNN, de quelle race appartient le chien.\
Nous analyserons les images à l'aide d'un modele réalisé from scratch et d'un modele VGG16 entrainés sur une base de données de race de chiens.\
(Base de données : [Students Performance](https://www.kaggle.com/spscientist/students-performance-in-exams))

La mini-app permet aussi d'enregistrer les images et de les classifier en fonction de la race du chien analyser. Elle permet aussi de visualiser
les différents chiens présents dans chaque race existante dans la base de données.
# Etapes pour manipuler cette application (Python >= 3.7.6)

## 1 Initialisez votre environnement virtuel
`py -m venv ./venv`

## 2 Démarrez votre environnement virtuel
```
.\venv\Scripts\Activate.ps1 (PowerShell)
.\venv\Scripts\Activate.bat (Windows)
source ./venv/bin/activate (MAC/Linux)
```

## 3 Installez les dépendances du projet
`pip install -r requirements.txt`

### Infos supplémentaires (Lorsque vous êtes dans l'environnement)
- `streamlit run app/app.py` Lancer l'application streamlit
- `jupyter notebook` Accéder au notebook jupyter

### Liens
- [Vers le trello](https://trello.com/b/rgbw9N0k/wwprojet-deep-learning)
- [Vers le set de données](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)

## Credits & Licence
12/03/2021 - GPL3 Licence (Open Source)

**Noé ABDEL KALEK**  - *Front-End & Back-End Developer (Project Manager)*

**Jéremie VANG FOUA**  - *Front-End & Back-End Developer Developer*

**Brahim KADDAR**  - *Back-End Developer*    

**Mohamed BOUROUCHE** - *Back-End Developer*

**Nicolas AUBE** - *Back-End Developer*
