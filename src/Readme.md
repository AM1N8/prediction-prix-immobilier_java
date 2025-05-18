# Système de Prédiction des Prix Immobiliers

Un système d'intelligence artificielle pour prédire les prix immobiliers basé sur un réseau de neurones artificiels développé en Java avec DL4J.

## Description

Cette application utilise un réseau de neurones artificiels (RNA) pour prédire les prix des maisons en fonction de diverses caractéristiques immobilières comme la surface, le nombre de chambres, et d'autres attributs. Elle offre une interface graphique conviviale permettant aux utilisateurs d'entrer les caractéristiques d'une propriété et d'obtenir une estimation de son prix.

## Fonctionnalités

- **Prédiction de Prix**: Entrez les caractéristiques d'une maison pour obtenir une estimation précise
- **Entraînement du Modèle**: Possibilité d'entraîner le modèle sur un jeu de données immobilières
- **Visualisation des Corrélations**: Analyse des relations entre différentes caractéristiques et leur impact sur le prix
- **Interface Graphique Intuitive**: Interface utilisateur simple et facile à utiliser
- **Conversion de Devise**: Affichage des prix en roupies indiennes (INR) et en dollars américains (USD)
- **Visualisation du Réseau de Neurones**: Représentation visuelle de l'architecture du modèle

## Architecture Technique

### Structure du Réseau de Neurones
- **Couche d'entrée**: 12 neurones (correspondant aux caractéristiques immobilières)
- **Première couche cachée**: 20 neurones avec activation ReLU
- **Deuxième couche cachée**: 20 neurones avec activation ReLU
- **Couche de sortie**: 1 neurone (prédiction du prix)

### Détails d'Implémentation
- **Framework d'IA**: DL4J (DeepLearning4J)
- **Algorithme d'optimisation**: Adam (Adaptive Moment Estimation)
- **Fonction de perte**: Erreur quadratique moyenne (MSE)
- **Initialisation des poids**: Xavier
- **Interface graphique**: Java Swing

## Caractéristiques Utilisées

L'application utilise les caractéristiques suivantes pour la prédiction:

1. Surface (en m²)
2. Nombre de chambres
3. Nombre de salles de bain
4. Nombre d'étages
5. Proximité d'une route principale (oui/non)
6. Présence d'une chambre d'amis (oui/non)
7. Présence d'un sous-sol (oui/non)
8. Système d'eau chaude (oui/non)
9. Climatisation (oui/non)
10. Nombre de places de parking
11. Zone préférentielle (oui/non)
12. État d'ameublement (meublé/semi-meublé/non meublé)

## Installation et Exécution

### Prérequis
- JDK 11 ou supérieur
- Maven (pour la gestion des dépendances)

### Installation
1. Clonez le dépôt:
   ```
   git clone https://github.com/AM1N8/prediction-prix-immobilier_java.git
   ```

2. Naviguez vers le répertoire du projet:
   ```
   cd prediction-prix-immobilier_java
   ```

3. Compilez le projet avec Maven:
   ```
   mvn clean package
   ```

### Exécution
Lancez l'application avec la commande:
```
java -jar target/house-price-prediction-1.0.jar
```

## Guide d'Utilisation

### Prédiction de Prix
1. Accédez à l'onglet "Prédiction"
2. Remplissez les champs avec les caractéristiques de la propriété:
    - Surface (en m²)
    - Nombre de chambres
    - Nombre de salles de bain
    - Etc.
3. Cliquez sur "Prédire le Prix" pour obtenir l'estimation

### Entraînement du Modèle
1. Accédez à l'onglet "Prédiction"
2. Cliquez sur "Entraîner le Modèle"
3. Attendez que le processus d'entraînement se termine
4. Consultez les métriques de performance affichées dans la zone de résultat

### Analyse des Corrélations
1. Accédez à l'onglet "Corrélations"
2. Consultez la matrice de corrélation pour comprendre les relations entre les différentes caractéristiques
3. Utilisez la légende pour interpréter les niveaux de corrélation

## Structure du Projet

Le projet est organisé comme suit:

- `HousePricePredictionANN.java`: Classe principale contenant l'interface graphique et le modèle d'IA
- `HousingDataLoader.java`: Classe responsable du chargement et du prétraitement des données
- `resources/Housing.csv`: Jeu de données immobilières utilisé pour l'entraînement

## Jeu de Données

Le jeu de données utilisé contient des informations sur diverses propriétés immobilières, incluant leurs caractéristiques et leurs prix. Les données sont normalisées avant d'être utilisées pour l'entraînement du modèle afin d'améliorer les performances.

## Performance du Modèle

Les performances typiques du modèle après entraînement sont:
- **R² Score**: ~0.85 (indiquant que le modèle explique environ 85% de la variance des prix)
- **Erreur moyenne**: ~10-15% sur les données de test
- **RMSE**: Varie selon les données d'entraînement

## Contribution

Les contributions à ce projet sont les bienvenues. Pour contribuer:

1. Forkez le dépôt
2. Créez une branche pour votre fonctionnalité (`git checkout -b nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -m 'Ajout de nouvelle-fonctionnalite'`)
4. Poussez vers la branche (`git push origin nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## Licence

Ce projet est distribué sous la licence MIT. 

## Remerciements

-  Réalisé dans le cadre d’un projet Java expert, sous la supervision du professeur Hajji Tarik.
- Utilise le framework DL4J pour l'implémentation du réseau de neurones