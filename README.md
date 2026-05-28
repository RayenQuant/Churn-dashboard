# Tableau de Bord Intelligent — Prédiction du Churn Bancaire

## C'est quoi ce projet ?

Dans le secteur bancaire, un client qui part (qu'on appelle un **churner**) représente une perte directe de revenus. Reconquérir un client coûte en moyenne **5 à 7 fois plus cher** que de le retenir. L'enjeu est donc simple : **identifier les clients qui risquent de partir avant qu'ils ne le fassent**, et agir en conséquence.

Ce tableau de bord fait exactement ça. Il combine de l'ingénierie de données, du machine learning et de la visualisation interactive pour donner à une équipe bancaire les outils nécessaires pour anticiper le churn, comprendre ses causes, segmenter sa clientèle et simuler le risque client en temps réel.

---

## Les données

On travaille sur le **Bank Customer Churn Dataset**, un jeu de données de référence disponible sur Kaggle contenant **10 000 clients** d'une banque fictive répartis sur trois pays : France, Allemagne et Espagne.

Chaque client est décrit par 14 variables :

| Variable | Description |
|---|---|
| `CreditScore` | Score de crédit du client (300 à 850) |
| `Geography` | Pays du client (France, Germany, Spain) |
| `Gender` | Genre (Male / Female) |
| `Age` | Âge du client |
| `Tenure` | Depuis combien d'années il est client de la banque |
| `Balance` | Solde de son compte bancaire |
| `NumOfProducts` | Nombre de produits bancaires souscrits (carte, crédit, assurance…) |
| `HasCrCard` | Possède-t-il une carte de crédit ? (0 = Non, 1 = Oui) |
| `IsActiveMember` | Est-il un membre actif ? (0 = Non, 1 = Oui) |
| `EstimatedSalary` | Salaire estimé annuel |
| `Exited` | **La cible** — a-t-il quitté la banque ? (0 = Non, 1 = Oui) |

Sur les 10 000 clients, environ **20 à 30% ont churné** selon les filtres appliqués. Les clients allemands et les clients inactifs sont statistiquement les plus à risque.

> Si le fichier CSV n'est pas présent au démarrage, l'application génère automatiquement un dataset synthétique réaliste de 10 000 lignes avec les mêmes distributions statistiques. L'application fonctionne donc sans aucune configuration préalable.

---

## Ce que fait l'application

L'application est divisée en **4 onglets**, chacun répondant à une question métier précise.

### Onglet 1 — Vue Exécutive
*Pour qui : directeurs, responsables commerciaux*

Une vue synthétique de la santé de la base clients. On y voit en un coup d'œil :
- Le **taux de churn global** et le nombre total de clients
- Le **solde moyen des clients perdus** (pour estimer leur valeur)
- Le **revenu estimé à risque** (calculé comme 15% du salaire estimé des clients à haut risque)
- Des graphiques sur la répartition du churn par pays, par nombre de produits et dans le temps

Les **filtres dans la barre latérale** (pays, genre, produits, statut actif) s'appliquent en temps réel sur cet onglet.

---

### Onglet 2 — Segmentation K-Means
*Pour qui : équipes marketing, chargés de clientèle*

On regroupe automatiquement les clients en **segments homogènes** grâce à l'algorithme K-Means. L'idée est simple : des clients qui se ressemblent doivent recevoir des offres de rétention différentes.

Pour choisir le bon nombre de segments, on utilise deux méthodes scientifiques :
- La **méthode du coude** (Elbow Method) : on cherche le point où ajouter un cluster supplémentaire n'améliore plus significativement la qualité du regroupement
- Le **score de silhouette** : mesure à quel point chaque client est bien assigné à son cluster plutôt qu'à un autre

Chaque segment reçoit ensuite un **label métier** (ex: "Clients Premium Engagés", "Seniors à Risque") et un **plan d'action de rétention** personnalisé.

---

### Onglet 3 — Modèles Prédictifs
*Pour qui : data scientists, équipes analytiques*

On entraîne et compare **3 modèles de machine learning** sur les données :

| Modèle | Principe |
|---|---|
| **Régression Logistique** | Modèle linéaire simple, très interprétable |
| **Random Forest** | Ensemble d'arbres de décision, robuste et précis |
| **XGBoost** | Gradient boosting, généralement le plus performant |

Chaque modèle est évalué sur 5 métriques (Accuracy, Précision, Rappel, F1, ROC-AUC) et le meilleur est automatiquement sélectionné pour le simulateur.

On utilise également **SHAP** (SHapley Additive exPlanations), une technique issue de la théorie des jeux, pour expliquer pourquoi le modèle prédit ce qu'il prédit. Cela répond à la question : *"Quelles variables ont le plus poussé ce client vers le churn ?"*

---

###  Onglet 4 — Simulateur Anti-Churn en Temps Réel
*Pour qui : conseillers clientèle, lors d'un entretien client*

Un formulaire interactif où l'on saisit les caractéristiques d'un client (âge, solde, pays, ancienneté…) et l'on obtient instantanément :
- Une **jauge de probabilité de churn** (0 à 100%)
- Un **niveau de risque** parmi 4 catégories : 🟢 Faible / 🟡 Modéré / 🟠 Élevé / 🔴 Critique
- Un **plan de rétention personnalisé** adapté au niveau de risque
- Les **3 facteurs de risque principaux** pour ce client spécifique (via SHAP)

---

## Architecture technique

```
app.py        → Interface Streamlit (frontend, 4 onglets)
pipeline.py   → Ingestion CSV → SQLite, vues SQL analytiques, couche de requêtes
utils.py      → Nettoyage des données, K-Means, modèles ML, SHAP, graphiques
```

**Choix techniques notables :**
- Toutes les requêtes analytiques passent par **SQL** via SQLite — pas de filtrage brut sur DataFrame
- Le préprocessing utilise un **ColumnTransformer** scikit-learn (StandardScaler + OneHotEncoder) intégré dans un pipeline reproductible
- Les résultats du clustering et des modèles sont mis en **cache Streamlit** pour des performances optimales
- Les graphiques sont tous en **Plotly** (interactifs, zoomables, exportables)

---

## Lancer l'application en local

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer
streamlit run app.py
```

L'application sera disponible sur [http://localhost:8501](http://localhost:8501).

Au premier lancement, comptez **30 à 60 secondes** le temps que les modèles s'entraînent. Les lancements suivants sont instantanés grâce au cache.

---



Aucune configuration supplémentaire n'est nécessaire. Le dataset synthétique est généré automatiquement si le CSV Kaggle n'est pas présent.
