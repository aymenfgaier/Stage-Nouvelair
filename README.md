# Plateforme d'optimisation de la consommation de fuel — Nouvelair

Prototype de plateforme (stage ingénieur, thème "optimisation prédictive de la
consommation de carburant aérien"), inspiré de l'architecture décrite dans le
rapport de référence (plateforme *FuelTrack*, PFE Mastère Sciences de Données)
mais adapté à une stack plus légère et à vos propres sprints.

## Stack

- **Frontend/backend** : Streamlit (une seule application Python)
- **Base de données** : SQLite (`db/fuel_platform.db`)
- **Traitement des données** : pandas
- **ML** : scikit-learn, XGBoost, SHAP

## Structure

```
FuelOptimPlatform/
  app/
    Home.py                          # page d'accueil = plan du site (navigation vers toutes les pages)
    ingestion.py                     # pipeline de nettoyage/centralisation (Sprint 1)
    datasource.py                    # abstraction source de données : fichier vs base
    pages/
      1_Import_Centralisation.py     # deux sources : nouveau fichier CSV ou base déjà centralisée
      2_Dashboard.py                 # 3 onglets : Carburant & Avion, Réseau & Routes, Opérationnel & Prédictif
      3_Modelisation_ML.py           # entraînement, évaluation, SHAP, anomalies (Sprint 2)
      4_Prediction.py                # prédiction manuelle ou automatique + recommandation + Fuel Score
  data/
    raw/SetNouvelair.csv             # dataset source (45 104 vols)
  db/
    fuel_platform.db                 # base centralisée (table flights, predictions, anomaly_scores, models_metadata)
  ml/
    model_training.py                # baseline, régression linéaire, XGBoost, SHAP, Isolation Forest
    recommendation.py                # moteur de recommandation (carburant optimal, cost index) et Fuel Score
    models/                          # modèles entraînés (.joblib) : fuel_reel_consomme (+ variante _manuel)
  docs/
    uml/*.mmd                        # diagrammes UML (Mermaid) : cas d'utilisation,
                                      # classes, architecture, séquences par sprint
```

## Lancer la plateforme

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

## État d'avancement

- **Sprint 0** (conception) : diagrammes UML disponibles dans `docs/uml/`
  (cas d'utilisation général, classes générales, architecture système).
- **Sprint 1** (centralisation) : fait — deux sources de données possibles
  (nouveau fichier CSV nettoyé, ou relecture directe de la base déjà
  centralisée), nettoyage (Trip Fuel invalide, fuel réel négatif/nul,
  outliers > 99,5e centile, imputation passagers, normalisation des dates,
  feature engineering) et dashboard exploratoire.
- **Sprint 2** (modélisation ML) : fait — **deux variantes** du modèle
  **Fuel réel total** (Block Off − Block On), entraînées ensemble mais
  utilisées dans des contextes différents :
  - **Modèle principal** (`fuel_reel_consomme`) — utilise Trip Fuel/Taxi Fuel
    planifiés en plus des caractéristiques du vol. C'est le seul affiché/évalué
    sur la page *Modélisation ML*, et le seul utilisé en mode Automatique et en
    mode Manuel "vol existant" (le plan réel est alors connu). Baseline
    (moyenne historique par paire d'aéroports), régression linéaire et
    XGBoost, split aléatoire 80/20 : **MAE=0,226 t, R²=0,963, MAPE=4,4 %**.
  - **Modèle "saisie manuelle"** (`fuel_reel_consomme_manuel`) — n'utilise
    aucune donnée de plan de vol (ni Trip Fuel, ni Taxi Fuel), uniquement les
    caractéristiques du vol (type d'avion, aéroports, passagers, durée
    programmée, moyenne historique par paire d'aéroports). Réservé au mode
    Manuel "saisir manuellement" (vol hypothétique, sans plan de vol
    disponible) : **MAE=0,272 t, R²=0,945, MAPE=5,2 %**.

  Block Off, Block On et les réserves de carburant sont volontairement
  exclus des features des deux modèles : vérification empirique, Block Off =
  Trip Fuel + Taxi Fuel + réserves (corrélation 0,999) — ce sont des
  grandeurs calculées à partir de Trip Fuel/Taxi Fuel, pas des mesures
  indépendantes ; les inclure produirait un score artificiellement élevé mais
  un modèle inutilisable sur un vol futur. Pour la même raison,
  `flight_duration_min` (durée réalisée, identique a posteriori à
  `duree_vol_reelle_h`) est exclu au profit de `duree_prevue_min` (durée
  programmée, réellement connue avant le vol).

  Explicabilité SHAP et détection de dérive par immatriculation (Isolation
  Forest) inclus (modèle principal).
- **Prédiction** : fait — deux modes sur la page *Prédiction* :
  - **Manuel** : l'utilisateur choisit un vol existant de la base (Trip
    Fuel/Taxi Fuel affichés pour référence, utilisés par le modèle principal)
    ou saisit librement les caractéristiques d'un vol — aéroports affichés en
    toutes lettres, aucun plan de vol requis (modèle "saisie manuelle") — et
    lance la prédiction du Fuel réel total à la demande.
  - **Automatique** : score l'intégralité de la base centralisée (jamais un
    fichier) avec le modèle principal, et enregistre les résultats dans la
    table `predictions`.
- **Sprint 3** (dashboard prédictif & recommandations) : en cours —
  - **Dashboard restructuré en 3 onglets** : *Carburant & Avion* (KPI globaux,
    consommation par flotte, alertes de dérive), *Réseau & Routes* (top
    routes, consommation par mois), *Opérationnel & Prédictif* (résultats du
    mode Automatique : Fuel Score moyen, répartition des recommandations,
    vols nécessitant un complément de carburant, alertes de dérive).
  - **Moteur de recommandation** (`ml/recommendation.py`) — module à règles
    simples (pas un optimiseur de performance avion) : compare la
    consommation prédite au plan de vol (Trip Fuel + Taxi Fuel) et propose un
    carburant optimal (+2 % de marge) ainsi qu'une indication qualitative
    d'ajustement de cost index. Intégré à la page *Prédiction* (manuel et
    automatique) et à l'onglet *Opérationnel & Prédictif*.
  - **Fuel Score** (0-100 par vol) — mesure l'écart entre consommation réelle
    et prédite (50 = conforme, >50 = a consommé moins que prévu). Limité au
    niveau vol/avion/route : **pas de score par pilote**, le dataset ne
    contenant aucun identifiant pilote.
  - **Filtres de visualisation** — page *Import & Centralisation* : le
    tableau du référentiel s'affiche automatiquement, que la source soit un
    nouveau fichier importé ou la base déjà centralisée (plus besoin de
    cliquer sur un bouton pour le voir). Page *Dashboard* : filtre période
    (date de début/fin) global appliqué aux trois onglets, complété par un
    filtre type d'avion (onglet *Carburant & Avion*) et un filtre route
    (onglet *Réseau & Routes*), propres à chaque onglet. Page *Modélisation
    ML* : sélecteur de période dédié permettant d'entraîner et d'évaluer le
    modèle (baseline, régression linéaire, XGBoost, SHAP) sur un
    sous-ensemble temporel choisi plutôt que sur tout l'historique.
  - Reste à faire : rédaction du chapitre Sprint 3 dans le rapport, limites/
    perspectives, support de soutenance.

## Diagrammes UML

Les fichiers `.mmd` dans `docs/uml/` sont éditables et rendus dans l'artifact
publié pendant cette session. Ils peuvent être régénérés en images (PNG/SVG)
pour être insérés directement dans le rapport de stage (section 3.9
« Architecture cible »).
