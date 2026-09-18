"""
Module de recommandation et de scoring (Sprint 3).

Deux briques, toutes deux derivees des predictions du modele (module
`model_training`) plutot que d'une modelisation physique du vol :

  - generate_recommendation() : compare la consommation predite a la
    consommation planifiee (Trip Fuel + Taxi Fuel) lorsque celle-ci est
    disponible, et propose un carburant optimal ainsi qu'une indication
    qualitative d'ajustement de cost index. Il s'agit d'un moteur a
    REGLES SIMPLES (seuils sur l'ecart predit/planifie), pas d'un
    optimiseur de performance avion -- un reglage fin de cost index
    necessiterait un modele de performance (vent, altitude, poids reel)
    hors perimetre des donnees disponibles.

  - compute_fuel_score() : indicateur synthetique 0-100 par vol, mesurant
    l'ecart entre consommation reelle et consommation predite (ou
    planifiee). Note : le dataset ne contient pas d'identifiant pilote,
    le score est donc calcule par vol / avion / route uniquement, pas
    par pilote (contrairement au module "Fuel Score" envisage au
    tableau 3.18, qui sera limite a ce perimetre par manque de donnee).
"""

from __future__ import annotations

from dataclasses import dataclass

# Seuil (en tonnes) au-dela duquel l'ecart predit/planifie est juge significatif
ECART_SIGNIFICATIF_T = 0.15

# Echelle de conversion ecart relatif -> score (voir compute_fuel_score)
FUEL_SCORE_SCALE = 4.0


@dataclass
class Recommendation:
    fuel_optimal_propose: float
    ecart_vs_plan: float | None
    message: str
    cost_index_suggestion: str

    def as_dict(self) -> dict:
        return {
            "fuel_optimal_propose": self.fuel_optimal_propose,
            "ecart_vs_plan": self.ecart_vs_plan,
            "message": self.message,
            "cost_index_suggestion": self.cost_index_suggestion,
        }


def generate_recommendation(predicted_fuel: float, planned_fuel: float | None = None) -> Recommendation:
    """Genere une recommandation carburant a partir de la prediction du modele.

    predicted_fuel : consommation reelle predite par le modele (t)
    planned_fuel    : Trip Fuel + Taxi Fuel planifies (t), si disponibles
    """
    # Marge de securite de 2% ajoutee a la prediction pour proposer un
    # carburant a embarquer (les predictions sont un point central, pas
    # une borne superieure garantie).
    fuel_optimal = predicted_fuel * 1.02

    if planned_fuel is None:
        return Recommendation(
            fuel_optimal_propose=fuel_optimal,
            ecart_vs_plan=None,
            message="Estimation fournie à titre indicatif (aucun plan de vol de référence disponible pour comparaison).",
            cost_index_suggestion="Non applicable sans plan de vol de référence.",
        )

    ecart = predicted_fuel - planned_fuel

    if ecart > ECART_SIGNIFICATIF_T:
        message = (
            f"Le modèle anticipe une consommation supérieure de {ecart:.2f} t au plan actuel. "
            f"Envisager un complément de carburant discrétionnaire d'environ {ecart:.2f} t."
        )
        cost_index_suggestion = "Cost index inchangé recommandé — priorité à la marge de sécurité carburant."
    elif ecart < -ECART_SIGNIFICATIF_T:
        message = (
            f"Le modèle anticipe une consommation inférieure de {abs(ecart):.2f} t au plan actuel. "
            f"Une réduction du carburant discrétionnaire peut être envisagée, sous réserve de validation opérationnelle."
        )
        cost_index_suggestion = "Une légère augmentation du cost index (vitesse de croisière) peut être envisagée pour valoriser la marge disponible."
    else:
        message = "La consommation prédite est cohérente avec le plan de vol actuel ; aucun ajustement n'est recommandé."
        cost_index_suggestion = "Cost index inchangé."

    return Recommendation(
        fuel_optimal_propose=fuel_optimal,
        ecart_vs_plan=ecart,
        message=message,
        cost_index_suggestion=cost_index_suggestion,
    )


def compute_fuel_score(actual_fuel: float, reference_fuel: float) -> float:
    """Score d'efficience 0-100 : 50 = conforme à la référence (prédiction ou
    baseline), >50 = a consommé moins que prévu (efficient), <50 = plus que
    prévu. reference_fuel doit être strictement positif."""
    if reference_fuel <= 0:
        return 50.0
    ecart_pct = (actual_fuel - reference_fuel) / reference_fuel
    score = 50.0 - ecart_pct * 100.0 * FUEL_SCORE_SCALE
    return float(max(0.0, min(100.0, score)))
