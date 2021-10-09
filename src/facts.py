from typing import Tuple

from spacingmodel import SpacingModel, Fact


def load_facts(model: SpacingModel) -> None:
    def properties(loc: Tuple[float, float], population: float) -> dict:
        longitude, latitude = loc
        return {
            'longitude': longitude,
            'latitude': latitude,
            'population': population,
        }

    belgium = Fact('BE', 'Belgium.png', 'Belgium', properties((50.5039, 4.4699), 11_560_000))
    netherlands = Fact('NL', 'Netherlands.png', 'Netherlands', properties((52.1326, 5.2913), 17_440_000))
    germany = Fact('DE', 'Germany.png', 'Germany', properties((51.1657, 10.4515), 83_240_000))

    model.add_fact(belgium)
    model.add_fact(netherlands)
    model.add_fact(germany)

    model.normalize_properties({'longitude': 0.5, 'latitude': 0.5, 'population': 1})
