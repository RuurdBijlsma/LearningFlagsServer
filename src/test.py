from typing import Tuple

from spacingmodel import *


def print_activations(model: SpacingModel, time: int, msg: str = 'Activations'):
    print(f'{msg}:')
    for fact in model.facts:
        activation = model.calculate_activation(time, fact)
        rof, _ = model.calculate_alpha(time, fact)
        print(f'[{fact.fact_id:^5}]\t{fact.question:<25}{fact.answer:<25}{activation:<13.5}{rof:<10}')


def main():
    model = SpacingModel()

    def properties(loc: Tuple[float, float], population: float) -> dict:
        longitude, latitude = loc
        return {
            'longitude': longitude,
            'latitude': latitude,
            'population': population,
        }

    belgium = Fact(1, 'Belgium.png', 'Belgium', properties((50.5039, 4.4699), 11_560_000))
    netherlands = Fact(2, 'Netherlands.png', 'Netherlands', properties((52.1326, 5.2913), 17_440_000))
    germany = Fact(3, 'Germany.png', 'Germany', properties((51.1657, 10.4515), 83_240_000))

    model.add_fact(belgium)
    model.add_fact(netherlands)
    model.add_fact(germany)

    model.normalize_properties({'longitude': 0.5, 'latitude': 0.5, 'population': 1})

    rt = 2_000
    time = 10_000

    model.register_response(Response(belgium, time, rt, True))

    time += 10_000

    # NL is unseen, don't propagate
    assert model.calculate_activation(time, netherlands) == -float("inf")

    print_activations(model, time)
    belgium_act = model.calculate_activation(time, belgium)

    time += 10_000
    print_activations(model, time)

    model.register_response(Response(netherlands, time - 10_000, rt, True))
    print_activations(model, time)

    assert model.calculate_activation(time, belgium) > belgium_act

    for _ in range(4):
        time += 10_000
        model.register_response(Response(belgium, time, rt, True))

    time += 10_000

    print_activations(model, time)


if __name__ == '__main__':
    main()
