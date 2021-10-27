import facts
from spacingmodel import *


def print_activations(model: SpacingModel, time: int, msg: str = 'Activations'):
    print(f'{msg}:')
    for fact in model.facts:
        activation = model.calculate_activation(time, fact)
        rof, _ = model.calculate_alpha(time, fact)
        print(f'[{fact.fact_id:^5}]\t{fact.question:<25}{fact.answer:<25}{activation:<13.5}{rof:<10}')


def main():
    model = SpacingModel(enable_propagation=True)

    facts.load_facts(model, subset_id=None)

    belgium = next(fact for fact in model.facts if fact.question == 'BE')
    netherlands = next(fact for fact in model.facts if fact.question == 'NL')
    germany = next(fact for fact in model.facts if fact.question == 'DE')

    test_facts = [netherlands, germany, belgium]

    rt = 2_000
    time = 10_000

    for fact in test_facts:
        model.register_response(Response(fact, time, rt, correct=True))
        time += 1

    time = 20_000
    model.register_response(Response(netherlands, start_time=time, rt=rt, correct=True))

    be_act, _ = model.calculate_alpha(time + 1, belgium)
    de_act, _ = model.calculate_alpha(time + 1, germany)

    # Belgium is more like the netherlands than Germany
    assert be_act > de_act

    time = 30_000
    model.register_response(Response(germany, start_time=time, rt=rt, correct=True))

    be_act, _ = model.calculate_alpha(time + 1, belgium)
    nl_act, _ = model.calculate_alpha(time + 1, netherlands)

    assert nl_act > be_act

    print_activations(model, time + 1)

    # model.register_response(Response(belgium, time, rt, True))
    #
    # time += 10_000
    #
    # # NL is unseen, don't propagate
    # assert model.calculate_activation(time, netherlands) == -float("inf")
    #
    # print_activations(model, time)
    # belgium_act = model.calculate_activation(time, belgium)
    #
    # time += 10_000
    # print_activations(model, time)
    #
    # model.register_response(Response(netherlands, time - 10_000, rt, True))
    # print_activations(model, time)
    #
    # assert model.calculate_activation(time, belgium) > belgium_act
    #
    # for _ in range(4):
    #     time += 10_000
    #     model.register_response(Response(belgium, time, rt, True))
    #
    # time += 10_000
    #
    # print_activations(model, time)


if __name__ == '__main__':
    main()
