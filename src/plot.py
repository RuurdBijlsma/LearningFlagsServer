import random

import matplotlib.pyplot as plt

import facts
from spacingmodel import *


def main():
    fig, (axs1, axs2) = plt.subplots(2, 2, figsize=(10, 10))
    prop_true = run_test(axs1, prop=True)
    prop_false = run_test(axs2, prop=False)

    print(f'{prop_true=}\t{prop_false=}')

    fig.show()


def run_test(axs, prop: bool, n=100) -> int:
    model = SpacingModel(enable_propagation=prop)
    facts.load_facts(model, subset_id=1)

    acts = []
    rofs = []

    ts = 5_000
    rt = 2_000

    random.seed(10)

    for i in range(n):
        t = ts * i
        fact, new = model.get_next_fact(t)
        print(f'showing {new=}', fact.question, t, )
        model.register_response(Response(fact, t + rt, rt, correct=random.randint(0, 1) == 1))

        activation = [
            model.calculate_activation(t, m_fact)
            for m_fact in model.facts
        ]
        rof = [
            model.get_rate_of_forgetting(t, m_fact)
            for m_fact in model.facts
        ]

        acts.append(activation)
        rofs.append(rof)

    ax1, ax2 = axs

    ax1.set_title(f'Activtion {prop=}')
    ax1.plot(acts)

    ax2.set_title(f'ROF {prop=}')
    ax2.plot(rofs)

    return len({r.fact.fact_id for r in model.responses})


if __name__ == '__main__':
    main()
