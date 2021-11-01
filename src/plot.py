import random

import matplotlib.pyplot as plt

import facts
from spacingmodel import *


def main():
    fig, (axs1, axs2) = plt.subplots(2, 2, figsize=(10, 10))
    run_test(axs1, prop=True)
    run_test(axs2, prop=False)

    fig.show()


def run_test(axs, prop: bool, n=50):
    model = SpacingModel(enable_propagation=prop)
    facts.load_facts(model, subset_id=1)
    # model.facts = model.facts[:20]
    # model.facts = [fact for fact in model.facts if fact.question in {'BE', 'NL', 'DE'}]

    acts = []
    rofs = []

    # FIXME The issue
    #  Most recent fact does not appear to seen_facts i.e. activation = -inf
    #  Could be the root of all evil, who knows
    #   showing new=True EE 35000
    #   len(seen_facts)=3, len(not_seen_facts)=216, 219
    #   EE already seen 55
    #   ['UA', 'GA', 'BI']
    #   showing new=True EE 40000
    #   len(seen_facts)=3, len(not_seen_facts)=216, 219
    #   EE already seen 55
    #   ['UA', 'GA', 'BI']
    #   showing new=True EE 45000
    #   len(seen_facts)=3, len(not_seen_facts)=216, 219
    #   EE already seen 55
    #   ['UA', 'GA', 'BI']
    #   showing new=True EE 50000

    ts = 5_000
    rt = 2_000

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


if __name__ == '__main__':
    main()
