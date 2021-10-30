import matplotlib.pyplot as plt

import facts
from spacingmodel import *


def main():
    run_test(True)
    run_test(False)


def run_test(prop: bool, n=50):
    model = SpacingModel(enable_propagation=prop)
    facts.load_facts(model, subset_id=1)
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
        model.register_response(Response(fact, t + rt, rt, correct=True))

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

    plt.title(f'Activtion {prop=}')
    plt.plot(acts)
    plt.show()

    plt.title(f'ROF {prop=}')
    plt.plot(rofs)
    plt.show()


if __name__ == '__main__':
    main()
