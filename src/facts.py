import pathlib
import random
from typing import List, Optional

import pandas as pd

import spacingmodel
from spacingmodel import SpacingModel, Fact

BASE_PATH = pathlib.Path('../data')

# These countries seem valid, but should not be included for some other reason
BLACKLIST = [
    # Shares flag with France
    'GF',
    'RE',
    'MQ',
    'PM',
    'GP',
    'YT',
    'NC',
]


def load_facts(model: SpacingModel, subset_id: Optional[int]) -> None:
    # subset size
    n = 60
    data = merge_data()
    facts_df = make_facts_from_df(data)

    if subset_id is not None:
        # Shuffle all facts so the subsets are random
        fixed_rng = random.Random(4)
        fixed_rng.shuffle(facts_df)
        print(f'Loading subset {subset_id}')
        facts_subsets = [facts_df[i * n:(i + 1) * n] for i in range((len(facts_df) + n - 1) // n)]
        facts = facts_subsets[subset_id]
        assert len(facts) == n, len(facts)
    else:
        # Outside the experiment we want to learn all facts, use rng without set seed to randomize
        random.shuffle(facts_df)
        facts = facts_df

    for fact in facts:
        model.add_fact(fact)

    model.normalize_properties({'coordinate': (1, False), 'population': (.6, True)})


def make_facts_from_df(countries: pd.DataFrame) -> List[Fact]:
    facts = []
    for idx, (country, properties) in enumerate(countries.iterrows()):
        country: str

        name = properties['name']
        population = properties['population']
        longitude = properties['longitude']
        latitude = properties['latitude']

        facts.append(Fact(
            fact_id=idx,
            question=country,
            answer=name,
            properties={
                'population': population,
                'coordinate': (longitude, latitude),
            },
        ))

    return facts


def merge_data() -> pd.DataFrame:
    countries = pd.read_json(BASE_PATH / 'countries.json', orient='index')
    population = pd.read_csv(BASE_PATH / 'population.csv', index_col=['cca2'])
    location = pd.read_csv(BASE_PATH / 'flags-latlng.tsv', sep='\t', index_col='country')
    countries.columns = ['name']
    countries['population'] = population['pop2021']
    countries['latitude'] = location['latitude']
    countries['longitude'] = location['longitude']
    countries.dropna(inplace=True)
    countries.drop(BLACKLIST, inplace=True)
    return countries


def make_similarity_square(facts: List[Fact]) -> pd.DataFrame:
    countries = [fact.question for fact in facts]

    df = pd.DataFrame(index=countries, columns=countries)

    for a in facts:
        df_part = pd.Series(index=countries, dtype='float64')
        for b in facts:
            df_part[b.question] = a.similarity_to(b)

        df[a.question] = df_part

    return df


def main() -> None:
    model = spacingmodel.SpacingModel(enable_propagation=True)
    load_facts(model, subset_id=None)

    df = make_similarity_square(model.facts)

    countries = pd.read_json(BASE_PATH / 'countries.json', orient='index')

    for idx, row in df.iterrows():
        most_similar = row.sort_values(ascending=False).index.values[1:20]
        most_similar = (countries.loc[code][0] for code in most_similar)
        name = countries.loc[idx][0]
        output = '\n'.join(f'\t{item}' for item in most_similar)
        print(f'{name}:\n{output}')


if __name__ == '__main__':
    main()
