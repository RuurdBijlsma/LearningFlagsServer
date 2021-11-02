import dataclasses
import itertools
import operator
from typing import Optional, Union, Dict, List, Tuple

import math
import pandas as pd


def tuple_op(op, a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        a1, a2 = a
        b1, b2 = b
        return op(a1, b1), op(a2, b2)
    if isinstance(a, tuple):
        a1, a2 = a
        return op(a1, b), op(a2, b)
    else:
        return op(a, b)


@dataclasses.dataclass(frozen=True)
class Fact:
    fact_id: int
    question: str
    answer: str
    properties: Dict[str, float]

    def similarity_to(self, other: 'Fact') -> float:
        distance = 0
        distances = []
        for prop in self.properties:
            value = self.properties[prop]
            other_value = other.properties[prop]
            if isinstance(value, tuple):
                assert isinstance(other_value, tuple)
                d = math.sqrt((value[0] - other_value[0]) ** 2 + (value[1] - other_value[1]) ** 2)
            else:
                d = (value - other_value) ** 2
            distances.append(d)
            distance += d

        if distance == 0:
            return float('inf')

        similarity = 1 / math.sqrt(distance)
        return similarity

    def __str__(self) -> str:
        return f'Fact({self.fact_id}, Q={self.question}, A={self.answer})'


@dataclasses.dataclass(frozen=True)
class Response:
    fact: Fact
    start_time: float
    rt: float
    correct: bool


@dataclasses.dataclass(frozen=True)
class Encounter:
    activation: float
    time: float
    reaction_time: float
    decay: float


class SpacingModel:
    # Model constants
    LOOKAHEAD_TIME = 15000
    FORGET_THRESHOLD = -0.8
    DEFAULT_ALPHA = 0.3
    C = 0.25
    F = 1.0
    PROPAGATION_RATE = 0.01
    ROF_PROPAGATION_RATE = 0.1  # Default = 0.1
    ACTIVATION_PROPAGATION_RATE = 1  # Default = 1

    def __init__(self, enable_propagation: bool) -> None:
        print(f'Creating model with propagation={enable_propagation}')

        self.enable_propagation = enable_propagation
        self.facts = []
        self.responses = []
        self.knowledge_factor = {}

    def add_fact(self, fact: Fact) -> None:
        """
        Add a fact to the list of study items.
        """
        # Ensure that a fact with this ID does not exist already
        if next((f for f in self.facts if f.fact_id == fact.fact_id), None):
            raise RuntimeError(
                f"Error while adding fact: There is already a fact with the same ID: {fact.fact_id}. "
                "Each fact must have a unique ID")

        self.knowledge_factor[fact.fact_id] = 0
        self.facts.append(fact)

    def normalize_properties(self, weight_values: Dict[str, Tuple[float, bool]]) -> None:
        """
        Normalize the properties and assign weights.
        :param weight_values: Dict of property names and their weights + whether to apply log
        """

        print('Normalizing properties with weights:', weight_values)

        def weird_log(a, b):
            if a == 0:
                return float('inf')
            return math.log(a, b)

        for fact in self.facts:
            for property_name, (_, apply_log) in weight_values.items():

                if apply_log:
                    fact.properties[property_name] = tuple_op(weird_log, fact.properties[property_name], 2)

        ranges = {}
        for key in weight_values:
            values = [fact.properties[key] for fact in self.facts]
            if isinstance(values[0], tuple):
                min_value = min(itertools.chain(*values))
                max_value = max(itertools.chain(*values))
            else:
                min_value = min(values)
                max_value = max(values)
            ranges[key] = (min_value, max_value)

        for fact in self.facts:
            for property_name, (weight, _) in weight_values.items():
                value = fact.properties[property_name]

                min_value, max_value = ranges[property_name]
                value = tuple_op(operator.sub, value, min_value)
                value = tuple_op(operator.truediv, value, (max_value - min_value))
                value = tuple_op(operator.mul, value, weight)

                fact.properties[property_name] = value

        pass

    def register_response(self, response: Response) -> None:
        """
        Register a response.
        """
        # Prevent duplicate responses
        if next((r for r in self.responses if r.start_time == response.start_time), None):
            raise RuntimeError("Error while registering response: A response has already been logged at this "
                               f"start_time: {response.start_time}. Each response must occur at a unique start_time.")

        self.responses.append(response)

        if self.enable_propagation:
            self.propagate_response(response)

    def propagate_response(self, response: Response) -> None:
        for other_fact in (fact for fact in self.facts if fact != response.fact):
            # Don't propagate to unseen facts
            if other_fact not in (response.fact for response in self.responses):
                continue

            similarity = response.fact.similarity_to(other_fact)
            # print(f'{response.fact} ~ {other_fact}: {similarity}')

            magnitude = self.PROPAGATION_RATE * similarity
            self.knowledge_factor[other_fact.fact_id] += magnitude * (1 if response.correct else -1)

    def get_next_fact(self, current_time: float) -> (Fact, bool):
        """
        Returns a tuple containing the fact that needs to be repeated most urgently and a boolean indicating whether
        this fact is new (True) or has been presented before (False).
        If none of the previously studied facts needs to be repeated right now, return a new fact instead.
        """
        # Calculate all fact activations in the near future
        fact_activations = [(f, self.calculate_activation(current_time + self.LOOKAHEAD_TIME, f)) for f in self.facts]

        seen_facts = [(f, a) for (f, a) in fact_activations if a > -float("inf")]
        not_seen_facts = [(f, a) for (f, a) in fact_activations if a == -float("inf")]

        print(f'{len(seen_facts)=}, {len(not_seen_facts)=}, {len(self.facts)}')
        # Prevent an immediate repetition of the same fact

        if len(seen_facts) > 1:
            last_response = self.responses[-1]
            seen_facts = [(f, a) for (f, a) in seen_facts if f.fact_id != last_response.fact.fact_id]
            print(last_response.fact.question, 'already seen', last_response.fact.fact_id)

        # Reinforce the weakest fact with an activation below the threshold
        seen_facts_below_threshold = [(f, a) for (f, a) in seen_facts if a < self.FORGET_THRESHOLD]
        if len(not_seen_facts) == 0 or len(seen_facts_below_threshold) > 0:
            weakest_fact = min(seen_facts, key=lambda t: t[1])
            return weakest_fact[0], False

        # If none of the previously seen facts has an activation below the threshold, return a new fact
        return not_seen_facts[0][0], True

    def calculate_alpha(self, time: float, fact: Fact) -> Tuple[float, List[Encounter]]:
        """
        Calculate alpha and list of previous encounters
        :return:
        """

        encounters = []
        responses_for_fact = [r for r in self.responses if r.fact.fact_id == fact.fact_id and r.start_time < time]
        alpha = self.DEFAULT_ALPHA
        # Calculate the activation by running through the sequence of previous responses
        for response in responses_for_fact:
            activation = self.calculate_activation_from_encounters(encounters, response.start_time)

            encounters.append(Encounter(
                activation,
                response.start_time,
                self.normalise_reaction_time(response),
                self.DEFAULT_ALPHA,
            ))
            alpha = self.estimate_alpha(encounters, activation, response, alpha)

            # Update decay estimates of previous encounters
            encounters = [
                dataclasses.replace(encounter, decay=self.calculate_decay(encounter.activation, alpha))
                for encounter in encounters
            ]

        return alpha, encounters

    def get_rate_of_forgetting(self, time: float, fact: Fact) -> float:
        """
        Return the estimated rate of forgetting of the fact at the specified time
        """
        alpha, _ = self.calculate_alpha(time, fact)

        return alpha

    def calculate_activation(self, time: float, fact: Fact) -> float:
        """
        Calculate the activation of a fact at the given time.
        """

        _, encounters = self.calculate_alpha(time, fact)

        # Add knowledge factor, high knowledge factor value means more known, so activation should be slightly higher
        return self.calculate_activation_from_encounters(encounters, time) + \
               self.knowledge_factor[fact.fact_id] * self.ACTIVATION_PROPAGATION_RATE

    def calculate_decay(self, activation: float, alpha: float) -> float:
        """
        Calculate activation-dependent decay
        """
        return self.C * math.exp(activation) + alpha

    def estimate_alpha(self, encounters: [Encounter], activation: float, response: Response,
                       previous_alpha: float) -> float:
        """
        Estimate the rate of forgetting parameter (alpha) for an item.
        """
        if len(encounters) < 3:
            return self.DEFAULT_ALPHA

        a_fit = previous_alpha
        reading_time = self.get_reading_time(response.fact.question)
        estimated_rt = self.estimate_reaction_time_from_activation(activation, reading_time)
        est_diff = estimated_rt - self.normalise_reaction_time(response)

        if est_diff < 0:
            # Estimated RT was too short (estimated activation too high), so actual decay was larger
            a0 = a_fit
            a1 = a_fit + 0.05

        else:
            # Estimated RT was too long (estimated activation too low), so actual decay was smaller
            a0 = a_fit - 0.05
            a1 = a_fit

        # Binary search between previous fit and proposed alpha
        for _ in range(6):
            # Adjust all decays to use the new alpha
            a0_diff = a0 - a_fit
            a1_diff = a1 - a_fit
            d_a0 = [dataclasses.replace(e, decay=e.decay + a0_diff) for e in encounters]
            d_a1 = [dataclasses.replace(e, decay=e.decay + a1_diff) for e in encounters]

            # Calculate the reaction times from activation and compare against observed RTs
            encounter_window = encounters[max(1, len(encounters) - 5):]
            total_a0_error = self.calculate_predicted_reaction_time_error(encounter_window, d_a0, reading_time)
            total_a1_error = self.calculate_predicted_reaction_time_error(encounter_window, d_a1, reading_time)

            # Adjust the search area based on the lowest total error
            ac = (a0 + a1) / 2
            if total_a0_error < total_a1_error:
                a1 = ac
            else:
                a0 = ac

        # The new alpha estimate is the average value in the remaining bracket
        # Add knowledge factor as well, more known = higher factor, so rof should be lower for them
        return (a0 + a1) / 2 + self.knowledge_factor[response.fact.fact_id] * -1 * self.ROF_PROPAGATION_RATE

    @staticmethod
    def calculate_activation_from_encounters(encounters: [Encounter], current_time: float) -> float:
        included_encounters = [e for e in encounters if e.time < current_time]

        if len(included_encounters) == 0:
            return -float("inf")

        decays = [e.decay for e in included_encounters]
        if max(decays) > 5:
            print(f'max decays {max(decays)}')
        try:
            total = 0
            for e in included_encounters:
                total += math.pow((current_time - e.time) / 1000, -e.decay)
        except OverflowError:
            total = float('inf')

        if total == 0:
            return -float('inf')

        return math.log(total)

    def calculate_predicted_reaction_time_error(self, test_set: [Encounter], decay_adjusted_encounters: [Encounter],
                                                reading_time: float) -> float:
        """
        Calculate the summed absolute difference between observed response times and those predicted based on a decay
        adjustment.
        """
        activations = [
            self.calculate_activation_from_encounters(decay_adjusted_encounters, e.time - 100)
            for e in test_set
        ]
        rt = [self.estimate_reaction_time_from_activation(a, reading_time) for a in activations]
        rt_errors = [abs(e.reaction_time - rt) for (e, rt) in zip(test_set, rt)]
        return sum(rt_errors)

    def estimate_reaction_time_from_activation(self, activation: float, reading_time: float) -> float:
        """
        Calculate an estimated reaction time given a fact's activation and the expected reading time 
        """
        return (self.F * math.exp(-activation) + (reading_time / 1000)) * 1000

    def get_max_reaction_time_for_fact(self, fact: Fact) -> float:
        """
        Return the highest response time we can reasonably expect for a given fact
        """
        reading_time = self.get_reading_time(fact.question)
        max_rt = 1.5 * self.estimate_reaction_time_from_activation(self.FORGET_THRESHOLD, reading_time)
        return max_rt

    @staticmethod
    def get_reading_time(text: str) -> float:
        """
        Return expected reading time in milliseconds for a given string
        """
        word_count = len(text.split())

        if word_count > 1:
            character_count = len(text)
            return max((-157.9 + character_count * 19.5), 300.)

        return 300.

    def normalise_reaction_time(self, response: Response) -> float:
        """
        Cut off extremely long responses to keep the reaction time within reasonable bounds
        """
        rt = response.rt if response.correct else 60000
        max_rt = self.get_max_reaction_time_for_fact(response.fact)
        return min(rt, max_rt)

    def export_data(self, path: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        Save the response data to the specified csv file, and return a copy of the pandas DataFrame.
        If no path is specified, return a CSV-formatted copy of the data instead.
        """

        def calc_rof(row):
            return self.get_rate_of_forgetting(row["start_time"] + 1, Fact(**row["fact"]))

        dat_resp = pd.DataFrame(self.responses)
        dat_facts = pd.DataFrame([r.fact for r in self.responses])
        dat = pd.concat([dat_resp, dat_facts], axis=1)

        # Add column for rate of forgetting estimate after each observation
        dat["alpha"] = dat.apply(calc_rof, axis=1)
        dat.drop(columns="fact", inplace=True)

        # Add trial number column
        dat.index.name = "trial"
        dat.index = dat.index + 1

        # Save to CSV file if a path was specified, otherwise return the CSV-formatted output
        if path is not None:
            dat.to_csv(path, encoding="UTF-8")
            return dat

        return dat.to_csv()
