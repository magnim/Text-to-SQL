import random


class Sampler:

    def sample(self,probabilities: list[float],) -> int:
        random_value = random.random()
        cumulative_probability = 0.0

        for index, probability in enumerate(probabilities):
            cumulative_probability += probability

            if random_value <= cumulative_probability:
                return index
        return len(probabilities) - 1