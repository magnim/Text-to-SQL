class TopPSampler:

    def __init__(self,p: float) -> None:
        if not isinstance(p, (int, float)):
            raise TypeError("p must be numeric.")
        if p <= 0.0 or p > 1.0:
            raise ValueError("p must be between 0 and 1.")
        self.p = p

    def get_top_p(self,indices: list[int],probabilities: list[float]) -> tuple[list[int], list[float]]:
        selected_indices = []
        selected_probabilities = []
        cumulative_probability = 0.0
        for index, probability in zip(indices, probabilities):
            selected_indices.append(index)
            selected_probabilities.append(probability)
            cumulative_probability += probability
            if cumulative_probability >= self.p:
                break
        return selected_indices,selected_probabilities