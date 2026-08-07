class TopKSampler:

    def __init__(self,k: int) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = k

    def get_top_k(self,logits: list[float]) -> tuple[list[int], list[float]]:
        if self.k > len(logits):
            raise ValueError("k cannot be greater than the number of logits.")
        working_logits = logits.copy()
        top_indices = []
        top_logits = []
        for _ in range(self.k):
            best_index = 0
            for index in range(1, len(working_logits)):
                if working_logits[index] > working_logits[best_index]:
                    best_index = index
            top_indices.append(best_index)
            top_logits.append(working_logits[best_index])
            working_logits[best_index] = float("-inf")

        return top_indices, top_logits