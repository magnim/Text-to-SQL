class BPETrainer:
    """
    Trains a simple Byte Pair Encoding tokenizer.

    Input:
        words: list of words from training corpus

    Example:
        ["low", "lower", "lowest", "low"]
    """

    def __init__(self, words: list[str]):
        self.words = words
        self.corpus = self._build_initial_corpus()

    def _build_initial_corpus(self) -> dict[tuple[str, ...], int]:
        """
        Convert words into character tuples and store frequencies.

        Example:
            ["low", "low", "lower"]

        becomes:
            {
                ("l", "o", "w"): 2,
                ("l", "o", "w", "e", "r"): 1
            }
        """
        corpus = {}

        for word in self.words:
            token_tuple = tuple(word)
            if token_tuple not in corpus:
                corpus[token_tuple] = 0
            corpus[token_tuple] += 1

        return corpus

    def count_pairs(self) -> dict[tuple[str, str], int]:
        """
        Count adjacent token-pair frequencies in the current corpus.

        Example:
            ("l", "o", "w"): 2

        contributes:
            ("l", "o") += 2
            ("o", "w") += 2
        """
        pair_counts = {}

        for tokens, frequency in self.corpus.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair not in pair_counts:
                    pair_counts[pair] = 0
                pair_counts[pair] += frequency
        return pair_counts

    def find_best_pair(self) -> tuple[str, str] | None:
        """
        Find the most frequent adjacent token pair.

        Returns:
            The pair with the highest frequency.
            If there are no pairs, return None.
        """
        pair_counts = self.count_pairs()

        if not pair_counts:
            return None

        # Find and return the pair with the highest frequency.
        best_pair = None
        best_count = -1
        for pair, count in pair_counts.items():
            if count > best_count:
                best_pair = pair
                best_count = count

        return best_pair
