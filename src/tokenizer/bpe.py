import re
import json

class BPETrainer:
    """
    Trains a simple Byte Pair Encoding tokenizer.

    Input:
        words: list of words from training corpus

    Example:
        ["low", "lower", "lowest", "low"]
    """

    def __init__(self, words: list[str]):
        self.words = []
        for word in words:
            for piece in self._pre_tokenize(word):
                if piece != " ":
                    self.words.append(piece)
        self.corpus = self._build_initial_corpus()
        self.merge_rules: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}

    def _pre_tokenize(self, text: str) -> list[str]:
        pieces = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\w\s]|\s+", text)
        normalized = []
        for piece in pieces:
            if piece.isspace():
                normalized.append(" ")
            else:
                normalized.append(piece)
        return normalized

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

    def merge_pair(self,pair_to_merge: tuple[str, str]) -> None:
        """
        Merge a selected adjacent pair everywhere in the corpus.

        Example:
            pair_to_merge = ("l", "o")

            ("l", "o", "w") becomes ("lo", "w")
        """
        new_corpus = {}
        # Loop through self.corpus
        # Replace matching adjacent pairs
        # Store the new token tuple with the same frequency

        for tokens, frequency in self.corpus.items():
            new_tokens = self._merge_tokens(tokens, pair_to_merge)
            new_token_tuple = tuple(new_tokens)
            if new_token_tuple not in new_corpus:
                new_corpus[new_token_tuple] = 0
            new_corpus[new_token_tuple] += frequency

        self.corpus = new_corpus

    def train(self, num_merges: int):
        for _ in range(num_merges):
            best_pair = self.find_best_pair()
            if best_pair is None:
                break

            self.merge_rules.append(best_pair)
            self.merge_pair(best_pair)
        self._build_vocab()

    def _merge_tokens(self,tokens: tuple[str, ...] | list[str],pair_to_merge: tuple[str, str]) -> list[str]:
        """
        Merge a single adjacent pair inside one token sequence.

        Example:
            tokens = ("l", "o", "w")
            pair_to_merge = ("l", "o")

        Returns:
            ["lo", "w"]
        """
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                    i < len(tokens) - 1
                    and tokens[i] == pair_to_merge[0]
                    and tokens[i + 1] == pair_to_merge[1]
            ):
                merged_token = tokens[i] + tokens[i + 1]
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def encode(self, text: str) -> list[str]:
        encoded_tokens = []
        for piece in self._pre_tokenize(text):
            if piece == " ":
                encoded_tokens.append(" ")
                continue
            tokens = list(piece)
            for merge_rule in self.merge_rules:
                tokens = self._merge_tokens(tokens, merge_rule)
            encoded_tokens.extend(tokens)
        return encoded_tokens

    def _build_vocab(self) -> None:
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.vocab["<BOS>"] = 2
        self.vocab["<EOS>"] = 3
        self.vocab[" "] = 4

        for word in self.words:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        for left, right in self.merge_rules:
            merged_token = left + right

            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

    def encode_ids(self, text: str) -> list[int]:
        """
        Encode text into vocabulary IDs.

        A deterministic instance-local cache avoids recomputing the full BPE
        merge sequence for repeated schema identifiers and SQL fragments.
        """
        cache = getattr(self, "_encode_ids_cache", None)
        if cache is None:
            cache = {}
            self._encode_ids_cache = cache
        if text in cache:
            return list(cache[text])

        tokens = self.encode(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["<UNK>"])
        cache[text] = tuple(ids)
        return list(ids)

    def decode_ids(self, token_ids: list[int]) -> str:
        id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        decoded_tokens = []
        for token_id in token_ids:
            token = id_to_token.get(token_id, "<UNK>")
            if token == "<BOS>":
                continue
            if token == "<EOS>":
                break
            if token == "<PAD>":
                continue
            decoded_tokens.append(token)
        return "".join(decoded_tokens)

    def save(self, file_path: str) -> None:
        data = {
            "vocab": self.vocab,
            "merge_rules": self.merge_rules,
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        tokenizer = cls(words=[])

        tokenizer.vocab = {
            token: int(token_id)
            for token, token_id in data["vocab"].items()
        }

        tokenizer.merge_rules = [
            (left, right)
            for left, right in data["merge_rules"]
        ]

        return tokenizer
