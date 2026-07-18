class TextDataset:

    def __init__(self,token_ids: list[int],context_size: int,):
        if context_size <= 0:
            raise ValueError("context_size must be greater than zero.")
        if len(token_ids) <= context_size:
            raise ValueError( "token_ids must contain more tokens than context_size.")
        self.token_ids = token_ids
        self.context_size = context_size
    def __len__(self) -> int:
        return len(self.token_ids) - self.context_size
    def __getitem__(self,index: int,) -> tuple[list[int], int]:
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index is out of range.")
        start_index = index
        end_index = (start_index + self.context_size)
        context = self.token_ids[start_index:end_index]
        target = self.token_ids[end_index]
        return context, target


