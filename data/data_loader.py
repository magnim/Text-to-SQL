import random
from collections.abc import Iterator
from data.batching import collate_examples,Batch
from data.text_dataset import TextDataset


class DataLoader:
    def __init__(self,dataset: TextDataset,batch_size: int,shuffle: bool = False,drop_last: bool = False,pad_token_id: int = 0,):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        dataset_size = len(self.dataset)
        if self.drop_last:
            return dataset_size // self.batch_size
        return (dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        indexes = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indexes)
        for start_index in range(0, len(indexes), self.batch_size):
            batch_indexes = indexes[start_index:start_index + self.batch_size]
            if self.drop_last and len(batch_indexes) < self.batch_size:
                continue
            examples = [self.dataset[index] for index in batch_indexes]
            yield collate_examples(examples, self.pad_token_id)