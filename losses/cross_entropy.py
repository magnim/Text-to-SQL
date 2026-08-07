import math
from layers.softmax import Softmax

class CrossEntropyLoss:
    def __init__(self):
        self.last_logits = None
        self.last_target_ids = None
        self.last_probabilities = None
        self.last_loss = None
        self.softmax = Softmax()

    def _validate_inputs(self,logits,target_ids):
        if not isinstance(logits, list):
            raise TypeError("Logits must be a list.")

        if not isinstance(target_ids, list):
            raise TypeError("Target IDs must be a list.")

        if not logits:
            raise ValueError("Logits cannot be empty.")

        if not target_ids:
            raise ValueError("Target IDs cannot be empty.")

        if len(logits) != len(target_ids):
            raise ValueError("The number of logit rows must match the number of target IDs.")

        if not isinstance(logits[0], list):
            raise TypeError("Every logit row must be a list.")

        vocabulary_size = len(logits[0])

        if vocabulary_size == 0:
            raise ValueError("Logit rows cannot be empty.")

        for position_logits in logits:
            if not isinstance(position_logits,list):
                raise TypeError("Every logit row must be a list.")

            if len(position_logits)!= vocabulary_size:
                raise ValueError("Every logit row must have ""the same vocabulary size.")

            for logit in position_logits:
                if not isinstance(logit,(int, float)):
                    raise TypeError("Every logit must be numeric.")

        for target_id in target_ids:
            if not isinstance(target_id,int):
                raise TypeError("Every target ID must be an integer.")
            if target_id < 0 or target_id >= vocabulary_size:
                raise ValueError(f"Target ID {target_id} is outside the vocabulary range.")


    # def _softmax(self, logits):
    #     maximum_logit = max(logits)
    #     exponentials = [math.exp(logit - maximum_logit) for logit in logits]
    #     exponential_sum = sum(exponentials)
    #     probabilities = [exponential / exponential_sum for exponential in exponentials]
    #     return probabilities

    def forward(self,logits,target_ids):
        self._validate_inputs(logits,target_ids)
        self.last_logits = [row.copy()for row in logits]
        self.last_target_ids = (target_ids.copy())
        self.last_probabilities = []
        total_loss = 0.0
        epsilon = 1e-12

        for (position_logits,target_id) in zip(logits,target_ids):
            probabilities = (self.softmax.forward(position_logits))
            self.last_probabilities.append(probabilities)
            correct_probability = max(probabilities[target_id],epsilon)
            position_loss = -math.log(correct_probability)
            total_loss += position_loss
        average_loss = (total_loss / len(target_ids))
        self.last_loss = average_loss
        return average_loss

    def backward(self):
        if self.last_probabilities is None:
            raise RuntimeError("forward() must be called before backward().")

        sequence_length = len(self.last_target_ids)
        logits_gradients = []

        for (probabilities,target_id) in zip(self.last_probabilities,self.last_target_ids):
            position_gradient = (probabilities.copy())
            position_gradient[target_id] -= 1.0
            position_gradient = [gradient / sequence_length for gradient in position_gradient]
            logits_gradients.append(position_gradient)

        return logits_gradients