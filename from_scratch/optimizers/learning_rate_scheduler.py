class LearningRateScheduler:

    def __init__(self,initial_learning_rate: float,decay_factor: float,decay_every: int) -> None:

        if initial_learning_rate <= 0.0:
            raise ValueError("initial_learning_rate must be positive.")

        if not (0.0 < decay_factor < 1.0):
            raise ValueError("decay_factor must be between 0 and 1.")

        if decay_every <= 0:
            raise ValueError("decay_every must be positive.")

        self.initial_learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.decay_every = decay_every

    def get_learning_rate(self,epoch: int) -> float:
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        decay_steps = epoch // self.decay_every
        current_learning_rate = (self.initial_learning_rate * (self.decay_factor ** decay_steps))
        return current_learning_rate