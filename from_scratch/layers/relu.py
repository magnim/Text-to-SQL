class ReLU():
    # def __init__(self):
    #     self.last_input = []
    #     self.weight_gradients = []

    def forward(self, x: list[float]) -> list[float]:
        self.last_input = x
        outputs = []
        for value in x:
            if value <= 0:
                outputs.append(0.0)
            else:
                outputs.append(value)
        return outputs

    def backward(self, gradients: list[float]) -> list[float]:
        output_gradients = []
        for value, gradient in zip(self.last_input, gradients):
            if value <= 0:
                output_gradients.append(0.0)
            else:
                output_gradients.append(gradient)
        return output_gradients