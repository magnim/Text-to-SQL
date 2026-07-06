class ReLU():
    # def __init__(self):
    #     self.last_input = []
    #     self.weight_gradients = []

    def forward(self, x: float) -> float:
        self.last_input = x
        if x<0:
            return 0.0
        return x

    def backward(self,gradient: float) -> float:
        if self.last_input<=0:
            return 0.0
        else:
            return gradient