class MSE:
    def forward(self, prediction: float, target: float) -> float:
        # error = prediction - target
        # loss = error ** 2
        return (prediction - target) ** 2

    def backward(self,prediction: float,target: float) -> float:
        # error = prediction - target
        # gradient = 2 * error
        return 2 * (prediction - target)
