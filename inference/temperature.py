class Temperature:
    def forward(self,logits: list[float],temperature: float) -> list[float]:
        if not isinstance(logits, list):
            raise TypeError("logits must be a list.")
        if not logits:
            raise ValueError("logits cannot be empty.")
        if any(not isinstance(logit, (int, float)) for logit in logits):
            raise TypeError("Every logit must be numeric.")
        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be numeric.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        return [logit / temperature for logit in logits]