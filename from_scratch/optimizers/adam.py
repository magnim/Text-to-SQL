import math

class Adam:

    def __init__(self,learning_rate: float = 0.001,beta1: float = 0.9,beta2: float = 0.999,epsilon: float = 1e-8):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not (0 < beta1 < 1):
            raise ValueError("beta1 must be between 0 and 1.")
        if not (0 < beta2 < 1):
            raise ValueError("beta2 must be between 0 and 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time_step = 0

    def start_step(self) -> None:
        self.time_step += 1

    def _update_value(self,parameter: float,gradient: float,first_moment: float,second_moment: float) -> tuple[float, float, float]:

        new_first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * gradient
        new_second_moment = self.beta2 * second_moment + (1.0 - self.beta2) * (gradient * gradient)
        corrected_first_moment = new_first_moment / (1.0 - (self.beta1 ** self.time_step))
        corrected_second_moment = new_second_moment / (1.0 - (self.beta2 ** self.time_step))
        new_parameter = (parameter -
                         self.learning_rate * (
                                     corrected_first_moment / (math.sqrt(corrected_second_moment) + self.epsilon)))

        return new_parameter, new_first_moment, new_second_moment

    def update(self,parameter,gradient,first_moment,second_moment):
        if self.time_step == 0:
            raise RuntimeError(
                "start_step() must be called before update()."
            )

        updated_parameter = []
        updated_first_moment = []
        updated_second_moment = []

        if isinstance(parameter[0], list):
            for row in range(len(parameter)):
                updated_parameter_row = []
                updated_first_moment_row = []
                updated_second_moment_row = []

                for column in range(len(parameter[row])):
                    (
                        new_parameter,
                        new_first_moment,
                        new_second_moment,
                    ) = self._update_value(
                        parameter[row][column],
                        gradient[row][column],
                        first_moment[row][column],
                        second_moment[row][column],
                    )

                    updated_parameter_row.append(new_parameter)
                    updated_first_moment_row.append(new_first_moment)
                    updated_second_moment_row.append(new_second_moment)

                updated_parameter.append(updated_parameter_row)
                updated_first_moment.append(updated_first_moment_row)
                updated_second_moment.append(updated_second_moment_row)

        else:
            for index in range(len(parameter)):
                (
                    new_parameter,
                    new_first_moment,
                    new_second_moment,
                ) = self._update_value(
                    parameter[index],
                    gradient[index],
                    first_moment[index],
                    second_moment[index],
                )

                updated_parameter.append(new_parameter)
                updated_first_moment.append(new_first_moment)
                updated_second_moment.append(new_second_moment)

        return (
            updated_parameter,
            updated_first_moment,
            updated_second_moment,
        )
