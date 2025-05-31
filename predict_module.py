import csv
import random

class Perceptron:
    def __init__(self, history_length, weight_limit=127):
        self.weights = [0] * (history_length + 1) 
        self.weight_limit = weight_limit

    def predict(self, history):
        y = self.weights[0]
        for i in range(len(history)):
            y += self.weights[i + 1] * history[i]
        return y

    def train(self, history, actual, theta):
        y = self.predict(history)
        prediction = 1 if y >= 0 else -1
        if prediction != actual or abs(y) <= theta:
            self.weights[0] += actual  # bias
            for i in range(len(history)):
                self.weights[i + 1] += actual * history[i]
                self.weights[i + 1] = max(-self.weight_limit, min(self.weights[i + 1], self.weight_limit))


class BranchPredictor:
    def __init__(self, num_perceptrons=128, history_length=16, weight_limit=127, theta=None):
        self.history_length = history_length
        self.weight_limit = weight_limit
        self.theta = theta if theta is not None else int(1.93 * history_length + 14)
        self.perceptrons = [Perceptron(history_length, weight_limit) for _ in range(num_perceptrons)]
        self.global_history = [-1] * history_length

    def _index(self, pc):
        return pc % len(self.perceptrons)

    def predict(self, pc):
        idx = self._index(pc)
        y = self.perceptrons[idx].predict(self.global_history)
        return 1 if y >= 0 else -1

    def update(self, pc, actual):
        idx = self._index(pc)
        self.perceptrons[idx].train(self.global_history, actual, self.theta)
        self.global_history = [actual] + self.global_history[:-1]

    def get_perceptron_weights(self):
        return [p.weights for p in self.perceptrons]



class LoopPredictor:
    def __init__(self, size=128):
        self.entries = [{} for _ in range(size)]

    def _index(self, pc):
        return pc % len(self.entries)

    def predict(self, pc):
        entry = self.entries[self._index(pc)]
        if not entry or "trip_count" not in entry or entry["confidence"] < 2:
            return None
        if entry["current_iter"] + 1 < entry["trip_count"]:
            return 1
        else:
            return -1

    def update(self, pc, actual):
        idx = self._index(pc)
        entry = self.entries[idx]

        if not entry:
            entry.update({
                "trip_count": None,
                "current_iter": 0,
                "last_outcome": actual,
                "confidence": 0,
            })
        else:
            if entry["last_outcome"] == 1 and actual == 1:
                entry["current_iter"] += 1
            elif entry["last_outcome"] == 1 and actual == -1:
                if entry["trip_count"] == entry["current_iter"]:
                    entry["confidence"] += 1
                else:
                    entry["trip_count"] = entry["current_iter"]
                    entry["confidence"] = 1
                entry["current_iter"] = 0
            else:
                entry["current_iter"] = 0

            entry["last_outcome"] = actual

        self.entries[idx] = entry


def load_trace_from_file(path):
    trace = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                pc = int(parts[0])
                actual = int(parts[1])
                trace.append((pc, actual))
    return trace


def run_predictor(trace_file="trace_alt.txt", output_csv="results.csv", predictor = None, loop_predictor = None):
    if predictor is None:
        predictor = BranchPredictor()
    if loop_predictor is None:    
        loop_predictor = LoopPredictor()
    trace = load_trace_from_file(trace_file)

    correct = 0
    step_data = []

    for step, (pc, actual) in enumerate(trace):
        source = "perceptron"
        loop_guess = loop_predictor.predict(pc)
        if loop_guess is not None:
            prediction = loop_guess
            source = "loop"
        else:
            prediction = predictor.predict(pc)

        if prediction == actual:
            correct += 1
        accuracy = correct / (step + 1)

        step_data.append([step, pc, prediction, actual, accuracy, source])

        predictor.update(pc, actual)
        loop_predictor.update(pc, actual)

    print(f"Final Accuracy: {accuracy:.2%}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "PC", "Prediction", "Actual", "Accuracy", "Source"])
        writer.writerows(step_data)


if __name__ == "__main__":
    run_predictor()
