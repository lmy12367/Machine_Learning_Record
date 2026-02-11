import math


def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    if len(numbers) < 2: return 0.0 
    avg = mean(numbers)
    variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def calculate_probability(x, mean, stdev):
    if stdev == 0: return 0 
    exponent = math.exp(-((x - mean)**2 / (2 * stdev**2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def group_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated

def summarize_dataset(dataset):
    summaries = []
    for column in zip(*dataset):
        summaries.append((mean(column), stdev(column)))
    del summaries[-1] 
    return summaries

def summarize_by_class(dataset):
    separated = group_by_class(dataset)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def calculate_class_probabilities(summaries, row):
    probabilities = {}
    for class_value, class_summaries in summaries.items():

        probabilities[class_value] = 1 
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = row[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label, probabilities



dataset = [
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [4.7, 3.2, 1.3, 0.2, 0],
    [7.0, 3.2, 4.7, 1.4, 1],
    [6.4, 3.2, 4.5, 1.5, 1],
    [6.9, 3.1, 4.9, 1.5, 1]
]

model = summarize_by_class(dataset)
print("模型统计参数 (均值, 标准差):")
for k, v in model.items():
    print(f"类别 {k}: {v}")

print("-" * 30)

test_sample = [5.0, 3.4, 1.5, 0.2] 
result, probs = predict(model, test_sample)

print(f"测试样本: {test_sample}")
print(f"各类别概率分值: {probs}")
print(f"预测结果: 类别 {result} ({'Setosa' if result==0 else 'Versicolor'})")