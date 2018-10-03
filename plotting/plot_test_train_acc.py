import json
import numpy as np
import matplotlib.pyplot as plt

with open('run_120k_samples_holdout_01_amp5-tag-train_mse.json', 'r') as file:
    train_data = json.load(file)

with open('run_120k_samples_holdout_01_amp5-tag-test_mse.json', 'r') as file:
    test_data = json.load(file)


test_steps = []
test_mse = []
for step in test_data:
    test_steps.append(step[1])
    test_mse.append(step[2])


train_steps = []
train_mse = []
for step in train_data:
    train_steps.append(step[1])
    train_mse.append(step[2])



plt.plot(test_steps, test_mse, label='Test Data', color='green')
plt.plot(train_steps, train_mse, label='Train Data', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Average MSE')
plt.title('Accuracy')
plt.ylim([0.002, 0.008])
plt.legend()
plt.savefig('./accuracy.png')
plt.show()






