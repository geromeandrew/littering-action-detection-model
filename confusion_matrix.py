import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# Define the confusion matrix values
fn = 19
fp = 17
tn = 751
tp = 173

# Create the confusion matrix as a 2D array
confusion_matrix = np.array([[tn, fp], [fn, tp]])

# Define the class labels
labels = ['Negative', 'Positive']

# Create the heatmap using Seaborn
# sns.set(font_scale=1.4)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Reds', xticklabels=labels, yticklabels=labels, cbar=True)

# Set the axis labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Set the title
plt.title('Confusion Matrix (Validation)')

# Rotate the tick labels for better alignment
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)

# Display the plot
plt.show()
