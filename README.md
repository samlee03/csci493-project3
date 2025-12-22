# Project 3

## Algorithm Justification

The algorithms that I chose were Decision Tree and Random Forest. These were initially chosen due to their attribute of being a classification model and thus having the strengths of treating a classification problem. In this case, we want to determine whether the samples are malignant or benign. Additionally, I would like to see the comparison of how well Random Forest does against Decision trees in terms of accuracy, precision, recall, and F1 score. I will later talk about which works better for this dataset.

## Training Procedure

The training procedure I used for both the Decision Tree and Random Forest is using a 5-fold cross validation. The dataset is divided into five folds (20% of the data is for testing, 80% is for training). I repeat this process 5 times for each fold so that the folds are used as the testing set once.  
I've computed the accuracy for each fold as well as calculated the predictions to get the overall precision, recall, and F1 score.

## Testing Results

### Random Forest

**Accuracies (on each fold):**
- 0.9561
- 0.9474
- 0.9737
- 0.9561
- 0.9469

**Overall Metrics:**
- Average Accuracy: **0.9560**
- Precision: **0.9652**
- Recall: **0.9151**
- F1 Score: **0.9395**

### Decision Tree

**Accuracies (on each fold):**
- 0.8947
- 0.9737
- 0.9737
- 0.9561
- 0.9381

**Overall Metrics:**
- Average Accuracy: **0.9473**
- Precision: **0.9550**
- Recall: **0.9009**
- F1 Score: **0.9272**

After attaining the metrics for both Random Forest and Decision Tree, the results are what I expected. RF did better than Decision Tree since it essentially combines multiple decision trees. The Decision Tree did not perform horrible. The Random Forest had a F1 Score of 0.9395, while the Decision Tree had an F1 Score of 0.9272 after calculating the metrics with 5-fold cross validation.  

## Discussion and Future Improvement


In terms of future improvement, I would like to experiment with adjusting the parameters such as the tree depth. Perhaps a different machine learning algorithm would be better overall. Especially dealing with cancerous tissues, having a high recall score and F1 score would be ideal.
