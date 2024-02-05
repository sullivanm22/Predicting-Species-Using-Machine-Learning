# Dry Beans Species Dataset Analysis

This project delves into the world of data-driven predictions using the Dry Beans Dataset from the UC Irvine Machine Learning Repository. The dataset comprises seven registered dry bean species, featuring 16 distinct attributes across over 13,000 instances.

## Dataset Exploration

Initially inspired by my work on the Iris dataset, the goal expanded to predict species across seven different beans, presenting a more complex challenge than the previous three-species task.

## Methodology

1. **Principal Component Analysis (PCA):**
   - Reduced dataset dimensionality.
   - Selected significant components based on their contribution to variance.
   - Focused analysis on PC1 and PC2, encompassing 82% of the total variance.

2. **Feature Selection:**
   - Identified crucial features: AspectRation, EquivDiameter, Compactness, and ShapeFactor3.
   - Adjusted attributes for optimal plot visibility and outcome reliability.

3. **Training and Testing Sets:**
   - Split data into an 80-20 ratio for training and testing, respectively.

4. **Machine Learning Techniques:**
   - Applied various techniques for species prediction.

## Results Overview

### Overall and Species-Specific Accuracy by Technique

| Technique                            | Overall Accuracy  | Barbunya Accuracy | Bombay Accuracy  | Cali Accuracy | Dermason Accuracy | Horoz Accuracy | Seker Accuracy | Sira Accuracy |
|--------------------------------------|-------------------|-------------------|------------------|---------------|-------------------|----------------|----------------|---------------|
| Naive Bayes                          | 85.9%             | 85.2%             | 99.1%            | 70.6%         | 89.2%             | 93.6%          | 91.7%          | 79.9%         |
| Naive Bayes with Normal Distribution | 87.9%             | 82.8%             | 99.1%            | 75.7%         | 92.1%             | 94.4%          | 94.7%          | 80.5%         |
| Support Vector Machines              | 88.49%            | 68.8%             | 100%             | 88.12%        | 90.3%             | 93.6%          | 93.4%          | 86.1%         |
| K-Nearest Neighbors                  | 81%               | 77.2%             | 100%             | 60.6%         | 85.7%             | 91.8%          | 94.9%          | 70.2%         |
| Decision Tree                        | 86.8%             | 66%               | 98.5%            | 84.9%         | 89.5%             | 87.9%          | 92.6%          | 87%           |
| Logistic Regression                  | 23.3%             | 0%                | 78.5%            | 48.1%         | 0.2%              | 71.8%          | 29%            | 0%            |

## Machine Learning Techniques

### 1. Naive Bayes

- **Methodology:**
  - Utilized the Naive Bayes classification method based on Bayes' theorem.
  - Efficiently handled multiple attributes within the algorithm.
  - Executed using the `naiveBayes` function in R.
- **Outcome:**
  - Achieved an overall accuracy of 85.9%.
  - Provided a solid baseline for classification.

### 2. Naive Bayes (Normal)

- **Methodology:**
  - Repeated Naive Bayes classification with a normal distribution.
- **Outcome:**
  - Enhanced overall accuracy to 87.9%.
  - Improved accuracy compared to standard Naive Bayes.

### 3. Support Vector Machines (SVM)

- **Methodology:**
  - Implemented SVM for optimal hyperplane identification.
  - Utilized the `svm` function in R.
- **Outcome:**
  - Attained the highest overall accuracy at 88.49%.
  - Successfully handled non-linear data with potential class overlaps.
  
### 4. K-Nearest Neighbors

- **Methodology:**
  - Applied K-Nearest Neighbors classification.
  - Explored different values for K to optimize accuracy.
- **Outcome:**
  - Achieved an accuracy of 81%.
  - Predicted based on the similarity between a new data point and its neighbors.

### 5. Decision Tree

- **Methodology:**
  - Utilized Decision Tree algorithm for classification.
  - Executed using the `rpart` function in R.
- **Outcome:**
  - Delivered an accuracy of 86.8%.
  - Created a tree-like structure where each node represents a decision based on a feature.

### 6. Logistic Regression

- **Methodology:**
  - Applied Logistic Regression with an overall accuracy of 23.3%.
  - Faced challenges due to linear regression's inherent nature.
- **Outcome:**
  - Resulted in lower accuracy for this classification problem.
  - Generated predictions not perfectly aligned with class numbers.

## Plots

### Plot 1: 2D Gaussian Distribution EquivDiameter vs AspectRation

![plot1](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/5ba69a5f-38bd-4df8-bc28-4d7c17cf8002)

### Plot 2: 2D Gaussian Distribution EquivDiameter vs Compactness

![plot2](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/0eecdb61-5a38-4c78-9ff4-de05f932aa55)

### Plot 3: SVM Classification Plot

![plot3](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/dedc24a6-bb4c-4c2a-bcc7-3a4b10799dbe)

### Plot 4: SVM Classification Plot (Subset)

![plot4](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/b924c98a-cf69-40b4-9108-9350da2de777)

### Plot 5: K Means Number of Clusters

![plot5](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/791af578-f40d-4992-8036-bd1f5209705a)

### Plot 6: K Means Clustering On Dry Beans Dataset

![plot6](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/1963d6ad-2617-4712-bed0-4eda0ee4d4cf)

### Plot 7: Decision Tree

![plot7](https://github.com/sullivanm22/Predicting-Species-Using-Machine-Learning/assets/59747399/61174119-d2e1-420e-b80a-6aaf99b74f88)

## Technique Insights

### Support Vector Machines (SVM)
   - Highest overall accuracy at 88.49%.
   - Effective in predicting non-linear data with potential overlaps between classes.

### Naive Bayes (Normal)
   - Second-highest accuracy at 87.9%.
   - Improved accuracy compared to standard Naive Bayes.

### Insights and Considerations
   - SVM and Naive Bayes (Normal) excelled in accuracy.
   - Species like Barbunya, Cali, and Sira proved challenging to predict.
   - Further feature refinement could enhance model accuracy.

## Conclusion

In conclusion, the project offers valuable insights into effective techniques for predicting dry bean species. SVM emerged as the most successful method, showcasing the potential of these models for similar challenges in the future. Despite the challenges posed by certain species, the overall performance of the models provides a solid foundation for future endeavors in the realm of machine learning and data analysis.

