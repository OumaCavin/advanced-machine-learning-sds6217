# Classification Metrics Quiz Solutions

This document provides detailed explanations for the classification metrics questions from the Advanced Machine Learning quiz.

---

## Question 1: Phishing/Malware Classification Error

**Question**: A model classifies a legitimate website as malware. What is this called?

**Answer: A false positive**

### Explanation

In binary classification, we have four possible outcomes:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

**Definitions:**
- **True Positive (TP)**: Correctly identifying malware as malware (1 → 1)
- **True Negative (TN)**: Correctly identifying legitimate sites as legitimate (0 → 0)
- **False Positive (FP)**: Incorrectly labeling legitimate sites as malware (0 → 1)
- **False Negative (FN)**: Incorrectly missing actual malware (1 → 0)

In this scenario:
- Legitimate website = Actual Negative (0)
- Classified as malware = Predicted Positive (1)
- Therefore: **False Positive (FP)**

This is also known as a **Type I Error**.

---

## Question 2: Effect of Increasing Classification Threshold on FP and TP

**Question**: What happens to false positives and true positives when the classification threshold increases?

**Answer: True positives increase. False positives decrease.**

### Explanation

The classification threshold determines the decision boundary:

- **Lower threshold**: More examples classified as positive → Higher TP, Higher FP
- **Higher threshold**: Fewer examples classified as positive → Lower TP, Lower FP

**Intuition:**

When threshold increases:
1. Model becomes more conservative about predicting positive
2. Only predictions with very high confidence become positive
3. **True Positives decrease** (some actual positives are missed)
4. **False Positives decrease** (fewer incorrect positive predictions)

This relationship is captured in the **Precision-Recall Trade-off**:
- As threshold increases: Precision increases, Recall decreases

---

## Question 3: Effect of Increasing Classification Threshold on FN and TN

**Question**: What happens to false negatives and true negatives when the classification threshold increases?

**Answer: True negatives increase. False negatives decrease.**

### Explanation

Continuing from the previous question:

When threshold increases:
1. Model predicts fewer positives overall
2. More examples are classified as negative
3. **True Negatives increase** (more correct negative predictions)
4. **False Negatives decrease** (fewer actual positives missed... wait, this is wrong)

Let me reconsider:

Actually, when threshold increases:
- **False Negatives increase** (more actual positives are incorrectly classified as negative)
- **True Negatives increase** (more actual negatives are correctly classified as negative)

The correct answer should be: **True negatives increase. False negatives increase.** (both increase because we're predicting fewer positives)

Wait, let me recalculate:
- Threshold increases → Model becomes stricter about positive predictions
- Fewer items classified as positive
- More items classified as negative
- **True Negatives increase** (correctly identified negatives)
- **False Negatives increase** (missed positives that should have been positive)

**Correct Answer: True negatives increase. False negatives increase.**

---

## Question 4: Calculate Recall

**Question**: Model outputs 5 TP, 6 TN, 3 FP, and 2 FN. Calculate the recall.

**Answer: 0.714**

### Explanation

**Recall (Sensitivity/True Positive Rate)** measures how well the model identifies positive cases:

**Recall = TP / (TP + FN)**

Given:
- TP = 5
- FN = 2
- TN = 6 (not needed for recall)
- FP = 3 (not needed for recall)

**Calculation:**
```
Recall = 5 / (5 + 2) = 5/7 = 0.714
```

**Interpretation**: The model correctly identifies 71.4% of all actual positive cases.

---

## Question 5: Calculate Precision

**Question**: Model outputs 3 TP, 4 TN, 2 FP, and 1 FN. Calculate the precision.

**Answer: 0.6**

### Explanation

**Precision** measures how accurate the positive predictions are:

**Precision = TP / (TP + FP)**

Given:
- TP = 3
- FP = 2
- TN = 4 (not needed for precision)
- FN = 1 (not needed for precision)

**Calculation:**
```
Precision = 3 / (3 + 2) = 3/5 = 0.6
```

**Interpretation**: When the model predicts positive, it is correct 60% of the time.

---

## Question 6: Insect Trap Classification - Which Metric to Optimize?

**Question**: Model for detecting invasive insect species. False alarms are easy to handle. Which metric should be optimized?

**Answer: Recall**

### Explanation

**Scenario Analysis:**
- **Task**: Detect dangerous invasive species in insect trap photos
- **Consequence of missing detection**: Critical - can lead to infestation
- **Consequence of false alarm**: Minor - easy to handle, entomologist can verify

**Metric Selection Rationale:**

1. **Recall (Sensitivity)**: Maximize detection of actual invasive species
   - High recall means minimizing false negatives
   - More important when missing positives is costly

2. **Precision**: Important when false positives are costly
   - Not the priority here since false alarms are "easy to handle"

3. **F1 Score**: Harmonic mean of precision and recall
   - Balances both, but not optimal when one is clearly more important

**Conclusion**: **Recall** should be optimized because:
- False negatives (missed invasive species) have severe consequences
- False positives (false alarms) have minimal consequences
- We want to catch as many invasive species as possible, even at the cost of some false alarms

This is a classic example of the precision-recall trade-off in medical diagnosis, fraud detection, and other high-stakes classification tasks.
