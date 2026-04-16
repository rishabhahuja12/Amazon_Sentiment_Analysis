## Cell 3

- Output 1:
  [notice] A new release of pip is available: 23.0.1 -> 26.0.1
  [notice] To update, run: python.exe -m pip install --upgrade pip

## Cell 8

- Output 1:
  [nltk_data] Downloading package stopwords to
  [nltk_data]     C:\Users\asus\AppData\Roaming\nltk_data...
  [nltk_data]   Package stopwords is already up-to-date!

## Cell 29

- Output 1:
  Dataset Keys:
  dict_keys(['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase'])

## Cell 32

- Output 1:
  Building Balanced Dataframes (This may take a moment)...
  Clothing DF shape: (19998, 2)
  Groceries DF shape: (19998, 2)
  Appliances DF shape: (19998, 2)

## Cell 34

- Output 1:
  Original: Don't be fooled by the description. I was freezing my butt off at 40 degrees! These are very thin, so to call them "fleece-lined" feels deceptive. Not a fan! I'd rather pay for a pair of Kerrits, which are truly warm! I ordered a medium but could have sized down to small.
  Cleaned: fool description freeze butt degree thin call fleeceline feel deceptive fan rather pay pair kerrit truly warm order medium could size small

## Cell 36

- Output 1:
  Cleaning Clothing text...

## Cell 38

- Output 1:
  Cleaning Groceries text...

## Cell 40

- Output 1:
  Cleaning Appliances text...

## Cell 46

- Output 1:
  Results for Clothing:
  Training Logistic Regression...
    -> Accuracy: 0.7053
  Training Linear SVM...
    -> Accuracy: 0.6720
  Training SGD Classifier...
    -> Accuracy: 0.7065
  Training Multinomial NB...
    -> Accuracy: 0.6945
  Training Decision Tree...
    -> Accuracy: 0.5513
  Training Random Forest...
    -> Accuracy: 0.6753
  Training Extra Trees...
    -> Accuracy: 0.6887
  Training Bagging...
    -> Accuracy: 0.6135
  Training LightGBM...
    -> Accuracy: 0.6957
  Training XGBoost...
    -> Accuracy: 0.6805

## Cell 49

- Output 1:
  Results for Groceries:
  Training Logistic Regression...
    -> Accuracy: 0.6953
  Training Linear SVM...
    -> Accuracy: 0.6747
  Training SGD Classifier...
    -> Accuracy: 0.6960
  Training Multinomial NB...
    -> Accuracy: 0.6873
  Training Decision Tree...
    -> Accuracy: 0.5423
  Training Random Forest...
    -> Accuracy: 0.6573
  Training Extra Trees...
    -> Accuracy: 0.6813
  Training Bagging...
    -> Accuracy: 0.5897
  Training LightGBM...
    -> Accuracy: 0.6720
  Training XGBoost...
    -> Accuracy: 0.6617

## Cell 52

- Output 1:
  Results for Appliances:
  Training Logistic Regression...
    -> Accuracy: 0.6670
  Training Linear SVM...
    -> Accuracy: 0.6405
  Training SGD Classifier...
    -> Accuracy: 0.6640
  Training Multinomial NB...
    -> Accuracy: 0.6623
  Training Decision Tree...
    -> Accuracy: 0.5347
  Training Random Forest...
    -> Accuracy: 0.6395
  Training Extra Trees...
    -> Accuracy: 0.6435
  Training Bagging...
    -> Accuracy: 0.5940
  Training LightGBM...
    -> Accuracy: 0.6567
  Training XGBoost...
    -> Accuracy: 0.6408

## Cell 56

- Output 1:
  Classification Report (Clothing):
                precision    recall  f1-score   support
      Negative       0.68      0.69      0.69      1333
       Neutral       0.61      0.58      0.60      1333
      Positive       0.81      0.84      0.83      1334
      accuracy                           0.71      4000
     macro avg       0.70      0.71      0.70      4000
  weighted avg       0.70      0.71      0.70      4000
- Output 2:
  <Figure size 600x500 with 2 Axes>

## Cell 58

- Output 1:
  Sample Clothing Errors:
- Output 2:
                                                      text  label
  15138  Reminds me of the color of the stone from the ...      2
  5726   Even though it fits, it doesn't fit that comfo...      0
  11277  my earrings were all one color-not what I want...      1
  9289   These are really cute, very light, but the are...      1
  825    Disappointed !<br />I expected Moissanite to l...      0

## Cell 61

- Output 1:
  Classification Report (Groceries):
                precision    recall  f1-score   support
      Negative       0.69      0.65      0.67      1333
       Neutral       0.59      0.62      0.60      1333
      Positive       0.81      0.81      0.81      1334
      accuracy                           0.70      4000
     macro avg       0.70      0.70      0.70      4000
  weighted avg       0.70      0.70      0.70      4000
- Output 2:
  <Figure size 600x500 with 2 Axes>

## Cell 63

- Output 1:
  Sample Groceries Errors:
- Output 2:
                                                      text  label
  3736   I never taste a poptart that was so dry and ta...      0
  19252  I like that this Za’atar seasoning is Organic ...      2
  9723   Not s bad tasting tea but it is very powdery a...      1
  1254   Not a very good tasting caramel. Has a bit of ...      0
  4178   Good points:<br />Convenient, easy to open, go...      0

## Cell 66

- Output 1:
  Classification Report (Appliances):
                precision    recall  f1-score   support
      Negative       0.67      0.68      0.67      1333
       Neutral       0.57      0.56      0.56      1333
      Positive       0.77      0.77      0.77      1334
      accuracy                           0.67      4000
     macro avg       0.67      0.67      0.67      4000
  weighted avg       0.67      0.67      0.67      4000
- Output 2:
  <Figure size 600x500 with 2 Axes>

## Cell 68

- Output 1:
  Sample Appliances Errors:
- Output 2:
                                                      text  label
  16978           Beautiful way to protect glass stove top      2
  11277  Disappointed that the inner part of lid gets l...      1
  19958      Looks like the original.  Couldn't be better.      2
  14240  We have a black refrigerator and no matter how...      2
  9671   NIce little unit for the price. Easy fill easy...      1

## Cell 71

- Output 1:
  Review: The material quality is good but the size fits terrible.
  Prediction: Negative

## Cell 75

- Output 1:
  Tuning Logistic Regression...
  C=0.1 -> accuracy=0.6947
  C=0.5 -> accuracy=0.7083
  C=1 -> accuracy=0.7043
  C=2 -> accuracy=0.6977
  C=5 -> accuracy=0.6883

## Cell 79

- Output 1:
  Tuning Linear SVM...
  C=0.01 -> accuracy=0.6887
  C=0.1 -> accuracy=0.7075
  C=1 -> accuracy=0.6720
  C=2 -> accuracy=0.6550
  C=5 -> accuracy=0.6360

## Cell 82

- Output 1:
  Tuning SGD Classifier...
  alpha=0.0001 -> accuracy=0.7063
  alpha=0.001 -> accuracy=0.6867
  alpha=0.01 -> accuracy=0.6630

## Cell 85

- Output 1:
  Tuning Logistic Regression (Groceries)...
  C=0.1 -> accuracy=0.6793
  C=0.5 -> accuracy=0.6950
  C=1 -> accuracy=0.6945
  C=2 -> accuracy=0.6917
  C=5 -> accuracy=0.6865

## Cell 87

- Output 1:
  Tuning Linear SVM (Groceries)...
  C=0.01 -> accuracy=0.6670
  C=0.1 -> accuracy=0.6967
  C=1 -> accuracy=0.6747
  C=2 -> accuracy=0.6617
  C=5 -> accuracy=0.6478

## Cell 89

- Output 1:
  Tuning SGD Classifier (Groceries)...
  alpha=0.0001 -> accuracy=0.6973
  alpha=0.001 -> accuracy=0.6585
  alpha=0.01 -> accuracy=0.6405

## Cell 92

- Output 1:
  Tuning Logistic Regression (Appliances)...
  C=0.1 -> accuracy=0.6587
  C=0.5 -> accuracy=0.6697
  C=1 -> accuracy=0.6687
  C=2 -> accuracy=0.6627
  C=5 -> accuracy=0.6500

## Cell 94

- Output 1:
  Tuning Linear SVM (Appliances)...
  C=0.01 -> accuracy=0.6532
  C=0.1 -> accuracy=0.6687
  C=1 -> accuracy=0.6405
  C=2 -> accuracy=0.6305
  C=5 -> accuracy=0.6160

## Cell 96

- Output 1:
  Tuning SGD Classifier (Appliances)...
  alpha=0.0001 -> accuracy=0.6655
  alpha=0.001 -> accuracy=0.6500
  alpha=0.01 -> accuracy=0.6220

## Cell 99

- Output 1:
  Finding best C per domain (this runs 3-fold CV for each C value)...
  Clothing  -> Best C = 0.5  (Weighted F1 = 0.6945)
  Groceries -> Best C = 2  (Weighted F1 = 0.6789)
  Appliances -> Best C = 0.5  (Weighted F1 = 0.6607)
  Retraining final LR models with best C values...
  Final LR Accuracy (Clothing):   0.7073
  Final LR Accuracy (Groceries):  0.6927
  Final LR Accuracy (Appliances): 0.6700
  lr_c, lr_g, lr_a are now updated with the optimal C for the rest of the notebook.

## Cell 106

- Output 1:
  Training Stacked Model (This may take time)...
  Stacking Accuracy (Clothing): 0.7157

## Cell 108

- Output 1:
  Training Stacked Model (This may take time)...
  Stacking Accuracy (Groceries): 0.7015

## Cell 110

- Output 1:
  Training Stacked Model (This may take time)...
  Stacking Accuracy (Appliances): 0.6737

## Cell 113

- Output 1:
  [notice] A new release of pip is available: 23.0.1 -> 26.0.1
  [notice] To update, run: python.exe -m pip install --upgrade pip

## Cell 125

- Output 1:
  <Figure size 800x950 with 2 Axes>

## Cell 128

- Output 1:
  <Figure size 2000x300 with 1 Axes>

## Cell 132

- Output 1:
  SHAP values computed for Groceries!

## Cell 135

- Output 1:
  <Figure size 800x950 with 2 Axes>

## Cell 136

- Output 1:
  <Figure size 2000x300 with 1 Axes>

## Cell 139

- Output 1:
  SHAP values computed for Appliances!

## Cell 141

- Output 1:
  <Figure size 800x950 with 2 Axes>

## Cell 142

- Output 1:
  <Figure size 2000x300 with 1 Axes>

## Cell 147

- Output 1:
  Training XGBoost models for SHAP analysis...
    -> Clothing XGBoost trained
    -> Groceries XGBoost trained
    -> Appliances XGBoost trained
  All XGBoost models trained successfully!

## Cell 150

- Output 1:
  Computing XGBoost SHAP values for Clothing...
  Generating XGBoost SHAP summary plot...
- Output 2:
  <Figure size 800x950 with 2 Axes>

## Cell 153

- Output 1:
  Computing XGBoost SHAP values for Groceries...
  Generating XGBoost SHAP summary plot...
- Output 2:
  <Figure size 800x950 with 2 Axes>

## Cell 155

- Output 1:
  Computing XGBoost SHAP values for Appliances...
  Generating XGBoost SHAP summary plot...
- Output 2:
  <Figure size 800x950 with 2 Axes>

## Cell 159

- Output 1:
  StackFull ensemble defined with 10 base models

## Cell 161

- Output 1:
  Training StackFull on all domains...
  (This may take several minutes)
  Training StackFull - Clothing...
    StackFull Clothing - Weighted F1: 0.7186, MCC: 0.5785
  Training StackFull - Groceries...
    StackFull Groceries - Weighted F1: 0.7041, MCC: 0.5546
  Training StackFull - Appliances...
    StackFull Appliances - Weighted F1: 0.6761, MCC: 0.5126
  StackFull training complete!

## Cell 164

- Output 1:
  Analyzing base model importance (Clothing domain)...
  Model Importance Ranking:
    mnb: 1.0692
    sgd: 0.9231
    lr: 0.4353
    svm: 0.3479
    et: 0.2987
    lgbm: 0.2939
    xgb: 0.2819
    rf: 0.0821
    bag: 0.0624
    dt: 0.0308

## Cell 166

- Output 1:
  <Figure size 1000x600 with 1 Axes>

## Cell 169

- Output 1:
  Training StackSHAP on Clothing domain...
  StackSHAP using top 5 models: ['mnb', 'sgd', 'lr', 'svm', 'et']
    StackSHAP Clothing - Weighted F1: 0.7250, MCC: 0.5887

## Cell 170

- Output 1:
  ============================================================
  ENSEMBLE COMPARISON (Clothing Domain)
  ============================================================
         Metric StackFull (10 models) StackSHAP (5 models)
       Accuracy                0.7190               0.7258
    Weighted F1                0.7186               0.7250
       Macro F1                0.7185               0.7250
            MCC                0.5785               0.5887
  Cohen's Kappa                0.5785               0.5886
         G-Mean                0.7126               0.7193
   Balanced Acc                0.7190               0.7257
  ✓ StackSHAP achieves comparable performance with fewer models!

## Cell 174

- Output 1:
  === Cross-Domain Accuracy Comparison ===
              Logistic Regression  Stacking Ensemble
  Clothing                 0.7072             0.7158
  Groceries                0.6928             0.7015
  Appliances               0.6700             0.6738

## Cell 177

- Output 1:
  <Figure size 1000x600 with 1 Axes>

## Cell 183

- Output 1:
  <Figure size 1500x400 with 6 Axes>

## Cell 188

- Output 1:
  =================================================================
    CONSOLIDATED LEADERBOARD — Weighted F1-Score
  =================================================================
                         Clothing Groceries Appliances
  Model                                               
  Logistic Regression      0.7035    0.6961     0.6667
  Linear SVM               0.6691    0.6751     0.6391
  SGD Classifier           0.7039    0.6953      0.666
  Multinomial NB           0.6937    0.6897     0.6634
  Decision Tree            0.5518    0.5411     0.5355
  Random Forest            0.6716    0.6576     0.6383
  Extra Trees               0.686    0.6814     0.6427
  Bagging                  0.6135    0.5886      0.594
  LightGBM                 0.6938    0.6729     0.6575
  XGBoost                  0.6781    0.6614     0.6425
  StackFull (Clothing)     0.7186         -          -
  StackFull (Groceries)         -    0.7041          -
  StackFull (Appliances)        -         -     0.6761
  Best overall model per domain:
    Clothing: StackFull (Clothing) (0.7186)
    Groceries: StackFull (Groceries) (0.7041)
    Appliances: StackFull (Appliances) (0.6761)

## Cell 190

- Output 1:
  <Figure size 1800x500 with 3 Axes>

## Cell 192

- Output 1:
  <Figure size 1800x500 with 3 Axes>
- Output 2:
  === AUC Summary Table ===
  Class       Clothing      Groceries     Appliances    
  ------------------------------------------------------
  Negative    0.8678        0.8567        0.8550        
  Neutral     0.7871        0.7767        0.7635        
  Positive    0.9424        0.9375        0.9079        

## Cell 195

- Output 1:
  Logistic Regression models saved!

## Cell 198

- Output 1:
  TF-IDF vectorizers saved!

## Cell 201

- Output 1:
  Stacking ensemble models saved!
  All models saved to '../saved_models/' directory.

## Cell 204

- Output 1:
  Review: This dress is absolutely beautiful and fits perfectly!
  Predicted Sentiment: Positive

## Cell 207

- Output 1:
  === Saved Model Files ===
    lr_appliances.pkl: 0.36 MB
    lr_clothing.pkl: 0.40 MB
    lr_groceries.pkl: 0.33 MB
    stack_appliances.pkl: 1.07 MB
    stack_clothing.pkl: 1.20 MB
    stack_groceries.pkl: 0.98 MB
    tfidf_appliances.pkl: 0.58 MB
    tfidf_clothing.pkl: 0.66 MB
    tfidf_groceries.pkl: 0.54 MB

## Cell 210

- Output 1:
  ✓ Models saved to models/ directory
    - models/lr_clothing.joblib
    - models/tfidf_clothing.joblib
  You can now run the Flask app with:
    python app.py


TOTAL_CELLS_WITH_OUTPUTS=62
TOTAL_TEXT_OUTPUT_LINES=312
