# Notebook Implementation Key Settings

## Dataset URLs
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Appliances.jsonl
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Grocery_and_Gourmet_Food.jsonl

## Sampling/Balancing Signals
- build_balanced_dataset(dataset_stream, total_samples=20000):
- build_balanced_dataset(clothing_ds, total_samples=20000)
- build_balanced_dataset(groceries_ds, total_samples=20000)
- build_balanced_dataset(appliances_ds, total_samples=20000)
- build_balanced_dataset(clothing_ds, ...)`**: Executes the balancing logic we wrote in Section 3. It scans the stream until it finds 6,666 Positive, 6,666 Neutral, and 6,666 Negative reviews.
- total_samples=20000
- total_samples=20000
- total_samples=20000
- total_samples=20000
- target_per_class = total_samples // 3
- target_per_class = total_samples // 3`**: Calculates how many reviews we need per category (e.g., 20000 / 3 = 6666).
- stratify=clothing_df["label"]
- stratify=clothing_df["label"]`**: CRITICAL for fair evaluation!
- stratify=groceries_df["label"]
- stratify=appliances_df["label"]

## Preprocessing Operations
- present: lower(
- present: re.sub(
- present: langdetect
- present: emoji
- missing: BeautifulSoup
- present: spacy.load
- present: stop_words
- present: lemma_
- present: clean_texts_batch

## TF-IDF Parameters
- TfidfVectorizer(
    ngram_range=(1, 2)
- TfidfVectorizer(...)
- TfidfVectorizer(ngram_range=(1, 2)
- TfidfVectorizer(ngram_range=(1, 2)

## Model Classes Mentioned
- LogisticRegression: 26
- LinearSVC: 13
- SGDClassifier: 13
- MultinomialNB: 6
- DecisionTreeClassifier: 5
- RandomForestClassifier: 5
- ExtraTreesClassifier: 5
- BaggingClassifier: 5
- LGBMClassifier: 4
- XGBClassifier: 7
- CatBoostClassifier: 0
- Albert: 0
- ALBERT: 0
- StackingClassifier: 9

## Stacking / CV Settings
- StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
- StackingClassifier(...)
- StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000)
- StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000)
- cv=3
- cv=5
- cv=5
- cv=5
- cv=5
- cv=3
- cv=5
- cv=3
- cross_val_score(model, X_train_vec, y_train, cv=3, scoring='f1_weighted')

## Statistical Significance Testing Presence
- mcnemar: missing
- statsmodels: missing
- scipy.stats: missing
- ttest: missing
- wilcoxon: missing
- p_value: present
- p-value: missing

## SHAP Implementation Details
- 1. SHAP explainability analysis
- ## 11. Explainable AI (SHAP Analysis)
- **SHAP (SHapley Additive exPlanations)** answers these questions by assigning an importance score to every single word in the vocabulary. It tells us exactly how much each word pushed the prediction toward or away from a particular class.
- ### Why SHAP Works Exceptionally Well Here
- Our pipeline (TF-IDF + Logistic Regression) is a **linear model**. SHAP has a specialized `LinearExplainer` that is:
- ### 11.1 Install SHAP Library
- 1. **`!pip install shap`**: Downloads and installs the SHAP library from the Python Package Index. SHAP is not included by default, so we must install it.
- ### 11.2 Import SHAP and Select Model to Explain
- 2. **`import numpy as np`**: Imports Numpy, which SHAP uses internally for array operations.
- 4. **`training_data = X_train_vec_c`**: SHAP needs access to the training data to understand what "normal" data looks like. It uses this as a baseline for comparisons.
- ### 11.3 Create the SHAP Explainer Object
- explainer = shap.LinearExplainer(
- 1. **`shap.LinearExplainer(...)`**: This creates an explainer object specifically optimized for linear models like Logistic Regression and SVM. It knows the math behind these models and can compute exact importance scores.
- 3. **`training_data`**: We pass the TF-IDF vectors from training. SHAP uses this as a "background" dataset to understand what typical word patterns look like.
- 4. **`feature_perturbation="interventional"`**: This is a technical setting that tells SHAP how to handle correlated features. In text, words often appear together (e.g., "very good"). The "interventional" mode handles this correctly by treating each word independently.
- ### 11.4 Compute SHAP Values for Test Samples
- Since we have 3 classes (Negative, Neutral, Positive), SHAP will return 3 arrays:
- 1. **`explainer.shap_values(test_data)`**: This is the core calculation. For every review in our test set, SHAP computes the importance of every word. The result is a list of 3 matrices (one per class).
- - **X-axis**: The SHAP value (impact on prediction)
- shap.summary_plot(
- 2. **`shap_values[2]`**: We select the SHAP values for the **Positive** class. You could change this to `[0]` for Negative or `[1]` for Neutral.
- shap.force_plot(
- shap_values[:, :, 2][idx],       # SHAP values for this instance & class
- 1. **`shap.initjs()`**: Initializes the Javascript library required to render interactive SHAP plots in the notebook.
- 4. **`shap_values[2][idx]`**: The specific SHAP values for this particular review. It tells us how much each word pushed the prediction above or below the baseline.
- ### 11.11 Overall SHAP Conclusion
- After running SHAP analysis on all three domains, you can confidently conclude:
- 3. **Neutral reviews have weaker signals**: SHAP values are smaller in magnitude for neutral predictions across all domains.
- ### 11.8 SHAP Analysis: Groceries Domain
- We now apply the same SHAP analysis to the Groceries domain to understand which words drive sentiment in food reviews.
- explainer_g = shap.LinearExplainer(
- # Compute SHAP values
- print("SHAP values computed for Groceries!")
- 3. **`shap_values_g`**: SHAP importance scores specific to grocery reviews.
- # Groceries SHAP Explainer
- explainer_g = shap.LinearExplainer(
- shap.summary_plot(
- shap.force_plot(
- ### 11.9 SHAP Analysis: Appliances Domain
- explainer_a = shap.LinearExplainer(
- # Compute SHAP values
- print("SHAP values computed for Appliances!")
- 3. **`shap_values_a`**: SHAP scores for appliance review words.
- shap.summary_plot(
- shap.force_plot(
- # Appliances SHAP Explainer
- explainer_a = shap.LinearExplainer(
- After running SHAP on all three domains, you can compare:
- ### 11.9 XGBoost SHAP Analysis
- To compare how different model types 'think', we now analyze XGBoost - a gradient boosting model - using SHAP.
- # Train XGBoost models for all domains (for SHAP analysis)
- print("Training XGBoost models for SHAP analysis...")
- ### Code Explanation: Training XGBoost for SHAP Analysis
- The benchmark trains many models but doesn't save them all. For SHAP, we need specific model instances:
- #### 11.9.1 XGBoost SHAP: Clothing Domain
- # XGBoost SHAP - Clothing
- print("Computing XGBoost SHAP values for Clothing...")
- # Sample for efficiency (SHAP on full dataset is slow)
- explainer_xgb_c = shap.TreeExplainer(xgb_c)
- print("Generating XGBoost SHAP summary plot...")
- shap.summary_plot(
- plt.title("XGBoost SHAP: Clothing (Positive Class)")
- ### Code Explanation: XGBoost SHAP with TreeExplainer
- | Model Type | SHAP Explainer | Speed |
- TreeExplainer uses the tree structure to compute exact SHAP values efficiently.
- - SHAP computation is O(n × features × trees)
- #### Interpreting XGBoost SHAP Values:
- - **Non-linear effects**: "not" + "good" together may have different SHAP than separately
- #### 11.9.2 XGBoost SHAP: Groceries Domain
- # XGBoost SHAP - Groceries
- print("Computing XGBoost SHAP values for Groceries...")
- explainer_xgb_g = shap.TreeExplainer(xgb_g)
- print("Generating XGBoost SHAP summary plot...")
- shap.summary_plot(
- plt.title("XGBoost SHAP: Groceries (Positive Class)")
- #### 11.9.3 XGBoost SHAP: Appliances Domain
- # XGBoost SHAP - Appliances
- print("Computing XGBoost SHAP values for Appliances...")
- explainer_xgb_a = shap.TreeExplainer(xgb_a)
- print("Generating XGBoost SHAP summary plot...")
- shap.summary_plot(
- plt.title("XGBoost SHAP: Appliances (Positive Class)")
- ### 11.10 LR vs XGBoost SHAP Comparison
- 2. **StackSHAP**: Uses SHAP-selected subset of best-performing models
- ### 10.2 StackSHAP Ensemble (SHAP-Guided Model Selection)
- The paper's key contribution: Use SHAP to analyze model contributions, then build a reduced ensemble using only the most important base models.
- # Model-level SHAP: Analyze base model contributions
- # (True SHAP on stacking requires expensive computation)
- ### Code Explanation: Model-Level SHAP (Model Importance Analysis)
- #### Why This Approach (Not Full SHAP)?
- - **True SHAP on stacking** would require computing SHAP for the meta-learner with 30+ input features
- print(f"StackSHAP using top {top_n} models: {top_models}")
- ### Code Explanation: StackSHAP - The Paper's Core Contribution
- #### What is StackSHAP?
- A **reduced stacking ensemble** that uses only the most important base models, selected using SHAP/importance analysis.
- Step 5: Build StackSHAP with only selected models
- StackSHAP often achieves **comparable or better performance** than StackFull while using **half the models**. This demonstrates that:
- - SHAP-guided selection is an effective pruning strategy
- # Train and evaluate StackSHAP on Clothing domain
- print("Training StackSHAP on Clothing domain...")
- print(f"  StackSHAP Clothing - Weighted F1: {metrics_shap_c['Weighted F1']:.4f}, MCC: {metrics_shap_c['MCC']:.4f}")
- # Compare StackFull vs StackSHAP
- 'StackSHAP (5 models)': [f"{v:.4f}" for v in metrics_shap_c.values()]
- print("\n✓ StackSHAP achieves comparable performance with fewer models!")
- | If StackSHAP... | It Means... |
- 1. **Efficiency claim**: StackSHAP is faster (fewer models to run)
- The comparison table is the **key evidence** supporting the paper's claim that SHAP-guided model selection produces effective, efficient ensembles.
- 4. **SHAP explanations**: What words drive each domain?
- ### 12.8 SHAP Confirms Domain Differences
- From our SHAP analysis in Step 11, we observed that each domain has characteristic words:
- > **"The proposed framework performs best on Clothing reviews (69.1% accuracy) due to stronger sentiment cues and clear polarity. Groceries achieve medium accuracy (67.8%) as taste descriptions are inherently subjective. Appliances exhibit the lowest accuracy (65.3%) due to technical language and mixed sentiment expressions within single reviews. SHAP analysis confirms that domain-specific keywords drive predictions, validating the model's semantic understanding across product categories."**
