# PCA-breast_cancer
### Appley PCA on breast-cancer dataset
---
## 1. The Dataset

* **Source**: It uses scikit-learn’s built-in Breast Cancer Wisconsin dataset (`load_breast_cancer()`).
* **Size & Features**:

  * **569 samples**, each with **30 real-valued features** (e.g. radius, texture, perimeter, area, smoothness, etc.) extracted from digitized images of breast mass biopsies.
  * **Target**: a binary label (0 = malignant, 1 = benign).

---

## 2. DataFrame Construction & Initial Exploration

1. **Extract data & target**

   ```python
   breast = load_breast_cancer()
   data = breast.data                # shape (569, 30)
   target = breast.target            # shape (569,)
   ```
2. **Build a single array** by concatenating features + target column:

   ```python
   data = np.concatenate([data, target.reshape(-1,1)], axis=1)
   columns = np.append(breast.feature_names, 'label')
   df = pd.DataFrame(data)
   df.columns = columns
   ```
3. **Map numeric labels to strings** for readability:

   ```python
   df['label'].replace(0, 'malignant', inplace=True)
   df['label'].replace(1, 'benign',    inplace=True)
   ```
4. **Quick checks**:

   * `df.head()` & `df.sample(10)` to peek at the data.
   * `df['label'].value_counts().plot(kind='bar')` to visualize class balance (malignant vs. benign).

---

## 3. Feature Scaling & Train/Test Split

* **Why scale?** PCA and many algorithms perform better when features have zero mean and unit variance.
* **Scaling**:

  ```python
  from sklearn.preprocessing import StandardScaler
  X = df.drop('label', axis=1).values
  y = df['label'].values
  X_scaled = StandardScaler().fit_transform(X)
  ```
* **Train/test split** (though not used further for modeling here):

  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X_scaled, y,
      test_size=0.2,
      random_state=42
  )
  ```

---

## 4. Principal Component Analysis (PCA)

* **Algorithm**: PCA from scikit-learn (`PCA(n_components=2)`)
* **Purpose**:

  * Reduce 30-dimensional data down to 2 dimensions for **visualization**.
  * See how well the two classes separate in the space of the first two principal components.
* **Steps**:

  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  breast_pca = pca.fit_transform(X_scaled)       # shape (569, 2)
  breast_pca_df = pd.DataFrame(
      breast_pca,
      columns=['PC1', 'PC2']
  )
  breast_pca_df['label'] = y
  ```
* **Inspection**:

  ```python
  pca.n_components                  # → 2
  pca.explained_variance_ratio_     # % variance captured by PC1 & PC2
  pca.explained_variance_ratio_.sum()  # total variance explained (e.g. ~70-80%)
  ```

---

## 5. Visualization of PCA Results

* Scatter-plot the two principal components, coloring points by class (“malignant” vs. “benign”):

  ```python
  plt.figure(figsize=(10,6))
  for color, label in zip(['blue','red'], ['Malignant','Benign']):
      subset = breast_pca_df[breast_pca_df['label']==label.lower()]
      plt.scatter(subset['PC1'], subset['PC2'],
                  color=color, label=label, s=50, alpha=0.6)
  plt.title('PCA of Breast Cancer Dataset')
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.legend(loc='upper right')
  plt.show()
  ```
* **Main insight**: You can often see distinct clusters or at least some separation between malignant and benign tumors in these two dimensions.

---

## 6. Main Idea Behind the Notebook

1. **Data exploration**: Load and inspect the breast cancer dataset in a pandas DataFrame.
2. **Preprocessing**: Clean up labels, scale features to zero mean/unit variance.
3. **Dimensionality reduction**: Apply PCA to boil 30 features down to 2 for visualization.
4. **Visualization**: Plot the first two principal components to see how well the classes separate.

No supervised model is actually trained here—the focus is purely on **exploratory data analysis** and **unsupervised dimensionality reduction** to get intuition about the structure of the data.
