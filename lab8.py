import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\lab8\ITA105_Lab_8.csv")

print(df[['Description','NoiseFeature']].head(10))
print(df.dtypes)
print(df.head())

class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.Q1 = np.percentile(X, 25, axis=0)
        self.Q3 = np.percentile(X, 75, axis=0)
        self.IQR = self.Q3 - self.Q1
        return self

    def transform(self, X):
        lower = self.Q1 - 1.5 * self.IQR
        upper = self.Q3 + 1.5 * self.IQR
        return np.clip(X, lower, upper)

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.to_datetime(X.squeeze(), errors='coerce')
        df = pd.DataFrame()
        df['month'] = X.dt.month
        df['quarter'] = X.dt.quarter
        df['year'] = X.dt.year
        return df

class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.squeeze().astype(str)
    

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.fillna('').astype(str).squeeze()

text_pipeline = Pipeline([
    ('clean', TextCleaner()),
    ('tfidf', TfidfVectorizer(stop_words='english'))
])
    
num_cols = ['LotArea', 'SalePrice', 'Rooms', 'NoiseFeature']
cat_cols = ['HasGarage', 'Neighborhood', 'Condition']
text_cols = ['Description']
date_cols = ['SaleDate']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier', OutlierClipper()),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_pipeline = ColumnTransformer([
    ('desc', Pipeline([
        ('clean', TextCleaner()),
        ('tfidf', TfidfVectorizer(stop_words='english'))
    ]), 'Description')
])

date_pipeline = Pipeline([
    ('extract', DateFeatureExtractor()),
    ('imputer', SimpleImputer(strategy='median'))
])


full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
    ('text', text_pipeline, ['Description']),
    ('date', date_pipeline, date_cols)
])

sample_df = df.head(10)
X_processed = full_pipeline.fit_transform(sample_df)

print(X_processed.shape)

def get_feature_names(column_transformer):
    feature_names = []

    for name, transformer, cols in column_transformer.transformers_:

        if name == 'num':
            feature_names.extend(cols)

        elif name == 'cat':
            ohe = transformer.named_steps['onehot']
            feature_names.extend(ohe.get_feature_names_out(cols))

        elif name == 'text':
            for sub_name, sub_trans, col in transformer.transformers_:
                
                tfidf = sub_trans.named_steps['tfidf']
                
                names = tfidf.get_feature_names_out()
                feature_names.extend([f"{col[0]}_{n}" for n in names])

        elif name == 'date':
            feature_names.extend(['month', 'quarter', 'year'])

    return feature_names


feature_names = get_feature_names(full_pipeline)

print("Số feature:", len(feature_names))
print("Ví dụ feature:", feature_names[:20])

df_full = df.copy()

df_missing = df.copy()
df_missing['LotArea'] = df_missing['LotArea'].astype(float)
df_missing['LotArea'] = np.nan
df_missing.loc[:, 'Description'] = np.nan

df_skewed = df.copy()
df_skewed['SalePrice'] = df_skewed['SalePrice'] ** 3

df_unseen = df.copy()
df_unseen.loc[0, 'Neighborhood'] = 'Z'

df_wrong_type = df.copy()

df_wrong_type['LotArea'] = df_wrong_type['LotArea'].astype(object)

df_wrong_type.loc[0, 'LotArea'] = 'abc'

def test_pipeline(data, name):
    print(f"\n===== TEST: {name} =====")
    
    try:
        X = full_pipeline.fit_transform(data)
        
        print("✔ Pipeline chạy OK")
        print("Shape:", X.shape)
        print("Type:", type(X))
        
        # kiểm tra numeric matrix
        if hasattr(X, "toarray"):
            X_check = X.toarray()
        else:
            X_check = X
        
        print("✔ Output numeric:", np.issubdtype(X_check.dtype, np.number))
        
    except Exception as e:
        print("❌ Lỗi:", e)

test_pipeline(df_full, "Full Data")
test_pipeline(df_missing, "Missing Data")
test_pipeline(df_skewed, "Skewed Data")
test_pipeline(df_unseen, "Unseen Category")
test_pipeline(df_wrong_type, "Wrong Type")


plt.hist(df['SalePrice'], bins=30)
plt.title("Before Pipeline")
plt.show()


X = full_pipeline.fit_transform(df)

if hasattr(X, "toarray"):
    X = X.toarray()

plt.hist(X[:, 1], bins=30)
plt.title("After Pipeline")
plt.show()

print("Before:\n", df['SalePrice'].describe())
print("After:\n", pd.Series(X[:, 1]).describe())

pipe_lr = Pipeline([
    ('preprocess', full_pipeline),
    ('model', LinearRegression())
])

pipe_rf = Pipeline([
    ('preprocess', full_pipeline),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])


scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

def evaluate(model, X, y):
    scores = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=False)
    
    return {
        'RMSE': -scores['test_neg_root_mean_squared_error'],
        'MAE': -scores['test_neg_mean_absolute_error'],
        'R2': scores['test_r2']
    }

X = df.drop(columns=['SalePrice'])
y = df['LotArea']

res_lr = evaluate(pipe_lr, X, y)
res_rf = evaluate(pipe_rf, X, y)

def summarize(name, res):
    print(f"\n=== {name} ===")
    print("RMSE:", res['RMSE'].mean(), "±", res['RMSE'].std())
    print("MAE :", res['MAE'].mean(), "±", res['MAE'].std())
    print("R2  :", res['R2'].mean(), "±", res['R2'].std())

summarize("Linear Regression", res_lr)
summarize("Random Forest", res_rf)

pipe_rf.fit(X, y)

model = pipe_rf.named_steps['model']
importances = model.feature_importances_

feature_names = get_feature_names(full_pipeline)

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feat_imp.head(10))