import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="회귀 모델 분석", page_icon="📊", layout="wide")
st.title("당뇨병 데이터셋으로 회귀 모델 분석")

# --- 세션스테이트 초기화 ---
if "prev_model" not in st.session_state:
    st.session_state.prev_model = None
if "history" not in st.session_state:
    st.session_state.history = []

# 데이터 준비
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 모델 선택
model_name = st.selectbox(
    "▶️ 사용할 회귀 모델을 선택하세요",
    ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "Polynomial"],
)

# 모델명이 바뀌었으면 history 초기화
if st.session_state.prev_model != model_name:
    st.session_state.history = []
    st.session_state.prev_model = model_name

# 파라미터 입력
alpha = None
l1_ratio = None
degree = None
if model_name in ["Ridge", "Lasso", "ElasticNet"]:
    alpha = st.slider("🔧 alpha", 0.0001, 1.0, 0.001, 0.0001, format="%.4f")
if model_name == "ElasticNet":
    l1_ratio = st.slider("🔧 l1_ratio", 0.0, 1.0, 0.5, 0.1, format="%.1f")
if model_name == "Polynomial":
    degree = st.slider("🔧 degree", 1, 5, 2)

model = None
# 모델 생성
if model_name == "LinearRegression":
    model = LinearRegression()
elif model_name == "Ridge":
    model = Ridge(alpha=alpha)
elif model_name == "Lasso":
    model = Lasso(alpha=alpha, max_iter=10000)
elif model_name == "ElasticNet":
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
else:
    model = make_pipeline(
        PolynomialFeatures(degree=degree), StandardScaler(), LinearRegression()
    )
if model_name == "KNN_classifier":
    from sklearn.neighbors import KNeighborsClassifier

    n_neighbors = st.slider("이웃 수 (k)", 1, 20, value=5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

elif model_name == "KNN_regression":
    from sklearn.neighbors import KNeighborsRegressor

    n_neighbors = st.slider("이웃 수 (k)", 1, 20, value=5)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)

elif model_name == "SVC":
    from sklearn.svm import SVC

    C = st.number_input("C (규제 강도)", 0.01, 100.0, value=1.0)
    kernel = st.selectbox("커널", ["linear", "poly", "rbf", "sigmoid"])
    gamma = st.selectbox("gamma", ["scale", "auto"])
    model = SVC(C=C, kernel=kernel, gamma=gamma)

elif model_name == "SVR":
    from sklearn.svm import SVR

    C = st.number_input("C (규제 강도)", 0.01, 100.0, value=1.0)
    epsilon = st.number_input("epsilon (오차 허용)", 0.0, 1.0, value=0.1)
    kernel = st.selectbox("커널", ["linear", "poly", "rbf", "sigmoid"])
    gamma = st.selectbox("gamma", ["scale", "auto"])
    model = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)

# 학습·예측
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 평가 지표

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
train_r2 = model.score(X_train, y_train)

# 기록 업데이트
st.session_state.history.append(
    {
        "모델": model_name,
        "alpha": alpha or "",
        "l1_ratio": l1_ratio or "",
        "degree": degree or "",
        "RMSE": f"{rmse:.2f}",
        "R2": f"{r2:.3f}",
    }
)

# 화면에 표시
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📝 모델 평가 지표")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("Train R²", f"{train_r2:.3f}")
    st.metric("Pred R²", f"{r2:.3f}")

with col2:
    st.subheader("📋 테스트 기록")
    df_hist = pd.DataFrame(st.session_state.history)
    st.table(df_hist)

with col3:
    st.subheader("🔍 실제 vs 예측")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("실제값")
    ax.set_ylabel("예측값")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
