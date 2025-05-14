import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_breast_cancer,
    load_wine,
    load_iris,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import confusion_matrix

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
st.set_page_config(page_title="KNN & SVM", page_icon="📊", layout="wide")
st.title("KNN & SVM 모델 분석")

# --- 세션스테이트 초기화 ---
if "history" not in st.session_state:
    st.session_state.history = []

# 1) 모델 선택
model_name = st.selectbox(
    "▶️ 사용할 모델을 선택하세요",
    ["KNN_classifier", "KNN_regression", "SVC", "SVR"],
)

# 2) 분류/회귀 구분 + 데이터셋 제한
if "classifier" in model_name or model_name == "SVC":
    task_type = "classification"
    available_datasets = ["load_breast_cancer", "load_wine", "load_iris"]
else:
    task_type = "regression"
    available_datasets = ["load_diabetes", "fetch_california_housing"]

# 3) 데이터셋 선택
data_name = st.selectbox(
    f"🧬 사용할 데이터셋을 선택하세요 ({task_type})", available_datasets
)

# 4) 데이터 로딩
dataset_loader = {
    "load_diabetes": load_diabetes,
    "fetch_california_housing": fetch_california_housing,
    "load_breast_cancer": load_breast_cancer,
    "load_wine": load_wine,
    "load_iris": load_iris,
}
dataset = dataset_loader[data_name]()
X, y = dataset.data, dataset.target

# 5) 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6) 스케일링 옵션
scaled = st.checkbox("표준화 하기", value=True)
if scaled:
    scaler_name = st.selectbox("스케일러 선택", ["StandardScaler", "MinMaxScaler"])
    scaler_cls = StandardScaler if scaler_name == "StandardScaler" else MinMaxScaler
    scaler = scaler_cls()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# 7) 하이퍼파라미터 UI & 모델 생성
hyperparams_desc = ""
if model_name == "KNN_classifier":
    k = st.slider("이웃 수 (k)", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    hyperparams_desc = f"k={k}"

elif model_name == "KNN_regression":
    k = st.slider("이웃 수 (k)", 1, 20, 5)
    model = KNeighborsRegressor(n_neighbors=k)
    hyperparams_desc = f"k={k}"

elif model_name == "SVC":
    C = st.slider("C (규제 강도)", 0.01, 10.0, 1.0, 0.01, format="%.2f")
    kernel = st.selectbox("커널", ["linear", "poly", "rbf", "sigmoid"])
    gamma = st.selectbox("gamma", ["scale", "auto"])
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    hyperparams_desc = f"C={C}, kernel={kernel}, gamma={gamma}"

else:  # "SVR"
    C = st.slider("C (규제 강도)", 0.01, 10.0, 1.0, 0.01, format="%.2f")
    epsilon = st.slider("epsilon (오차 허용)", 0.0, 1.0, 0.1, 0.01)
    kernel = st.selectbox("커널", ["linear", "poly", "rbf", "sigmoid"])
    gamma = st.selectbox("gamma", ["scale", "auto"])
    model = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    hyperparams_desc = f"C={C}, eps={epsilon}, kernel={kernel}, gamma={gamma}"

# 8) 학습·예측
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9) 평가 지표 계산
if task_type == "classification":
    metric_name = "Accuracy"
    metric_value = accuracy_score(y_test, y_pred)
    train_score = model.score(X_train, y_train)  # 분류면 accuracy
else:
    metric_name = "MSE"
    metric_value = mean_squared_error(y_test, y_pred)
    train_score = model.score(X_train, y_train)  # 회귀면 R²

# 10) 결과 화면에 표시
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 모델 평가 지표")
    st.metric(metric_name, f"{metric_value:.4f}")
    st.metric(f"Train {metric_name}", f"{train_score:.4f}")

with col2:
    if task_type == "classification":
        st.subheader("📊 혼동 행렬")
        cm = confusion_matrix(y_test, y_pred)
        labels = model.classes_ if hasattr(model, "classes_") else np.unique(y_test)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_xlabel("예측 레이블")
        ax.set_ylabel("실제 레이블")
        ax.set_title("Confusion Matrix")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    else:  # regression
        st.subheader("🔍 실제 vs 예측")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        ax.set_xlabel("실제값")
        ax.set_ylabel("예측값")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


st.subheader("📋 사용 기록")
# 새로운 기록 추가
st.session_state.history.append(
    {
        "model": model_name,
        "dataset": data_name,
        "hyperparams": hyperparams_desc,
        metric_name: f"{metric_value:.4f}",
        f"Train {metric_name}": f"{train_score:.4f}",
    }
)
# 테이블로 표시
df_hist = pd.DataFrame(st.session_state.history)
st.table(df_hist)
