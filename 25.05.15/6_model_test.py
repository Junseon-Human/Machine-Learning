import streamlit as st
import re
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.model_selection import cross_validate
import numpy as np

st.title("모델 성능 테스트 앱")

# 1. 모델 선택
model_name = st.sidebar.selectbox(
    "모델 선택",
    [
        # 분류 모델
        "KNN",
        "Bagging KNN",
        "SVC",
        "Bagging SVC",
        "Decision Tree",
        "Bagging DT",
        "RandomForest",
        "ExtraTrees",
        "AdaBoost",
        "GradientBoosting",
        # 회귀 모델
        "KNNRegressor",
        "Bagging KNNRegressor",
        "SVR",
        "Bagging SVR",
        "DecisionTreeRegressor",
        "Bagging DTR",
        "AdaBoostRegressor",
        "GradientBoostingRegressor",
    ],
)

# 2. 분류/회귀 여부 판별
is_classifier = model_name in [
    "KNN",
    "Bagging KNN",
    "SVC",
    "Bagging SVC",
    "Decision Tree",
    "Bagging DT",
    "RandomForest",
    "ExtraTrees",
    "AdaBoost",
    "GradientBoosting",
]

# 3. 데이터셋 선택 (모델 유형에 따라 달라짐)
dataset_options = (
    ["iris", "wine", "breast_cancer"] if is_classifier else ["diabetes", "california"]
)
dataset_name = st.sidebar.selectbox("데이터셋 선택", dataset_options)

# 4. 파라미터 입력 UI
params = {}
if "KNN" in model_name:
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 15, 5)
if "Bagging" in model_name:
    params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 50, 10)
if model_name in ["SVC", "Bagging SVC", "SVR", "Bagging SVR"]:
    params["C"] = st.sidebar.slider("C", 0.1, 10.0, 1.0)
if "Decision Tree" in model_name or model_name == "DecisionTreeRegressor":
    params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 5)
if model_name in ["RandomForest", "ExtraTrees"]:
    params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 100)
if model_name == "RandomForest":
    params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 5)
if model_name in [
    "AdaBoost",
    "AdaBoostRegressor",
    "GradientBoosting",
    "GradientBoostingRegressor",
]:
    params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 50)
    params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1)

# 5. 데이터 로드
if dataset_name == "iris":
    data = load_iris()
elif dataset_name == "wine":
    data = load_wine()
elif dataset_name == "breast_cancer":
    data = load_breast_cancer()
elif dataset_name == "diabetes":
    data = load_diabetes()
else:
    data = fetch_california_housing()
X, y = data.data, data.target


# 6. 모델 생성 함수
def make_model(name, params):
    if name == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5)),
        )
    if name == "Bagging KNN":
        base = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5)),
        )
        return BaggingClassifier(base, n_estimators=params.get("n_estimators", 10))
    if name == "SVC":
        return make_pipeline(StandardScaler(), SVC(C=params.get("C", 1.0)))
    if name == "Bagging SVC":
        base = make_pipeline(StandardScaler(), SVC(C=params.get("C", 1.0)))
        return BaggingClassifier(base, n_estimators=params.get("n_estimators", 10))
    if name == "Decision Tree":
        return make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(max_depth=params.get("max_depth", 5)),
        )
    if name == "Bagging DT":
        base = make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(max_depth=params.get("max_depth", 5)),
        )
        return BaggingClassifier(base, n_estimators=params.get("n_estimators", 10))
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
        )
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=params.get("n_estimators", 100))
    if name == "AdaBoost":
        return make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(
                n_estimators=params.get("n_estimators", 50),
                learning_rate=params.get("learning_rate", 0.1),
            ),
        )
    if name == "GradientBoosting":
        return make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 50),
                learning_rate=params.get("learning_rate", 0.1),
            ),
        )
    if name == "KNNRegressor":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=params.get("n_neighbors", 5)),
        )
    if name == "Bagging KNNRegressor":
        base = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=params.get("n_neighbors", 5)),
        )
        return BaggingRegressor(base, n_estimators=params.get("n_estimators", 10))
    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=params.get("C", 1.0)))
    if name == "Bagging SVR":
        base = make_pipeline(StandardScaler(), SVR(C=params.get("C", 1.0)))
        return BaggingRegressor(base, n_estimators=params.get("n_estimators", 10))
    if name == "DecisionTreeRegressor":
        return make_pipeline(
            StandardScaler(),
            DecisionTreeRegressor(max_depth=params.get("max_depth", None)),
        )
    if name == "Bagging DTR":
        base = make_pipeline(
            StandardScaler(),
            DecisionTreeRegressor(max_depth=params.get("max_depth", None)),
        )
        return BaggingRegressor(base, n_estimators=params.get("n_estimators", 10))
    if name == "AdaBoostRegressor":
        return make_pipeline(
            StandardScaler(),
            AdaBoostRegressor(
                n_estimators=params.get("n_estimators", 50),
                learning_rate=params.get("learning_rate", 0.1),
            ),
        )
    if name == "GradientBoostingRegressor":
        return make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 50),
                learning_rate=params.get("learning_rate", 0.1),
            ),
        )
    raise ValueError(f"알 수 없는 모델: {name}")


# 7. 모델 생성
model = make_model(model_name, params)

# 8. scoring 설정
scoring = (
    ["accuracy", "f1_macro"] if is_classifier else ["neg_mean_squared_error", "r2"]
)
scoring_str = re.sub(r"[\[\]']", "", str(scoring))
params_str = re.sub(r"[\{\}']", "", str(params))
# 9. 옵션 표시
st.write(f"**선택된 모델:** {model_name}")
st.write(f"**선택된 데이터셋:** {dataset_name}")
st.write(f"**사용 파라미터:** {params_str}")
st.write(f"**교차검증 scoring:** {scoring_str}")

# 10. 교차검증 및 결과 표시
with st.spinner("교차검증 실행 중..."):
    results = cross_validate(
        estimator=model, X=X, y=y, cv=5, scoring=scoring, return_train_score=False
    )

st.write("**평균 학습 시간 (s):**", np.mean(results["fit_time"]).round(4))
st.write("**평균 예측 시간 (s):**", np.mean(results["score_time"]).round(4))
if is_classifier:
    st.write("**평균 정확도:**", np.mean(results["test_accuracy"]).round(4))
    st.write("**평균 F1:**", np.mean(results["test_f1_macro"]).round(4))
else:
    st.write("**평균 MSE:**", -np.mean(results["test_neg_mean_squared_error"]).round(4))
    st.write("**평균 R²:**", np.mean(results["test_r2"]).round(4))
