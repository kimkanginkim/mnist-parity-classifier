import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


############## Train 데이터 로드 및 전처리 ##############
# Train 데이터 로드
train_data = np.load("train.npz", allow_pickle=True)
train_images = train_data["images"]
train_labels = train_data["labels"]

# 데이터를 2D 형태로 변환 (28x28 이미지를 784 크기 벡터로 변환)
X = train_images.reshape(train_images.shape[0], -1)
y = train_labels.astype(int)  # 레이블을 정수형으로 변환

# Train, Validation, Test 데이터 분리
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5758
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=5758
)  # Validation Set = 10% of Total

# StandardScaler를 사용하여 Train 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Train 데이터로 스케일러 학습 및 변환
X_val_scaled = scaler.transform(X_val)  # Validation 데이터 변환
X_test_scaled = scaler.transform(X_test)  # Test 데이터 변환

# PCA를 사용하여 SVM 전용 데이터로 차원 축소 (784 → 100)
from sklearn.decomposition import PCA

pca = PCA(n_components=100, random_state=5758)  # 주성분 100개로 축소
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)
############## Train 데이터 로드 및 전처리 ##############

############## 모델 선언 및 학습 ##############
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Logistic Regression
logistic_model = LogisticRegression(random_state=5758)
logistic_model.fit(X_train_scaled, y_train)  # 원본 스케일된 데이터 사용
logistic_val_pred = logistic_model.predict(X_val_scaled)
logistic_val_acc = accuracy_score(y_val, logistic_val_pred)

# Random Forest
rf_model = RandomForestClassifier(random_state=5758, n_estimators=100, max_depth=10)
rf_model.fit(X_train_scaled, y_train)  # 원본 스케일된 데이터 사용
rf_val_pred = rf_model.predict(X_val_scaled)
rf_val_acc = accuracy_score(y_val, rf_val_pred)

# SVM
svm_model = SVC(random_state=5758)
svm_model.fit(X_train_pca, y_train)  # PCA로 축소된 데이터로 학습
svm_val_pred = svm_model.predict(X_val_pca)
svm_val_acc = accuracy_score(y_val, svm_val_pred)

# 가장 높은 Validation 정확도를 가진 모델 선택
model = None
if logistic_val_acc >= rf_val_acc and logistic_val_acc >= svm_val_acc:
    model = logistic_model
    print("Best Model: Logistic Regression with Validation Accuracy:", logistic_val_acc)
elif rf_val_acc >= logistic_val_acc and rf_val_acc >= svm_val_acc:
    model = rf_model
    print("Best Model: Random Forest with Validation Accuracy:", rf_val_acc)
else:
    model = svm_model
    print("Best Model: SVM with Validation Accuracy:", svm_val_acc)
############## 모델 선언 및 학습 ##############

############## 내부 테스트 데이터로 평가 ##############
if model == svm_model:
    y_pred = model.predict(X_test_pca)  # SVM은 PCA 데이터 사용
else:
    y_pred = model.predict(X_test_scaled)  # 나머지는 원본 스케일된 데이터 사용

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Internal Test Accuracy:", accuracy)
print("Internal Test F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
############## 내부 테스트 데이터로 평가 ##############

############## Unseen Test 데이터 로드 ##############
# Test 데이터 로드
test_data = np.load("test.npz",allow_pickle=True)
test_images = test_data["images"]

# Test 데이터를 2D 형태로 변환
X_test_ext = test_images.reshape(test_images.shape[0], -1)
############## Unseen Test 데이터 로드 ##############

############## Train 데이터 전처리 코드 적용 ##############
# Test 데이터 스케일링
X_test_ext = scaler.transform(X_test_ext)  # 동일한 스케일러를 외부 테스트 데이터에 적용해야함!
X_test_ext = pca.transform(X_test_ext)  # 동일한 PCA 모델을 사용해 차원 축소
############## Train 데이터 전처리 코드 적용 ##############

############## Test 데이터에 예측 수행 ##############
test_predictions = model.predict(X_test_ext)

# 예측 결과 저장
test_ids = np.arange(len(test_predictions))  # Test 데이터의 ID를 생성
test_output = pd.DataFrame({
    'id': test_ids,
    'prediction': test_predictions
})

output_filename = f"{student_id}_{student_name}_MNIST_predictions.csv"  # 파일명 생성
test_output.to_csv(output_filename, index=False)

print(f"Predictions saved to '{output_filename}'")
############## Test 데이터에 예측 수행 ##############
