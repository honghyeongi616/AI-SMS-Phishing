import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# 1. 샘플 데이터 생성
data = {
    'text': [
        '무료 쿠폰을 받으세요',
        '당신의 계좌가 정지되었습니다',
        '안녕하세요 오늘 점심 같이 하실래요?',
        '지금 바로 클릭하세요',
        '회식 일정 확인 부탁드립니다',
        '이 링크를 클릭하면 선물을 드립니다',
        '어제 회의록 공유드립니다',
        '고객님 당첨을 축하드립니다',
        '내일 일정 괜찮으신가요?',
        '카드 승인 실패. 즉시 확인하세요'
    ],
    'label': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = 피싱, 0 = 정상
}

df = pd.read_csv('dataset.csv')

# 2. 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. 모델 학습
model = MultinomialNB()
model.fit(X, y)

# 4. 모델 및 벡터 저장
os.makedirs('model', exist_ok=True)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ 모델과 벡터 저장 완료!")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# train/test 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 예측 정확도 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"📊 모델 정확도: {acc * 100:.2f}%")

from sklearn.metrics import classification_report

# 모델 평가 및 결과 저장
report = classification_report(y_test, model.predict(X_test), output_dict=False)
with open('score.txt', 'w', encoding='utf-8') as f:
    f.write(report)
