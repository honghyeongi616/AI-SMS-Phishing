from flask import Flask, render_template, request
import pickle
import os

# Flask 앱 초기화
app = Flask(__name__)

# 저장된 모델 및 벡터라이저 불러오기
model_path = os.path.join('model', 'model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]

    if prediction == 1:
        result = "⚠️ 피싱으로 의심됩니다!"
    else:
        result = "✅ 정상 메시지로 보입니다."

    # 🔥 로그 저장
    with open('log.csv', 'a', encoding='utf-8') as log_file:
        log_file.write(f'"{user_input}",{prediction}\n')

    return render_template('result.html', message=user_input, result=result)

@app.route('/logs')
def logs():
    logs = []
    if os.path.exists('log.csv'):
        with open('log.csv', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    msg, label = line.strip().rsplit(',', 1)  # 뒤에서부터 1번만 split
                    logs.append((msg.strip('"'), label))
                except ValueError:
                    # 형식이 이상한 줄은 무시
                    continue
    return render_template('log.html', logs=logs)


@app.route('/score')
def score():
    if os.path.exists('score.txt'):
        with open('score.txt', 'r', encoding='utf-8') as f:
            report = f.read()
    else:
        report = "아직 모델 평가 결과가 없습니다."
    return render_template('score.html', report=report)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
