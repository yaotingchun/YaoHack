from flask import Flask, render_template, Response, jsonify
import cv2
from posture_analyzer import PostureAnalyzer

app = Flask(__name__)
camera = cv2.VideoCapture(0)

posture_analyzer = PostureAnalyzer()
detection_result = {"hand": "N/A", "pose": "N/A", "score": 0}

def generate_frames():
    global detection_result
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame, posture_metrics = posture_analyzer.analyze_frame(frame)
            detection_result = posture_metrics

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html')

@app.route('/ai_coach')
def ai_coach():
    return render_template('ai_coach.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_result')
def get_detection_result():
    return jsonify(detection_result)

if __name__ == '__main__':
    app.run(debug=True)
