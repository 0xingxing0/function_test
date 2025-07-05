# import os
# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = os.path.join('static', 'uploads')
# RESULT_FOLDER = os.path.join('static', 'results')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# def yolo_prdict(predict_path):
#     model = YOLO("yolo11n.pt")
#     # Perform object detection on an image
#     results = model.predict(source=predict_path, save=False)
#     results[0].save("./static/results/result.jpg")
#
#     result = results[0]
#     # 获取检测框信息
#     boxes = result.boxes  # Boxes 对象
#
#     # 获取类别id、类别名、置信度
#     class_ids = boxes.cls.cpu().numpy().astype(int)  # 类别id数组
#     confidences = boxes.conf.cpu().numpy()  # 置信度数组
#     # 获取类别名（需要模型类名）
#     class_names = model.names  # dict: id -> name
#     # 统计检测目标个数
#     num_targets = len(class_ids)
#     print(f"检测到 {num_targets} 个目标:")
#
#     for i, cls_id in enumerate(class_ids):
#         name = class_names[cls_id]
#         conf = confidences[i]
#         print(f"目标 {i + 1}: 类别={name}, 置信度={conf:.2f}")
#
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     original_img_url = None
#     result_img_url = None
#
#     if request.method == 'POST':
#         print("检测开始了")
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(save_path)
#
#             original_img_url = url_for('static', filename=f'uploads/{filename}')
#             print(original_img_url)
#
#             # 开始检测
#             predict_path = '.' + original_img_url
#             yolo_prdict(predict_path)
#             # 显示检测结果图
#             result_filename = 'result.jpg'  # 需提前放入 static/results 文件夹
#             result_img_url = url_for('static', filename=f'results/{result_filename}')
#
#             return render_template('index.html', original_img_url=original_img_url, result_img_url=result_img_url)
#
#     return render_template('index.html', original_img_url=original_img_url, result_img_url=result_img_url)
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#

from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename

from ultralytics import YOLO


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def yolo_prdict(predict_path):
    model = YOLO("yolo11n.pt")
    # Perform object detection on an image
    results = model.predict(source=predict_path, save=False)
    results[0].save("./static/results/result.jpg")

    result = results[0]
    # 获取检测框信息
    boxes = result.boxes  # Boxes 对象

    # 获取类别id、类别名、置信度
    class_ids = boxes.cls.cpu().numpy().astype(int)  # 类别id数组
    confidences = boxes.conf.cpu().numpy()  # 置信度数组
    # 获取类别名（需要模型类名）
    class_names = model.names  # dict: id -> name
    # 统计检测目标个数
    num_targets = len(class_ids)
    print(f"检测到 {num_targets} 个目标:")

    for i, cls_id in enumerate(class_ids):
        name = class_names[cls_id]
        conf = confidences[i]
        print(f"目标 {i + 1}: 类别={name}, 置信度={conf:.2f}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(save_path)
    file.save(save_path)

    # 模拟检测结果，真实场景可以替换为模型推理结果
    predict_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    yolo_prdict(predict_path)

    return jsonify({
        'original': url_for('static', filename='uploads/' + filename),
        'result': url_for('static', filename='results/result.jpg')
    })


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
