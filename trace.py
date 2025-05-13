%matplotlib inline
!pip install opencv-python-headless==4.7.0.72

from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
# 1. .mp4形式の動画ファイルを読み込む（ユーザーがアップロードするファイルを選ぶ）
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
"""
cap = cv2.VideoCapture('test.MP4')

# 2. 動画ファイルのフレームレートを60fps→2fpsに変更する（連続する30フレームのうち29フレームを消去する）
frame_rate = cap.get(cv2.CAP_PROP_FPS)
skip_frames = int(frame_rate / 2) - 1  # 60fps を 2fps に変更する場合

# オブジェクトの重心の軌跡を保存するリスト
centroids = []

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % (skip_frames + 1) != 0:  # フレームをスキップ
        continue

    # 3. 画像を二値化する。黒いオブジェクト（最も面積の大きいもの）が1つだけ残るような閾値を設定する。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)  # 閾値を100に指定


    # 4. 各フレームについて、オブジェクトの重心の座標を出す。
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)  # 最も面積の大きい輪郭を抽出
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    # 5. 元画像に重ねてオブジェクトの重心の軌跡を表示する。
    # ループ内で軌跡を描画
    for i in range(len(centroids) - 1):
        cv2.line(frame, centroids[i], centroids[i + 1], (0, 0, 255), 2)  # 赤い線で軌跡を描画
        
    # 画像を表示
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Object Trajectory")
    plt.show() # plt.show() を明示的に呼び出す
    plt.clf() # 現在の図形をクリア

plt.show()
