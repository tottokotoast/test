!pip install opencv-python-headless==4.7.0.72 numpy==1.23.5

%matplotlib inline

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

"""
# 1. .mp4形式の動画ファイルを読み込む（ユーザーがアップロードするファイルを選ぶ）
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
"""
cap = cv2.VideoCapture('test2.MP4')

# 2. 動画ファイルのフレームレートを60fps→2fpsに変更する（連続する30フレームのうち29フレームを消去する）
frame_rate = cap.get(cv2.CAP_PROP_FPS)
skip_frames = int(frame_rate / 2) - 1  # 60fps を 2fps に変更する場合 (60fps / 2fps - 1 = 29)

# オブジェクトの重心の軌跡を保存するリスト
centroids = []

frame_count = 0 # 元動画のフレームカウンター
processed_frame_count = 0 # スキップされずに処理されたフレームのカウンター

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 29フレームをスキップし、30フレーム目に処理を行う
    if frame_count % (skip_frames + 1) != 0:
        continue

    # スキップされずに処理されたフレームのカウンターをインクリメント
    processed_frame_count += 1

    # 60処理フレームごとに経過時間を表示 (2Hzで60フレームは30秒に相当)
    if processed_frame_count % 60 == 0:
        elapsed_seconds = (processed_frame_count / 2) # 2Hzなので、カウントを2で割ると経過秒数
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        print(f"{minutes:02}m{seconds:02}s / 05m00s")

    # ここから元の処理（スキップされなかったフレームに対して重心を計算）
    # 3. 画像を二値化する。黒いオブジェクト（最も面積の大きいもの）が1つだけ残るような閾値を設定する。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)  # 閾値を20に指定

    # 4. 各フレームについて、オブジェクトの重心の座標を出す。
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)  # 最も面積の大きい輪郭を抽出
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    # 軌跡の描画はループ内で行わない

cap.release() # キャプチャを解放

# 5. ループ終了後、最後のフレームに重ねてオブジェクトの重心の軌跡の全長を表示する。
# この部分については、平均化フレームではなく、最後に読み込んだフレームに対して描画する必要があります。
# 動画全体をメモリに保持しない場合、最後のフレームを別途取得する必要があります。
# 簡単のために、今回は重心リストの長さに応じて軌跡を表示します。
# もし最後のフレームに描画したい場合は、動画を再度開き、最後のフレームを読み込む処理を追加してください。

if len(centroids) > 0:
    # 軌跡の描画には元の画像が必要ですが、ループ終了後に最後のフレームを取得する簡単な方法がないため、
    # ここでは例として空白の画像に軌跡を描画するか、または重心座標のみを表示します。
    # 最後のフレームに描画したい場合は、別途その処理を追加してください。

    # 例：重心座標のみを表示
    # print("Object Centroid Trajectory (x, y):")
    # for centroid in centroids:
    #     print(centroid)

    # もし、最後のフレームに描画したい場合は、以下のコメントアウトを解除し、
    # 適切な方法で最後のフレームを取得してください。
    # (例: capを再度開き、seekで最後に移動、readする)

    if cap.open('test2.MP4'): # 動画ファイルを再度開く
        last_frame_index = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        ret, last_frame = cap.read()
        if ret:
            for i in range(len(centroids) - 1):
                cv2.line(last_frame, centroids[i], centroids[i + 1], (0, 0, 255), 2)
            plt.imshow(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
            plt.title("Overall Object Trajectory")
            plt.show()
        cap.release()

# 総移動距離を計算して表示
total_distance_pixels = 0 # ピクセル単位の総移動距離を保持する変数名を変更
for i in range(len(centroids) - 1):
    x1, y1 = centroids[i]
    x2, y2 = centroids[i+1]
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    total_distance_pixels += distance

# ピクセルをcmに換算
pixels_per_cm = 700 / 30  # 700ピクセルが30cmなので、1cmあたりのピクセル数
total_distance_cm = total_distance_pixels / pixels_per_cm

print(f"Total object movement distance: {total_distance_cm:.2f} cm")

# 平均速度を計算して表示 (cm/s)
if processed_frame_count > 0: # 処理されたフレームがあるか確認
    elapsed_seconds = processed_frame_count / 2 # 2Hzなので、処理されたフレーム数を2で割ると経過秒数
    if elapsed_seconds > 0: # 経過時間が0より大きいか確認
        average_speed_cm_per_s = total_distance_cm / elapsed_seconds
        print(f"Average object speed: {average_speed_cm_per_s:.2f} cm/s")
    else:
        print("Cannot calculate average speed: Elapsed time is zero.")
else:
    print("Cannot calculate average speed: No frames were processed.")


print("Processing finished.")
