import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import files # 必要であればアップロードに使用
# from google.colab.patches import cv2_imshow # 画像表示はファイル保存に切り替えるためコメントアウト

# 処理する動画ファイルがあるフォルダのパスを指定
# 例：Colabの左側のファイルアイコンからフォルダを作成し、動画をアップロードした場合
video_folder = '/content/your_video_folder' # ここを実際のフォルダパスに変更してください

# 必要なライブラリがインストールされていることを確認
# !pip install opencv-python-headless==4.7.0.72 numpy==1.23.5 # 環境に合わせて適切なバージョンを指定してください

# 各動画の処理結果を保存するリスト（総移動距離のみ）
all_video_results = []

# フォルダが存在するか確認
if not os.path.isdir(video_folder):
    print(f"Error: Folder not found at {video_folder}")
else:
    # フォルダ内の動画ファイルをリストアップ
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mov') or f.endswith('.mp4')]

    if not video_files:
        print(f"No video files (.mov or .mp4) found in {video_folder}")
    else:
        print(f"Found {len(video_files)} video files in {video_folder}: {video_files}")

        # 各動画ファイルに対して処理を実行
        for video_file_name in video_files:
            video_path = os.path.join(video_folder, video_file_name)
            print(f"\nProcessing video: {video_file_name}")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Could not open video file {video_file_name}")
                continue

            # 2. 動画ファイルのフレームレートを60fps→2fpsに変更する（連続する30フレームのうち29フレームを消去する）
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            if frame_rate == 0: # フレームレートが取得できない場合のエラーハンドリング
                 print(f"Warning: Could not get frame rate for {video_file_name}. Skipping.")
                 cap.release()
                 continue

            skip_frames = int(frame_rate / 2) - 1  # 60fps を 2fps に変更する場合 (60fps / 2fps - 1 = 29)
            if skip_frames < 0: # 2fpsより低いフレームレートの場合
                skip_frames = 0
                print(f"Warning: Frame rate is already low ({frame_rate} fps). No frames will be skipped.")


            # オブジェクトの重心の軌跡を保存するリスト
            centroids = []

            frame_count = 0 # 元動画のフレームカウンター
            processed_frame_count = 0 # スキップされずに処理されたフレームのカウンター

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
                    print(f"  Processing time: {minutes:02}m{seconds:02}s")


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

            # 5. ループ終了後、最後のフレームに重ねてオブジェクトの重心の軌跡の全長を描画し、画像として保存する。
            if len(centroids) > 0:
                cap_render = cv2.VideoCapture(video_path)
                if cap_render.isOpened():
                    last_frame_index = int(cap_render.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                    # processed_frame_countはスキップされたフレームを除いた数なので、
                    # 最後の処理フレームの元の動画でのインデックスを計算
                    last_processed_original_index = (processed_frame_count - 1) * (skip_frames + 1)
                    if last_processed_original_index >= 0:
                         cap_render.set(cv2.CAP_PROP_POS_FRAMES, last_processed_original_index)
                         ret, last_frame = cap_render.read()
                         if ret:
                             for i in range(len(centroids) - 1):
                                 cv2.line(last_frame, centroids[i], centroids[i + 1], (0, 0, 255), 2) # 赤い線で軌跡を描画

                             # 画像をファイルとして保存
                             output_image_name = f"Trajectory for {os.path.splitext(video_file_name)[0]}.jpg" # .movや.mp4拡張子を除去
                             output_image_path = os.path.join(video_folder, output_image_name)
                             cv2.imwrite(output_image_path, last_frame)
                             print(f"  Trajectory image saved to: {output_image_path}")

                         else:
                             print(f"Warning: Could not read the last processed frame for {video_file_name}")
                    else:
                         print(f"Warning: No frames were processed for {video_file_name}, cannot render trajectory.")

                    cap_render.release()
                else:
                     print(f"Error: Could not re-open video file {video_file_name} for rendering.")

            # 総移動距離を計算
            total_distance_pixels = 0
            for i in range(len(centroids) - 1):
                x1, y1 = centroids[i]
                x2, y2 = centroids[i+1]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance_pixels += distance

            # ピクセルをcmに換算
            pixels_per_cm = 700 / 30  # 700ピクセルが30cmなので、1cmあたりのピクセル数
            total_distance_cm = total_distance_pixels / pixels_per_cm

            # この動画の総移動距離をリストに追加
            all_video_results.append({
                "file_name": video_file_name,
                "total_distance_cm": total_distance_cm,
            })

            print("-" * 30) # 区切り線を表示

        print("All video files processed.")

        # 全動画の総移動距離をまとめて表示
        print("\n--- Summary of Total Distances ---")
        for result in all_video_results:
            print(f"{result['file_name']}: {result['total_distance_cm']:.2f} cm")
            # print("-" * 10) # 各動画の結果間の区切り線は不要になるのでコメントアウト
