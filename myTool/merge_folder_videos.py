import cv2
import os
import glob
import sys

def merge_videos_with_padding(input_folder: str, output_path: str,
                              max_width: int = 1920, max_height: int = 1080,
                              exts: tuple = ('.mp4', '.avi', '.mov', '.mkv')):
    # 取得資料夾下所有影片檔，並依檔名排序
    files = sorted(
        [f for f in glob.glob(os.path.join(input_folder, '*'))
         if os.path.isfile(f) and f.lower().endswith(exts)]
    )
    if not files:
        print("找不到任何影片檔，請確認路徑與副檔名設定。")
        sys.exit(1)

    # 以第一支影片設定輸出參數（fps、編碼器等）
    cap0 = cv2.VideoCapture(files[0])
    fps = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 若需 AVI 可改 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (max_width, max_height))

    for idx, vf in enumerate(files, 1):
        cap = cv2.VideoCapture(vf)
        print(f"[{idx}/{len(files)}] 正在處理：{os.path.basename(vf)}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            # 計算縮放比例，確保不超過上下限
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            # 計算 Padding 大小，置中對齊
            top = (max_height - new_h) // 2
            bottom = max_height - new_h - top
            left = (max_width - new_w) // 2
            right = max_width - new_w - left

            # 用黑色邊框填充
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            out.write(padded)

        cap.release()

    out.release()
    print(f"完成！已輸出合併影片：{output_path}")

if __name__ == "__main__":
    # 使用範例：請將下面路徑替換為你的資料夾與輸出檔名
    input_folder = r"E:\temp\lilin_MRT_test_dataset"
    output_path  = r"E:\temp\test_merged.mp4"
    merge_videos_with_padding(input_folder, output_path)


"""
# 將四個影片合併為一個四分割畫面
# 直接執行於 cmd 或 PowerShell
# 需要安裝 ffmpeg

ffmpeg ^
  -i "e:\temp\demo\lilin_MRT_test_video.mp4" ^
  -i "e:\temp\demo\demo2000.avi" ^
  -i "e:\temp\demo\demo4000.avi" ^
  -i "e:\temp\demo\demo8000.avi" ^
  -filter_complex ^
  "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [A]; [1:v] setpts=PTS-STARTPTS [B]; [2:v] setpts=PTS-STARTPTS [C]; [3:v] setpts=PTS-STARTPTS [D]; [base][A]overlay=shortest=1[tmp1]; [tmp1][B]overlay=shortest=1:x=960[tmp2]; [tmp2][C]overlay=shortest=1:y=540[tmp3]; [tmp3][D]overlay=shortest=1:x=960:y=540" ^
  -c:v libx264 -crf 23 -preset veryfast ^
  -y "e:\temp\demo\demo-merge.avi"


ffmpeg -i "e:\temp\demo\lilin_MRT_test_video.mp4" ^
       -i "e:\temp\demo\demo2000.avi" ^
       -i "e:\temp\demo\demo4000.avi" ^
       -i "e:\temp\demo\demo8000.avi" ^
-filter_complex "[0:v]scale=960:540[p0];[1:v]scale=960:540[p1];[2:v]scale=960:540[p2];[3:v]scale=960:540[p3];[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540[out]" ^
-map "[out]" -c:v libx264 -crf 23 -preset veryfast "e:\temp\demo\output_2x2.mp4"


"""