import os
from multiprocessing import Pool
import torchvision.io as tvio

DATA_ROOT = "/root/maihathm/AVASR/data/avsr_self"  # Đường dẫn đến thư mục gốc data
NUM_PROCESSES = 72  # Số lượng CPU muốn tận dụng

def get_num_frames(video_path: str) -> int:
    """
    Hàm đọc video và trả về số khung hình (frames).
    """
    try:
        # đọc video với output_format="THWC" => [T, H, W, C]
        video, _, _ = tvio.read_video(
            video_path, 
            pts_unit="sec",
            output_format="THWC"
        )
        return video.shape[0]  # T = số khung hình
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {video_path}: {e}")
        return 0

def collect_video_paths(data_root: str):
    """
    Hàm gom tất cả đường dẫn .mp4 trong các thư mục train/val/test.
    """
    all_video_paths = []
    for split in ["train", "val", "test"]:
        video_dir = os.path.join(data_root, split, f"{split}_video_seg12s")
        if not os.path.isdir(video_dir):
            # Nếu không tồn tại, bỏ qua
            continue

        # Duyệt đệ quy mọi file .mp4
        for root, dirs, files in os.walk(video_dir):
            for fname in files:
                if fname.endswith(".mp4"):
                    fpath = os.path.join(root, fname)
                    all_video_paths.append(fpath)

    return all_video_paths

def main():
    # Thu thập tất cả các đường dẫn .mp4
    video_paths = collect_video_paths(DATA_ROOT)
    print(f"🟢 Tổng số video cần duyệt: {len(video_paths)}")

    # Dùng multiprocessing Pool để đọc frame count
    with Pool(processes=NUM_PROCESSES) as pool:
        frame_counts = pool.map(get_num_frames, video_paths)

    max_frame_count = max(frame_counts) if frame_counts else 0
    print(f"✅ Số frame dài nhất trong toàn bộ dataset: {max_frame_count}")

if __name__ == "__main__":
    main()
