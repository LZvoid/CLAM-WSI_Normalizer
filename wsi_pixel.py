import openslide
import sys
import os

def get_level0_dimensions(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"文件不存在: {wsi_path}")

    slide = openslide.OpenSlide(wsi_path)
    level0_dim = slide.level_dimensions[0]  # (width, height)
    slide.close()

    return level0_dim

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python get_wsi_level0_size.py <WSI路径>")
        sys.exit(1)

    # 获取WSI文件路径,采用对话框
    #wsi_path = 
    wsi_path = sys.argv[1]
    try:
        width, height = get_level0_dimensions(wsi_path)
        print(f"Level 0 尺寸: 宽度={width} 像素, 高度={height} 像素")
    except Exception as e:
        print(f"读取失败: {e}")
