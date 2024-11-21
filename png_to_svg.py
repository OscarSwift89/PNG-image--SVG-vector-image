import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter  # 确保导入 ImageFilter

def extract_contours(image_path):
    # 加载图像并转换为灰度
    image = Image.open(image_path).convert('L')
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)  # 提高分辨率
    image = image.point(lambda p: p > 128 and 255)  # 二值化

    # 转换为 NumPy 数组
    bitmap = np.array(image)

    # 使用 OpenCV 提取边缘
    edges = cv2.Canny(bitmap, 50, 150)  # 边缘检测（调整阈值范围以提取更精细的轮廓）
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 平滑轮廓
    smoothed_contours = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)  # 轮廓简化比例
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(smoothed_contour)

    return smoothed_contours, image.size

def draw_contours_to_png(contours, image_size, output_path):
    # 创建黑色背景图像
    output_image = Image.new('RGB', image_size, 'black')
    draw = ImageDraw.Draw(output_image)

    # 创建一个临时白色遮罩图像
    mask_image = Image.new('L', image_size, 'black')  # 单通道图像，用于遮罩
    mask_draw = ImageDraw.Draw(mask_image)

    # 绘制白色轮廓到遮罩
    for contour in contours:
        points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
        mask_draw.polygon(points, fill='white')  # 填充轮廓为白色

    # 将遮罩叠加到背景图上
    output_image.paste('white', mask=mask_image)

    # 平滑图像
    output_image = output_image.filter(ImageFilter.SMOOTH_MORE)  # 进一步平滑
    output_image.save(output_path)
    print(f"矢量图已保存为 PNG 文件：{output_path}")

# 主程序
input_image_path = 'weex.png'  # 替换为你的输入文件路径
output_png_path = 'output_image.png'

# 提取轮廓并绘制
contours, image_size = extract_contours(input_image_path)
draw_contours_to_png(contours, image_size, output_png_path)
