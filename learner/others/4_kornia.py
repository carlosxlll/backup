import torch
import kornia
import cv2

# 读取图像
img = cv2.imread('./data/1.png')
# 将图像转换为 PyTorch 张量
img_tensor = kornia.image_to_tensor(img, keepdim=False).float() / 255.0
# 定义旋转角度
angle = torch.tensor([30.0])
# 创建旋转矩阵
rotation_matrix = kornia.rotation_matrix(angle, center=torch.zeros(1, 2))
# 对图像进行旋转
img_rotated = kornia.warp_affine(img_tensor, rotation_matrix)
# 将 PyTorch 张量转换为图像
img_rotated = kornia.tensor_to_image(img_rotated.clamp(0.0, 1.0) * 255.0)
# 保存图像
cv2.imwrite('./data/1_rotated.png', img_rotated)