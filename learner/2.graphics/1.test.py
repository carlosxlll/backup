import pygame
import numpy as np

# 初始化Pygame
pygame.init()

# 设置窗口大小
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# 创建一个3D长方体
cube_points = np.array([
    [100, 100, 100],
    [100, -100, 100],
    [-100, -100, 100],
    [-100, 100, 100],
    [100, 100, -100],
    [100, -100, -100],
    [-100, -100, -100],
    [-100, 100, -100]
])

# 创建长方体的边
cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    # 清除屏幕
    screen.fill((0, 0, 0))

    # 透视投影
    fov = 500  # 视场（越大，物体越小；越小，物体越大）
    for edge in cube_edges:
        point1 = cube_points[edge[0]]
        point2 = cube_points[edge[1]]
        perspective_point1 = point1 * fov / (fov + point1[2]) + np.array([width/2, height/2, 0])[:2]
        perspective_point2 = point2 * fov / (fov + point2[2]) + np.array([width/2, height/2, 0])[:2]
        pygame.draw.line(screen, (255, 255, 255), perspective_point1[:2], perspective_point2[:2])

    # 更新屏幕
    pygame.display.flip()
