import numpy as np

def calculate_normal_vector(p1 : list[float], p2 : list[float], p3 : list[float]):
    """
    计算三维空间中三点构成的平面的法向量

    Parameters:
    P1,p2,p3 -- 三维空间中的三点

    Returns:
    计算得到的法向量
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return np.cross(v1, v2)

if __name__ == '__main__':
    p1 = [0., 0., 0.]
    p2 = [1., 0.4, 0.]
    p3 = [0., 1., 0.]
    normal_vector = calculate_normal_vector(p1, p2, p3)
    print(normal_vector)