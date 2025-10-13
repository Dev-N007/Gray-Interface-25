import numpy as np

# euclidean distance
def e_dist(v1, v2):
    d = 0
    for i in range(len(v1)):
        d += (v1[i] - v2[i]) ** 2
    print(f"Euclidean distance is : {np.sqrt(d)}")


# manhattan distance
def m_dist(v1, v2):
    d = 0
    for i in range(len(v1)):
        d += np.abs(v1[i] - v2[i])
    print(f"Manhattan distance is: {d}")


# cosine distnace
def c_dist(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    print(f"Cosine distance is : {1 - np.dot(v1,v2)/(m1*m2)}")  # c_dist = 1 - c_sim


# Angle b/w vectors
def angle(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    c_theta = np.clip(
        np.dot(v1, v2) / (m1 * m2), -1.0, 1.0
    )  # clipping to avoid floating point precision errors
    print(f"The angle b/w the vectors is {np.degrees(np.arccos(c_theta))} degrees")


v1 = np.array(list(map(float, input("Enter v1 : ").split())))
v2 = np.array(list(map(float, input("Enter v2 : ").split())))

e_dist(v1, v2)
m_dist(v1, v2)
c_dist(v1, v2)
angle(v1, v2)
