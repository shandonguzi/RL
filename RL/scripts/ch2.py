import numpy as np

"""
给定 P 计算 state value
"""

def build_gridworld(gamma=0.9, n=5, forbid=None, target=None, arrow_map=None):
    n_states = n * n
    reward = np.zeros(n_states)
    P = np.zeros((n_states, n_states))

    if forbid is None:
        forbid = []
    if target is None:
        target = 0

    for i in range(n):
        for j in range(n):
            s = i * n + j
            # target state
            if s == target:
                P[s, s] = 1
                reward[s] = 1
                continue
            action = arrow_map[i][j]
            ni, nj = i + action[0], j + action[1]
            if 0 <= ni < n and 0 <= nj < n:
                s2 = ni * n + nj
                P[s, s2] = 1
                # forbidden state
                if s2 in forbid:
                    reward[s] = -1
                # target state
                elif s2 == target:
                    reward[s] = 1
            # boundary state
            else:
                P[s, s] = 1
                reward[s] = -1
    return P, reward

def value_iteration(P, reward, gamma=0.9, n_iter=100):
    v = np.zeros_like(reward)
    for _ in range(n_iter):
        v = reward + gamma * P @ v
    return v

def value_closed_form(P, reward, gamma=0.9):
    n_states = len(reward)
    I = np.eye(n_states)
    v = np.linalg.solve(I - gamma * P, reward)
    return v

def main():
    gamma = 0.9
    n = 5
    forbid = [6, 7, 12, 16, 18, 21]
    target = 17
    """
    actions = {
        (0, 1): 1,     # right
        (0, -1): -1,   # left
        (1, 0): 5,     # down
        (-1, 0): -5    # up
    }
    """
    arrow_map = [
        [(0,1),(0,1),(0,1),(1,0),(1,0)],
        [(-1,0),(-1,0),(0,1),(1,0),(1,0)],
        [(-1,0),(0,-1),(1,0),(0,1),(1,0)],
        [(-1,0),(0,1),(0,0),(0,-1),(1,0)],
        [(-1,0),(0,1),(-1,0),(0,-1),(0,-1)]
    ]
    P, reward = build_gridworld(gamma, n, forbid, target, arrow_map)

    # 迭代法
    v_iter = value_iteration(P, reward, gamma)
    print("值迭代法结果:")
    print(np.round(v_iter.reshape(n, n), 2))

    # 闭式解法
    v_closed = value_closed_form(P, reward, gamma)
    print("\n矩阵公式法结果:")
    print(np.round(v_closed.reshape(n, n), 2))

if __name__ == "__main__":
    main()
