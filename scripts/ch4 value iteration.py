import numpy as np


def calculate_q(n, s, a, v, r, action_choice, gamma, forbidden, target):
    """
    计算 q 值
    """
    now_i, now_j = s // n, s % n
    offset = action_choice[a]
    next_i, next_j = now_i + offset[0], now_j + offset[1]

    # 检查是否越界
    if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
        return r['boundary'] + gamma * v[s]

    next_s = next_i * n + next_j
    if next_s in forbidden:
        return r['forbidden'] + gamma * v[next_s]
    if next_s == target:
        return r['target'] + gamma * v[next_s]

    return r['otherwise'] + gamma * v[next_s]


def value_iteration(n, v, pi, reward, action_choice, forbidden, target, gamma=0.9, n_iter=10000, epsilon=1e-6):
    iteration = 0
    v_old = np.ones_like(v)

    while iteration < n_iter and np.max(np.abs(v - v_old)) > epsilon:
        v_old = v.copy()
        # for each state
        for s in range(n**2):
            # for each action
            q_final, a_final = -np.inf, -1
            for a in range(5):
                q = calculate_q(n, s, a, v, reward, action_choice, gamma, forbidden, target)
                if q > q_final:
                    q_final = q
                    a_final = a
            pi[s] = a_final
            v[s] = q_final
        iteration += 1
            
    return v, pi, iteration


def print_policy(pi, n, action_str):
    res = np.zeros((n, n), dtype=str)
    for i in range(n):
        for j in range(n):
            s = i * n + j
            action = pi[s]
            res[i, j] = action_str[action]
    return res


def print_state_value(v, n):
    res = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = i * n + j
            res[i, j] = v[s]
    return res


def main():
    reward = {
        'boundary': -1,
        'target': 1,
        'forbidden': -1,
        'otherwise': 0
    }
    action_choice = {
        0: (0, 1),   # right
        1: (-1, 0),   # up
        2: (0, -1),  # left
        3: (1, 0),  # down
        4: (0, 0)    # stay
    }
    action_str = {
        0: '→',   # right
        1: '↑',   # up
        2: '←',   # left
        3: '↓',   # down
        4: '◦'  # stay
    }
    gamma = 0.9

    # n = 2
    # forbidden = [1]
    # target = 3
    # n_states = n * n

    n = 5
    forbidden = [6, 7, 12, 16, 18, 21]
    target = 17
    n_states = n * n

    v = np.zeros(n_states)
    pi = np.zeros(n_states, dtype=int)
    v, pi, iteration = value_iteration(n, v, pi, reward, action_choice, forbidden, target, gamma=gamma)
    print(f"Value Iteration completed in {iteration} iterations.")
    print("Optimal Policy:")
    print(print_policy(pi, n, action_str))
    print("State Values:")
    print(print_state_value(v, n))


if __name__ == "__main__":
    main()