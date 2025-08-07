import numpy as np
from tqdm import tqdm


def generate_episode(pi, n, action_choice, reward, forbidden, target, length):
    s = np.zeros(length, dtype=int)
    a = np.zeros(length, dtype=int)
    r = np.zeros(length, dtype=float)

    s[0] = np.random.choice(n * n)

    for i in tqdm(range(1, length+1), desc="Generating episodes", unit="step"):
        s_t_1 = s[i-1]
        a_t_1 = np.random.choice(len(pi[s_t_1]))#, p=pi[s_t_1])
        a[i-1] = a_t_1
        offset_i, offset_j = action_choice[a_t_1]
        next_i, next_j = s_t_1 // n + offset_i, s_t_1 % n + offset_j
        if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
            r[i-1] = reward['boundary']
            if i < length:
                s[i] = s_t_1
        else:
            s_t = next_i * n + next_j
            if s_t in forbidden:
                r[i-1] = reward['forbidden']
            elif s_t in target:
                r[i-1] = reward['target']
            else:
                r[i-1] = reward['otherwise']
            
            if i < length:
                s[i] = s_t

    return s, a, r


def q_to_v(q, pi, n):
    v = np.zeros(n * n)
    for s in range(n * n):
        v[s] = np.sum(pi[s] * q[s])
    return v


def MC_epsilon_greedy(n, pi, length, gamma, action_choice, reward, forbidden, target, epsilon, tolerance=0.5):
    n_states = n * n
    returns = np.zeros((n_states, len(action_choice)))
    nums = np.zeros((n_states, len(action_choice)))
    q = np.random.random(size=(n_states, len(action_choice)))

    norm_list = []

    while True:

        if len(norm_list) >= 3 and max(norm_list[-3:]) < tolerance:
            break
        
        q_old = q.copy()
        s, a, r = generate_episode(pi, n, action_choice, reward, forbidden, target, length)  
        g = 0

        for i in tqdm(range(length - 1, -1, -1), desc="Updating Q and pi", unit="step"):
            s_t_1 = s[i]
            a_t_1 = a[i]
            r_t = r[i]

            g = gamma * g + r_t
            returns[s_t_1, a_t_1] += g
            nums[s_t_1, a_t_1] += 1

            # policy evaluation
            q[s_t_1, a_t_1] = returns[s_t_1, a_t_1] / nums[s_t_1, a_t_1]
            # policy improvement
            a_star = np.argmax(q[s_t_1])

            # epsilon-greedy policy
            pi_s_t_1 = np.zeros(len(action_choice))
            pi_s_t_1[:] = epsilon / len(action_choice)
            pi_s_t_1[a_star] = 1 - epsilon + epsilon / len(action_choice)

            pi[s_t_1] = pi_s_t_1.copy()
        
        print(np.linalg.norm(q - q_old, ord=1))
        norm_list.append(np.linalg.norm(q - q_old, ord=1))
    
    v = q_to_v(q, pi, n)

    return pi, v, q, returns, nums


def main():
    reward = {
        'boundary': -1,
        'target': +1,
        'forbidden': -1,
        'otherwise': 0
    }
    action_choice = [
        (0, 1),   # right
        (-1, 0),   # up
        (0, -1),  # left
        (1, 0),  # down
        (0, 0)    # stay
    ]

    # 网格信息
    n = 5
    forbidden = [6, 7, 12, 16, 18, 21]
    target = [17]

    # 折扣因子与探索率
    gamma = 0.9
    epsilon = 0.1

    # episode长度
    length = 1000

    # 所有状态的初始策略均匀分布
    pi = np.full((n * n, len(action_choice)), 1 / len(action_choice))
    # pi = np.array([
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10], 
    #     [0.10, 0.10, 0.10, 0.60, 0.10],
    #     [0.10, 0.60, 0.10, 0.10, 0.10],
    #     [0.10, 0.60, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.10, 0.10, 0.10, 0.60, 0.10],
    #     [0.10, 0.60, 0.10, 0.10, 0.10],
    #     [0.10, 0.10, 0.60, 0.10, 0.10],
    #     [0.10, 0.10, 0.10, 0.60, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.10, 0.10, 0.10, 0.60, 0.10],
    #     [0.10, 0.60, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.10, 0.10, 0.10, 0.10, 0.60],
    #     [0.10, 0.10, 0.60, 0.10, 0.10],
    #     [0.10, 0.10, 0.10, 0.60, 0.10],
    #     [0.10, 0.60, 0.10, 0.10, 0.10],
    #     [0.60, 0.10, 0.10, 0.10, 0.10],
    #     [0.10, 0.60, 0.10, 0.10, 0.10],
    #     [0.10, 0.10, 0.60, 0.10, 0.10],
    #     [0.10, 0.10, 0.60, 0.10, 0.10],
    # ])

    pi, v, q, returns, nums = MC_epsilon_greedy(n, pi, length, gamma, action_choice, reward, forbidden, target, epsilon)

    print(f"Final Policy: {pi}")
    print(f"Final State Value: {v}")
    print(f"Final Q-values: {q}")


if __name__ == "__main__":
    main()