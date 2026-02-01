import numpy as np
from scipy.special import gamma as gamma_func
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
import config

# --- HELPER FUNCTIONS ---
def Levy(dim):
    beta = 1.5
    # Tính sigma
    num = gamma_func(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma_func((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)
    
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    
    # [FIX 4] Levy scale: Trả về giá trị thô, việc scale sẽ làm ở trong thuật toán
    step = u / (np.abs(v) ** (1 / beta))
    return step

# --- ALGORITHM CORE (RBMO & SBOA) - UPDATED ---

def SBOA(N, steps_to_run, dim, X, y, fitness_func, PopPos_in, PopFit_in, global_iter_start, global_max_iter, **kwargs):
    """
    SBOA Optimized:
    - [FIX 6] Trả về best_x, PopPos, PopFit để không phải tính lại.
    - [FIX 3] Nhận steps_to_run (actual_k) để không chạy lố.
    """
    lb, ub = -10, 10
    
    # [FIX 9] Copy population để an toàn, tránh side-effect
    PopPos = PopPos_in.copy() if PopPos_in is not None else np.random.rand(N, dim) * (ub - lb) + lb
    
    # Tận dụng Fitness cũ nếu có
    if PopFit_in is None:
        pop_fit = np.array([fitness_func(p, X, y, config, **kwargs)[0] for p in PopPos])
    else:
        pop_fit = PopFit_in.copy()
    
    best_idx = np.argmin(pop_fit)
    best_f = pop_fit[best_idx]
    best_x = PopPos[best_idx, :].copy()

    for it in range(steps_to_run):
        curr_global = global_iter_start + it + 1
        ratio = curr_global / global_max_iter 
        
        for i in range(N):
            # [FIX 7] Chọn r1, r2 tối ưu hơn
            choices = np.delete(np.arange(N), i)
            r1, r2 = np.random.choice(choices, 2, replace=False)

            # Phase 1: Hunting Strategies
            if curr_global <= global_max_iter / 3:
                new_pos = PopPos[i, :] + (PopPos[r1, :] - PopPos[r2, :]) * np.random.rand(dim)
            elif curr_global <= (2 * global_max_iter / 3):
                new_pos = best_x + np.exp(ratio**4) * (np.random.randn(dim) - 0.5) * (best_x - PopPos[i, :])
            else:
                # [FIX 4] Scale Levy flight để tránh nổ giá trị (nhân 0.01)
                levy_step = 0.01 * Levy(dim) 
                new_pos = best_x + (1 - ratio)**(2*ratio) * PopPos[i, :] * levy_step
            
            # Boundary check
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config, **kwargs)
            
            # Greedy Selection
            if new_fit < pop_fit[i]: 
                PopPos[i, :] = new_pos
                pop_fit[i] = new_fit

        # Phase 2: Escape Strategies
        for i in range(N):
            if np.random.rand() < 0.5:
                r2_val = np.random.rand(dim) 
                new_pos = np.random.uniform(0, 2, dim) * best_x + (2 * r2_val - 1) * ((1 - ratio)**2) * PopPos[i, :]
            else:
                rand_idx = np.random.randint(0, N)
                # Đổi công thức nhiễu nhẹ lại một chút để ổn định hơn
                new_pos = np.random.uniform(0, 2, dim) * PopPos[i, :] + \
                          np.random.randn(dim) * (PopPos[rand_idx, :] - PopPos[i, :])
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config, **kwargs)
            
            if new_fit < pop_fit[i]: 
                PopPos[i, :] = new_pos
                pop_fit[i] = new_fit

        # Cập nhật Best Global của SBOA
        current_best_val = np.min(pop_fit)
        if current_best_val < best_f:
            best_idx = np.argmin(pop_fit)
            best_f = current_best_val
            best_x = PopPos[best_idx, :].copy()
            
    # [FIX 6] Trả về đầy đủ
    return best_f, best_x, PopPos, pop_fit

def RBMO(N, steps_to_run, dim, X, y, fitness_func, PopPos_in, PopFit_in, global_iter_start, global_max_iter, epsilon=0.5, **kwargs):
    lb, ub = -10, 10
    
    PopPos = PopPos_in.copy() if PopPos_in is not None else np.random.rand(N, dim) * (ub - lb) + lb
    
    if PopFit_in is None:
        pop_fit = np.array([fitness_func(p, X, y, config, **kwargs)[0] for p in PopPos])
    else:
        pop_fit = PopFit_in.copy()
    
    best_idx = np.argmin(pop_fit)
    best_f = pop_fit[best_idx]
    best_x = PopPos[best_idx, :].copy()

    for it in range(steps_to_run):
        curr_global = global_iter_start + it + 1
        ratio = curr_global / global_max_iter 

        # [FIX 5] Xử lý q_val an toàn khi N nhỏ
        # Đảm bảo low < high cho randint. Nếu N <= 2 thì q_val min là 1
        low_bound = 2
        high_bound = max(3, int(N + 1)) # Đảm bảo luôn lớn hơn 2
        
        # Phase 1: Cooperative
        for i in range(N):
            if np.random.rand() < epsilon: 
                q_val = np.random.randint(low_bound, min(6, high_bound))
            else: 
                q_val = np.random.randint(min(10, high_bound-1), high_bound)
                
            # Chọn m_idx an toàn
            pool = np.delete(np.arange(N), i)
            actual_q = min(q_val, len(pool))
            if actual_q < 1: actual_q = 1
            
            m_idx = np.random.choice(pool, actual_q, replace=False)
            sum_Xm = np.mean(PopPos[m_idx], axis=0)
            
            # Chọn 1 cá thể ngẫu nhiên khác i
            rand_other = np.random.choice(pool)
            
            new_pos = PopPos[i,:] + (sum_Xm - PopPos[rand_other, :]) * np.random.rand()
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config, **kwargs)
            
            if new_fit < pop_fit[i]:
                PopPos[i, :] = new_pos
                pop_fit[i] = new_fit
                if new_fit < best_f: best_f, best_x = new_fit, new_pos.copy()
        
        # Phase 2: Scrounging
        for i in range(N):
            if np.random.rand() < epsilon: 
                 q_val = np.random.randint(low_bound, min(6, high_bound))
            else: 
                 q_val = np.random.randint(min(10, high_bound-1), high_bound)
            
            pool = np.delete(np.arange(N), i)
            actual_q = min(q_val, len(pool))
            m_idx = np.random.choice(pool, actual_q, replace=False)
            sum_Xm = np.mean(PopPos[m_idx], axis=0)
            
            CF = (1 - ratio)**(2*ratio) 
            new_pos = best_x + CF * (sum_Xm - PopPos[i, :]) * np.random.randn(dim)
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config, **kwargs)
            
            if new_fit < pop_fit[i]:
                PopPos[i, :] = new_pos
                pop_fit[i] = new_fit
                if new_fit < best_f: best_f, best_x = new_fit, new_pos.copy()

        # Phase 3: Self-learning
        for i in range(N):
            if np.random.rand() < 0.5:
                rand_other = np.random.randint(0, N)
                new_pos = best_x + np.random.rand() * (best_x - PopPos[i, :]) + \
                          np.random.randn() * (PopPos[rand_other, :] - PopPos[i, :])
            else:
                new_pos = PopPos[i, :] * (1 + np.random.randn(dim) * (1 - ratio))
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config, **kwargs)
            
            if new_fit < pop_fit[i]:
                PopPos[i, :] = new_pos
                pop_fit[i] = new_fit
                if new_fit < best_f: best_f, best_x = new_fit, new_pos.copy()
        
    return best_f, best_x, PopPos, pop_fit

# --- MAIN HYBRID RUNNER ---

def run_rbmo_sboa(X, y, fitness_func, **kwargs):
    dim = X.shape[1]
    N = config.POP_SIZE // 2
    T, k = config.MAX_ITER, config.EXCHANGE_INTERVAL
    
    # Khởi tạo
    pop_r, pop_s = None, None
    fit_r, fit_s = None, None # Lưu Fitness để truyền qua lại
    
    best_overall_f = np.inf
    best_overall_z = None
    fitness_history = []
    
    no_improve_count = 0
    last_best_f = np.inf

    # Vòng lặp chính chạy theo Block
    for curr_it in range(0, T, k):
        # [FIX 3] Tính số vòng lặp thực tế cho block này để không vượt quá T
        actual_k = min(k, T - curr_it)
        if actual_k <= 0: break

        # 1. Chạy RBMO
        f_r, x_r, pop_r, fit_r = RBMO(N, actual_k, dim, X, y, fitness_func, 
                                      PopPos_in=pop_r, PopFit_in=fit_r,
                                      global_iter_start=curr_it, global_max_iter=T, **kwargs)
                          
        # 2. Chạy SBOA
        f_s, x_s, pop_s, fit_s = SBOA(N, actual_k, dim, X, y, fitness_func, 
                                      PopPos_in=pop_s, PopFit_in=fit_s,
                                      global_iter_start=curr_it, global_max_iter=T, **kwargs)

        # 3. Information Exchange (Trao đổi thông tin) - [FIX 1: LOGIC TRAO ĐỔI AN TOÀN]
        
        # Tìm cá thể tốt nhất hiện tại của mỗi quần thể
        idx_best_s = np.argmin(fit_s)
        val_best_s = fit_s[idx_best_s]
        vec_best_s = pop_s[idx_best_s].copy() # Copy để không bị tham chiếu
        
        idx_best_r = np.argmin(fit_r)
        val_best_r = fit_r[idx_best_r]
        vec_best_r = pop_r[idx_best_r].copy()

        # Chọn vị trí ngẫu nhiên để thay thế (tránh thay thế chính cá thể tốt nhất của mình nếu có thể)
        rand_idx_r = np.random.randint(0, N)
        rand_idx_s = np.random.randint(0, N)

        # Thực hiện trao đổi: Ghi đè Gen và cập nhật Fitness tương ứng
        # (Không cần tính lại fitness vì ta biết gen đó có fitness bao nhiêu từ nguồn)
        pop_r[rand_idx_r] = vec_best_s
        fit_r[rand_idx_r] = val_best_s
        
        pop_s[rand_idx_s] = vec_best_r
        fit_s[rand_idx_s] = val_best_r

        # 4. Cập nhật Best Global - [FIX 2: LOGIC CHỌN BEST OVERALL]
        # So sánh trực tiếp f_r và f_s trả về từ hàm con
        if f_r < f_s:
            current_min_f = f_r
            current_min_z = x_r
        else:
            current_min_f = f_s
            current_min_z = x_s
            
        if current_min_f < best_overall_f:
            best_overall_f = current_min_f
            best_overall_z = current_min_z.copy() # [FIX] Copy để an toàn
        
        # 5. Lưu lịch sử [FIX 8: Lưu dạng block nhưng chính xác hơn]
        for _ in range(actual_k): fitness_history.append(best_overall_f)
        
        if config.PRINT_PROGRESS: 
            print(f"> [{config.ALGO_NAME}] Iter {curr_it + actual_k}/{T} | Best Fitness: {best_overall_f:.5f}")

        # 6. Early Stopping
        if best_overall_f < last_best_f:
            last_best_f, no_improve_count = best_overall_f, 0
        else:
            no_improve_count += actual_k
        
        if no_improve_count >= config.PATIENCE:
            # Fill nốt history nếu dừng sớm
            remaining = T - len(fitness_history)
            if remaining > 0: fitness_history.extend([best_overall_f] * remaining)
            break

    return best_overall_z, best_overall_f, fitness_history

# --- GA & PSO RUNNERS (GIỮ NGUYÊN) ---
def run_ga(X, y, fitness_func, **kwargs):
    dim = X.shape[1]
    N, T = config.POP_SIZE, config.MAX_ITER
    lb, ub = -10, 10
    mutation_rate = 0.1
    
    PopPos = np.random.uniform(lb, ub, (N, dim))
    PopFit = np.array([fitness_func(p, X, y, config, **kwargs)[0] for p in PopPos])
    
    best_idx = np.argmin(PopFit)
    best_f, best_z = PopFit[best_idx], PopPos[best_idx].copy()
    history = []
    
    no_improve_count = 0
    last_best_f = best_f

    for it in range(T):
        new_pop = []
        for _ in range(N): 
            candidates = np.random.randint(0, N, 3)
            best_cand = candidates[np.argmin(PopFit[candidates])]
            new_pop.append(PopPos[best_cand].copy())
        new_pop = np.array(new_pop)
        
        next_gen = []
        for i in range(0, N, 2): 
            p1, p2 = new_pop[i], new_pop[(i+1)%N]
            pt = np.random.randint(1, dim)
            c1 = np.concatenate((p1[:pt], p2[pt:]))
            c2 = np.concatenate((p2[:pt], p1[pt:]))
            for child in [c1, c2]: 
                mask = np.random.rand(dim) < mutation_rate
                child[mask] = np.random.uniform(lb, ub, np.sum(mask))
                next_gen.append(np.clip(child, lb, ub))
        
        PopPos = np.array(next_gen[:N])
        PopFit = np.array([fitness_func(p, X, y, config, **kwargs)[0] for p in PopPos])

        curr_best = np.min(PopFit)
        if curr_best < best_f:
            best_f = curr_best
            best_z = PopPos[np.argmin(PopFit)].copy()
        
        history.append(best_f)
        if config.PRINT_PROGRESS and (it+1)%10==0: print(f"> [GA] Iter {it+1}/{T} | Best: {best_f:.5f}")

        if best_f < last_best_f: last_best_f = best_f; no_improve_count = 0
        else: no_improve_count += 1
        if no_improve_count >= config.PATIENCE:
            history.extend([best_f] * (T - len(history)))
            break
            
    return best_z, best_f, history

def run_pso(X, y, fitness_func, **kwargs):
    dim = X.shape[1]
    N, T = config.POP_SIZE, config.MAX_ITER
    lb, ub = -10, 10
    w_max, w_min, c1, c2 = 0.9, 0.4, 2.0, 2.0
    
    PopPos = np.random.uniform(lb, ub, (N, dim))
    PopVel = np.zeros((N, dim))
    PopFit = np.array([fitness_func(p, X, y, config, **kwargs)[0] for p in PopPos])
    PBestPos, PBestFit = PopPos.copy(), PopFit.copy()
    GBestPos, GBestFit = PopPos[np.argmin(PopFit)].copy(), np.min(PopFit)
    history = []
    
    no_improve_count = 0
    last_best_f = GBestFit

    for it in range(T):
        w = w_max - it * ((w_max - w_min) / T)
        for i in range(N):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            PopVel[i] = w*PopVel[i] + c1*r1*(PBestPos[i]-PopPos[i]) + c2*r2*(GBestPos-PopPos[i])
            PopPos[i] = np.clip(PopPos[i] + PopVel[i], lb, ub)
            fit, _ = fitness_func(PopPos[i], X, y, config, **kwargs)
            
            if fit < PBestFit[i]:
                PBestFit[i], PBestPos[i] = fit, PopPos[i].copy()
                if fit < GBestFit: GBestFit, GBestPos = fit, PopPos[i].copy()
        
        history.append(GBestFit)
        if config.PRINT_PROGRESS and (it+1)%10==0: print(f"> [PSO] Iter {it+1}/{T} | Best: {GBestFit:.5f}")

        if GBestFit < last_best_f: last_best_f = GBestFit; no_improve_count = 0
        else: no_improve_count += 1
        if no_improve_count >= config.PATIENCE:
            history.extend([GBestFit] * (T - len(history)))
            break
            
    return GBestPos, GBestFit, history