import math
import matplotlib.pyplot as plt

def cosine_annealing_warmup_restarts(total_steps, first_cycle_steps=50000, cycle_mult=1.5,
                                      max_lr=2e-4, min_lr=1e-7, warmup_steps=2000, gamma=0.8):
    lr_list = []
    cur_cycle_steps = first_cycle_steps
    step_in_cycle = -1
    cycle = 0
    base_max_lr = max_lr

    for epoch in range(total_steps):
        # 周期调度
        step_in_cycle += 1
        if step_in_cycle >= cur_cycle_steps:
            cycle += 1
            step_in_cycle = step_in_cycle - cur_cycle_steps
            cur_cycle_steps = int((cur_cycle_steps - warmup_steps) * cycle_mult) + warmup_steps

        max_lr = base_max_lr * (gamma ** cycle)

        if step_in_cycle == -1:
            lr = min_lr
        elif step_in_cycle < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * step_in_cycle / warmup_steps
        else:
            lr = min_lr + (max_lr - min_lr) * \
                (1 + math.cos(math.pi * (step_in_cycle - warmup_steps) /
                              (cur_cycle_steps - warmup_steps))) / 2

        lr_list.append(lr)

    return lr_list

# 配置参数
total_steps = 300000
lr_values = cosine_annealing_warmup_restarts(total_steps)

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(lr_values)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("CosineAnnealingWarmupRestarts Learning Rate Schedule")
plt.grid(True)
plt.tight_layout()
plt.show()
