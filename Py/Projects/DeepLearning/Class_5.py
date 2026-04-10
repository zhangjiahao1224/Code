from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 第 5 课：深度强化学习
#
# 这份脚本对应 MIT 6.S191 L5 中最核心的两条主线：
# 1. 价值学习：用 Deep Q-Network (DQN) 解决离散动作任务。
# 2. 策略学习：用 Policy Gradient 解决连续动作任务。
#
# 为了让脚本完全自包含、方便教学阅读，这里没有依赖 gym 等外部环境，
# 而是直接用纯 Python / Torch 构造了两个最小环境：
# - LineWorldEnv：离散状态、离散动作，演示 DQN。
# - ContinuousDriveEnv：连续控制，演示策略梯度。
#
# 目标不是追求高性能 benchmark，而是把强化学习里最重要的几个概念
# 串成一条可以直接运行和阅读的链路：
# 状态、动作、奖励、Bellman target、经验回放、折扣回报、随机策略更新。
# ------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽量让实验结果可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """统计模型中的可训练参数量。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """计算折扣回报 G_t = r_t + gamma * G_{t+1}。"""

    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        # 从后往前递推，是实现折扣回报最直接的方式：
        # G_t = r_t + gamma * G_{t+1}
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def moving_average(values: list[float], window: int = 20) -> float:
    """返回最近 `window` 个数值的平均值。"""

    if not values:
        return 0.0
    tail = values[-window:]
    return sum(tail) / len(tail)


def save_text_report(text: str, path: str) -> str:
    """把文本报告保存到磁盘。"""

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return str(file)


def _load_pillow_font(size: int = 16):
    """尽量加载支持中文的字体；如果失败，再退回默认字体。"""

    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidate_fonts = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
        "C:/Windows/Fonts/STSONG.TTF",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in candidate_fonts:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue

    return ImageFont.load_default()


def _ascii_fallback(text: str) -> str:
    """在极端字体缺失时，把文本降级为 ASCII，避免绘图阶段报错。"""

    fallback = text.encode("ascii", errors="ignore").decode("ascii")
    return fallback if fallback else "text"


def _safe_draw_text(draw, position: tuple[int, int], text: str, font, fill: str = "black") -> None:
    """安全绘制文本；若字体不支持中文，就自动退回 ASCII。"""

    try:
        draw.text(position, text, fill=fill, font=font)
    except UnicodeEncodeError:
        draw.text(position, _ascii_fallback(text), fill=fill, font=font)


def save_curve_plot(
    values: list[float],
    path: str,
    title: str,
    y_label: str,
    width: int = 960,
    height: int = 540,
) -> str | None:
    """如果环境里有 Pillow，就保存一张轻量级折线图。"""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("未安装 Pillow，跳过曲线图导出。")
        return None

    if not values:
        return None

    # 整个仓库尽量保持轻依赖，因此这里直接用 Pillow 画图，
    # 不额外依赖 matplotlib 之类的可视化库。
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_pillow_font(size=16)

    margin_left = 70
    margin_right = 30
    margin_top = 45
    margin_bottom = 70
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    _safe_draw_text(draw, (margin_left, 15), title, font=font, fill="black")
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)
    _safe_draw_text(draw, (20, plot_top - 5), y_label, font=font, fill="gray")
    _safe_draw_text(draw, (plot_right - 45, plot_bottom + 12), "step", font=font, fill="gray")

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0

    def project(idx: int, value: float) -> tuple[int, int]:
        x_ratio = idx / max(len(values) - 1, 1)
        y_ratio = (value - vmin) / (vmax - vmin)
        x = plot_left + int(x_ratio * (plot_right - plot_left))
        y = plot_bottom - int(y_ratio * (plot_bottom - plot_top))
        return x, y

    points = [project(i, v) for i, v in enumerate(values)]
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(33, 150, 243))
    else:
        draw.line(points, fill=(33, 150, 243), width=3)

    _safe_draw_text(draw, (plot_left, plot_bottom + 12), "0", font=font, fill="gray")
    _safe_draw_text(draw, (plot_right - 12, plot_bottom + 12), str(len(values)), font=font, fill="gray")
    _safe_draw_text(draw, (plot_left - 50, plot_bottom - 8), f"{vmin:.2f}", font=font, fill="gray")
    _safe_draw_text(draw, (plot_left - 50, plot_top - 8), f"{vmax:.2f}", font=font, fill="gray")

    img.save(file)
    return str(file)


def save_training_history_report(
    returns: list[float],
    successes: list[float],
    path: str,
    title: str,
    window: int = 20,
) -> str:
    """保存逐回合训练记录，方便回看回报和成功率变化。"""

    lines = [title, ""]
    lines.append("episode | return | success | avg_return(window) | avg_success(window)")

    for idx, (episode_return, success) in enumerate(zip(returns, successes), start=1):
        start = max(0, idx - window)
        return_window = returns[start:idx]
        success_window = successes[start:idx]
        avg_return = sum(return_window) / len(return_window)
        avg_success = sum(success_window) / len(success_window)
        lines.append(
            f"{idx:03d} | {episode_return:+.4f} | {success:.0f} | "
            f"{avg_return:+.4f} | {avg_success:.4f}"
        )

    return save_text_report("\n".join(lines), path)


class LineWorldEnv:
    """
    一个给 DQN 用的极简离散强化学习环境。

    状态：
    - 一条 1D 线上共有 `num_states` 个位置；
    - 观测用 one-hot 向量表示当前位置。

    动作：
    - 0 = 向左移动
    - 1 = 原地不动
    - 2 = 向右移动

    奖励：
    - 抵达最右侧目标点，奖励 +1.0
    - 掉进最左侧陷阱，奖励 -1.0
    - 其余时间每步都有轻微时间惩罚 -0.02
    """

    def __init__(self, num_states: int = 9, max_steps: int = 16):
        self.num_states = num_states
        self.max_steps = max_steps
        self.goal_state = num_states - 1
        self.pit_state = 0
        self.position = 0
        self.steps = 0

    def _obs(self) -> torch.Tensor:
        obs = torch.zeros(self.num_states, dtype=torch.float32)
        obs[self.position] = 1.0
        return obs

    def reset(self) -> torch.Tensor:
        self.position = random.randint(2, self.num_states - 3)
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, float | int | bool]]:
        self.steps += 1
        move = {0: -1, 1: 0, 2: 1}[int(action)]
        self.position = max(self.pit_state, min(self.goal_state, self.position + move))

        reward = -0.02
        done = False
        success = False

        if self.position == self.goal_state:
            reward = 1.0
            done = True
            success = True
        elif self.position == self.pit_state:
            reward = -1.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -0.25
            done = True

        info = {
            "position": self.position,
            "success": success,
            "steps": self.steps,
        }
        return self._obs(), reward, done, info


class ReplayBuffer:
    """DQN 使用的简单经验回放缓冲区。"""

    def __init__(self, capacity: int = 5000):
        self.buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append((state.clone(), int(action), float(reward), next_state.clone(), float(done)))

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 随机采样而不是按时间顺序取数据，是经验回放的关键：
        # 它能削弱轨迹相邻样本的强相关性，让训练更稳定。
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )


class DQN(nn.Module):
    """一个小型 MLP，把状态映射成各个动作的 Q 值。"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_dqn_action(
    model: nn.Module,
    state: torch.Tensor,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    """使用 epsilon-greedy 规则选动作。"""

    # 探索-利用权衡：
    # - 以 epsilon 的概率随机探索；
    # - 否则就利用当前 Q 网络给出的最优动作。
    if random.random() < epsilon:
        return random.randrange(action_dim)

    with torch.no_grad():
        q_values = model(state.unsqueeze(0).to(device))
        return int(q_values.argmax(dim=1).item())


def optimize_dqn(
    q_net: nn.Module,
    target_net: nn.Module,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    """基于 Bellman target 执行一次 DQN 参数更新。"""

    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)
    # q_net(states) 会给出每个动作的 Q 值；
    # gather 只取“这次样本里真实执行过的动作”对应的 Q(s, a)。
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        # DQN 的 Bellman target：
        # target = r + gamma * max_a' Q_target(s', a')
        # 如果已经终止，就不再加未来回报。
        targets = rewards + gamma * (1.0 - dones) * next_q

    # 让当前 Q 预测逼近 Bellman target。
    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_dqn(
    model: nn.Module,
    env: LineWorldEnv,
    episodes: int,
    device: torch.device,
) -> tuple[float, float]:
    """评估 DQN 的贪心策略表现。"""

    returns: list[float] = []
    successes = 0

    for _ in range(episodes):
        # 评估时不再随机探索，而是固定走当前最优动作，
        # 这样才能客观反映模型现在的真实水平。
        state = env.reset()
        done = False
        episode_return = 0.0
        info: dict[str, float | int | bool] = {"success": False}

        while not done:
            q_values = model(state.unsqueeze(0).to(device))
            action = int(q_values.argmax(dim=1).item())
            state, reward, done, info = env.step(action)
            episode_return += reward

        returns.append(episode_return)
        if bool(info.get("success", False)):
            successes += 1

    avg_return = sum(returns) / len(returns)
    success_rate = successes / len(returns)
    return avg_return, success_rate


def save_dqn_policy_report(model: nn.Module, env: LineWorldEnv, device: torch.device) -> str:
    """把每个状态下学到的贪心动作写入文本报告。"""

    arrows = {0: "left", 1: "stay", 2: "right"}
    lines = [
        "DQN 在 LineWorld 上学到的策略",
        f"陷阱位置={env.pit_state}, 目标位置={env.goal_state}",
        "",
    ]

    with torch.no_grad():
        for position in range(env.num_states):
            state = torch.zeros(env.num_states, dtype=torch.float32)
            state[position] = 1.0
            q_values = model(state.unsqueeze(0).to(device)).squeeze(0).cpu()
            action = int(q_values.argmax().item())
            lines.append(
                f"状态={position:02d} | 贪心动作={arrows[action]:>5} | "
                f"Q={q_values.tolist()}"
            )

    return save_text_report("\n".join(lines), "artifacts/dqn_policy_report.txt")


def save_dqn_rollout_report(model: nn.Module, env: LineWorldEnv, device: torch.device) -> str:
    """保存一次 DQN 贪心 rollout，方便检查学到的行为。"""

    arrows = {0: "left", 1: "stay", 2: "right"}
    state = env.reset()
    done = False
    lines = ["DQN 贪心 rollout 轨迹", ""]
    step_idx = 0
    total_reward = 0.0

    while not done:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0).to(device)).squeeze(0).cpu()
        action = int(q_values.argmax().item())
        next_state, reward, done, info = env.step(action)
        lines.append(
            f"步数={step_idx:02d} | 当前位置={int(state.argmax().item())} | "
            f"动作={arrows[action]} | 奖励={reward:+.2f} | 下一位置={int(next_state.argmax().item())}"
        )
        state = next_state
        total_reward += reward
        step_idx += 1

    lines.append("")
    lines.append(f"结束={done} | 是否成功={bool(info.get('success', False))} | 总回报={total_reward:.2f}")
    return save_text_report("\n".join(lines), "artifacts/dqn_rollout.txt")


def save_dqn_checkpoint(model: nn.Module, env: LineWorldEnv, path: str = "artifacts/dqn_lineworld.pt") -> str:
    """保存 DQN checkpoint。"""

    ckpt = {
        "state_dict": model.state_dict(),
        "state_dim": env.num_states,
        "action_dim": 3,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, file)
    return str(file)


def load_dqn_checkpoint(path: str, device: torch.device) -> DQN:
    """加载 DQN checkpoint。"""

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DQN(state_dim=int(ckpt["state_dim"]), action_dim=int(ckpt["action_dim"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class ContinuousDriveEnv:
    """
    一个给基础策略梯度使用的极简连续控制环境。

    观测：
    - 当前位置 position
    - 当前速度 velocity
    - 与目标的有符号距离 target - position

    动作：
    - 连续加速度，范围在 [-1, 1]

    奖励设计：
    - 如果离目标更近，就给正奖励
    - 动作过大有轻微惩罚
    - 到达目标附近且速度足够小，会给额外成功奖励
    """

    def __init__(self, max_steps: int = 35):
        self.max_steps = max_steps
        self.target = 0.85
        self.position = 0.0
        self.velocity = 0.0
        self.steps = 0

    def _obs(self) -> torch.Tensor:
        return torch.tensor(
            [self.position, self.velocity, self.target - self.position],
            dtype=torch.float32,
        )

    def reset(self) -> torch.Tensor:
        self.position = random.uniform(-1.0, 0.1)
        self.velocity = random.uniform(-0.05, 0.05)
        self.steps = 0
        return self._obs()

    def step(self, action: float) -> tuple[torch.Tensor, float, bool, dict[str, float | int | bool]]:
        self.steps += 1
        action = max(-1.0, min(1.0, float(action)))
        prev_distance = abs(self.target - self.position)

        self.velocity = max(-1.2, min(1.2, 0.86 * self.velocity + 0.36 * action))
        self.position = max(-1.5, min(1.5, self.position + 0.11 * self.velocity))
        new_distance = abs(self.target - self.position)

        reward = 2.0 * (prev_distance - new_distance) - 0.02 * (action**2)
        success = new_distance < 0.05 and abs(self.velocity) < 0.10
        done = False

        if success:
            reward += 2.0
            done = True
        elif self.steps >= self.max_steps:
            reward -= 0.25 * new_distance
            done = True

        info = {
            "position": self.position,
            "velocity": self.velocity,
            "distance": new_distance,
            "success": success,
            "steps": self.steps,
        }
        return self._obs(), reward, done, info


class GaussianPolicy(nn.Module):
    """用于连续控制的高斯策略网络。"""

    def __init__(self, state_dim: int, action_dim: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = torch.tanh(self.mean_head(h))
        log_std = self.log_std.clamp(-2.0, 0.8).expand_as(mean)
        std = torch.exp(log_std)
        return mean, std


def sample_policy_action(
    policy: nn.Module,
    state: torch.Tensor,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    """从策略中采样动作，或按均值确定性地选择动作。"""

    state_batch = state.unsqueeze(0).to(device)
    mean, std = policy(state_batch)
    dist = torch.distributions.Normal(mean, std)

    if deterministic:
        # 测试或 rollout 展示时，直接取均值动作更稳定。
        action_tensor = mean
        return float(action_tensor.squeeze(0).squeeze(-1).item()), None, None

    # 训练时保留随机采样，这样策略才有探索能力。
    raw_action = dist.sample()
    clipped_action = raw_action.clamp(-1.0, 1.0)
    log_prob = dist.log_prob(raw_action).sum(dim=-1).squeeze(0)
    entropy = dist.entropy().sum(dim=-1).squeeze(0)
    return float(clipped_action.squeeze(0).squeeze(-1).item()), log_prob, entropy


def train_policy_gradient(
    policy: nn.Module,
    env: ContinuousDriveEnv,
    device: torch.device,
    episodes: int = 320,
    gamma: float = 0.97,
    lr: float = 2e-3,
    entropy_coef: float = 1e-3,
) -> tuple[list[float], list[float]]:
    """用带折扣回报的 REINFORCE 训练一个基础策略梯度智能体。"""

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    return_history: list[float] = []
    success_history: list[float] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []
        info: dict[str, float | int | bool] = {"success": False}

        while not done:
            action, log_prob, entropy = sample_policy_action(policy, state, device, deterministic=False)
            next_state, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            state = next_state

        # 一条完整轨迹结束后，先计算每一步的折扣回报 G_t。
        returns = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32, device=device)
        # 再做标准化，减少方差，让 REINFORCE 训练更稳一些。
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        # REINFORCE 的核心损失：
        # loss = - sum_t log pi(a_t | s_t) * G_t
        # 额外减去少量熵项，是为了鼓励早期探索，避免过快塌缩成单一动作。
        loss = -(log_probs_tensor * returns).sum() - entropy_coef * entropies_tensor.sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

        episode_return = float(sum(rewards))
        return_history.append(episode_return)
        success_history.append(1.0 if bool(info.get("success", False)) else 0.0)

        if episode == 1 or episode % 25 == 0 or episode == episodes:
            print(
                f"[PG] 回合={episode:03d} | "
                f"本回合回报={episode_return:+.3f} | "
                f"最近20回合平均回报={moving_average(return_history, 20):+.3f} | "
                f"最近20回合成功率={moving_average(success_history, 20):.2f}"
            )

    return return_history, success_history


@torch.no_grad()
def evaluate_policy_gradient(
    policy: nn.Module,
    env: ContinuousDriveEnv,
    device: torch.device,
    episodes: int = 40,
) -> tuple[float, float]:
    """评估策略网络的确定性均值策略。"""

    returns: list[float] = []
    successes = 0

    for _ in range(episodes):
        # 评估阶段固定走 mean action，避免随机采样带来的波动。
        state = env.reset()
        done = False
        episode_return = 0.0
        info: dict[str, float | int | bool] = {"success": False}

        while not done:
            action, _, _ = sample_policy_action(policy, state, device, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_return += reward

        returns.append(episode_return)
        if bool(info.get("success", False)):
            successes += 1

    return sum(returns) / len(returns), successes / len(returns)


@torch.no_grad()
def save_policy_rollout_report(policy: nn.Module, env: ContinuousDriveEnv, device: torch.device) -> tuple[str, list[float]]:
    """保存一次连续控制策略的确定性 rollout。"""

    state = env.reset()
    done = False
    positions = [float(state[0].item())]
    lines = ["策略梯度确定性 rollout 轨迹", ""]
    total_reward = 0.0
    step_idx = 0
    info: dict[str, float | int | bool] = {"success": False}

    while not done:
        action, _, _ = sample_policy_action(policy, state, device, deterministic=True)
        next_state, reward, done, info = env.step(action)
        positions.append(float(next_state[0].item()))
        lines.append(
            f"步数={step_idx:02d} | 位置={float(state[0].item()):+.3f} | "
            f"速度={float(state[1].item()):+.3f} | 动作={action:+.3f} | "
            f"奖励={reward:+.3f} | 下一位置={float(next_state[0].item()):+.3f}"
        )
        total_reward += reward
        state = next_state
        step_idx += 1

    lines.append("")
    lines.append(f"是否成功={bool(info.get('success', False))} | 总回报={total_reward:+.3f}")
    report_path = save_text_report("\n".join(lines), "artifacts/policy_gradient_rollout.txt")
    return report_path, positions


def save_policy_report(policy: nn.Module, device: torch.device) -> str:
    """查看若干典型状态下策略输出的均值和标准差。"""

    example_states = [
        (-1.00, 0.00, 1.85),
        (-0.50, 0.00, 1.35),
        (0.00, 0.00, 0.85),
        (0.40, 0.00, 0.45),
        (0.80, 0.00, 0.05),
    ]
    lines = ["策略梯度状态探针", ""]

    with torch.no_grad():
        for state_tuple in example_states:
            state = torch.tensor(state_tuple, dtype=torch.float32).unsqueeze(0).to(device)
            mean, std = policy(state)
            lines.append(
                f"状态={state_tuple} | 动作均值={float(mean.item()):+.3f} | 标准差={float(std.item()):.3f}"
            )

    return save_text_report("\n".join(lines), "artifacts/policy_gradient_report.txt")


def save_policy_checkpoint(
    policy: nn.Module,
    state_dim: int,
    path: str = "artifacts/policy_gradient_drive.pt",
) -> str:
    """保存策略梯度 checkpoint。"""

    ckpt = {
        "state_dict": policy.state_dict(),
        "state_dim": state_dim,
        "action_dim": 1,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, file)
    return str(file)


def load_policy_checkpoint(path: str, device: torch.device) -> GaussianPolicy:
    """加载策略梯度 checkpoint。"""

    ckpt = torch.load(path, map_location=device, weights_only=True)
    policy = GaussianPolicy(state_dim=int(ckpt["state_dim"]), action_dim=int(ckpt["action_dim"])).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def train_dqn_agent(
    q_net: nn.Module,
    target_net: nn.Module,
    env: LineWorldEnv,
    device: torch.device,
    episodes: int = 260,
    gamma: float = 0.97,
    lr: float = 1e-3,
    batch_size: int = 64,
    target_sync_every: int = 25,
) -> tuple[list[float], list[float]]:
    """利用经验回放和目标网络训练 DQN。"""

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=5000)
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.988
    action_dim = 3

    return_history: list[float] = []
    success_history: list[float] = []

    for episode in range(1, episodes + 1):
        # 每个 episode 都是一条完整交互轨迹：
        # 从 reset 开始，直到成功、失败或超时终止。
        state = env.reset()
        done = False
        episode_return = 0.0
        info: dict[str, float | int | bool] = {"success": False}

        while not done:
            action = select_dqn_action(q_net, state, epsilon, action_dim, device)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            optimize_dqn(
                q_net=q_net,
                target_net=target_net,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )
            state = next_state
            episode_return += reward

        # epsilon 逐步衰减：前期鼓励探索，后期逐渐转向利用。
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        return_history.append(episode_return)
        success_history.append(1.0 if bool(info.get("success", False)) else 0.0)

        if episode % target_sync_every == 0:
            # 目标网络不是每步同步，而是隔一段时间更新一次，
            # 这样 Bellman target 不会抖得太厉害。
            target_net.load_state_dict(q_net.state_dict())

        if episode == 1 or episode % 20 == 0 or episode == episodes:
            print(
                f"[DQN] 回合={episode:03d} | "
                f"epsilon={epsilon:.3f} | "
                f"最近20回合平均回报={moving_average(return_history, 20):+.3f} | "
                f"最近20回合成功率={moving_average(success_history, 20):.2f}"
            )

    return return_history, success_history


def run_dqn_lesson(device: torch.device, force_retrain: bool = False) -> None:
    """运行第 5 课中 DQN 的教学实验。"""

    print("\n" + "=" * 68)
    print("第 5 课 A：离散动作环境上的 DQN")

    env = LineWorldEnv(num_states=9, max_steps=16)
    ckpt_path = "artifacts/dqn_lineworld.pt"
    return_curve_path = "artifacts/dqn_returns.png"
    success_curve_path = "artifacts/dqn_success_rate.png"
    history_report_path = "artifacts/dqn_training_history.txt"

    if Path(ckpt_path).exists() and not force_retrain:
        print(f"检测到已有 DQN 权重，直接加载：{ckpt_path}")
        q_net = load_dqn_checkpoint(ckpt_path, device)
    else:
        q_net = DQN(state_dim=env.num_states, action_dim=3).to(device)
        target_net = DQN(state_dim=env.num_states, action_dim=3).to(device)
        target_net.load_state_dict(q_net.state_dict())
        print(f"开始训练新的 DQN，参数量：{count_parameters(q_net)}")
        return_history, success_history = train_dqn_agent(q_net, target_net, env, device)
        save_curve_plot(
            return_history,
            path=return_curve_path,
            title="DQN 训练回报曲线",
            y_label="return",
        )
        save_curve_plot(
            success_history,
            path=success_curve_path,
            title="DQN 训练成功率变化",
            y_label="success",
        )
        save_training_history_report(
            return_history,
            success_history,
            path=history_report_path,
            title="DQN 训练过程记录",
        )
        save_dqn_checkpoint(q_net, env, ckpt_path)

    # 不论是读取旧权重还是刚训练完，都统一导出评估和策略报告。
    avg_return, success_rate = evaluate_dqn(q_net, env, episodes=40, device=device)
    policy_report = save_dqn_policy_report(q_net, env, device)
    rollout_report = save_dqn_rollout_report(q_net, env, device)

    print(f"DQN 评估结果：平均回报={avg_return:+.3f}，成功率={success_rate:.2f}")
    print(f"DQN 策略报告已保存到：{policy_report}")
    print(f"DQN rollout 报告已保存到：{rollout_report}")
    if Path(return_curve_path).exists():
        print(f"DQN 回报曲线已保存到：{return_curve_path}")
    if Path(success_curve_path).exists():
        print(f"DQN 成功率曲线已保存到：{success_curve_path}")
    if Path(history_report_path).exists():
        print(f"DQN 训练过程记录已保存到：{history_report_path}")


def run_policy_gradient_lesson(device: torch.device, force_retrain: bool = False) -> None:
    """运行第 5 课中策略梯度的教学实验。"""

    print("\n" + "=" * 68)
    print("第 5 课 B：连续控制任务上的策略梯度")

    # 这部分单独固定随机种子，避免前面的 DQN 已经消耗过随机数，
    # 进而把策略梯度带到一条明显更差的训练轨迹上。
    policy_seed = 42
    env = ContinuousDriveEnv(max_steps=35)
    ckpt_path = "artifacts/policy_gradient_drive.pt"
    return_curve_path = "artifacts/policy_gradient_returns.png"
    success_curve_path = "artifacts/policy_gradient_success_rate.png"
    history_report_path = "artifacts/policy_gradient_training_history.txt"
    # 如果已保存的权重明显过差，就不盲目复用，而是自动重训。
    min_success_rate = 0.80

    if Path(ckpt_path).exists() and not force_retrain:
        print(f"检测到已有策略网络权重，直接加载：{ckpt_path}")
        policy = load_policy_checkpoint(ckpt_path, device)
        warmup_avg_return, warmup_success_rate = evaluate_policy_gradient(
            policy,
            env,
            device=device,
            episodes=40,
        )
        if warmup_success_rate < min_success_rate:
            print(
                f"检测到当前策略成功率仅为 {warmup_success_rate:.2f}，"
                "说明旧权重质量较差，自动重新训练。"
            )
            set_seed(policy_seed)
            policy = GaussianPolicy(state_dim=3, action_dim=1).to(device)
            print(f"开始训练新的策略网络，参数量：{count_parameters(policy)}")
            return_history, success_history = train_policy_gradient(policy, env, device)
            save_curve_plot(
                return_history,
                path=return_curve_path,
                title="策略梯度训练回报曲线",
                y_label="return",
            )
            save_curve_plot(
                success_history,
                path=success_curve_path,
                title="策略梯度训练成功率变化",
                y_label="success",
            )
            save_training_history_report(
                return_history,
                success_history,
                path=history_report_path,
                title="策略梯度训练过程记录",
            )
            save_policy_checkpoint(policy, state_dim=3, path=ckpt_path)
        else:
            print(
                f"已加载策略通过体检：平均回报={warmup_avg_return:+.3f}，"
                f"成功率={warmup_success_rate:.2f}"
            )
    else:
        set_seed(policy_seed)
        policy = GaussianPolicy(state_dim=3, action_dim=1).to(device)
        print(f"开始训练新的策略网络，参数量：{count_parameters(policy)}")
        return_history, success_history = train_policy_gradient(policy, env, device)
        save_curve_plot(
            return_history,
            path=return_curve_path,
            title="策略梯度训练回报曲线",
            y_label="return",
        )
        save_curve_plot(
            success_history,
            path=success_curve_path,
            title="策略梯度训练成功率变化",
            y_label="success",
        )
        save_training_history_report(
            return_history,
            success_history,
            path=history_report_path,
            title="策略梯度训练过程记录",
        )
        save_policy_checkpoint(policy, state_dim=3, path=ckpt_path)

    # 最后统一导出评估、策略探针和 rollout，方便把“学到了什么”直接看出来。
    avg_return, success_rate = evaluate_policy_gradient(policy, env, device=device, episodes=40)
    policy_report = save_policy_report(policy, device)
    rollout_report, positions = save_policy_rollout_report(policy, env, device)
    rollout_curve = save_curve_plot(
        positions,
        path="artifacts/policy_gradient_position_curve.png",
        title="策略 rollout 位置变化曲线",
        y_label="position",
    )

    print(f"策略梯度评估结果：平均回报={avg_return:+.3f}，成功率={success_rate:.2f}")
    print(f"策略探针报告已保存到：{policy_report}")
    print(f"策略 rollout 报告已保存到：{rollout_report}")
    if rollout_curve is not None:
        print(f"策略 rollout 曲线已保存到：{rollout_curve}")
    if Path(return_curve_path).exists():
        print(f"策略梯度回报曲线已保存到：{return_curve_path}")
    if Path(success_curve_path).exists():
        print(f"策略梯度成功率曲线已保存到：{success_curve_path}")
    if Path(history_report_path).exists():
        print(f"策略梯度训练过程记录已保存到：{history_report_path}")


def main() -> None:
    """程序入口。"""

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_training_process = True
    print(f"当前使用设备：{device}")
    if show_training_process:
        print("已开启训练过程展示：本次会重新训练模型，并输出回报/成功率变化。")
    run_dqn_lesson(device=device, force_retrain=show_training_process)
    run_policy_gradient_lesson(device=device, force_retrain=show_training_process)


if __name__ == "__main__":
    main()
