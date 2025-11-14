import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class PeakSeekingEnv(gym.Env):
    """寻找山顶环境：代理需找到接近山顶的位置"""
    
    def __init__(self, render_mode=None, size=20, max_steps=100):
        # --- “内部”变量，我们自己用 ---
        self._size = size  # 环境大小
        self._max_steps = max_steps  # 每个episode的最大步骤数
        
        # --- “公共”变量，Gym 框架需要 ---
        # 动作空间: 上下左右移动
        self.action_space = spaces.Discrete(4)
        # 观察空间: 代理位置 (注意: high 这里用到了我们的内部变量 _size)
        self.observation_space = spaces.Box(low=0, high=self._size-1, shape=(2,), dtype=int)
        
        # --- 更多“内部”变量，用于游戏逻辑 ---
        self._height_map = None
        self._global_max = None
        self._threshold = None
        self._agent_pos = [0, 0]  # 初始化代理位置
        self._step_count = 0
        
        # --- “内部”变量，用于渲染 ---
        self._fig, self._ax = plt.subplots()  # 添加用于绘制图形的属性
        self.render_mode = render_mode # render_mode 是 gym 的标准属性，保持公共
        
    def _generate_height_map(self, peak_value):
        """生成具有指定峰值高度的高度图"""
        # 使用内部变量 self._size
        x = np.linspace(-2, 2, self._size)
        y = np.linspace(-2, 2, self._size)
        X, Y = np.meshgrid(x, y)
        height_map = np.exp(-(X**2 + Y**2)) * peak_value  # 使用高斯分布生成单峰
        return height_map
    
    # reset 是 gym 规定的“公共”方法
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机生成山峰的最高点
        random_peak_value = self.np_random.uniform(5, 10) # self.np_random 是 gym 提供的
        
        # 设置所有“内部”游戏状态
        self._height_map = self._generate_height_map(random_peak_value)
        self._global_max = self._height_map.max()
        self._threshold = self._global_max * 0.98   # 接近山顶即成功
        
        # 初始化代理位置 (使用内部变量 self._size)
        self._agent_pos = [
            self.np_random.integers(1, self._size - 1),
            self.np_random.integers(1, self._size - 1)
        ]
        self._step_count = 0
        
        if self.render_mode == "human":
            self._render_frame() # 调用我们自己的“内部”渲染函数

        # 返回“公共”的观察值和信息
        return self._get_obs(), {}
    
    # step 是 gym 规定的“公共”方法
    def step(self, action):
        self._step_count += 1
        
        # 根据动作更新代理的“内部”位置
        if action == 0:  # 向上移动
            self._agent_pos[0] = max(self._agent_pos[0] - 1, 0)
        elif action == 1:  # 向右移动
            # 使用内部变量 self._size
            self._agent_pos[1] = min(self._agent_pos[1] + 1, self._size - 1)
        elif action == 2:  # 向下移动
            # 使用内部变量 self._size
            self._agent_pos[0] = min(self._agent_pos[0] + 1, self._size - 1)
        elif action == 3:  # 向左移动
            self._agent_pos[1] = max(self._agent_pos[1] - 1, 0)
        
        # 计算奖励 (全部使用“内部”变量)
        current_height = self._height_map[self._agent_pos[0], self._agent_pos[1]]
        
        done = current_height >= self._threshold or self._step_count >= self._max_steps
        
        reward = 1 if current_height >= self._threshold else 0

        if self.render_mode == "human":
            self._render_frame() # 调用我们自己的“内部”渲染函数
        
        # 返回“公共”的五元组
        return self._get_obs(), reward, done, False, {}

    # 这是一个“内部”辅助函数
    def _get_obs(self):
        """返回当前观察值（即代理的位置）"""
        return self._agent_pos
    
    # 这是一个“内部”辅助函数
    def _render_frame(self):
        # 全部使用“内部”变量
        self._ax.clear()
        self._ax.imshow(self._height_map, cmap='viridis', extent=[0, self._size, 0, self._size])
        self._ax.plot(self._agent_pos[1], self._agent_pos[0], 'ro')  # 画出代理的位置
        plt.title(f'Step {self._step_count}, Height: {self._height_map[self._agent_pos[0], self._agent_pos[1]]:.2f}')
        plt.pause(0.1)

# 示例用法 (这部分代码不需要改变)
# 因为 `env` 对象的使用者只调用“公共”方法 (reset 和 step)
# 他们不需要知道 `env` 内部的变量名是 `self.size` 还是 `self._size`
if __name__ == "__main__":
    env = PeakSeekingEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 随机选择动作
        obs, reward, done, truncated, info = env.step(action)