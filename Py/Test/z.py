import tkinter as tk
import random, math, colorsys

"""
简短说明：
这是一个使用 Tkinter 绘制简单烟花效果的小演示程序。
按下鼠标左键会在点击位置产生一次烟花爆裂，程序也会定时自动触发爆裂。
"""

# 窗口和渲染参数
WIDTH, HEIGHT = 900, 600
PARTICLES_PER_BURST = 120  # 每次爆裂的粒子数量基准
GRAVITY = 0.12             # 对粒子的垂直加速度（重力模拟）
FRAMERATE_MS = 16          # 帧间隔（毫秒）


class Particle:
    """表示单个烟花粒子。

    属性：
        x, y: 粒子当前坐标
        vx, vy: 粒子速度分量
        hue: 色相（0-1）用于生成颜色
        life: 剩余寿命（帧数）
        size: 粒子半径基准
    """

    def __init__(self, x, y, vx, vy, hue, life, size):
        self.x = x; self.y = y
        self.vx = vx; self.vy = vy
        self.hue = hue
        self.life = life
        self.size = size

    def update(self):
        """每帧更新速度、位置和寿命。"""
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def color(self):
        """根据色相和剩余寿命返回 RGB 十六进制颜色字符串。

        随着寿命减少，亮度会降低以产生淡出效果。
        """
        r, g, b = colorsys.hsv_to_rgb(self.hue, 1.0, max(0.2, self.life / 40))
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


class FireworksApp:
    """主应用类，负责创建窗口、处理事件与渲染粒子。"""

    def __init__(self, root):
        self.root = root
        root.title("Fireworks — Demo")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='black')
        self.canvas.pack()
        self.particles = []
        # 点击触发爆裂
        self.canvas.bind("<Button-1>", self.click)
        # 自动触发计时器
        self._auto_timer = 0
        # 启动主循环
        self.loop()

    def burst(self, x=None, y=None, count=PARTICLES_PER_BURST):
        """在 (x, y) 位置产生一组粒子；若未指定位置则随机生成。

        每个粒子有随机角度、速度、寿命与大小，色相以基准色微调。
        """
        if x is None:
            x = random.randint(int(WIDTH*0.2), int(WIDTH*0.8))
        if y is None:
            y = random.randint(int(HEIGHT*0.1), int(HEIGHT*0.5))
        hue = random.random()
        speed = random.uniform(2.5, 6.5)
        for i in range(count):
            angle = random.random() * 2 * math.pi
            sp = random.random()**0.8 * speed
            vx = math.cos(angle) * sp
            vy = math.sin(angle) * sp
            life = random.randint(18, 38)
            size = random.uniform(1.5, 4.5)
            p = Particle(x, y, vx, vy, (hue + random.uniform(-0.05, 0.05)) % 1.0, life, size)
            self.particles.append(p)

    def click(self, event):
        """鼠标点击回调：在点击位置生成更密集的爆裂效果。"""
        self.burst(event.x, event.y, count=PARTICLES_PER_BURST+40)

    def loop(self):
        """主渲染/更新循环：清空画布、更新粒子并重绘，然后安排下一帧。"""
        self.canvas.delete("all")
        # 定期自动触发爆裂以保持演示活跃
        self._auto_timer += 1
        if self._auto_timer > 40:
            self._auto_timer = 0
            self.burst()
        new_particles = []
        for p in self.particles:
            p.update()
            # 只在粒子仍在有效寿命且未完全离开画布时绘制
            if p.life > 0 and  -50 < p.x < WIDTH+50 and p.y < HEIGHT+50:
                c = p.color()
                s = p.size * (0.6 + (p.life / 40))
                x0, y0 = p.x - s, p.y - s
                x1, y1 = p.x + s, p.y + s
                self.canvas.create_oval(x0, y0, x1, y1, fill=c, outline="")
                new_particles.append(p)
        self.particles = new_particles
        # 屏幕左下角显示简单提示文字
        self.canvas.create_text(10, HEIGHT-10, anchor="sw", text="Click to burst · Auto bursts every few seconds", fill="#888", font=("Arial", 10))
        self.root.after(FRAMERATE_MS, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = FireworksApp(root)
    root.mainloop()