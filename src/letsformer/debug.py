from rich.console import Console
import matplotlib.pyplot as plt
import time
import numpy as np

DEBUG = True
console = Console()

def plot_loss_curve_basic(loss_list, title="Training Loss Curve", save_path=None):
    """
    基础版本 - 绘制loss曲线
    
    参数:
    loss_list: 包含loss值的列表
    title: 图表标题
    save_path: 图片保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制loss曲线
    plt.plot(range(1, len(loss_list) + 1), loss_list, 
             color='royalblue', linewidth=2, label='Loss')
    
    # 添加标记点
    min_loss_idx = np.argmin(loss_list)
    plt.scatter(min_loss_idx + 1, loss_list[min_loss_idx], 
                color='red', zorder=5, label=f'Min Loss: {loss_list[min_loss_idx]:.4f}')
    
    # 设置图表属性
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()

class LossMonitor:
    """实时Loss监控器"""
    
    def __init__(self, title: str ="Loss Monitor", window_size=10, update_interval=0.1):
        """
        参数:
            window_size: 移动平均窗口大小
            update_interval: 图表更新间隔（秒）
        """
        # 开启交互模式
        plt.ion()
        
        # 创建图形和子图
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 初始化数据
        self.epochs = []
        self.losses = []
        self.moving_avg = []
        self.title = title
        self.window_size = window_size
        
        # 预计算统计变量，避免重复计算
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.sum_loss = 0.0
        self.sum_squared_loss = 0.0
        self.count = 0
        self.first_loss = None
        
        # 创建线条
        self.loss_line, = self.ax1.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Loss')
        self.avg_line, = self.ax1.plot([], [], 'r-', linewidth=2, label=f'{window_size}-Epoch MA')
        
        # 创建散点图（最近的点）
        self.recent_scatter = self.ax1.scatter([], [], color='green', s=50, zorder=5, label='Recent')
        
        # 创建文本统计信息
        self.stats_text = self.ax2.text(0.05, 0.95, '', transform=self.ax2.transAxes, 
                                        verticalalignment='top', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置图表属性
        self.ax1.set_title(title, fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # 隐藏第二个坐标轴（用于显示文本）
        self.ax2.axis('off')
        
        # 设置更新间隔
        self.update_interval = update_interval
        self.last_update_time = time.time()
        
        plt.tight_layout()
    
    def _calculate_moving_average(self):
        """计算移动平均"""
        if len(self.losses) >= self.window_size:
            window = self.losses[-self.window_size:]
            return np.mean(window)
        return None
    
    def _update_stats_text(self):
        """更新统计信息文本"""
        if self.count == 0:
            return
        
        avg_loss = self.sum_loss / self.count
        variance = (self.sum_squared_loss / self.count) - (avg_loss ** 2)
        std_loss = variance ** 0.5
        
        stats = [
            f"Current Epoch: {self.count}",
            f"Current Loss: {self.losses[-1]:.6f}",
            f"Min Loss: {self.min_loss:.6f}",
            f"Max Loss: {self.max_loss:.6f}",
            f"Average Loss: {avg_loss:.6f}",
            f"Loss Std: {std_loss:.6f}",
            f"Loss Rate: {100*(self.first_loss-self.losses[-1])/self.first_loss:.2f}%"
        ]
        
        if len(self.moving_avg) > 0:
            stats.append(f"Moving Average: {self.moving_avg[-1]:.6f}")
        
        self.stats_text.set_text('\n'.join(stats))
    
    def add_loss(self, epoch, loss) -> bool:
        """添加Loss数据, 返回是否为最小Loss"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # 实时更新统计变量
        if self.first_loss is None:
            self.first_loss = loss
        self.sum_loss += loss
        self.sum_squared_loss += loss * loss
        self.count += 1
        
        is_min_loss = loss < self.min_loss
        if is_min_loss:
            self.min_loss = loss
        if loss > self.max_loss:
            self.max_loss = loss
        
        # 计算移动平均
        avg = self._calculate_moving_average()
        if avg is not None:
            self.moving_avg.append(avg)
        
        # 检查是否需要更新图表
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._update_plot()
            self.last_update_time = current_time
        
        return is_min_loss
    
    def _update_plot(self):
        """更新图表"""
        if len(self.epochs) == 0:
            return
        
        # 更新Loss曲线
        self.loss_line.set_data(self.epochs, self.losses)
        
        # 更新移动平均曲线
        if len(self.moving_avg) > 0:
            avg_epochs = self.epochs[self.window_size-1:]
            self.avg_line.set_data(avg_epochs, self.moving_avg)
        
        # 更新最近点散点图（显示最近5个点）
        recent_count = min(5, len(self.epochs))
        recent_epochs = self.epochs[-recent_count:]
        recent_losses = self.losses[-recent_count:]
        self.recent_scatter.set_offsets(np.column_stack([recent_epochs, recent_losses]))
        
        # 调整坐标轴范围
        if len(self.epochs) > 1:
            self.ax1.set_xlim(0, max(self.epochs) * 1.05)
            y_min = self.min_loss * 0.95
            y_max = self.max_loss * 1.05
            self.ax1.set_ylim(y_min, y_max)
        
        # 更新统计信息
        self._update_stats_text()
        
        # 重绘图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def finalize(self, save_path="./data/model/loss_curve.png"):
        """完成训练，显示最终图表"""
        # 确保最后一次更新
        self._update_plot()
        
        # 关闭交互模式
        plt.ioff()
        
        # 添加最终标记
        if len(self.losses) > 0:
            min_idx = np.argmin(self.losses)
            self.ax1.plot(self.epochs[min_idx], self.losses[min_idx], 'r*', 
                         markersize=15, label=f'Min Loss: {self.losses[min_idx]:.4f}')
            self.ax1.legend()
        
        # 保存一个 Loss 图
        plot_loss_curve_basic(self.losses, title=self.title, save_path=save_path)

        plt.show()