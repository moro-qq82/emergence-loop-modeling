import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from reaction_loop import ReactionLoop3D

class Visualization:
    @staticmethod
    def visualize_loops(sim: ReactionLoop3D) -> None:
        """反応ループとその空間的関係を可視化"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 膜が存在する場合はプロット
        if sim.membrane_bounds:
            min_z, max_z = sim.membrane_bounds
            x = y = np.linspace(0, sim.box_size, 10)
            X, Y = np.meshgrid(x, y)

            for z in [min_z, max_z]:
                Z = np.full_like(X, z)
                ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

        # 検出された各ループをプロット
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sim.detected_loops)))
        for loop, color in zip(sim.detected_loops, colors):
            positions = np.array(loop['positions'])
            # 反応サイトをプロット
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=[color], alpha=0.6)
            # 反応サイトを線で接続
            for i in range(len(positions)-1):
                ax.plot([positions[i,0], positions[i+1,0]],
                       [positions[i,1], positions[i+1,1]],
                       [positions[i,2], positions[i+1,2]],
                       c=color, alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('反応ループの可視化')
        plt.show()

    @staticmethod
    def plot_loop_statistics(sim: ReactionLoop3D) -> None:
        """検出されたループに関する統計をプロット"""
        if not sim.detected_loops:
            print("ループはまだ検出されていません。")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ループサイクルタイムをプロット
        cycle_times = [loop['cycle_time'] for loop in sim.detected_loops]
        ax1.hist(cycle_times, bins='auto')
        ax1.set_xlabel('ループサイクルタイム')
        ax1.set_ylabel('頻度')
        ax1.set_title('ループサイクルタイムの分布')
        
        # ループ作成タイムラインをプロット
        ax2.plot(sim.loop_creation_times, range(len(sim.loop_creation_times)), 'o-')
        ax2.set_xlabel('シミュレーション時間')
        ax2.set_ylabel('累積ループ')
        ax2.set_title('ループ形成タイムライン')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_history(sim: ReactionLoop3D) -> None:
        """シミュレーションの包括的な履歴をプロット"""
        if not sim.history['time']:
            print("履歴データはありません")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 濃度をプロット
        for species in sim.history['concentrations']:
            ax1.plot(sim.history['time'], 
                    sim.history['concentrations'][species],
                    label=species)
        
        ax1.set_xlabel('時間')
        ax1.set_ylabel('濃度')
        ax1.set_title('時間経過に伴う種の濃度')
        ax1.legend()
        ax1.grid(True)
        
        # ループ統計が利用可能な場合はプロット
        if 'loop_stats' in sim.history and sim.history['loop_stats']:
            times = [stat['time'] for stat in sim.history['loop_stats']]
            active_loops = [stat['active_loops'] for stat in sim.history['loop_stats']]
            avg_cycles = [stat['avg_cycle_time'] for stat in sim.history['loop_stats']]
            
            ax2.plot(times, active_loops, 'b-', label='アクティブなループ')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(times, avg_cycles, 'r--', label='平均サイクルタイム')
            
            ax2.set_xlabel('時間')
            ax2.set_ylabel('アクティブなループの数', color='b')
            ax2_twin.set_ylabel('平均サイクルタイム', color='r')
            
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        plt.tight_layout()
        plt.show()