from reaction_loop import ReactionLoop3D
from visualization import Visualization

# 使用例
def run_simulation():
    # 球状の膜でシミュレーションを初期化
    membrane_bounds = (5.0, 5.0, 5.0, 3.0)  # 中心(5, 5, 5), 半径3の球状膜
    sim = ReactionLoop3D(box_size=10.0, membrane_bounds=membrane_bounds)

    # シミュレーションを実行
    dt = 0.01
    total_steps = 1000

    for step in range(total_steps):
        current_time = step * dt
        sim.step(dt, current_time)

        # 定期的に可視化
        if step % 200 == 0:
            Visualization.visualize_loops(sim)
            Visualization.plot_loop_statistics(sim)
            Visualization.plot_history(sim)

if __name__ == "__main__":
    run_simulation()