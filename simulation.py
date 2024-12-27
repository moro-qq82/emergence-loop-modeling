from reaction_loop import ReactionLoop3D
from visualization import Visualization

# 使用例
def run_simulation():
    # 膜でシミュレーションを初期化
    membrane_bounds = (4.0, 6.0)  # z=4とz=6の間の膜
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