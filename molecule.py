import numpy as np
from typing import Union, Tuple

class Molecule:
    def __init__(self, position: np.ndarray, species_type: str, diffusion_coefficient: float) -> None:
        """
        3D空間における分子を反応履歴とともに初期化

        Args:
            position (np.array): 3D座標 [x, y, z]
            species_type (str): 化学種識別子
            diffusion_coefficient (float): 拡散係数
        """
        self.position = np.array(position, dtype=float)
        self.species_type = species_type
        self.diffusion_coefficient = diffusion_coefficient
        self.velocity = np.zeros(3)

        # 反応履歴トラッキング
        self.history = {
            'reactions': [],  # (時間、位置、反応物、生成物)タプルのリスト
            'species_sequence': [species_type],  # 反応チェーンにおける種系列
            'positions': [position.copy()],  # 反応が起こった位置
            'times': [0.0]  # 反応が起こった時間
        }
        self.loop_detected = False
        self.current_loop = None

    def diffuse(self, dt: float, membrane_bounds: Union[Tuple[float, float], None] = None) -> None:
        """
        膜の制約を考慮したブラウン運動をシミュレート

        Args:
            dt (float): タイムステップ
            membrane_bounds (tuple): 膜境界のオプション (min_z, max_z)
        """
        random_displacement = np.random.normal(0, np.sqrt(2*self.diffusion_coefficient*dt), 3)
        new_position = self.position + random_displacement

        # 指定されている場合は膜の制約を適用
        if membrane_bounds and self.species_type.startswith('A'):
            min_z, max_z = membrane_bounds
            if new_position[2] < min_z or new_position[2] > max_z:
                new_position[2] = self.position[2]  # 膜から跳ね返る

        self.position = new_position

    def update_history(self, time: float, position: np.ndarray, reactants: Tuple[str, str], product: str) -> None:
        """反応履歴を更新"""
        self.history['reactions'].append((time, position.copy(), reactants, product))
        self.history['species_sequence'].append(product)
        self.history['positions'].append(position.copy())
        self.history['times'].append(time)
        self._check_for_loop()

    def _check_for_loop(self) -> None:
        """履歴における反応ループをチェック"""
        sequence = self.history['species_sequence']
        if len(sequence) < 3:  # ループには少なくとも3つの種が必要
            return

        # 任意のサイズのループをチェック
        current_species = sequence[-1]
        for i, species in enumerate(sequence[:-1]):
            if species == current_species:
                # ループが見つかった
                self.loop_detected = True
                self.current_loop = {
                    'species': sequence[i:],
                    'positions': self.history['positions'][i:],
                    'times': self.history['times'][i:],
                    'cycle_time': self.history['times'][-1] - self.history['times'][i]
                }
                return