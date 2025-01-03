from typing import List, Tuple, Dict, Union, DefaultDict
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from molecule import Molecule  # Moleculeクラスをインポート
from utils import *

class ReactionLoop3D:
    def __init__(self, 
                 box_size: float, 
                 membrane_bounds: Union[Tuple[float, float], None] = None, 
                 initial_molecules: Union[Dict, None] = None) -> None:
        """
        膜を伴う3D反応ループシミュレーションを初期化

        Args:
            box_size (float): 立方体シミュレーションボックスのサイズ
            membrane_bounds (tuple): 膜境界のオプション (min_z, max_z)
            initial_molecules (dict): 初期分子構成
        """
        self.box_size = box_size
        self.membrane_bounds = membrane_bounds
        self.molecules = []
        self.reaction_constants = {
            ('A1', 'F1'): {'product': 'A2', 'rate': 0.1},
            ('A2', 'F2'): {'product': 'A3', 'rate': 0.1},
            ('A3', 'F3'): {'product': 'A1', 'rate': 0.1}
        }

        # 検出されたすべてのループを追跡
        self.detected_loops = []
        self.loop_creation_times = []

        if initial_molecules is None:
            self._initialize_default_molecules()
        else:
            self._initialize_custom_molecules(initial_molecules)

        self.history = {
            'time': [],
            'concentrations': defaultdict(list),
            'loop_events': []  # ループ形成/破壊イベントを追跡
        }

    def _initialize_default_molecules(self) -> None:
        """膜の考慮事項を含むデフォルトの初期分子分布を設定"""
        species_configs = {
            'A1': {'count': 100, 'diffusion_coef': 0.1},
            'A2': {'count': 100, 'diffusion_coef': 0.1},
            'A3': {'count': 100, 'diffusion_coef': 0.1},
            'F1': {'count': 300, 'diffusion_coef': 0.2},
            'F2': {'count': 300, 'diffusion_coef': 0.2},
            'F3': {'count': 300, 'diffusion_coef': 0.2}
        }

        for species, config in species_configs.items():
            for _ in range(config['count']):
                # Aタイプの分子については、指定されている場合は膜の境界内で初期化
                if self.membrane_bounds and species.startswith('A'):
                    min_z, max_z = self.membrane_bounds
                    position = np.array([
                        np.random.uniform(0, self.box_size),
                        np.random.uniform(0, self.box_size),
                        np.random.uniform(min_z, max_z)
                    ])
                else:
                    position = np.random.uniform(0, self.box_size, 3)

                molecule = Molecule(position, species, config['diffusion_coef'])
                self.molecules.append(molecule)

    def _initialize_custom_molecules(self, initial_molecules: Dict) -> None:
        """カスタム初期分子分布を設定"""
        for species, config in initial_molecules.items():
            for _ in range(config['count']):
                position = np.random.uniform(0, self.box_size, 3)
                molecule = Molecule(position, species, config['diffusion_coef'])
                self.molecules.append(molecule)

    def step(self, dt: float, current_time: float) -> None:
        """1つのシミュレーションタイムステップを実行"""
        # 拡散
        for molecule in self.molecules:
            molecule.diffuse(dt, self.membrane_bounds)
            molecule.position = apply_periodic_boundary(molecule.position, self.box_size)

        # 反応
        new_molecules = []
        molecules_to_remove = set()

        for i, mol1 in enumerate(self.molecules):
            if mol1 in molecules_to_remove:
                continue

            for j, mol2 in enumerate(self.molecules[i+1:], i+1):
                if mol2 in molecules_to_remove:
                    continue

                distance = np.linalg.norm(
                    minimum_image_distance(mol1.position, mol2.position, self.box_size)
                )

                if distance < 1.0:  # 反応半径
                    reaction = check_reaction(mol1, mol2, self.reaction_constants)
                    if reaction and np.random.random() < reaction['rate'] * dt:
                        new_pos = (mol1.position + mol2.position) / 2
                        new_pos = apply_periodic_boundary(new_pos, self.box_size)

                        # 組み合わせた履歴を持つ新しい分子を作成
                        new_mol = Molecule(new_pos, reaction['product'], 0.1)
                        new_mol.update_history(
                            current_time,
                            new_pos,
                            (mol1.species_type, mol2.species_type),
                            reaction['product']
                        )

                        # 反応物から履歴を転送
                        if mol1.history['reactions']:
                            new_mol.history['reactions'].extend(mol1.history['reactions'])
                        if mol2.history['reactions']:
                            new_mol.history['reactions'].extend(mol2.history['reactions'])

                        new_molecules.append(new_mol)
                        molecules_to_remove.add(mol1)
                        molecules_to_remove.add(mol2)

                        # これが新しいループを作成するかどうかを確認
                        if new_mol.loop_detected:
                            self._process_new_loop(new_mol, current_time)

                        break

        # 分子リストを更新
        self.molecules = [m for m in self.molecules if m not in molecules_to_remove] + new_molecules

        # 現在の状態を記録
        self._update_history(dt, current_time)

    def _process_new_loop(self, molecule: Molecule, current_time: float) -> None:
        """新しく検出された反応ループを処理および分析"""
        if molecule.current_loop:
            self.detected_loops.append(molecule.current_loop)
            self.loop_creation_times.append(current_time)

            # 他のループとの空間的近接性を分析
            self._analyze_loop_proximity(molecule.current_loop)

            # ループイベントを記録
            self.history['loop_events'].append({
                'time': current_time,
                'type': 'creation',
                'loop_data': molecule.current_loop
            })

    def _analyze_loop_proximity(self, new_loop: Dict) -> None:
        """ループ間の空間的近接性を分析"""
        if len(self.detected_loops) < 2:
            return

        # 新しいループの重心を計算
        new_centroid = np.mean(new_loop['positions'], axis=0)

        # 既存のループと比較
        for existing_loop in self.detected_loops[:-1]:  # 新しいループを除外
            existing_centroid = np.mean(existing_loop['positions'], axis=0)

            # 各ループの最大内部距離を計算
            new_max_dist = max(np.linalg.norm(p - new_centroid) 
                             for p in new_loop['positions'])
            existing_max_dist = max(np.linalg.norm(p - existing_centroid) 
                                 for p in existing_loop['positions'])

            # 近接条件を確認
            centroid_distance = np.linalg.norm(new_centroid - existing_centroid)
            if centroid_distance < max(new_max_dist, existing_max_dist):
                # ループは空間的に近接している
                self.history['loop_events'].append({
                    'time': self.history['time'][-1],
                    'type': 'proximity',
                    'loops': (new_loop, existing_loop)
                })

    def _update_history(self, dt: float, current_time: float) -> None:
        """
        濃度履歴を更新し、ループイベントを記録

        Args:
            dt (float): タイムステップ
            current_time (float): 現在のシミュレーション時間
        """
        # 時間を記録
        self.history['time'].append(current_time)

        # 各タイプの分子をカウント
        counts = defaultdict(int)
        volume = self.box_size ** 3

        for molecule in self.molecules:
            counts[molecule.species_type] += 1

        # 考えられるすべての種の濃度を更新
        all_species = {'A1', 'A2', 'A3', 'F1', 'F2', 'F3'}
        for species in all_species:
            concentration = counts[species] / volume
            self.history['concentrations'][species].append(concentration)

        # このタイムステップからのループイベントを記録
        active_loops = []
        for molecule in self.molecules:
            if molecule.loop_detected and molecule.current_loop:
                loop_info = {
                    'time': current_time,
                    'species_sequence': molecule.current_loop['species'],
                    'positions': [pos.copy() for pos in molecule.current_loop['positions']],
                    'cycle_time': molecule.current_loop['cycle_time']
                }
                active_loops.append(loop_info)

        # ループ統計を更新
        if active_loops:
            self.history['loop_stats'] = self.history.get('loop_stats', []) + [{
                'time': current_time,
                'active_loops': len(active_loops),
                'avg_cycle_time': np.mean([loop['cycle_time'] for loop in active_loops]),
                'loop_details': active_loops
            }]

        # 必要に応じて追加の統計を計算
        if len(self.history['time']) > 1:
            # 濃度変化を計算
            for species in all_species:
                current_conc = self.history['concentrations'][species][-1]
                prev_conc = self.history['concentrations'][species][-2]

                # 大きな変化を記録 (オプション)
                if abs(current_conc - prev_conc) > 0.1:  # 大きな変化のしきい値
                    event = {
                        'time': current_time,
                        'type': 'concentration_change',
                        'species': species,
                        'change': current_conc - prev_conc
                    }
                    self.history['events'] = self.history.get('events', []) + [event]

        # メモリ管理 - オプション: 必要に応じて履歴の長さを制限
        max_history_length = 10000  # 必要に応じて調整
        if len(self.history['time']) > max_history_length:
            cutoff = len(self.history['time']) - max_history_length
            self.history['time'] = self.history['time'][cutoff:]
            for species in self.history['concentrations']:
                self.history['concentrations'][species] = \
                    self.history['concentrations'][species][cutoff:]
            if 'loop_stats' in self.history:
                self.history['loop_stats'] = [
                    stat for stat in self.history['loop_stats']
                    if stat['time'] > self.history['time'][0]
                ]
            if 'events' in self.history:
                self.history['events'] = [
                    event for event in self.history['events']
                    if event['time'] > self.history['time'][0]
                ]