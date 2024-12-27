from molecule import Molecule
from typing import Dict, Tuple, Union
import numpy as np

def apply_periodic_boundary(position: np.ndarray, box_size: float) -> np.ndarray:
    """周期的境界条件を適用"""
    return position % box_size

def check_reaction(mol1: Molecule, mol2: Molecule, reaction_constants: Dict) -> Union[Dict, None]:
    """2つの分子が反応できるかどうかを確認"""
    reaction_key = (mol1.species_type, mol2.species_type)
    reverse_key = (mol2.species_type, mol1.species_type)

    if reaction_key in reaction_constants:
        return reaction_constants[reaction_key]
    elif reverse_key in reaction_constants:
        return reaction_constants[reverse_key]
    return None

def minimum_image_distance(pos1: np.ndarray, pos2: np.ndarray, box_size: float) -> np.ndarray:
    """周期的境界条件下での最小画像距離を計算"""
    delta = pos1 - pos2
    delta = np.where(delta > box_size/2, delta - box_size, delta)
    delta = np.where(delta < -box_size/2, delta + box_size, delta)
    return delta