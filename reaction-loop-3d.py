import numpy as np
from scipy.integrate import odeint
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Molecule:
    def __init__(self, position, species_type, diffusion_coefficient):
        """
        Initialize a molecule in 3D space
        
        Args:
            position (np.array): 3D coordinates [x, y, z]
            species_type (str): Chemical species identifier
            diffusion_coefficient (float): Diffusion coefficient
        """
        self.position = np.array(position, dtype=float)
        self.species_type = species_type
        self.diffusion_coefficient = diffusion_coefficient
        self.velocity = np.zeros(3)  # Initial velocity

    def diffuse(self, dt):
        """Simulate Brownian motion"""
        random_displacement = np.random.normal(0, np.sqrt(2*self.diffusion_coefficient*dt), 3)
        self.position += random_displacement

class ReactionLoop3D:
    def __init__(self, box_size, initial_molecules=None):
        """
        Initialize a 3D reaction loop simulation
        
        Args:
            box_size (float): Size of cubic simulation box
            initial_molecules (dict): Initial molecule configurations
        """
        self.box_size = box_size
        self.molecules = []
        self.reaction_constants = {
            ('A1', 'F1'): {'product': 'A2', 'rate': 0.1},
            ('A2', 'F2'): {'product': 'A3', 'rate': 0.1},
            ('A3', 'F3'): {'product': 'A1', 'rate': 0.1}
        }
        
        # Default initial conditions if none provided
        if initial_molecules is None:
            self._initialize_default_molecules()
        else:
            self._initialize_custom_molecules(initial_molecules)
        
        # Concentration history for analysis
        self.history = {
            'time': [],
            'concentrations': {
                'A1': [], 'A2': [], 'A3': [],
                'F1': [], 'F2': [], 'F3': []
            }
        }

    def _initialize_default_molecules(self):
        """Set up default initial molecule distribution"""
        # Create initial molecules with random positions
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
                position = np.random.uniform(0, self.box_size, 3)
                molecule = Molecule(position, species, config['diffusion_coef'])
                self.molecules.append(molecule)

    def _initialize_custom_molecules(self, initial_molecules):
        """Set up custom initial molecule distribution"""
        for species, config in initial_molecules.items():
            for _ in range(config['count']):
                position = np.random.uniform(0, self.box_size, 3)
                molecule = Molecule(position, species, config['diffusion_coef'])
                self.molecules.append(molecule)

    def _apply_periodic_boundary(self, position):
        """Apply periodic boundary conditions"""
        return position % self.box_size

    def _check_reaction(self, mol1, mol2):
        """Check if two molecules can react"""
        reaction_key = (mol1.species_type, mol2.species_type)
        reverse_key = (mol2.species_type, mol1.species_type)
        
        if reaction_key in self.reaction_constants:
            return self.reaction_constants[reaction_key]
        elif reverse_key in self.reaction_constants:
            return self.reaction_constants[reverse_key]
        return None

    def step(self, dt):
        """Perform one simulation timestep"""
        # Diffusion
        for molecule in self.molecules:
            molecule.diffuse(dt)
            molecule.position = self._apply_periodic_boundary(molecule.position)
        
        # Reactions
        new_molecules = []
        molecules_to_remove = set()
        
        # Check for reactions between nearby molecules
        for i, mol1 in enumerate(self.molecules):
            if mol1 in molecules_to_remove:
                continue
                
            for j, mol2 in enumerate(self.molecules[i+1:], i+1):
                if mol2 in molecules_to_remove:
                    continue
                    
                # Calculate distance between molecules
                distance = np.linalg.norm(
                    self._minimum_image_distance(mol1.position, mol2.position)
                )
                
                # Check for reaction if molecules are close enough
                if distance < 1.0:  # Reaction radius
                    reaction = self._check_reaction(mol1, mol2)
                    if reaction and np.random.random() < reaction['rate'] * dt:
                        # Create product molecule
                        new_pos = (mol1.position + mol2.position) / 2
                        new_pos = self._apply_periodic_boundary(new_pos)
                        new_mol = Molecule(new_pos, reaction['product'], 0.1)
                        new_molecules.append(new_mol)
                        
                        # Mark reactants for removal
                        molecules_to_remove.add(mol1)
                        molecules_to_remove.add(mol2)
                        break
        
        # Update molecule list
        self.molecules = [m for m in self.molecules if m not in molecules_to_remove] + new_molecules
        
        # Record current state
        self._update_history(dt)

    def _minimum_image_distance(self, pos1, pos2):
        """Calculate minimum image distance under periodic boundary conditions"""
        delta = pos1 - pos2
        delta = np.where(delta > self.box_size/2, delta - self.box_size, delta)
        delta = np.where(delta < -self.box_size/2, delta + self.box_size, delta)
        return delta

    def _update_history(self, dt):
        """Update concentration history"""
        if not self.history['time']:
            self.history['time'].append(0)
        else:
            self.history['time'].append(self.history['time'][-1] + dt)
            
        # Count molecules of each type
        counts = {'A1': 0, 'A2': 0, 'A3': 0, 'F1': 0, 'F2': 0, 'F3': 0}
        for molecule in self.molecules:
            if molecule.species_type in counts:
                counts[molecule.species_type] += 1
        
        # Update concentration history
        volume = self.box_size ** 3
        for species in counts:
            concentration = counts[species] / volume
            self.history['concentrations'][species].append(concentration)

    def visualize_current_state(self):
        """Visualize current state of the system in 3D"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color scheme for different species
        colors = {
            'A1': 'red', 'A2': 'blue', 'A3': 'green',
            'F1': 'orange', 'F2': 'purple', 'F3': 'brown'
        }
        
        # Plot molecules
        for molecule in self.molecules:
            ax.scatter(*molecule.position, 
                      c=colors[molecule.species_type],
                      label=molecule.species_type)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Reaction Loop System')
        plt.show()

    def plot_concentration_history(self):
        """Plot concentration history"""
        plt.figure(figsize=(10, 6))
        
        for species in ['A1', 'A2', 'A3']:
            plt.plot(self.history['time'], 
                    self.history['concentrations'][species],
                    label=species)
        
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title('Species Concentration Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
def run_simulation():
    # Initialize simulation
    sim = ReactionLoop3D(box_size=10.0)
    
    # Run simulation for 1000 steps
    dt = 0.01
    for _ in range(1000):
        sim.step(dt)
        
        # Visualize every 100 steps
        if _ % 100 == 0:
            sim.visualize_current_state()
    
    # Plot final concentration history
    sim.plot_concentration_history()

if __name__ == "__main__":
    run_simulation()
