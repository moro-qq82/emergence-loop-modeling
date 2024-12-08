import numpy as np
from scipy.integrate import odeint
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

class Molecule:
    def __init__(self, position, species_type, diffusion_coefficient):
        """
        Initialize a molecule in 3D space with reaction history
        
        Args:
            position (np.array): 3D coordinates [x, y, z]
            species_type (str): Chemical species identifier
            diffusion_coefficient (float): Diffusion coefficient
        """
        self.position = np.array(position, dtype=float)
        self.species_type = species_type
        self.diffusion_coefficient = diffusion_coefficient
        self.velocity = np.zeros(3)
        
        # Reaction history tracking
        self.history = {
            'reactions': [],  # List of (time, position, reactants, product) tuples
            'species_sequence': [species_type],  # Sequence of species in reaction chain
            'positions': [position.copy()],  # Positions where reactions occurred
            'times': [0.0]  # Times when reactions occurred
        }
        self.loop_detected = False
        self.current_loop = None

    def diffuse(self, dt, membrane_bounds=None):
        """
        Simulate Brownian motion with optional membrane constraints
        
        Args:
            dt (float): Time step
            membrane_bounds (tuple): Optional (min_z, max_z) for membrane boundaries
        """
        random_displacement = np.random.normal(0, np.sqrt(2*self.diffusion_coefficient*dt), 3)
        new_position = self.position + random_displacement
        
        # Apply membrane constraints if specified
        if membrane_bounds and self.species_type.startswith('A'):
            min_z, max_z = membrane_bounds
            if new_position[2] < min_z or new_position[2] > max_z:
                new_position[2] = self.position[2]  # Bounce back from membrane
        
        self.position = new_position

    def update_history(self, time, position, reactants, product):
        """Update reaction history"""
        self.history['reactions'].append((time, position.copy(), reactants, product))
        self.history['species_sequence'].append(product)
        self.history['positions'].append(position.copy())
        self.history['times'].append(time)
        self._check_for_loop()

    def _check_for_loop(self):
        """Check for reaction loops in history"""
        sequence = self.history['species_sequence']
        if len(sequence) < 3:  # Need at least 3 species for a loop
            return
        
        # Check for loops of any size
        current_species = sequence[-1]
        for i, species in enumerate(sequence[:-1]):
            if species == current_species:
                # Loop found
                self.loop_detected = True
                self.current_loop = {
                    'species': sequence[i:],
                    'positions': self.history['positions'][i:],
                    'times': self.history['times'][i:],
                    'cycle_time': self.history['times'][-1] - self.history['times'][i]
                }
                return

class ReactionLoop3D:
    def __init__(self, box_size, membrane_bounds=None, initial_molecules=None):
        """
        Initialize a 3D reaction loop simulation with membrane
        
        Args:
            box_size (float): Size of cubic simulation box
            membrane_bounds (tuple): Optional (min_z, max_z) for membrane boundaries
            initial_molecules (dict): Initial molecule configurations
        """
        self.box_size = box_size
        self.membrane_bounds = membrane_bounds
        self.molecules = []
        self.reaction_constants = {
            ('A1', 'F1'): {'product': 'A2', 'rate': 0.1},
            ('A2', 'F2'): {'product': 'A3', 'rate': 0.1},
            ('A3', 'F3'): {'product': 'A1', 'rate': 0.1}
        }
        
        # Track all detected loops
        self.detected_loops = []
        self.loop_creation_times = []
        
        if initial_molecules is None:
            self._initialize_default_molecules()
        else:
            self._initialize_custom_molecules(initial_molecules)
        
        self.history = {
            'time': [],
            'concentrations': defaultdict(list),
            'loop_events': []  # Track loop formation/destruction events
        }

    def _initialize_default_molecules(self):
        """Set up default initial molecule distribution with membrane consideration"""
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
                # For A-type molecules, initialize within membrane bounds if specified
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

    def step(self, dt, current_time):
        """Perform one simulation timestep"""
        # Diffusion
        for molecule in self.molecules:
            molecule.diffuse(dt, self.membrane_bounds)
            molecule.position = self._apply_periodic_boundary(molecule.position)
        
        # Reactions
        new_molecules = []
        molecules_to_remove = set()
        
        for i, mol1 in enumerate(self.molecules):
            if mol1 in molecules_to_remove:
                continue
                
            for j, mol2 in enumerate(self.molecules[i+1:], i+1):
                if mol2 in molecules_to_remove:
                    continue
                    
                distance = np.linalg.norm(
                    self._minimum_image_distance(mol1.position, mol2.position)
                )
                
                if distance < 1.0:  # Reaction radius
                    reaction = self._check_reaction(mol1, mol2)
                    if reaction and np.random.random() < reaction['rate'] * dt:
                        new_pos = (mol1.position + mol2.position) / 2
                        new_pos = self._apply_periodic_boundary(new_pos)
                        
                        # Create new molecule with combined history
                        new_mol = Molecule(new_pos, reaction['product'], 0.1)
                        new_mol.update_history(
                            current_time,
                            new_pos,
                            (mol1.species_type, mol2.species_type),
                            reaction['product']
                        )
                        
                        # Transfer history from reactants
                        if mol1.history['reactions']:
                            new_mol.history['reactions'].extend(mol1.history['reactions'])
                        if mol2.history['reactions']:
                            new_mol.history['reactions'].extend(mol2.history['reactions'])
                        
                        new_molecules.append(new_mol)
                        molecules_to_remove.add(mol1)
                        molecules_to_remove.add(mol2)
                        
                        # Check if this creates a new loop
                        if new_mol.loop_detected:
                            self._process_new_loop(new_mol, current_time)
                        
                        break
        
        # Update molecule list
        self.molecules = [m for m in self.molecules if m not in molecules_to_remove] + new_molecules
        
        # Record current state
        self._update_history(dt, current_time)

    def _process_new_loop(self, molecule, current_time):
        """Process and analyze a newly detected reaction loop"""
        if molecule.current_loop:
            self.detected_loops.append(molecule.current_loop)
            self.loop_creation_times.append(current_time)
            
            # Analyze spatial proximity with other loops
            self._analyze_loop_proximity(molecule.current_loop)
            
            # Record loop event
            self.history['loop_events'].append({
                'time': current_time,
                'type': 'creation',
                'loop_data': molecule.current_loop
            })

    def _analyze_loop_proximity(self, new_loop):
        """Analyze spatial proximity between loops"""
        if len(self.detected_loops) < 2:
            return
        
        # Calculate centroid of the new loop
        new_centroid = np.mean(new_loop['positions'], axis=0)
        
        # Compare with existing loops
        for existing_loop in self.detected_loops[:-1]:  # Exclude the new loop
            existing_centroid = np.mean(existing_loop['positions'], axis=0)
            
            # Calculate maximum internal distance in each loop
            new_max_dist = max(np.linalg.norm(p - new_centroid) 
                             for p in new_loop['positions'])
            existing_max_dist = max(np.linalg.norm(p - existing_centroid) 
                                  for p in existing_loop['positions'])
            
            # Check proximity condition
            centroid_distance = np.linalg.norm(new_centroid - existing_centroid)
            if centroid_distance < max(new_max_dist, existing_max_dist):
                # Loops are spatially proximate
                self.history['loop_events'].append({
                    'time': self.history['time'][-1],
                    'type': 'proximity',
                    'loops': (new_loop, existing_loop)
                })

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
    
    def _minimum_image_distance(self, pos1, pos2):
        """Calculate minimum image distance under periodic boundary conditions"""
        delta = pos1 - pos2
        delta = np.where(delta > self.box_size/2, delta - self.box_size, delta)
        delta = np.where(delta < -self.box_size/2, delta + self.box_size, delta)
        return delta
    
    def _update_history(self, dt, current_time):
        """
        Update concentration history and record loop events
        
        Args:
            dt (float): Time step
            current_time (float): Current simulation time
        """
        # Record time
        self.history['time'].append(current_time)
            
        # Count molecules of each type
        counts = defaultdict(int)
        volume = self.box_size ** 3
        
        for molecule in self.molecules:
            counts[molecule.species_type] += 1
        
        # Update concentrations for all possible species
        all_species = {'A1', 'A2', 'A3', 'F1', 'F2', 'F3'}
        for species in all_species:
            concentration = counts[species] / volume
            self.history['concentrations'][species].append(concentration)
        
        # Record loop events from this timestep
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
        
        # Update loop statistics
        if active_loops:
            self.history['loop_stats'] = self.history.get('loop_stats', []) + [{
                'time': current_time,
                'active_loops': len(active_loops),
                'avg_cycle_time': np.mean([loop['cycle_time'] for loop in active_loops]),
                'loop_details': active_loops
            }]
            
        # Calculate additional statistics if needed
        if len(self.history['time']) > 1:
            # Calculate concentration changes
            for species in all_species:
                current_conc = self.history['concentrations'][species][-1]
                prev_conc = self.history['concentrations'][species][-2]
                
                # Record significant changes (optional)
                if abs(current_conc - prev_conc) > 0.1:  # Threshold for significant change
                    event = {
                        'time': current_time,
                        'type': 'concentration_change',
                        'species': species,
                        'change': current_conc - prev_conc
                    }
                    self.history['events'] = self.history.get('events', []) + [event]
        
        # Memory management - optional: limit history length if needed
        max_history_length = 10000  # Adjust as needed
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
    
    def visualize_loops(self):
        """Visualize reaction loops and their spatial relationships"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot membrane if present
        if self.membrane_bounds:
            min_z, max_z = self.membrane_bounds
            x = y = np.linspace(0, self.box_size, 10)
            X, Y = np.meshgrid(x, y)
            for z in [min_z, max_z]:
                Z = np.full_like(X, z)
                ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
        # Plot each detected loop
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.detected_loops)))
        for loop, color in zip(self.detected_loops, colors):
            positions = np.array(loop['positions'])
            # Plot reaction sites
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=[color], alpha=0.6)
            # Connect reaction sites with lines
            for i in range(len(positions)-1):
                ax.plot([positions[i,0], positions[i+1,0]],
                       [positions[i,1], positions[i+1,1]],
                       [positions[i,2], positions[i+1,2]],
                       c=color, alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Reaction Loops Visualization')
        plt.show()

    def plot_loop_statistics(self):
        """Plot statistics about detected loops"""
        if not self.detected_loops:
            print("No loops detected yet.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loop cycle times
        cycle_times = [loop['cycle_time'] for loop in self.detected_loops]
        ax1.hist(cycle_times, bins='auto')
        ax1.set_xlabel('Loop Cycle Time')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Loop Cycle Times')
        
        # Plot loop creation timeline
        ax2.plot(self.loop_creation_times, range(len(self.loop_creation_times)), 'o-')
        ax2.set_xlabel('Simulation Time')
        ax2.set_ylabel('Cumulative Loops')
        ax2.set_title('Loop Formation Timeline')
        
        plt.tight_layout()
        plt.show()

    def plot_history(self):
        """Plot comprehensive history of the simulation"""
        if not self.history['time']:
            print("No history data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot concentrations
        for species in self.history['concentrations']:
            ax1.plot(self.history['time'], 
                    self.history['concentrations'][species],
                    label=species)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Species Concentrations Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loop statistics if available
        if 'loop_stats' in self.history and self.history['loop_stats']:
            times = [stat['time'] for stat in self.history['loop_stats']]
            active_loops = [stat['active_loops'] for stat in self.history['loop_stats']]
            avg_cycles = [stat['avg_cycle_time'] for stat in self.history['loop_stats']]
            
            ax2.plot(times, active_loops, 'b-', label='Active Loops')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(times, avg_cycles, 'r--', label='Avg Cycle Time')
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Number of Active Loops', color='b')
            ax2_twin.set_ylabel('Average Cycle Time', color='r')
            
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        plt.tight_layout()
        plt.show()

# Example usage
def run_simulation():
    # Initialize simulation with membrane
    membrane_bounds = (4.0, 6.0)  # Membrane between z=4 and z=6
    sim = ReactionLoop3D(box_size=10.0, membrane_bounds=membrane_bounds)
    
    # Run simulation
    dt = 0.01
    total_steps = 1000
    
    for step in range(total_steps):
        current_time = step * dt
        sim.step(dt, current_time)
        
        # Visualize periodically
        if step % 200 == 0:
            sim.visualize_loops()
            sim.plot_loop_statistics()
            sim.plot_history()

if __name__ == "__main__":
    run_simulation()
