import math
from dataclasses import dataclass
from typing import List, Callable, Optional

# --- Environment Definitions ---
@dataclass
class State:
    grip_width: float      # Continuous feature: 0.0 to 10.0
    object_friction: float # Continuous feature: 0.0 to 1.0
    
    def distance(self, other: 'State') -> float:
        # Euclidean distance in latent space
        return math.hypot(self.grip_width - other.grip_width, 
                          self.object_friction - other.object_friction)

# --- Dreth Architecture Components ---
class Nethra:
    """A certified 'handle' or policy (e.g., a wrapped CNN or RL policy)"""
    def __init__(self, name: str, policy_fn: Callable[[State], str]):
        self.name = name
        self.policy_fn = policy_fn
        self.earned_tareth: List[State] = [] # The local failure ledger
        self.boundary_radius = 1.5 # Radius of invalidation around a failure

    def can_handle(self, state: State) -> bool:
        # The forward pass is physically blocked if state is inside a known failure boundary
        for failure_state in self.earned_tareth:
            if state.distance(failure_state) < self.boundary_radius:
                return False
        return True

    def execute(self, state: State) -> str:
        return self.policy_fn(state)

    def log_failure(self, state: State):
        self.earned_tareth.append(state)


class DrethController:
    """The Nethra-of-Nethras: routes execution, audits failures, prunes problem space."""
    def __init__(self, available_nethras: List[Nethra]):
        self.nethras = available_nethras
        self.trass_ledger: List[State] = [] # The graveyard of intractable space
        self.trass_radius = 1.0
        
        # Meta-ledger to track how many times a region has failed across ALL tools
        self.region_failure_counts = {}

    def is_trass(self, state: State) -> bool:
        for trass_state in self.trass_ledger:
            if state.distance(trass_state) < self.trass_radius:
                return True
        return False

    def forward_pass(self, state: State) -> str:
        # 1. Check Intractability (Trass) -> Structural Halt (No Hallucination)
        if self.is_trass(state):
            return "STRUCTURAL HALT: Space certified as TRASS (Intractable). Compute refused."

        # 2. Find a certified Nethra whose failure ledger does not block this state
        certified_nethra = next((n for n in self.nethras if n.can_handle(state)), None)
        
        if certified_nethra:
            print(f"  -> Routing to: {certified_nethra.name}")
            return certified_nethra.execute(state)
        
        return "STRUCTURAL HALT: No certified Nethras available. Composition required."

    def process_feedback(self, state: State, attempted_nethra_name: str, success: bool):
        if success:
            return
            
        print(f"  -> FAILURE AUDIT: {attempted_nethra_name} failed at {state}.")
        
        # Log failure in the specific Nethra's ledger (Earned Tareth)
        nethra = next(n for n in self.nethras if n.name == attempted_nethra_name)
        nethra.log_failure(state)
        
        # Grid approximation to track global regional failures
        region_key = (round(state.grip_width), round(state.object_friction, 1))
        self.region_failure_counts[region_key] = self.region_failure_counts.get(region_key, 0) + 1
        
        # Garbage Collection / Trass Designation:
        # If the space fails 2+ times across different handles, designate as intractable.
        if self.region_failure_counts[region_key] >= 2:
            print(f"  -> TRASS CERTIFIED: Region {region_key} added to global failure ledger.")
            self.trass_ledger.append(state)


# --- Simulation Environment ---
def physics_simulator(state: State, action: str) -> bool:
    """Mock physics engine returning success or failure."""
    if state.object_friction <= 0.1: 
        return False # Physically impossible to grasp (ice ball)
    if action == "Action: Apply standard 50N force" and state.grip_width > 5.0:
        return False # Object too wide for standard grip, slips out
    if action == "Action: Apply delicate 10N force" and state.object_friction < 0.5:
        return False # Delicate grip on semi-slick object slips
    return True

# --- Test Execution ---
if __name__ == "__main__":
    # Define Nethras (base models)
    standard_nethra = Nethra("Standard_Grasp_CNN", lambda s: "Action: Apply standard 50N force")
    delicate_nethra = Nethra("Delicate_Grasp_RL", lambda s: "Action: Apply delicate 10N force")
    
    controller = DrethController([standard_nethra, delicate_nethra])

    scenarios = [
        ("Routine Grasp", State(4.0, 0.8)),
        ("Wide Object Grasp (Fails Standard)", State(6.0, 0.6)),
        ("Wide Object Grasp (Attempt 2)", State(6.1, 0.6)), 
        ("Ice Ball (Impossible to Grasp)", State(2.0, 0.05)),
        ("Ice Ball (Attempt 2)", State(2.1, 0.05)),
        ("Ice Ball (Attempt 3)", State(2.0, 0.05)), # Should trigger Structural Halt
    ]

    for name, state in scenarios:
        print(f"\n--- Testing Scenario: {name} | {state} ---")
        
        # 1. Forward Pass
        action = controller.forward_pass(state)
        print(f"  Output: {action}")
        
        # 2. If an action was taken, get environmental feedback
        if "STRUCTURAL HALT" not in action:
            # Extract Nethra name used for auditing
            used_nethra = "Standard_Grasp_CNN" if "standard" in action else "Delicate_Grasp_RL"
            success = physics_simulator(state, action)
            controller.process_feedback(state, used_nethra, success)
            if success: print("  -> Environment: SUCCESS")