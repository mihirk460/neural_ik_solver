import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import numpy as np
import os

# --- ⚙️ CONTROL FLAGS ---
# Set these to True or False to control which parts of the script run
DO_DATA_COLLECTION = False
DO_TRAINING = False
DO_INFERENCE = True

# --- HYPERPARAMETERS ---
NUM_SAMPLES = 20000       # Number of data points to collect
EPOCHS = 100              # Number of training epochs
BATCH_SIZE = 128          # Batch size for training
LEARNING_RATE = 0.0001     # Learning rate for the optimizer

# --- FILE PATHS ---
DATA_FILE = './data/ik_data.pt'
MODEL_FILE = './models/ik_model.pth'

# --- 1. SETUP GPU DEVICE (MPS for Apple Silicon) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple MPS (GPU) for training.")
else:
    device = torch.device("cpu")
    print("🐌 MPS not available. Using CPU for training.")

# --- 2. DEFINE HELPER FUNCTIONS AND MODEL CLASS ---

def collect_data_ik(num_samples):
    """Generates high-quality data using PyBullet's IK solver."""
    print("Starting data collection with IK solver...")
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
    
    num_joints = p.getNumJoints(robot)
    end_effector_index = num_joints - 1
    
    positions = []
    joint_angles = []
    
    # Get the joint ranges
    joint_ranges = []
    for i in range(num_joints):
        info = p.getJointInfo(robot, i)
        joint_ranges.append((info[8], info[9])) # (lower limit, upper limit)

    collected_count = 0
    while collected_count < num_samples:
        # Generate a random target position within a reachable box
        target_pos = [
            np.random.uniform(0.3, 0.7),
            np.random.uniform(-0.4, 0.4),
            np.random.uniform(0.2, 0.8)
        ]

        # Use PyBullet's IK solver to find a valid joint configuration
        # This is slow, which is why we do it offline to generate a dataset
        joint_poses = p.calculateInverseKinematics(
            robot,
            end_effector_index,
            target_pos,
            lowerLimits=[r[0] for r in joint_ranges],
            upperLimits=[r[1] for r in joint_ranges],
            jointRanges=[r[1] - r[0] for r in joint_ranges]
        )

        if joint_poses and len(joint_poses) == num_joints:
            positions.append(target_pos)
            joint_angles.append(joint_poses)
            collected_count += 1

            if collected_count % 1000 == 0:
                print(f"  Collected {collected_count}/{num_samples} samples...")

    p.disconnect()
    print("Data collection complete.")
    return torch.tensor(positions, dtype=torch.float32), torch.tensor(joint_angles, dtype=torch.float32)

class IKNet(nn.Module):
    """A deeper neural network for more capacity."""
    def __init__(self, input_size, output_size):
        super(IKNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512) # Added a wider layer
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, output_size)
        # Dropout can help prevent overfitting in larger models
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.fc5(x) # No activation or dropout on the final output
        return x

# --- 3. MAIN SCRIPT LOGIC ---

# Get robot properties without a GUI connection
dummy_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robotId_dummy = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
NUM_JOINTS = p.getNumJoints(robotId_dummy)
END_EFFECTOR_INDEX = NUM_JOINTS - 1 # KUKA's last link is the end-effector
p.disconnect(dummy_client)

## DATA COLLECTION ##
if DO_DATA_COLLECTION or not os.path.exists(DATA_FILE):
    target_positions, target_joint_angles = collect_data_ik(num_samples=NUM_SAMPLES)
    print(f"Saving data to '{DATA_FILE}'...")
    torch.save((target_positions, target_joint_angles), DATA_FILE)
else:
    print(f"Loading existing data from '{DATA_FILE}'...")
    target_positions, target_joint_angles = torch.load(DATA_FILE)

## MODEL TRAINING ##
if DO_TRAINING:
    print("\n--- STARTING MODEL TRAINING ---")
    
    # Instantiate model and move it to the GPU
    model = IKNet(input_size=3, output_size=NUM_JOINTS).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create a dataset and dataloader for batching
    dataset = torch.utils.data.TensorDataset(target_positions, target_joint_angles)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Move data batches to the GPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.6f}')
        
    print("Training complete.")
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Model saved to '{MODEL_FILE}'.")

## INFERENCE (TESTING) ##
if DO_INFERENCE:
    print("\n--- RUNNING INFERENCE ---")
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found. Please train the model first.")
    else:
        # Load the trained model
        model = IKNet(input_size=3, output_size=NUM_JOINTS)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.to(device) # Move model to GPU for inference
        model.eval()     # Set model to evaluation mode

        # Connect to simulation with GUI
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        robotId = p.loadURDF("kuka_iiwa/model.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))
        
        # Define a target position in the robot's workspace
        target_pos_np = np.array([0.3, 0.5, 0.7])
        target_pos = torch.tensor(target_pos_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            predicted_angles = model(target_pos)
        
        # Move angles to CPU to use with PyBullet
        predicted_angles_cpu = predicted_angles.cpu().numpy()

        p.addUserDebugText("•", target_pos_np, textColorRGB=[1,0,0], textSize=2.0)
        
        for i in range(NUM_JOINTS):
            p.setJointMotorControl2(
                bodyIndex=robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=predicted_angles_cpu[i]
            )

        print("Model prediction sent to robot. Press Ctrl+C in terminal to exit.")
        try:
            while True:
                p.stepSimulation()
                time.sleep(1./240.)
        except KeyboardInterrupt:
            print("\nSimulation finished.")
        finally:
            p.disconnect()