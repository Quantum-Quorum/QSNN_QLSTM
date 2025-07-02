import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tqdm.notebook import tqdm
import numpy as np
import time
import os
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import traceback

import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)
pnp.random.seed(42)


class LorentzVector:
    """
    Represents a 4-vector in special relativity (px, py, pz, E).
    Provides methods for common high-energy physics calculations like pt, eta, phi, and mass.
    """

    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E
        self._epsilon = 1e-10

    def pt(self):
        """Calculates the transverse momentum."""
        pt_sq = self.px ** 2 + self.py ** 2
        return np.sqrt(np.maximum(0, pt_sq))

    def eta(self):
        """
        Calculates the pseudorapidity.
        Handles cases where momentum is zero or very close to zero to avoid infinities.
        """
        p_sq = self.px ** 2 + self.py ** 2 + self.pz ** 2
        p = np.sqrt(np.maximum(0, p_sq))
        p_plus_pz = p + self.pz
        p_minus_pz = p - self.pz

        if isinstance(p, np.ndarray):
            eta_vals = np.zeros_like(p, dtype=float)

            safe_mask = (p_minus_pz > self._epsilon) & (p_plus_pz > self._epsilon) & (p > self._epsilon)
            eta_vals[safe_mask] = 0.5 * np.log(p_plus_pz[safe_mask] / p_minus_pz[safe_mask])


            inf_mask_pos = (~safe_mask) & (p > self._epsilon) & (self.pz > 0)
            inf_mask_neg = (~safe_mask) & (p > self._epsilon) & (self.pz < 0)
            eta_vals[inf_mask_pos] = np.inf
            eta_vals[inf_mask_neg] = -np.inf


            zero_p_mask = (p <= self._epsilon)
            eta_vals[zero_p_mask] = 0.0
            return eta_vals
        else:
            if p <= self._epsilon: return 0.0
            if p_minus_pz <= self._epsilon: return np.inf
            if p_plus_pz <= self._epsilon: return -np.inf
            return 0.5 * np.log(p_plus_pz / p_minus_pz)

    def phi(self):
        """Calculates the azimuthal angle."""
        return np.arctan2(self.py, self.px)

    def mass(self):
        """Calculates the invariant mass."""
        mass_sq = self.E ** 2 - self.px ** 2 - self.py ** 2 - self.pz ** 2
        return np.sqrt(np.maximum(0.0, mass_sq))

    def __add__(self, other):
        """Adds two LorentzVectors component-wise."""
        if isinstance(other, LorentzVector):
            return LorentzVector(self.px + other.px,
                                 self.py + other.py,
                                 self.pz + other.pz,
                                 self.E + other.E)
        return NotImplemented


def extract_parton_features(partons_raw_df):
    """
    Computes kinematic features for partons (e.g., pt, eta, phi, mass) and
    derived features for the parton system (e.g., delta_eta, delta_phi, invariant mass).
    Handles potential non-finite values by replacing them or dropping rows.
    """
    features = []

    required_cols = [f'parton_{i}_{comp}' for i in range(2) for comp in ['px', 'py', 'pz', 'E', 'id', 'charge']]
    if not all(col in partons_raw_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in partons_raw_df.columns]
        raise ValueError(f"Missing required raw parton columns: {missing}")

    print("Extracting Parton Features...")
    for index, row in tqdm(partons_raw_df.iterrows(), total=partons_raw_df.shape[0]):
        try:

            p0_vec = LorentzVector(px=row['parton_0_px'], py=row['parton_0_py'], pz=row['parton_0_pz'],
                                   E=row['parton_0_E'])
            p1_vec = LorentzVector(px=row['parton_1_px'], py=row['parton_1_py'], pz=row['parton_1_pz'],
                                   E=row['parton_1_E'])

            p_system = p0_vec + p1_vec


            p0_pt, p0_eta, p0_phi, p0_mass = p0_vec.pt(), p0_vec.eta(), p0_vec.phi(), p0_vec.mass()
            p1_pt, p1_eta, p1_phi, p1_mass = p1_vec.pt(), p1_vec.eta(), p1_vec.phi(), p1_vec.mass()


            parton_delta_eta = p0_eta - p1_eta if np.isfinite(p0_eta) and np.isfinite(p1_eta) else np.nan


            dphi = p0_phi - p1_phi
            parton_delta_phi_wrap = (dphi + np.pi) % (2 * np.pi) - np.pi


            parton_delta_R = np.sqrt(parton_delta_eta ** 2 + parton_delta_phi_wrap ** 2) if np.isfinite(
                parton_delta_eta) and np.isfinite(parton_delta_phi_wrap) else np.nan


            event_features = {
                'event_id': index,
                'p0_pt': p0_pt, 'p0_eta': p0_eta, 'p0_phi': p0_phi, 'p0_mass': p0_mass,
                'p0_id': row['parton_0_id'], 'p0_charge': row['parton_0_charge'],
                'p1_pt': p1_pt, 'p1_eta': p1_eta, 'p1_phi': p1_phi, 'p1_mass': p1_mass,
                'p1_id': row['parton_1_id'], 'p1_charge': row['parton_1_charge'],
                'parton_delta_eta': parton_delta_eta,
                'parton_delta_phi_abs': np.abs(parton_delta_phi_wrap),
                'parton_delta_phi_wrap': parton_delta_phi_wrap,
                'parton_delta_R': parton_delta_R,
                'parton_system_pt': p_system.pt(),
                'parton_system_eta': p_system.eta(),
                'parton_system_phi': p_system.phi(),
                'parton_system_mass': p_system.mass(),
                'parton_dot_prod': p0_vec.E * p1_vec.E - p0_vec.px * p1_vec.px - p0_vec.py * p1_vec.py - p0_vec.pz * p1_vec.pz,
            }
            features.append(event_features)
        except Exception as e:
            print(f"Warning: Error processing partons in event {index}: {e}")
            continue

    feature_df = pd.DataFrame(features)

    large_finite_val = 1e5
    feature_df.replace([np.inf, -np.inf], [large_finite_val, -large_finite_val], inplace=True)

    initial_count = len(feature_df)
    feature_df.dropna(inplace=True)
    if len(feature_df) < initial_count:
        print(f"Dropped {initial_count - len(feature_df)} rows containing NaNs during feature extraction.")

    if feature_df.empty:
        print("Warning: Feature DataFrame is empty after processing and NaN handling.")
    else:
        feature_df['event_id'] = feature_df['event_id'].astype(int)
    return feature_df


def extract_jet_targets(jets_raw_df, num_max_jets):
    """
    Calculates target variables (n_jets, leading_pt, subleading_pt) from raw jet data.
    Filters out invalid jets (e.g., zero momentum) and sorts by pT.
    """
    targets = []
    print("Extracting Jet Targets...")
    for index, row in tqdm(jets_raw_df.iterrows(), total=jets_raw_df.shape[0]):
        valid_jets = []

        for j in range(num_max_jets):
            px_val = row.get(f'jet_{j}_px', 0)
            py_val = row.get(f'jet_{j}_py', 0)
            pz_val = row.get(f'jet_{j}_pz', 0)
            E_val = row.get(f'jet_{j}_E', 0)


            if px_val == 0 and py_val == 0 and pz_val == 0 and E_val == 0:
                continue

            try:
                pt_val = np.sqrt(px_val ** 2 + py_val ** 2)

                if pt_val > 1e-6 and np.isfinite(pt_val):
                    valid_jets.append({'pt': pt_val})
            except Exception as e:
                print(f"Warning: Error calculating pt for jet {j} in event {index}: {e}")
                continue


        valid_jets.sort(key=lambda x: x['pt'], reverse=True)

        n_jets = len(valid_jets)
        leading_pt = valid_jets[0]['pt'] if n_jets > 0 else 0.0
        subleading_pt = valid_jets[1]['pt'] if n_jets > 1 else 0.0

        targets.append({
            'event_id': index,
            'n_jets': n_jets,
            'leading_pt': leading_pt,
            'subleading_pt': subleading_pt
        })

    target_df = pd.DataFrame(targets)
    if target_df.empty:
        print("Warning: Target DataFrame is empty after processing.")
    else:
        target_df['event_id'] = target_df['event_id'].astype(int)
    return target_df


def process_training_data(file_path):
    """
    Loads parton and jet data from an HDF5 file, extracts features (X) from partons,
    extracts targets (y) from jets, and merges them into a single DataFrame.
    """
    parton_features_df = None
    jet_targets_df = None
    merged_df = None

    start_time = time.time()
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified file does not exist: {file_path}")

        with h5py.File(file_path, 'r') as f:
            print(f"\n--- Processing training data from: {file_path} ---")
            keys = list(f.keys())
            print(f'Keys found: {keys}')


            if 'partons' in keys:
                partons_data = f['partons'][:]
                print(f"Raw parton data shape: {partons_data.shape}")

                if partons_data.ndim == 3 and partons_data.shape[1] == 2 and partons_data.shape[2] == 6:
                    num_events = partons_data.shape[0]
                    column_names = [f'parton_{i}_{comp}' for i in range(2) for comp in
                                    ['px', 'py', 'pz', 'E', 'id', 'charge']]
                    partons_raw_df = pd.DataFrame(partons_data.reshape(num_events, -1), columns=column_names)
                    parton_features_df = extract_parton_features(partons_raw_df)
                else:
                    print(f"Error: Unexpected parton data shape: {partons_data.shape}. Expected (N, 2, 6).")
            else:
                print("Error: 'partons' dataset not found. Cannot extract features.")
                return None


            if 'jets' in keys:
                jets_data = f['jets'][:]
                print(f"Raw jet data shape: {jets_data.shape}")

                if jets_data.ndim == 3 and jets_data.shape[2] == 4:
                    num_events_jets = jets_data.shape[0]
                    num_max_jets = jets_data.shape[1]
                    jet_column_names = [f'jet_{j}_{comp}' for j in range(num_max_jets) for comp in
                                        ['px', 'py', 'pz', 'E']]
                    jets_raw_df = pd.DataFrame(jets_data.reshape(num_events_jets, -1), columns=jet_column_names)
                    jet_targets_df = extract_jet_targets(jets_raw_df, num_max_jets)
                else:
                    print(f"Error: Unexpected jet data shape: {jets_data.shape}. Expected (N, N_max, 4).")
            else:
                print("Error: 'jets' dataset not found. Cannot extract targets.")
                return None


            if parton_features_df is not None and not parton_features_df.empty and \
                    jet_targets_df is not None and not jet_targets_df.empty:
                print(
                    f"Merging features ({len(parton_features_df)} events) and targets ({len(jet_targets_df)} events)...")

                merged_df = pd.merge(parton_features_df, jet_targets_df, on='event_id', how='inner')
                print(f"Merged data shape: {merged_df.shape}")
                if merged_df.empty:
                    print("Error: Merged DataFrame is empty. Check processing steps and event IDs.")
                    return None
            else:
                print("Error: Cannot merge due to empty features or targets DataFrame.")
                return None

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error processing training data file: {str(e)}")
        traceback.print_exc()
        return None

    end_time = time.time()
    print(f"Data processing finished in {end_time - start_time:.2f} seconds.")
    return merged_df


def prepare_data_regression(feature_target_df, feature_columns, target_columns):
    """
    Prepares data for a regression task.
    Extracts features (X) and targets (y) from the DataFrame,
    scales them using StandardScaler, and returns the scaled data and scalers.
    Handles non-finite values before scaling.
    """
    print("\n--- Preparing data for regression ---")
    if feature_target_df is None or feature_target_df.empty:
        print("Input DataFrame is None or empty.")
        return None, None, None, None


    missing_features = [col for col in feature_columns if col not in feature_target_df.columns]
    missing_targets = [col for col in target_columns if col not in feature_target_df.columns]
    if missing_features or missing_targets:
        print(f"Error: Missing feature columns: {missing_features}")
        print(f"Error: Missing target columns: {missing_targets}")
        return None, None, None, None

    X = feature_target_df[feature_columns].values
    y = feature_target_df[target_columns].values
    print(f"Raw data shapes: X={X.shape}, y={y.shape}")



    if np.any(~np.isfinite(X)):
        print("Warning: Non-finite values found in feature data before scaling. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~np.isfinite(y)):
        print("Warning: Non-finite values found in target data before scaling. Replacing with 0.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_x = StandardScaler()
    try:
        X_scaled = scaler_x.fit_transform(X)
    except ValueError as e:
        print(f"Error during feature scaling: {e}. Check for constant columns or zero variance.")

        variances = np.var(X, axis=0)
        constant_cols_indices = np.where(variances < 1e-9)[0]
        if len(constant_cols_indices) > 0:
            constant_cols_names = [feature_columns[i] for i in constant_cols_indices]
            print(
                f"Detected constant feature columns (zero variance) that cause issues with StandardScaler: {constant_cols_names}.")
            print("Consider removing these columns from your feature set.")
        return None, None, None, None

    scaler_y = StandardScaler()
    try:
        y_scaled = scaler_y.fit_transform(y)
    except ValueError as e:
        print(f"Error during target scaling: {e}. Check for constant columns or zero variance.")
        return None, None, None, None

    print(f"Scaling complete: X_scaled shape {X_scaled.shape}, y_scaled shape {y_scaled.shape}")
    return X_scaled, y_scaled, scaler_x, scaler_y


class QLSTMLayer(nn.Module):
    """
    A conceptual Quantum Long Short-Term Memory (QLSTM) layer.
    This layer processes sequential input data, with its internal gates (forget, input,
    cell state candidate, output) being computed by a parameterized quantum circuit.

    Note: For simplicity and integration with the existing static data pipeline,
    this implementation treats each input sample (batch_size, input_features)
    as a sequence of length 1. For true sequential data processing, the input
    shape should be (batch_size, sequence_length, input_features).
    """

    def __init__(self, input_size: int, hidden_size: int, num_qubits: int, quantum_layers: int = 2):
        """
        Initializes the QLSTM layer.

        Args:
            input_size (int): The number of expected features in the input `x` at each time step.
            hidden_size (int): The number of features in the hidden state `h`.
            num_qubits (int): Number of qubits for the quantum circuit within the QLSTM cell.
                              Must be >= (input_size + hidden_size) for embedding.
            quantum_layers (int): Number of layers in the variational quantum circuit for the gates.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits
        self.quantum_layers = quantum_layers


        self.total_classical_input_size = input_size + hidden_size


        # if self.num_qubits < self.total_classical_input_size:
        #     raise ValueError(
        #         f"num_qubits ({num_qubits}) must be >= (input_size + hidden_size) "
        #         f"({self.total_classical_input_size}) for AngleEmbedding in QLSTM. "
        #         f"Please increase num_qubits."
        #     )
        if self.num_qubits < 4:
            raise ValueError(
                f"num_qubits ({num_qubits}) must be >= 4 for measuring 4 gate outputs in QLSTM. "
                f"Please increase num_qubits."
            )

        try:
            self.dev = qml.device("lightning.qubit", wires=self.num_qubits)
            print(f"QLSTMLayer: Using lightning.qubit device with {self.num_qubits} wires.")
        except qml.DeviceError:

            self.dev = qml.device("default.qubit", wires=self.num_qubits)
            print(
                f"QLSTMLayer: lightning.qubit not available, falling back to default.qubit with {self.num_qubits} wires.")

        @qml.qnode(self.dev, interface="torch")
        def quantum_lstm_cell_circuit(inputs, weights):
            """
            Parameterized quantum circuit for computing the LSTM gate values.
            It takes the concatenated input and previous hidden state, and outputs
            expectation values for the forget, input, cell state candidate, and output gates.

            Args:
                inputs (torch.Tensor): Concatenated input features and previous hidden state.
                weights (torch.Tensor): Trainable weights for the variational circuit.

            Returns:
                list: List of 4 expectation values (between -1 and 1) for the gates.
            """

            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='X')


            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))


            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.quantum_lstm_cell_circuit = quantum_lstm_cell_circuit



        shape_weights = qml.StronglyEntanglingLayers.shape(n_layers=self.quantum_layers, n_wires=self.num_qubits)
        self.q_weights = nn.Parameter(torch.rand(shape_weights) * 2 * np.pi)



        self.linear_f = nn.Linear(1, hidden_size)
        self.linear_i = nn.Linear(1, hidden_size)
        self.linear_C = nn.Linear(1, hidden_size)
        self.linear_o = nn.Linear(1, hidden_size)


        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None, c_prev: torch.Tensor = None):
        """
        Forward pass of the QLSTM layer.

        Args:
            x (torch.Tensor): Input batch of features. Expected shape:
                              (batch_size, sequence_length, input_size).
                              For this simplified integration, we will treat
                              (batch_size, input_size) as (batch_size, 1, input_size).
            h_prev (torch.Tensor, optional): Previous hidden state. If None, initialized to zeros.
                                             Shape: (batch_size, hidden_size).
            c_prev (torch.Tensor, optional): Previous cell state. If None, initialized to zeros.
                                             Shape: (batch_size, hidden_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - h_out (torch.Tensor): The final hidden state for the sequence.
                                        Shape: (batch_size, hidden_size).
                - c_out (torch.Tensor): The final cell state for the sequence.
                                        Shape: (batch_size, hidden_size).
        """
        batch_size = x.size(0)

        if x.ndim == 2:
            x = x.unsqueeze(1)

        sequence_length = x.size(1)


        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev

        for t in range(sequence_length):
            x_t = x[:, t, :]


            quantum_outputs_batch = []
            for i in range(batch_size):

                combined_input = torch.cat([x_t[i], h_t[i]], dim=-1)


                if combined_input.size(0) < self.num_qubits:
                    padding = torch.zeros(self.num_qubits - combined_input.size(0), device=x.device)
                    padded_combined_input = torch.cat([combined_input, padding])
                else:

                    padded_combined_input = combined_input[:self.num_qubits]


                q_out_list = self.quantum_lstm_cell_circuit(padded_combined_input, self.q_weights)
                quantum_outputs_batch.append(torch.stack(q_out_list))


            quantum_outputs_batch_tensor = torch.stack(quantum_outputs_batch, dim=0)




            f_val_q = quantum_outputs_batch_tensor[:, 0].unsqueeze(1)
            i_val_q = quantum_outputs_batch_tensor[:, 1].unsqueeze(1)
            C_tilde_val_q = quantum_outputs_batch_tensor[:, 2].unsqueeze(1)
            o_val_q = quantum_outputs_batch_tensor[:, 3].unsqueeze(1)


            f_gate = self.sigmoid(self.linear_f(f_val_q))
            i_gate = self.sigmoid(self.linear_i(i_val_q))
            C_tilde = self.tanh(self.linear_C(C_tilde_val_q))
            o_gate = self.sigmoid(self.linear_o(o_val_q))


            c_t = f_gate * c_t + i_gate * C_tilde
            h_t = o_gate * self.tanh(c_t)

        return h_t, c_t


class QSNNModel(nn.Module):
    """
    A simple Feedforward Quantum Spiking Neural Network (QSNN) for classification.
    It now uses a QLSTM layer for processing input features, followed by a
    classical readout layer.

    Note: This implementation treats each input sample as a sequence of length 1
    for the QLSTM layer, as the original data pipeline provides static features per event.
    For true sequential data, the data preparation would need to be adapted
    to produce (batch_size, sequence_length, input_features) inputs.
    """

    def __init__(self, input_features: int, hidden_neurons: int, output_classes: int,
                 num_qubits_qlstm: int = 4, quantum_layers_qlstm: int = 2):
        """
        Initializes the QSNNModel with a QLSTM layer.

        Args:
            input_features (int): Number of input features per sample (becomes input_size for QLSTM).
            hidden_neurons (int): Number of hidden neurons in the QLSTM layer (becomes hidden_size for QLSTM).
            output_classes (int): Number of output classes for classification.
            num_qubits_qlstm (int): Number of qubits for the QLSTM layer's quantum circuit.
            quantum_layers_qlstm (int): Number of layers in the QLSTM layer's quantum circuit.
        """
        super().__init__()

        # The QLSTM layer
        self.qlstm_layer = QLSTMLayer(
            input_size=input_features,
            hidden_size=hidden_neurons,
            num_qubits=num_qubits_qlstm,
            quantum_layers=quantum_layers_qlstm
        )


        self.readout = nn.Linear(hidden_neurons, output_classes)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the QSNN with QLSTM.

        Args:
            x (torch.Tensor): Input batch of static features (batch_size, input_features).
                              This will be treated as a sequence of length 1 by QLSTM.

        Returns:
            torch.Tensor: Logits for classification (batch_size, output_classes).
        """


        hidden_state, _ = self.qlstm_layer(x)


        logits = self.readout(hidden_state)
        return logits


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: Adam, num_epochs: int,
                device: torch.device):
    """
    Trains the QSNN model.

    Args:
        model (nn.Module): The QSNN model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (Adam): Optimizer for updating model parameters.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on ('cpu' or 'cuda').

    Returns:
        tuple[list, list]: Lists of training loss and accuracy per epoch.
    """
    model.train()
    print(f"\n--- Starting Training on {device} ---", flush=True)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)


        print(f"--- Epoch {epoch + 1} batches processed. Calculating epoch summary... ---", flush=True)


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}", flush=True)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

    print("Training finished.", flush=True)
    return train_losses, train_accuracies


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluates the QSNN model on the test set.

    Args:
        model (nn.Module): The trained QSNN model.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray]: Test loss, test accuracy,
                                                                 true labels, predicted binary labels,
                                                                 and predicted probabilities.
    """
    model.eval()
    print("\n--- Starting Evaluation ---")
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_binary_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            predicted_probs = torch.sigmoid(outputs)
            predicted_binary = (predicted_probs > 0.5).float()

            correct_predictions += (predicted_binary == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_binary_predictions.extend(predicted_binary.cpu().numpy().flatten())
            all_probabilities.extend(predicted_probs.cpu().numpy().flatten())

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct_predictions / total_samples

    print(f"Evaluation Results:")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")


    print("\nClassification Report (Binary):")
    print(classification_report(all_labels, all_binary_predictions))

    return avg_loss, accuracy, np.array(all_labels), np.array(all_binary_predictions), np.array(all_probabilities)




def plot_training_history(train_losses, train_accuracies, num_epochs):
    """Plots training loss and accuracy over epochs."""
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'o-', label='Training Accuracy', color='green')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('qlstm_training_history.png')
    plt.tight_layout()



def plot_confusion_matrix(y_true, y_pred_binary, class_names=['Class 0', 'Class 1']):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('qlstm_confusion.png')



def plot_roc_curve(y_true, y_prob):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('qlstm_roc_curve.png')


if __name__ == "__main__":

    hep_data_path = '/Users/gouthamarcot/Documents/personal/codebase/Quantum_Agoize/cern-2025-challenge-2/.aqora/data/data/pp-z-to-jets-500K-57246.h5'



    num_qubits_qlstm = 8
    hidden_neurons = 4
    output_classes = 1

    quantum_layers_qlstm = 2
    num_epochs = 10
    batch_size = 5
    learning_rate = 0.05
    test_size_split = 0.2


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    merged_hep_df = process_training_data(hep_data_path)

    if merged_hep_df is None or merged_hep_df.empty:
        print("Failed to load or process HEP data. Please check file path and data integrity. Exiting.")
        exit()


    feature_columns = [
        'p0_pt', 'p0_eta', 'p0_phi', 'p0_mass', 'p0_id', 'p0_charge',
        'p1_pt', 'p1_eta', 'p1_phi', 'p1_mass', 'p1_id', 'p1_charge',
        'parton_delta_eta', 'parton_delta_phi_abs', 'parton_delta_phi_wrap',
        'parton_delta_R', 'parton_system_pt', 'parton_system_eta',
        'parton_system_phi', 'parton_system_mass', 'parton_dot_prod'
    ]
    target_columns = ['n_jets', 'leading_pt',
                      'subleading_pt']


    X_scaled_hep, y_scaled_hep_dummy, scaler_x_hep, scaler_y_hep_dummy = prepare_data_regression(
        merged_hep_df, feature_columns, target_columns
    )

    if X_scaled_hep is None:
        print("Failed to scale HEP features. Exiting.")
        exit()


    original_input_features = X_scaled_hep.shape[1]


    if original_input_features > num_qubits_qlstm:
        print(f"\nWarning: Original HEP features ({original_input_features}) > num_qubits_qlstm ({num_qubits_qlstm}).")
        print(f"Taking only the first {num_qubits_qlstm} features for the quantum embedding in QLSTM.")
        X_for_qlstm = X_scaled_hep[:, :num_qubits_qlstm]
    else:
        X_for_qlstm = X_scaled_hep

    input_features_qlstm = X_for_qlstm.shape[1]


    n_jets_original = merged_hep_df['n_jets'].values
    Y_binary_hep = (n_jets_original >= 2).astype(np.float32)
    Y_binary_hep = torch.tensor(Y_binary_hep).unsqueeze(1)


    X_for_qlstm = torch.tensor(X_for_qlstm).float().to(device)
    Y_for_qlstm = Y_binary_hep.to(device)

    num_samples_actual = X_for_qlstm.shape[0]

    print(f"\nAdapted HEP Data Shapes for QLSTM Model:")
    print(f"  X (features after reduction/selection): {X_for_qlstm.shape}")
    print(f"  Y (binary labels): {Y_for_qlstm.shape}")
    print(f"  Actual num_samples: {num_samples_actual}")
    print(
        f"  Input features for QLSTM model: {input_features_qlstm} (reduced from {original_input_features} if applicable)")
    print(
        f"  Binary label distribution: 0s={torch.sum(Y_for_qlstm == 0).item()}, 1s={torch.sum(Y_for_qlstm == 1).item()}")


    X_train, X_test, Y_train, Y_test = train_test_split(
        X_for_qlstm, Y_for_qlstm, test_size=test_size_split, random_state=42, stratify=Y_for_qlstm.cpu()
    )


    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print(f"\n--- Initializing Quantum Long Short-Term Memory (QLSTM) Model ---")
    qlstm_model = QSNNModel(input_features=input_features_qlstm,
                            hidden_neurons=hidden_neurons,
                            output_classes=output_classes,
                            num_qubits_qlstm=num_qubits_qlstm,
                            quantum_layers_qlstm=quantum_layers_qlstm).to(device)


    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(qlstm_model.parameters(), lr=learning_rate)

    print(f"QLSTM Model Architecture:\n{qlstm_model}")
    print(f"Total trainable parameters: {sum(p.numel() for p in qlstm_model.parameters() if p.requires_grad)}")


    training_start_time = time.time()
    train_losses, train_accuracies = train_model(qlstm_model, train_loader, criterion, optimizer, num_epochs, device)
    training_end_time = time.time()
    print(f"\nTotal training duration: {training_end_time - training_start_time:.2f} seconds.")


    evaluation_start_time = time.time()
    test_loss, test_accuracy, true_labels_eval, binary_predictions_eval, probabilities_eval = evaluate_model(
        qlstm_model,
        test_loader,
        criterion,
        device)
    evaluation_end_time = time.time()
    print(f"Total evaluation duration: {evaluation_end_time - evaluation_start_time:.2f} seconds.")


    print("\n--- Generating Plots ---")
    plot_training_history(train_losses, train_accuracies, num_epochs)
    plot_confusion_matrix(true_labels_eval, binary_predictions_eval, class_names=['n_jets < 2', 'n_jets >= 2'])
    plot_roc_curve(true_labels_eval, probabilities_eval)
