import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ============================================================================
# 1. VANILLA RNN - FROM SCRATCH
# ============================================================================

class VanillaRNN:
    """A basic RNN implementation from scratch"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, X: np.ndarray, h_prev: np.ndarray = None) -> Tuple:
        """
        Forward pass through the RNN
        X: input sequence (seq_len, input_size)
        Returns: outputs, hidden states
        """
        seq_len = X.shape[0]
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        
        h_states = [h_prev]
        outputs = []
        
        for t in range(seq_len):
            x_t = X[t:t+1].T  # (input_size, 1)
            h_t = np.tanh(self.Wxh @ x_t + self.Whh @ h_prev + self.bh)
            y_t = self.Why @ h_t + self.by
            
            h_states.append(h_t)
            outputs.append(y_t)
            h_prev = h_t
        
        return np.hstack(outputs), h_states
    
    def backward(self, X: np.ndarray, Y: np.ndarray, h_states: List, 
                 outputs: np.ndarray) -> None:
        """Backpropagation through time (BPTT)"""
        seq_len = X.shape[0]
        
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        # Backprop through time
        for t in reversed(range(seq_len)):
            dy = outputs[:, t:t+1] - Y[t:t+1].T
            dWhy += dy @ h_states[t+1].T
            dby += dy
            
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - h_states[t+1]**2) * dh
            
            dWxh += dh_raw @ X[t:t+1]
            dWhh += dh_raw @ h_states[t].T
            dbh += dh_raw
            
            dh_next = self.Whh.T @ dh_raw
        
        # Clip gradients
        for dW in [dWxh, dWhh, dWhy]:
            np.clip(dW, -5, 5, out=dW)
        
        # Update weights
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
    
    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Single training step"""
        outputs, h_states = self.forward(X)
        loss = np.mean((outputs - Y.T)**2)
        self.backward(X, Y, h_states, outputs)
        return loss


# ============================================================================
# 2. LSTM - LONG SHORT-TERM MEMORY
# ============================================================================

class LSTMCell:
    """Single LSTM cell"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for input gate
        self.W_ii = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.W_if = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Cell gate
        self.W_ic = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output gate
        self.W_io = np.random.randn(hidden_size, input_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple:
        """
        LSTM forward pass
        x_t: input at time t (input_size, 1)
        h_prev: previous hidden state
        c_prev: previous cell state
        """
        # Input gate
        i_t = 1 / (1 + np.exp(-(self.W_ii @ x_t + self.W_hi @ h_prev + self.b_i)))
        
        # Forget gate
        f_t = 1 / (1 + np.exp(-(self.W_if @ x_t + self.W_hf @ h_prev + self.b_f)))
        
        # Cell gate
        c_tilde = np.tanh(self.W_ic @ x_t + self.W_hc @ h_prev + self.b_c)
        
        # Cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        o_t = 1 / (1 + np.exp(-(self.W_io @ x_t + self.W_ho @ h_prev + self.b_o)))
        
        # Hidden state
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t, (i_t, f_t, c_tilde, o_t)
    
    def get_parameters(self):
        """Return all parameters"""
        return {
            'W_ii': self.W_ii, 'W_hi': self.W_hi, 'b_i': self.b_i,
            'W_if': self.W_if, 'W_hf': self.W_hf, 'b_f': self.b_f,
            'W_ic': self.W_ic, 'W_hc': self.W_hc, 'b_c': self.b_c,
            'W_io': self.W_io, 'W_ho': self.W_ho, 'b_o': self.b_o,
        }


# ============================================================================
# 3. GRU - GATED RECURRENT UNIT
# ============================================================================

class GRUCell:
    """Single GRU cell"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate
        self.W_xr = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hr = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((hidden_size, 1))
        
        # Update gate
        self.W_xz = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))
        
        # Candidate hidden state
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> Tuple:
        """
        GRU forward pass
        x_t: input at time t (input_size, 1)
        h_prev: previous hidden state
        """
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        # Reset gate
        r_t = sigmoid(self.W_xr @ x_t + self.W_hr @ h_prev + self.b_r)
        
        # Update gate
        z_t = sigmoid(self.W_xz @ x_t + self.W_hz @ h_prev + self.b_z)
        
        # Candidate hidden state
        h_tilde = np.tanh(self.W_xh @ x_t + self.W_hh @ (r_t * h_prev) + self.b_h)
        
        # New hidden state
        h_t = (1 - z_t) * h_tilde + z_t * h_prev
        
        return h_t, (r_t, z_t, h_tilde)


# ============================================================================
# 4. EXAMPLE: SEQUENCE GENERATION
# ============================================================================

def example_sequence_generation():
    """Example: train RNN to generate sine wave"""
    np.random.seed(42)
    
    # Generate synthetic data
    t = np.linspace(0, 4*np.pi, 100)
    data = np.sin(t).reshape(-1, 1)
    
    # Prepare sequences
    seq_len = 10
    X_train = []
    Y_train = []
    
    for i in range(len(data) - seq_len):
        X_train.append(data[i:i+seq_len])
        Y_train.append(data[i+seq_len:i+seq_len+1])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # Train RNN
    rnn = VanillaRNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
    
    losses = []
    for epoch in range(50):
        epoch_loss = 0
        for i in range(len(X_train)):
            loss = rnn.train_step(X_train[i], Y_train[i])
            epoch_loss += loss
        losses.append(epoch_loss / len(X_train))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
    
    # Generate predictions
    predictions = []
    h_state = np.zeros((10, 1))
    
    for i in range(len(X_train)):
        output, _ = rnn.forward(X_train[i], h_state)
        predictions.append(output[0, -1])
        h_state = np.zeros((10, 1))  # Reset for new sequence
            
    print("\nTraining complete!")
    return losses, predictions, Y_train


if __name__ == "__main__":
    print("=" * 60)
    print("RNN IMPLEMENTATIONS")
    print("=" * 60)
    
    losses, predictions, targets = example_sequence_generation()
    
    print(f"\nFirst 5 predictions vs targets:")
    for i in range(5):
        print(f"Pred: {predictions[i]:.4f}, Target: {targets[i][0][0]:.4f}")