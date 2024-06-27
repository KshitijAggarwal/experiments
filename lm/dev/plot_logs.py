# %% 
import pylab as plt

def read_log_file(file_path):
    train_data = []
    val_data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                iteration = int(parts[0])
                data_type = parts[1]
                value = float(parts[2])
                
                if data_type == 'train':
                    train_data.append((iteration, value))
                elif data_type == 'val':
                    val_data.append((iteration, value))
    
    return train_data, val_data

def plot_data(train_data, val_data, label):
    train_iterations, train_values = zip(*train_data)
    val_iterations, val_values = zip(*val_data)
    
    # plt.figure(figsize=(10, 6))
    plt.plot(train_iterations, train_values, label=f'Train_{label}')
    plt.plot(val_iterations, val_values, label=f'Validation_{label}')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.ylim([3.5, 5])
    # plt.show()

# %% 

# Usage
file_path = '/root/experiments/lm/log/log_interp_29M.txt'
train_data, val_data = read_log_file(file_path)
plot_data(train_data, val_data, '29M')

file_path = '/root/experiments/lm/log/log_interp_17M.txt'
train_data, val_data = read_log_file(file_path)
plot_data(train_data, val_data, '17M')

# %% 