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

def plot_data(train_data, val_data):
    train_iterations, train_values = zip(*train_data)
    val_iterations, val_values = zip(*val_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_iterations, train_values, label='Train')
    plt.plot(val_iterations, val_values, label='Validation')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Training and Validation Data')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Usage
file_path = '/root/experiments/lm/log/log_interp_29M.txt'
train_data, val_data = read_log_file(file_path)
plot_data(train_data, val_data)

