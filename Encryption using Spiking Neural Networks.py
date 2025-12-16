#!/usr/bin/env python
# coding: utf-8

# # SNN_AES

# In[65]:


import numpy as np
import nest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import random

# Directory to store the results
results_dir = "TextResults/"

# Function to generate synthetic speaker data
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

# Convert text to ASCII values
def text_to_ascii(text):
    return [ord(c) for c in text]

# Encrypt ASCII values using AES in CBC mode
def encrypt_text(ascii_values, key):
    cipher = AES.new(key, AES.MODE_CBC)
    padded_ascii = pad(bytes(ascii_values), AES.block_size)
    encrypted = cipher.encrypt(padded_ascii)
    return cipher.iv + encrypted

# Decrypt ASCII values using AES in CBC mode
def decrypt_text(encrypted_data, key):
    iv = encrypted_data[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(encrypted_data[AES.block_size:]), AES.block_size)
    return list(decrypted)

# Define the ascii_to_current function
def ascii_to_current(ascii_value, scale=100):
    return ascii_value * scale

# Enhanced encoding: Encode ASCII value into a spiking pattern
def encode_to_spike_pattern(ascii_value, num_neurons, duration):
    spike_pattern = np.zeros((num_neurons, int(duration)))
    spike_times = np.linspace(0, duration, ascii_value % num_neurons + 1)
    for i, spike_time in enumerate(spike_times):
        if i < num_neurons:
            spike_pattern[i, int(spike_time):] = 1
    return spike_pattern

# Simulate raster plot for ASCII values across three layers with noise
def simulate_raster_plot_ascii_with_noise(ascii_values, current_func, sim_time=200.0, pd=13.0):
    print("Simulating...")

    nest.ResetKernel()
    nest.SetKernelStatus({'print_time': False})

    num_neurons = len(ascii_values)

    neuron_params = {
        "C_m": 250.0,
        "tau_m": 20.0,
        "t_ref": 2.0,
        "E_L": -70.0,
        "V_reset": -70.0,
        "V_th": -55.0,
    }

    # Create layers
    layer_L1 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L2 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L3 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)

    # Create noise layer
    noise_layer = nest.Create('poisson_generator', 1, params={'rate': 8000.0})  # Adjust rate as needed

    # Create spike recorders
    spikerecorder_L1 = nest.Create("spike_recorder")
    spikerecorder_L2 = nest.Create("spike_recorder")
    spikerecorder_L3 = nest.Create("spike_recorder")
    spikerecorder_noise = nest.Create("spike_recorder")

    # Encode the text into spiking patterns
    for i, ascii_value in enumerate(ascii_values):
        spike_pattern = encode_to_spike_pattern(ascii_value, num_neurons, sim_time)
        current = current_func(np.sum(spike_pattern), scale=100)
        nest.SetStatus(layer_L1[i:i + 1], {"I_e": current})

    # One-to-one connections between layers with increased weight and reduced delay
    nest.Connect(layer_L1, layer_L2, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})
    nest.Connect(layer_L2, layer_L3, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})

    # Connect noise layer to specific neurons in L2
    connection_mapping = {0: [1, 4, 7, 10]}  # Example: Connecting noise to 4 neurons in L2
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect([noise_layer[noise_neuron].get('global_id')], [layer_L2[target_neuron].get('global_id')],
                         syn_spec={"weight": 50.0})

    # Connect spike recorders
    nest.Connect(layer_L1, spikerecorder_L1)
    nest.Connect(layer_L2, spikerecorder_L2)
    nest.Connect(layer_L3, spikerecorder_L3)
    nest.Connect(noise_layer, spikerecorder_noise)

    nest.Simulate(sim_time + pd)
    print("Simulation completed.")

    for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
        events = spikerecorder.get("events")
        times = events["times"]
        senders = events["senders"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            print(f"Subtracting the propagation delay: {pd} ms.")
            times -= pd
            print(f"0 {int(np.min(times))} {int(np.max(times))}")

    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "w") as file:
        for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
            events = spikerecorder.get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(layer_name)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["simulation_time"] = sim_time

    plt.figure(figsize=(15, 10))

    for i, (layer_name, spikerecorder) in enumerate(
            [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]):
        plt.subplot(4, 1, i + 1)
        plt.title(f'ASCII Values - {layer_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()

        times = spikerecorder.get("events")["times"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            senders = spikerecorder.get("events")["senders"]
            plt.vlines(times, senders - 1, senders, color='blue', linewidths=0.5)

    # Plot the noise layer separately
    plt.subplot(4, 1, 4)
    plt.title('Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Noise Neuron Index')
    plt.grid()

    noise_times = spikerecorder_noise.get("events")["times"]
    noise_senders = spikerecorder_noise.get("events")["senders"]
    plt.vlines(noise_times, noise_senders - 1, noise_senders, color='red', linewidths=0.5)

    plt.tight_layout()
    plt.show()

# Function to decode spikes back to ASCII values from Layer 3 only
def decode_from_layer3(layer3_data, num_neurons):
    decoded_values = []
    for i in range(num_neurons):
        spike_times = np.where(layer3_data == i)[0]
        decoded_value = len(spike_times)
        decoded_values.append(decoded_value)
    return decoded_values

# Main Execution
key = get_random_bytes(16)  # Generate a random 16-byte AES key
print(f"Generated AES Key: {key.hex()}")  # Print the key in hexadecimal format

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate synthetic data
speakers_df = generate_speakers(categories, cities)

# Convert the synthetic data to a string representation (e.g., JSON or CSV format)
text = speakers_df.to_csv(index=False)

# Convert the text to ASCII values
ascii_values = text_to_ascii(text)

# Encrypt the ASCII values
encrypted_ascii = encrypt_text(ascii_values, key)

print(f"Length of encrypted data: {len(encrypted_ascii)} bytes")
print(f"Expected length (IV + ciphertext): {AES.block_size + len(pad(bytes(ascii_values), AES.block_size))} bytes")

# Simulate the neural spiking process
simulate_raster_plot_ascii_with_noise(encrypted_ascii, ascii_to_current)

# Now, to decode (This step would typically be handled in a separate process)
# Load spike data from Layer 3 and decode back to ASCII
with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "r") as file:
    senders_L3 = file["L3"]["senders"][:]

    # Decode directly from Layer 3 data
    decoded_ascii = decode_from_layer3(senders_L3, len(ascii_values))

try:
    decrypted_ascii = decrypt_text(encrypted_ascii, key)
    decoded_text = ''.join(chr(c) for c in decrypted_ascii)
    print(f"Original Text: {text}")
    print(f"Decoded Text: {decoded_text}")
except ValueError as e:
    print(f"Decryption error: {e}")


# In[67]:


import numpy as np
import nest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import random

results_dir = "TextResults/"

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

# Convert text to ASCII values
def text_to_ascii(text):
    return [ord(c) for c in text]

# Encrypt ASCII values using AES in CBC mode
def encrypt_text(ascii_values, key):
    cipher = AES.new(key, AES.MODE_CBC)
    padded_ascii = pad(bytes(ascii_values), AES.block_size)
    encrypted = cipher.encrypt(padded_ascii)
    return cipher.iv + encrypted

# Decrypt ASCII values using AES in CBC mode
def decrypt_text(encrypted_data, key):
    iv = encrypted_data[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(encrypted_data[AES.block_size:]), AES.block_size)
    return list(decrypted)

# Define the ascii_to_current function
def ascii_to_current(ascii_value, scale=100):
    return ascii_value * scale

# Enhanced encoding: Encode ASCII value into a spiking pattern
def encode_to_spike_pattern(ascii_value, num_neurons, duration):
    spike_pattern = np.zeros((num_neurons, int(duration)))
    spike_times = np.linspace(0, duration, ascii_value % num_neurons + 1)
    for i, spike_time in enumerate(spike_times):
        if i < num_neurons:
            spike_pattern[i, int(spike_time):] = 1
    return spike_pattern

# Simulate raster plot for ASCII values across three layers with noise
def simulate_raster_plot_ascii_with_noise(ascii_values, current_func, sim_time=200.0, pd=13.0):
    print("Simulating...")

    nest.ResetKernel()
    nest.SetKernelStatus({'print_time': False})

    num_neurons = len(ascii_values)

    neuron_params = {
        "C_m": 250.0,
        "tau_m": 20.0,
        "t_ref": 2.0,
        "E_L": -70.0,
        "V_reset": -70.0,
        "V_th": -55.0,
    }

    # Create layers
    layer_L1 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L2 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L3 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)

    # Create noise layer
    noise_layer = nest.Create('poisson_generator', 1, params={'rate': 8000.0})  # Adjust rate as needed

    # Create spike recorders
    spikerecorder_L1 = nest.Create("spike_recorder")
    spikerecorder_L2 = nest.Create("spike_recorder")
    spikerecorder_L3 = nest.Create("spike_recorder")
    spikerecorder_noise = nest.Create("spike_recorder")

    # Encode the text into spiking patterns
    for i, ascii_value in enumerate(ascii_values):
        spike_pattern = encode_to_spike_pattern(ascii_value, num_neurons, sim_time)
        current = current_func(np.sum(spike_pattern), scale=100)
        nest.SetStatus(layer_L1[i:i + 1], {"I_e": current})

    # One-to-one connections between layers with increased weight and reduced delay
    nest.Connect(layer_L1, layer_L2, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})
    nest.Connect(layer_L2, layer_L3, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})

    # Connect noise layer to specific neurons in L2
    connection_mapping = {0: [1, 4, 7, 10]}  # Example: Connecting noise to 4 neurons in L2
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect([noise_layer[noise_neuron].get('global_id')], [layer_L2[target_neuron].get('global_id')],
                         syn_spec={"weight": 50.0})

    # Connect spike recorders
    nest.Connect(layer_L1, spikerecorder_L1)
    nest.Connect(layer_L2, spikerecorder_L2)
    nest.Connect(layer_L3, spikerecorder_L3)
    nest.Connect(noise_layer, spikerecorder_noise)

    nest.Simulate(sim_time + pd)
    print("Simulation completed.")

    for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
        events = spikerecorder.get("events")
        times = events["times"]
        senders = events["senders"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            print(f"Subtracting the propagation delay: {pd} ms.")
            times -= pd
            print(f"0 {int(np.min(times))} {int(np.max(times))}")

    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "w") as file:
        for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
            events = spikerecorder.get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(layer_name)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["simulation_time"] = sim_time

    plt.figure(figsize=(15, 10))

    for i, (layer_name, spikerecorder) in enumerate(
            [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]):
        plt.subplot(4, 1, i + 1)
        plt.title(f'ASCII Values - {layer_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()

        times = spikerecorder.get("events")["times"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            senders = spikerecorder.get("events")["senders"]
            plt.vlines(times, senders - 1, senders, color='blue', linewidths=0.5)

    # Plot the noise layer separately
    plt.subplot(4, 1, 4)
    plt.title('Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Noise Neuron Index')
    plt.grid()

    noise_times = spikerecorder_noise.get("events")["times"]
    noise_senders = spikerecorder_noise.get("events")["senders"]
    plt.vlines(noise_times, noise_senders - 1, noise_senders, color='red', linewidths=0.5)

    plt.tight_layout()
    plt.show()

# Function to decode spikes back to ASCII values from Layer 3 only
def decode_from_layer3(layer3_data, num_neurons):
    decoded_values = []
    for i in range(num_neurons):
        spike_times = np.where(layer3_data == i)[0]
        decoded_value = len(spike_times)
        decoded_values.append(decoded_value)
    return decoded_values

# Main Execution
key = get_random_bytes(16)  # Generate a random 16-byte AES key
print(f"Generated AES Key: {key.hex()}")  # Print the key in hexadecimal format

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate synthetic data
speakers_df = generate_speakers(categories, cities)

# Convert the synthetic data to a string representation (e.g., JSON or CSV format)
text = speakers_df.to_csv(index=False)

# Convert the text to ASCII values
ascii_values = text_to_ascii(text)

# Encrypt the ASCII values
encrypted_ascii = encrypt_text(ascii_values, key)

print(f"Length of encrypted data: {len(encrypted_ascii)} bytes")
print(f"Expected length (IV + ciphertext): {AES.block_size + len(pad(bytes(ascii_values), AES.block_size))} bytes")

# Simulate the neural spiking process
simulate_raster_plot_ascii_with_noise(encrypted_ascii, ascii_to_current)

# Now, to decode (This step would typically be handled in a separate process)
# Load spike data from Layer 3 and decode back to ASCII
with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "r") as file:
    senders_L3 = file["L3"]["senders"][:]

    # Decode directly from Layer 3 data
    decoded_ascii = decode_from_layer3(senders_L3, len(ascii_values))

try:
    decrypted_ascii = decrypt_text(encrypted_ascii, key)
    decoded_text = ''.join(chr(c) for c in decrypted_ascii)
    print(f"Original Text: {text}")
    print(f"Decoded Text: {decoded_text}")
except ValueError as e:
    print(f"Decryption error: {e}")


# # SNN_S-DES

# In[69]:


import numpy as np
import nest
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import random

results_dir = "TextResults/"

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

# S-DES Encryption and Decryption functions
def s_des_encrypt(plaintext, key):
    def permute(bits, permutation):
        return [bits[i] for i in permutation]

    def ip(bits):
        return permute(bits, [1, 5, 2, 0, 3, 7, 4, 6])

    def ip_inv(bits):
        return permute(bits, [3, 0, 2, 4, 6, 1, 7, 5])

    def expand_permute(bits):
        return permute(bits, [3, 0, 1, 2, 1, 2, 3, 0])

    def sbox(bits, sbox):
        row = (bits[0] << 1) + bits[3]
        col = (bits[1] << 1) + bits[2]
        return [(sbox[row][col] & 0b10) >> 1, sbox[row][col] & 0b01]

    def f_k(bits, subkey):
        l, r = bits[:4], bits[4:]
        ep = expand_permute(r)
        xor_ep = [a ^ b for a, b in zip(ep, subkey)]
        s0 = sbox(xor_ep[:4], [[1, 0, 3, 2], [3, 2, 1, 0], [0, 2, 1, 3], [3, 1, 3, 2]])
        s1 = sbox(xor_ep[4:], [[0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 0], [2, 1, 0, 3]])
        p4 = permute(s0 + s1, [1, 3, 2, 0])
        return [a ^ b for a, b in zip(l, p4)] + r

    def swap(bits):
        return bits[4:] + bits[:4]

    def key_gen(key):
        p10 = permute(key, [2, 4, 1, 6, 3, 9, 0, 8, 7, 5])
        ls1 = lambda k: k[1:] + k[:1]
        ls2 = lambda k: k[2:] + k[:2]
        subkey1 = permute(ls1(p10[:5]) + ls1(p10[5:]), [5, 2, 6, 3, 7, 4, 9, 8])
        subkey2 = permute(ls2(p10[:5]) + ls2(p10[5:]), [5, 2, 6, 3, 7, 4, 9, 8])
        return subkey1, subkey2

    bits = [int(x) for x in format(plaintext, '08b')]
    key_bits = [int(x) for x in format(key, '010b')]
    subkey1, subkey2 = key_gen(key_bits)
    bits = ip(bits)
    bits = f_k(bits, subkey1)
    bits = swap(bits)
    bits = f_k(bits, subkey2)
    return int(''.join(map(str, ip_inv(bits))), 2)

def s_des_decrypt(ciphertext, key):
    def permute(bits, permutation):
        return [bits[i] for i in permutation]

    def ip(bits):
        return permute(bits, [1, 5, 2, 0, 3, 7, 4, 6])

    def ip_inv(bits):
        return permute(bits, [3, 0, 2, 4, 6, 1, 7, 5])

    def expand_permute(bits):
        return permute(bits, [3, 0, 1, 2, 1, 2, 3, 0])

    def sbox(bits, sbox):
        row = (bits[0] << 1) + bits[3]
        col = (bits[1] << 1) + bits[2]
        return [(sbox[row][col] & 0b10) >> 1, sbox[row][col] & 0b01]

    def f_k(bits, subkey):
        l, r = bits[:4], bits[4:]
        ep = expand_permute(r)
        xor_ep = [a ^ b for a, b in zip(ep, subkey)]
        s0 = sbox(xor_ep[:4], [[1, 0, 3, 2], [3, 2, 1, 0], [0, 2, 1, 3], [3, 1, 3, 2]])
        s1 = sbox(xor_ep[4:], [[0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 0], [2, 1, 0, 3]])
        p4 = permute(s0 + s1, [1, 3, 2, 0])
        return [a ^ b for a, b in zip(l, p4)] + r

    def swap(bits):
        return bits[4:] + bits[:4]

    def key_gen(key):
        p10 = permute(key, [2, 4, 1, 6, 3, 9, 0, 8, 7, 5])
        ls1 = lambda k: k[1:] + k[:1]
        ls2 = lambda k: k[2:] + k[:2]
        subkey1 = permute(ls1(p10[:5]) + ls1(p10[5:]), [5, 2, 6, 3, 7, 4, 9, 8])
        subkey2 = permute(ls2(p10[:5]) + ls2(p10[5:]), [5, 2, 6, 3, 7, 4, 9, 8])
        return subkey1, subkey2

    bits = [int(x) for x in format(ciphertext, '08b')]
    key_bits = [int(x) for x in format(key, '010b')]
    subkey1, subkey2 = key_gen(key_bits)
    bits = ip(bits)
    bits = f_k(bits, subkey2)
    bits = swap(bits)
    bits = f_k(bits, subkey1)
    return int(''.join(map(str, ip_inv(bits))), 2)

# Convert text to ASCII values
def text_to_ascii(text):
    return [ord(c) for c in text]

# Encrypt ASCII values using S-DES
def encrypt_text(ascii_values, key):
    return [s_des_encrypt(value, key) for value in ascii_values]

# Decrypt ASCII values using S-DES
def decrypt_text(encrypted_data, key):
    return [s_des_decrypt(value, key) for value in encrypted_data]

# Define the ascii_to_current function
def ascii_to_current(ascii_value, scale=100):
    return ascii_value * scale

# Enhanced encoding: Encode ASCII value into a spiking pattern
def encode_to_spike_pattern(ascii_value, num_neurons, duration):
    spike_pattern = np.zeros((num_neurons, int(duration)))
    spike_times = np.linspace(0, duration, ascii_value % num_neurons + 1)
    for i, spike_time in enumerate(spike_times):
        if i < num_neurons:
            spike_pattern[i, int(spike_time):] = 1
    return spike_pattern

# Simulate raster plot for ASCII values across three layers with noise
def simulate_raster_plot_ascii_with_noise(ascii_values, current_func, sim_time=200.0, pd=13.0):
    print("Simulating...")

    nest.ResetKernel()
    nest.SetKernelStatus({'print_time': False})

    num_neurons = len(ascii_values)

    neuron_params = {
        "C_m": 250.0,
        "tau_m": 20.0,
        "t_ref": 2.0,
        "E_L": -70.0,
        "V_reset": -70.0,
        "V_th": -55.0,
    }

    # Create layers
    layer_L1 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L2 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L3 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)

    # Create noise layer
    noise_layer = nest.Create('poisson_generator', 1, params={'rate': 8000.0})  # Adjust rate as needed

    # Create spike recorders
    spikerecorder_L1 = nest.Create("spike_recorder")
    spikerecorder_L2 = nest.Create("spike_recorder")
    spikerecorder_L3 = nest.Create("spike_recorder")
    spikerecorder_noise = nest.Create("spike_recorder")

    # Encode the text into spiking patterns
    for i, ascii_value in enumerate(ascii_values):
        spike_pattern = encode_to_spike_pattern(ascii_value, num_neurons, sim_time)
        current = current_func(np.sum(spike_pattern), scale=100)
        nest.SetStatus(layer_L1[i:i + 1], {"I_e": current})

    # One-to-one connections between layers with increased weight and reduced delay
    nest.Connect(layer_L1, layer_L2, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})
    nest.Connect(layer_L2, layer_L3, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})

    # Connect noise layer to specific neurons in L2
    connection_mapping = {0: [1, 4, 7, 10]}  # Example: Connecting noise to 4 neurons in L2
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect([noise_layer[noise_neuron].get('global_id')], [layer_L2[target_neuron].get('global_id')],
                         syn_spec={"weight": 50.0})

    # Connect spike recorders
    nest.Connect(layer_L1, spikerecorder_L1)
    nest.Connect(layer_L2, spikerecorder_L2)
    nest.Connect(layer_L3, spikerecorder_L3)
    nest.Connect(noise_layer, spikerecorder_noise)

    nest.Simulate(sim_time + pd)
    print("Simulation completed.")

    for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
        events = spikerecorder.get("events")
        times = events["times"]
        senders = events["senders"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            print(f"Subtracting the propagation delay: {pd} ms.")
            times -= pd
            print(f"0 {int(np.min(times))} {int(np.max(times))}")

    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "w") as file:
        for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
            events = spikerecorder.get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(layer_name)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["simulation_time"] = sim_time

    plt.figure(figsize=(15, 10))

    for i, (layer_name, spikerecorder) in enumerate(
            [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]):
        plt.subplot(4, 1, i + 1)
        plt.title(f'ASCII Values - {layer_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()

        times = spikerecorder.get("events")["times"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            senders = spikerecorder.get("events")["senders"]
            plt.vlines(times, senders - 1, senders, color='blue', linewidths=0.5)

    # Plot the noise layer separately
    plt.subplot(4, 1, 4)
    plt.title('Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Noise Neuron Index')
    plt.grid()

    noise_times = spikerecorder_noise.get("events")["times"]
    noise_senders = spikerecorder_noise.get("events")["senders"]
    plt.vlines(noise_times, noise_senders - 1, noise_senders, color='red', linewidths=0.5)

    plt.tight_layout()
    plt.show()

# Function to decode spikes back to ASCII values from Layer 3 only
def decode_from_layer3(layer3_data, num_neurons):
    decoded_values = []
    for i in range(num_neurons):
        spike_times = np.where(layer3_data == i)[0]
        decoded_value = len(spike_times)
        decoded_values.append(decoded_value)
    return decoded_values

# Main Execution
key = 0b1010000010  # 10-bit S-DES key
print(f"Generated S-DES Key: {format(key, '010b')}")  # Print the key in binary format

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate synthetic data
speakers_df = generate_speakers(categories, cities)

# Convert the synthetic data to a string representation (e.g., JSON or CSV format)
text = speakers_df.to_csv(index=False)

# Convert the text to ASCII values
ascii_values = text_to_ascii(text)

# Encrypt the ASCII values
encrypted_ascii = encrypt_text(ascii_values, key)

print(f"Encrypted ASCII: {encrypted_ascii}")

# Simulate the neural spiking process
simulate_raster_plot_ascii_with_noise(encrypted_ascii, ascii_to_current)

# Now, to decode (This step would typically be handled in a separate process)
# Load spike data from Layer 3 and decode back to ASCII
with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "r") as file:
    senders_L3 = file["L3"]["senders"][:]

    # Decode directly from Layer 3 data
    decoded_ascii = decode_from_layer3(senders_L3, len(ascii_values))

decrypted_ascii = decrypt_text(encrypted_ascii, key)
decoded_text = ''.join(chr(c) for c in decrypted_ascii)

print(f"Original Text: {text}")
print(f"Decoded Text: {decoded_text}")


# # SNN_RSA

# In[41]:


import numpy as np
import nest
import os
import h5py
import matplotlib.pyplot as plt
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes

results_dir = "TextResults/"

# RSA Encryption and Decryption functions
def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def rsa_encrypt(plaintext, public_key):
    rsa_key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    return cipher.encrypt(plaintext)

def rsa_decrypt(ciphertext, private_key):
    rsa_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    return cipher.decrypt(ciphertext)

# Convert text to ASCII values
def text_to_ascii(text):
    return text.encode('ascii')

# Encrypt ASCII values using RSA
def encrypt_text(ascii_values, public_key):
    return rsa_encrypt(ascii_values, public_key)

# Decrypt ASCII values using RSA
def decrypt_text(encrypted_data, private_key):
    return rsa_decrypt(encrypted_data, private_key)

# Define the ascii_to_current function
def ascii_to_current(ascii_value, scale=100):
    return ascii_value * scale

# Enhanced encoding: Encode ASCII value into a spiking pattern
def encode_to_spike_pattern(ascii_value, num_neurons, duration):
    spike_pattern = np.zeros((num_neurons, int(duration)))
    spike_times = np.linspace(0, duration, ascii_value % num_neurons + 1)
    for i, spike_time in enumerate(spike_times):
        if i < num_neurons:
            spike_pattern[i, int(spike_time):] = 1
    return spike_pattern

# Simulate raster plot for ASCII values across three layers with noise
def simulate_raster_plot_ascii_with_noise(ascii_values, current_func, sim_time=200.0, pd=13.0):
    print("Simulating...")

    nest.ResetKernel()
    nest.SetKernelStatus({'print_time': False})

    num_neurons = len(ascii_values)

    neuron_params = {
        "C_m": 250.0,
        "tau_m": 20.0,
        "t_ref": 2.0,
        "E_L": -70.0,
        "V_reset": -70.0,
        "V_th": -55.0,
    }

    # Create layers
    layer_L1 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L2 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)
    layer_L3 = nest.Create('iaf_psc_alpha', num_neurons, params=neuron_params)

    # Create noise layer
    noise_layer = nest.Create('poisson_generator', 1, params={'rate': 8000.0})  # Adjust rate as needed

    # Create spike recorders
    spikerecorder_L1 = nest.Create("spike_recorder")
    spikerecorder_L2 = nest.Create("spike_recorder")
    spikerecorder_L3 = nest.Create("spike_recorder")
    spikerecorder_noise = nest.Create("spike_recorder")

    # Encode the text into spiking patterns
    for i, ascii_value in enumerate(ascii_values):
        spike_pattern = encode_to_spike_pattern(ascii_value, num_neurons, sim_time)
        current = current_func(np.sum(spike_pattern), scale=100)
        nest.SetStatus(layer_L1[i:i + 1], {"I_e": current})

    # One-to-one connections between layers with increased weight and reduced delay
    nest.Connect(layer_L1, layer_L2, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})
    nest.Connect(layer_L2, layer_L3, conn_spec="one_to_one", syn_spec={"weight": 1000.0, "delay": 0.1})

    # Connect noise layer to specific neurons in L2
    connection_mapping = {0: [1, 4, 7, 10]}  # Example: Connecting noise to 4 neurons in L2
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect([noise_layer[noise_neuron].get('global_id')], [layer_L2[target_neuron].get('global_id')],
                         syn_spec={"weight": 50.0})

    # Connect spike recorders
    nest.Connect(layer_L1, spikerecorder_L1)
    nest.Connect(layer_L2, spikerecorder_L2)
    nest.Connect(layer_L3, spikerecorder_L3)
    nest.Connect(noise_layer, spikerecorder_noise)

    nest.Simulate(sim_time + pd)
    print("Simulation completed.")

    for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
        events = spikerecorder.get("events")
        times = events["times"]
        senders = events["senders"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            print(f"Subtracting the propagation delay: {pd} ms.")
            times -= pd
            print(f"0 {int(np.min(times))} {int(np.max(times))}")

    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "w") as file:
        for layer_name, spikerecorder in [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]:
            events = spikerecorder.get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(layer_name)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["simulation_time"] = sim_time

    plt.figure(figsize=(15, 10))

    for i, (layer_name, spikerecorder) in enumerate(
            [("L1", spikerecorder_L1), ("L2", spikerecorder_L2), ("L3", spikerecorder_L3)]):
        plt.subplot(4, 1, i + 1)
        plt.title(f'ASCII Values - {layer_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()

        times = spikerecorder.get("events")["times"]
        if len(times) == 0:
            print(f"Warning: No spikes recorded in {layer_name}!")
        else:
            senders = spikerecorder.get("events")["senders"]
            plt.vlines(times, senders - 1, senders, color='blue', linewidths=0.5)

    # Plot the noise layer separately
    plt.subplot(4, 1, 4)
    plt.title('Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Noise Neuron Index')
    plt.grid()

    noise_times = spikerecorder_noise.get("events")["times"]
    noise_senders = spikerecorder_noise.get("events")["senders"]
    plt.vlines(noise_times, noise_senders - 1, noise_senders, color='red', linewidths=0.5)

    plt.tight_layout()
    plt.show()

# Function to decode spikes back to ASCII values from Layer 3 only
def decode_from_layer3(layer3_data, num_neurons):
    decoded_values = []
    for i in range(num_neurons):
        spike_times = np.where(layer3_data == i)[0]
        decoded_value = len(spike_times)
        decoded_values.append(decoded_value)
    return decoded_values

# Main Execution
private_key, public_key = generate_rsa_keys()
print(f"Generated RSA Public Key:\n{public_key.decode('ascii')}")
print(f"Generated RSA Private Key:\n{private_key.decode('ascii')}")

text = "hello world"
ascii_values = text_to_ascii(text)

encrypted_ascii = encrypt_text(ascii_values, public_key)

print(f"Encrypted ASCII: {encrypted_ascii}")

simulate_raster_plot_ascii_with_noise(encrypted_ascii, ascii_to_current)

# Now, to decode (This step would typically be handled in a separate process)
# Load spike data from Layer 3 and decode back to ASCII
with h5py.File(os.path.join(results_dir, "ascii_spikes_with_noise.h5"), "r") as file:
    senders_L3 = file["L3"]["senders"][:]

    # Decode directly from Layer 3 data
    decoded_ascii = decode_from_layer3(senders_L3, len(ascii_values))

decrypted_ascii = decrypt_text(encrypted_ascii, private_key)
decoded_text = decrypted_ascii.decode('ascii')

print(f"Original Text: {text}")
print(f"Decoded Text: {decoded_text}")


# In[ ]:





# # SNN_AES_STDP 

# In[42]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']


# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)


speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)


# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)


save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)


# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)


loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)


# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]


def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)


# AES Encryption and Decryption Functions
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plain_text, AES.block_size))
    return cipher.iv + ct_bytes


def aes_decrypt(cipher_text, key):
    iv = cipher_text[:16]
    ct = cipher_text[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)


# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)


# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))


# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)


# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values


# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(12, 10))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights[0], label='Layer 1 -> Ih Layer', color='red')
    plt.plot(weights[1], label='Layer 2 -> Ih Layer', color='blue')
    plt.plot(weights[2], label='Ih Layer -> Layer 2', color='green')
    plt.xlabel('Simulation Step')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    os.makedirs("SNNresults", exist_ok=True)
    plt.savefig("SNNresults/SNNwithSTDP_AES_raster_plot.png")
    plt.show()


# Simulate the neural network with spike train input
def simulate_neural_network_with_input(spike_train, num_steps=10,
                                       simulation_duration=10.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons_layer1 = 10
    num_neurons_layer2 = 16
    num_neurons_layer3 = 10

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", num_neurons_layer1),
        'layer2': nest.Create("iaf_psc_alpha", num_neurons_layer2),
        'layer3': nest.Create("iaf_psc_alpha", num_neurons_layer3),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}
    print("Number of neurons in Layer 1:", len(neuron_layers['layer1']))

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1lih = {"synapse_model": "stdp_synapse", "weight": -1200.0}
    syn_spec_l2l3 = {"weight": 400.0}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_l2ih = {"synapse_model": "stdp_synapse", "weight": 1200.0}
    syn_spec_ihl2 = {"synapse_model": "stdp_synapse", "weight": -600.0}

    # Connect neurons between layers
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Define the connections for lateral inhibition all-to-all
    for i in range(num_neurons_layer1):
        for j in range(18):
            nest.Connect(neuron_layers['layer1'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l1lih)
    for i in range(num_neurons_layer2):
        for j in range(18):
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l2ih)
            nest.Connect(neuron_layers['lateral_ih'][j], neuron_layers['layer2'][i], syn_spec=syn_spec_ihl2)

    # Define synaptic weight recording for STDP connections
    stdp_synapse_weights_l1lih = []
    stdp_synapse_weights_l2ih = []
    stdp_synapse_weights_ihl2 = []

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=num_neurons_layer1)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
        nest.Simulate(simulation_duration)

        # Record synaptic weights after each simulation step
        stdp_synapse_weights_l1lih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer1'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_l2ih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_ihl2.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['lateral_ih'], neuron_layers['layer2']), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]

    # Convert lists to numpy arrays for easier manipulation
    stdp_synapse_weights_l1lih = np.array(stdp_synapse_weights_l1lih)
    stdp_synapse_weights_l2ih = np.array(stdp_synapse_weights_l2ih)
    stdp_synapse_weights_ihl2 = np.array(stdp_synapse_weights_ihl2)

    # Print or visualize the recorded synaptic weights
    print("Synaptic weights from layer 1 to lateral inhibition layer (l1 to lih):")
    print(stdp_synapse_weights_l1lih)

    print("Synaptic weights from layer 2 to lateral inhibition layer (l2 to lih):")
    print(stdp_synapse_weights_l2ih)

    print("Synaptic weights from lateral inhibition layer to layer 2 (lih to l2):")
    print(stdp_synapse_weights_ihl2)

    # Save the final weights
    save_weights(stdp_synapse_weights_l2ih[-1], "weights_l2ih.npy")

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    weights = [stdp_synapse_weights_l1lih, stdp_synapse_weights_l2ih, stdp_synapse_weights_ihl2]
    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights)

    return ts_layer1, ts_layer2, ts_layer3

def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)


# Purpura distance function
def purpura_distance(spiketrains, cost):
    """
    Calculates the Victor-Purpura distance for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                if len(st1) == 0 or len(st2) == 0:
                    continue  # Skip empty spike trains

                idx1, idx2 = 0, 0
                len_st1, len_st2 = len(st1), len(st2)

                while idx1 < len_st1 and idx2 < len_st2:
                    time_diff = st1[idx1] - st2[idx2]

                    if time_diff > 0:
                        distance += np.exp(-time_diff / cost)
                        idx2 += 1
                    elif time_diff < 0:
                        distance += np.exp(time_diff / cost)
                        idx1 += 1
                    else:
                        idx1 += 1
                        idx2 += 1

                # Add remaining spikes from longer spike train
                while idx1 < len_st1:
                    if len_st2 > 0:
                        distance += np.exp(-(st1[idx1] - st2[-1]) / cost)
                    else:
                        distance += np.exp(-(st1[idx1] - st1[-1]) / cost)
                    idx1 += 1

                while idx2 < len_st2:
                    if len_st1 > 0:
                        distance += np.exp(-(st1[-1] - st2[idx2]) / cost)
                    else:
                        distance += np.exp(-(st2[idx2] - st2[-1]) / cost)
                    idx2 += 1

    return distance


# van Rossum metric function
def rossum_metric(*spiketrains, tau):
    """
    Calculates the van Rossum metric for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                for spike_time1 in st1:
                    for spike_time2 in st2:
                        distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance


# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate AES key
key = get_random_bytes(16)  # AES-128 key

# Print the generated key in hexadecimal format
print("Generated AES Key:", key.hex())

# Encrypt and decrypt the message using AES for the binary values
plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = aes_encrypt(plain_text_combined, key)
decrypted_text_combined = aes_decrypt(cipher_text_combined, key)

decrypted_ascii_values_combined = list(decrypted_text_combined)

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key[0])  # Using the first byte of the AES key

# Simulate the neural network with the spike train as input
ts_layer1, ts_layer2, ts_layer3 = simulate_neural_network_with_input(spike_train_combined)

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key[0])

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)

# Calculate Purpura distance
cost = 1.0  # Example cost parameter
purpura_dist = purpura_distance([ts_layer1, ts_layer2, ts_layer3], cost)
print(f"Purpura Distance: {purpura_dist}")

# Calculate van Rossum metric
tau = 2.0  # Example tau parameter
rossum_dist = rossum_metric(ts_layer1, ts_layer2, ts_layer3, tau=tau)
print(f"van Rossum Metric: {rossum_dist}")


# SNN_AES-STDP and RESUME

# In[108]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# AES Encryption and Decryption Functions
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plain_text, AES.block_size))
    return cipher.iv + ct_bytes

def aes_decrypt(cipher_text, key):
    iv = cipher_text[:16]
    ct = cipher_text[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,
                senders_teaching_layer, ts_teaching_layer):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # Teaching Layer
    plt.subplot(3, 2, 6)
    plt.title('Spike Raster Plot - Teaching Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_teaching_layer, ts_teaching_layer):
        plt.vlines(spike_time, sender, sender + 1, color='pink')

    plt.tight_layout()
    plt.savefig("SNNresults/SNN_AES STDP and RESUME_raster_plot.png")
    plt.show()

# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)

# Simulate the neural network with spike train input
def simulate_neural_network_with_input(spike_train, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons = len(spike_train)

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", 10),
        'layer2': nest.Create("iaf_psc_alpha", 16),
        'layer3': nest.Create("iaf_psc_alpha", 10),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18),
        'teaching': nest.Create("iaf_psc_alpha", 10)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "stdp_triplet_synapse", "weight": 400.0, "Wmax": 900}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_tll3 = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -100.0}
    syn_spec_l3ih = {"weight": 350.0}

    # Connect layer 1 to layer 2
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)

    # Connect layer 2 to layer 3
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Connect teaching layer to layer 3
    for i in range(len(neuron_layers['teaching'])):
        nest.Connect(neuron_layers['teaching'][i], neuron_layers['layer3'][i], syn_spec=syn_spec_tll3)

    # Connect layer 3 to lateral inhibition layer
    nest.Connect(neuron_layers['layer3'], neuron_layers['lateral_ih'], syn_spec=syn_spec_l3ih)

    # Connect lateral inhibition layer to layer 2
    nest.Connect(neuron_layers['lateral_ih'], neuron_layers['layer2'], syn_spec=syn_spec_ihl2)

    # Connect noise layer to layer 2
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron], syn_spec=syn_spec_lnl2)

    # Simulation steps
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=2)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
            nest.SetStatus(neuron_layers['teaching'][i], {"I_e": current})

        # Simulate the network
        nest.Simulate(simulation_duration)
    print("Number of neurons in Layer 1:", len(neuron_layers['layer1']))
    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorders['teaching'], "events")[0]

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]

    # Save final weights
    print("Saving weights ...")
    save_weights(np.asarray(nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['layer3']), "weight")), "weights_l2l3.npy")

    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,
                senders_teaching_layer, ts_teaching_layer)
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text

simulate_neural_network_with_input([], num_steps=200, simulation_duration=50.0, min_current=300.0, max_current=600.0)

# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate AES key
key = get_random_bytes(16)  # AES-128 key

# Print the generated key in hexadecimal format
print("Generated AES Key:", key.hex())

# Encrypt and decrypt the message using AES for the binary values
plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = aes_encrypt(plain_text_combined, key)
decrypted_text_combined = aes_decrypt(cipher_text_combined, key)

decrypted_ascii_values_combined = list(decrypted_text_combined)

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key[0])  # Using the first byte of the AES key

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key[0])

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)


# # SNN with Lateral Inhibition- Training using resume with noise

# In[109]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# AES Encryption and Decryption Functions
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plain_text, AES.block_size))
    return cipher.iv + ct_bytes

def aes_decrypt(cipher_text, key):
    iv = cipher_text[:16]
    ct = cipher_text[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.scatter(range(len(weights)), weights, label='Layer 2 -> Layer 3', color='red')
    plt.xlabel('Layer 2 -> synapse index')
    plt.ylabel('Synaptic Weight')
    plt.title('Pre-trained Synaptic Weights')
    plt.grid()

    plt.tight_layout()
    plt.savefig("SNNresults/SNN_Training_Resume_Noise_raster_plot.png")
    plt.show()

def load_weights(filename):
    file_path = os.path.join("final_weights", filename)
    try:
        weights = np.load(file_path, allow_pickle=True)
        return weights
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def purpura_distance(spiketrain1, spiketrain2, cost):
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)
    distance = 0.0
    i = 0
    j = 0
    while i < len(spiketrain1) and j < len(spiketrain2):
        time_diff = spiketrain1[i] - spiketrain2[j]
        if time_diff > 0:
            distance += np.exp(-time_diff / cost)
            j += 1
        elif time_diff < 0:
            distance += np.exp(time_diff / cost)
            i += 1
        else:
            i += 1
            j += 1
    while i < len(spiketrain1):
        distance += np.exp(-(spiketrain1[i] - spiketrain2[-1]) / cost)
        i += 1
    while j < len(spiketrain2):
        distance += np.exp(-(spiketrain1[-1] - spiketrain2[j]) / cost)
        j += 1
    return distance

def rossum_metric(spiketrain1, spiketrain2, tau):
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)
    distance = 0.0
    for spike_time1 in spiketrain1:
        for spike_time2 in spiketrain2:
            distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)
    return distance

def simulate_neural_network_with_input(spike_train, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", 10),
        'layer2': nest.Create("iaf_psc_alpha", 16),  
        'layer3': nest.Create("iaf_psc_alpha", 10),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18),
        'teaching': nest.Create("iaf_psc_alpha", 10)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "stdp_triplet_synapse", "weight": 400.0, "Wmax": 900}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_tll3 = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -100.0}
    syn_spec_l3ih = {"weight": 350.0}

    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)

    print("Loading weights ...")
    weights_l2l3 = load_weights("weights_l2l3.npy")
    expected_size = len(neuron_layers['layer2']) * len(neuron_layers['layer3'])
    if weights_l2l3 is None or len(weights_l2l3) != expected_size:
        raise ValueError(
            f"Expected weights of size {expected_size}, but got {None if weights_l2l3 is None else len(weights_l2l3)}")

    for i in range(len(neuron_layers['layer2'])):
        for j in range(len(neuron_layers['layer3'])):
            weight_index = i * len(neuron_layers['layer3']) + j
            weight = weights_l2l3[weight_index]
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['layer3'][j], syn_spec={"weight": weight})

    for i in range(len(neuron_layers['teaching'])):
        nest.Connect(neuron_layers['teaching'][i], neuron_layers['layer3'][i], syn_spec=syn_spec_tll3)

    nest.Connect(neuron_layers['layer3'], neuron_layers['lateral_ih'], syn_spec=syn_spec_l3ih)
    nest.Connect(neuron_layers['lateral_ih'], neuron_layers['layer2'], syn_spec=syn_spec_ihl2)
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron], syn_spec=syn_spec_lnl2)

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        random_currents = np.random.uniform(min_current, max_current, size=10)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
            nest.SetStatus(neuron_layers['teaching'][i], {"I_e": current})

        nest.Simulate(simulation_duration)
    

    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorders['teaching'], "events")[0]

    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]
    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]
    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]
    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]
    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]
    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]

    cost = 1.0  
    tau = 1.0  

    n0_spikes_layer1 = ts_layer1[senders_layer1 == neuron_layers['layer1'][0]]
    n1_spikes_layer1 = ts_layer1[senders_layer1 == neuron_layers['layer1'][1]]
    n0_spikes_layer3 = ts_layer3[senders_layer3 == neuron_layers['layer3'][0]]
    n1_spikes_layer3 = ts_layer3[senders_layer3 == neuron_layers['layer3'][1]]

    purpura_distance_n0 = purpura_distance(n0_spikes_layer1, n0_spikes_layer3, cost)
    purpura_distance_n1 = purpura_distance(n1_spikes_layer1, n1_spikes_layer3, cost)

    rossum_metric_n0 = rossum_metric(n0_spikes_layer1, n0_spikes_layer3, tau)
    rossum_metric_n1 = rossum_metric(n1_spikes_layer1, n1_spikes_layer3, tau)

    print("Purpura Distance between n0 in Layer 1 and n0 in Layer 3:", purpura_distance_n0)
    print("Purpura Distance between n1 in Layer 1 and n1 in Layer 3:", purpura_distance_n1)
    print("Rossum Metric between n0 in Layer 1 and n0 in Layer 3:", rossum_metric_n0)
    print("Rossum Metric between n1 in Layer 1 and n1 in Layer 3:", rossum_metric_n1)

    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights_l2l3)

def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
text_data_channel1 = loaded_df_json.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_json.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

key = get_random_bytes(16) 

print("Generated AES Key:", key.hex())

plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = aes_encrypt(plain_text_combined, key)
decrypted_text_combined = aes_decrypt(cipher_text_combined, key)

decrypted_ascii_values_combined = list(decrypted_text_combined)
decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

spike_train_combined = ascii_to_spike_train(combined_ascii_values, key[0])  

simulate_neural_network_with_input(spike_train_combined, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0)

decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key[0])
decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)


# # SNN_RSA_STDP 

# In[110]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# RSA Encryption and Decryption Functions
def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return public_key, private_key

def rsa_encrypt(plain_text, public_key):
    recipient_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    encrypted_data = cipher_rsa.encrypt(plain_text)
    return encrypted_data

def rsa_decrypt(cipher_text, private_key):
    private_key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(private_key)
    decrypted_data = cipher_rsa.decrypt(cipher_text)
    return decrypted_data

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(12, 10))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights[0], label='Layer 1 -> Ih Layer', color='red')
    plt.plot(weights[1], label='Layer 2 -> Ih Layer', color='blue')
    plt.plot(weights[2], label='Ih Layer -> Layer 2', color='green')
    plt.xlabel('Simulation Step')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    os.makedirs("SNNresults", exist_ok=True)
    plt.savefig("SNNresults/SNNwithSTDP_RSA_raster_plot.png")
    plt.show()

# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)

# Simulate the neural network with spike train input
def simulate_neural_network_with_input(spike_train, num_steps=10, simulation_duration=10.0, min_current=300.0, max_current=600.0):
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons_layer1 = 10
    num_neurons_layer2 = 16
    num_neurons_layer3 = 10

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", num_neurons_layer1),
        'layer2': nest.Create("iaf_psc_alpha", num_neurons_layer2),
        'layer3': nest.Create("iaf_psc_alpha", num_neurons_layer3),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}
    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1lih = {"synapse_model": "stdp_synapse", "weight": -1200.0}
    syn_spec_l2l3 = {"weight": 400.0}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_l2ih = {"synapse_model": "stdp_synapse", "weight": 1200.0}
    syn_spec_ihl2 = {"synapse_model": "stdp_synapse", "weight": -600.0}

    # Connect neurons between layers
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Define the connections for lateral inhibition all-to-all
    for i in range(num_neurons_layer1):
        for j in range(18):
            nest.Connect(neuron_layers['layer1'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l1lih)
    for i in range(num_neurons_layer2):
        for j in range(18):
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l2ih)
            nest.Connect(neuron_layers['lateral_ih'][j], neuron_layers['layer2'][i], syn_spec=syn_spec_ihl2)

    # Define synaptic weight recording for STDP connections
    stdp_synapse_weights_l1lih = []
    stdp_synapse_weights_l2ih = []
    stdp_synapse_weights_ihl2 = []

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=num_neurons_layer1)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
        nest.Simulate(simulation_duration)

        # Record synaptic weights after each simulation step
        stdp_synapse_weights_l1lih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer1'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_l2ih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_ihl2.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['lateral_ih'], neuron_layers['layer2']), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]

    # Convert lists to numpy arrays for easier manipulation
    stdp_synapse_weights_l1lih = np.array(stdp_synapse_weights_l1lih)
    stdp_synapse_weights_l2ih = np.array(stdp_synapse_weights_l2ih)
    stdp_synapse_weights_ihl2 = np.array(stdp_synapse_weights_ihl2)

    # Print or visualize the recorded synaptic weights
    print("Synaptic weights from layer 1 to lateral inhibition layer (l1 to lih):")
    print(stdp_synapse_weights_l1lih)

    print("Synaptic weights from layer 2 to lateral inhibition layer (l2 to lih):")
    print(stdp_synapse_weights_l2ih)

    print("Synaptic weights from lateral inhibition layer to layer 2 (lih to l2):")
    print(stdp_synapse_weights_ihl2)

    # Save the final weights
    save_weights(stdp_synapse_weights_l2ih[-1], "weights_l2ih.npy")

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    weights = [stdp_synapse_weights_l1lih, stdp_synapse_weights_l2ih, stdp_synapse_weights_ihl2]
    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights)

    return ts_layer1, ts_layer2, ts_layer3
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)


# Purpura distance function
def purpura_distance(spiketrains, cost):
    """
    Calculates the Victor-Purpura distance for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                if len(st1) == 0 or len(st2) == 0:
                    continue  # Skip empty spike trains

                idx1, idx2 = 0, 0
                len_st1, len_st2 = len(st1), len(st2)

                while idx1 < len_st1 and idx2 < len_st2:
                    time_diff = st1[idx1] - st2[idx2]

                    if time_diff > 0:
                        distance += np.exp(-time_diff / cost)
                        idx2 += 1
                    elif time_diff < 0:
                        distance += np.exp(time_diff / cost)
                        idx1 += 1
                    else:
                        idx1 += 1
                        idx2 += 1

                # Add remaining spikes from longer spike train
                while idx1 < len_st1:
                    if len_st2 > 0:
                        distance += np.exp(-(st1[idx1] - st2[-1]) / cost)
                    else:
                        distance += np.exp(-(st1[idx1] - st1[-1]) / cost)
                    idx1 += 1

                while idx2 < len_st2:
                    if len_st1 > 0:
                        distance += np.exp(-(st1[-1] - st2[idx2]) / cost)
                    else:
                        distance += np.exp(-(st2[idx2] - st2[-1]) / cost)
                    idx2 += 1

    return distance


# van Rossum metric function
def rossum_metric(*spiketrains, tau):
    """
    Calculates the van Rossum metric for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                for spike_time1 in st1:
                    for spike_time2 in st2:
                        distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance


# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate RSA keys
public_key, private_key = generate_rsa_keys()

# Print the generated keys
print("Generated RSA Public Key:")
print(public_key.decode())
print("\nGenerated RSA Private Key:")
print(private_key.decode())

# Encrypt and decrypt the message using RSA for the binary values
plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = rsa_encrypt(plain_text_combined, public_key)
decrypted_text_combined = rsa_decrypt(cipher_text_combined, private_key)

decrypted_ascii_values_combined = list(decrypted_text_combined)

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
public_key_str = public_key.decode()  # Decode public key to string
spike_train_combined = ascii_to_spike_train(combined_ascii_values,
                                            ord(public_key_str[0]))  # Using the first character of the RSA public key

# Simulate the neural network with the spike train as input
ts_layer1, ts_layer2, ts_layer3 = simulate_neural_network_with_input(spike_train_combined)

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, ord(public_key_str[0]))

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)

# Calculate Purpura distance
cost = 1.0  # Example cost parameter
purpura_dist = purpura_distance([ts_layer1, ts_layer2, ts_layer3], cost)
print(f"Purpura Distance: {purpura_dist}")

# Calculate van Rossum metric
tau = 2.0  # Example tau parameter
rossum_dist = rossum_metric(ts_layer1, ts_layer2, ts_layer3, tau=tau)
print(f"van Rossum Metric: {rossum_dist}")


# # SNN_RSA-STDP and RESUME

# In[111]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# RSA Encryption and Decryption Functions
def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def rsa_encrypt(plain_text, public_key):
    rsa_key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    return cipher.encrypt(plain_text)

def rsa_decrypt(cipher_text, private_key):
    rsa_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    return cipher.decrypt(cipher_text)

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []
    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)
    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []
    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)
    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,
                senders_teaching_layer, ts_teaching_layer):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # Teaching Layer
    plt.subplot(3, 2, 6)
    plt.title('Spike Raster Plot - Teaching Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_teaching_layer, ts_teaching_layer):
        plt.vlines(spike_time, sender, sender + 1, color='pink')

    plt.tight_layout()
    plt.savefig("SNNresults/SNN_RSA_STDP_and_RESUME_raster_plot.png")
    plt.show()

# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)

# Simulate the neural network with spike train input
def simulate_neural_network_with_input(spike_train, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons = len(spike_train)

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", 10),
        'layer2': nest.Create("iaf_psc_alpha", 16),
        'layer3': nest.Create("iaf_psc_alpha", 10),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18),
        'teaching': nest.Create("iaf_psc_alpha", 10)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "stdp_triplet_synapse", "weight": 400.0, "Wmax": 900}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_tll3 = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -100.0}
    syn_spec_l3ih = {"weight": 350.0}

    # Connect layer 1 to layer 2
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)

    # Connect layer 2 to layer 3
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Connect teaching layer to layer 3
    for i in range(len(neuron_layers['teaching'])):
        nest.Connect(neuron_layers['teaching'][i], neuron_layers['layer3'][i], syn_spec=syn_spec_tll3)

    # Connect layer 3 to lateral inhibition layer
    nest.Connect(neuron_layers['layer3'], neuron_layers['lateral_ih'], syn_spec=syn_spec_l3ih)

    # Connect lateral inhibition layer to layer 2
    nest.Connect(neuron_layers['lateral_ih'], neuron_layers['layer2'], syn_spec=syn_spec_ihl2)

    # Connect noise layer to layer 2
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron], syn_spec=syn_spec_lnl2)

    # Simulation steps
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=2)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
            nest.SetStatus(neuron_layers['teaching'][i], {"I_e": current})

        # Simulate the network
        nest.Simulate(simulation_duration)

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorders['teaching'], "events")[0]

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]

    # Save final weights
    print("Saving weights ...")
    save_weights(np.asarray(nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['layer3']), "weight")), "weights_l2l3.npy")

    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,
                senders_teaching_layer, ts_teaching_layer)

simulate_neural_network_with_input([], num_steps=200, simulation_duration=50.0, min_current=300.0, max_current=600.0)
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate RSA keys
private_key, public_key = generate_rsa_keys()

# Print the generated RSA keys
print("Generated RSA Public Key:", public_key.decode())
print("Generated RSA Private Key:", private_key.decode())

# Encrypt and decrypt the message using RSA for the binary values
plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = rsa_encrypt(plain_text_combined, public_key)
decrypted_text_combined = rsa_decrypt(cipher_text_combined, private_key)

decrypted_ascii_values_combined = list(decrypted_text_combined)

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, int.from_bytes(public_key[:1], 'big'))  # Using the first byte of the RSA public key

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, int.from_bytes(public_key[:1], 'big'))

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)


# # SNN_RSA_with Lateral Inhibition- Training using resume with noise

# In[113]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# RSA Encryption and Decryption Functions
def rsa_encrypt(plain_text, public_key):
    cipher = PKCS1_OAEP.new(public_key)
    return cipher.encrypt(plain_text)

def rsa_decrypt(cipher_text, private_key):
    cipher = PKCS1_OAEP.new(private_key)
    return cipher.decrypt(cipher_text)

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []
    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)
    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []
    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)
    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.scatter(range(len(weights)), weights, label='Layer 2 -> Layer 3', color='red')
    plt.xlabel('Layer 2 -> synapse index')
    plt.ylabel('Synaptic Weight')
    plt.title('Pre-trained Synaptic Weights')
    plt.grid()

    plt.tight_layout()
    plt.savefig("SNNresults/SNN_Training_using_Resume_Noise_raster_plot.png")
    plt.show()

def load_weights(filename):
    file_path = os.path.join("final_weights", filename)
    try:
        weights = np.load(file_path, allow_pickle=True)
        return weights
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def purpura_distance(spiketrain1, spiketrain2, cost):
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)
    distance = 0.0
    i, j = 0, 0
    while i < len(spiketrain1) and j < len(spiketrain2):
        time_diff = spiketrain1[i] - spiketrain2[j]
        if time_diff > 0:
            distance += np.exp(-time_diff / cost)
            j += 1
        elif time_diff < 0:
            distance += np.exp(time_diff / cost)
            i += 1
        else:
            i += 1
            j += 1
    while i < len(spiketrain1):
        distance += np.exp(-(spiketrain1[i] - spiketrain2[-1]) / cost)
        i += 1
    while j < len(spiketrain2):
        distance += np.exp(-(spiketrain1[-1] - spiketrain2[j]) / cost)
        j += 1
    return distance

def rossum_metric(spiketrain1, spiketrain2, tau):
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)
    distance = 0.0
    for spike_time1 in spiketrain1:
        for spike_time2 in spiketrain2:
            distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)
    return distance

def simulate_neural_network_with_input(spike_train, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", 10),
        'layer2': nest.Create("iaf_psc_alpha", 16),
        'layer3': nest.Create("iaf_psc_alpha", 10),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18),
        'teaching': nest.Create("iaf_psc_alpha", 10)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "stdp_triplet_synapse", "weight": 400.0, "Wmax": 900}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_tll3 = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -100.0}
    syn_spec_l3ih = {"weight": 350.0}

    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)

    print("Loading weights ...")
    weights_l2l3 = load_weights("weights_l2l3.npy")
    expected_size = len(neuron_layers['layer2']) * len(neuron_layers['layer3'])
    if weights_l2l3 is None or len(weights_l2l3) != expected_size:
        raise ValueError(f"Expected weights of size {expected_size}, but got {None if weights_l2l3 is None else len(weights_l2l3)}")

    for i in range(len(neuron_layers['layer2'])):
        for j in range(len(neuron_layers['layer3'])):
            weight_index = i * len(neuron_layers['layer3']) + j
            weight = weights_l2l3[weight_index]
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['layer3'][j], syn_spec={"weight": weight})

    for i in range(len(neuron_layers['teaching'])):
        nest.Connect(neuron_layers['teaching'][i], neuron_layers['layer3'][i], syn_spec=syn_spec_tll3)

    nest.Connect(neuron_layers['layer3'], neuron_layers['lateral_ih'], syn_spec=syn_spec_l3ih)
    nest.Connect(neuron_layers['lateral_ih'], neuron_layers['layer2'], syn_spec=syn_spec_ihl2)
    
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron], syn_spec=syn_spec_lnl2)

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        random_currents = np.random.uniform(min_current, max_current, size=10)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
            nest.SetStatus(neuron_layers['teaching'][i], {"I_e": current})
        nest.Simulate(simulation_duration)

    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorders['teaching'], "events")[0]

    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]

    cost = 1.0
    tau = 1.0

    n0_spikes_layer1 = ts_layer1[senders_layer1 == neuron_layers['layer1'][0]]
    n1_spikes_layer1 = ts_layer1[senders_layer1 == neuron_layers['layer1'][1]]
    n0_spikes_layer3 = ts_layer3[senders_layer3 == neuron_layers['layer3'][0]]
    n1_spikes_layer3 = ts_layer3[senders_layer3 == neuron_layers['layer3'][1]]

    purpura_distance_n0 = purpura_distance(n0_spikes_layer1, n0_spikes_layer3, cost)
    purpura_distance_n1 = purpura_distance(n1_spikes_layer1, n1_spikes_layer3, cost)

    rossum_metric_n0 = rossum_metric(n0_spikes_layer1, n0_spikes_layer3, tau)
    rossum_metric_n1 = rossum_metric(n1_spikes_layer1, n1_spikes_layer3, tau)

    print("Purpura Distance between n0 in Layer 1 and n0 in Layer 3:", purpura_distance_n0)
    print("Purpura Distance between n1 in Layer 1 and n1 in Layer 3:", purpura_distance_n1)
    print("Rossum Metric between n0 in Layer 1 and n0 in Layer 3:", rossum_metric_n0)
    print("Rossum Metric between n1 in Layer 1 and n1 in Layer 3:", rossum_metric_n1)

    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights_l2l3)
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
text_data_channel1 = loaded_df_json.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_json.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

key_pair = RSA.generate(2048)
public_key = key_pair.publickey()
private_key = key_pair

print("Generated RSA Public Key:\n", public_key.export_key().decode())
print("Generated RSA Private Key:\n", private_key.export_key().decode())

plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = rsa_encrypt(plain_text_combined, public_key)
decrypted_text_combined = rsa_decrypt(cipher_text_combined, private_key)

decrypted_ascii_values_combined = list(decrypted_text_combined)
decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

spike_train_combined = ascii_to_spike_train(combined_ascii_values, public_key.n % 256)

simulate_neural_network_with_input(spike_train_combined, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0)

decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, public_key.n % 256)
decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)


# # SNN_S-DES-STDP 

# In[114]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# Initial Permutation (IP) and Inverse Permutation (IP-1) Functions
def initial_permutation(data):
    permutation = [1, 5, 2, 0, 3, 7, 4, 6]
    return [data[i] for i in permutation]

def inverse_initial_permutation(data):
    permutation = [3, 0, 2, 4, 6, 1, 7, 5]
    return [data[i] for i in permutation]

# S-Box and Permutation Functions
S0 = [[1, 0, 3, 2],
      [3, 2, 1, 0],
      [0, 2, 1, 3],
      [3, 1, 3, 2]]

S1 = [[0, 1, 2, 3],
      [2, 0, 1, 3],
      [3, 0, 1, 0],
      [2, 1, 0, 3]]

def s_box_lookup(s_box, input_bits):
    row = int(f"{input_bits[0]}{input_bits[3]}", 2)
    col = int(f"{input_bits[1]}{input_bits[2]}", 2)
    return list(map(int, format(s_box[row][col], '02b')))

def fk(L, R, sub_key):
    # Expansion and permutation
    E_P = [3, 0, 1, 2, 1, 2, 3, 0]
    E_P_R = [R[i] for i in E_P]

    # XOR with sub_key
    xor_result = [E_P_R[i] ^ sub_key[i] for i in range(8)]

    # Split and lookup in S-boxes
    left = xor_result[:4]
    right = xor_result[4:]
    S_output = s_box_lookup(S0, left) + s_box_lookup(S1, right)

    # Permutation P4
    P4 = [S_output[1], S_output[3], S_output[2], S_output[0]]

    # XOR with L
    result = [L[i] ^ P4[i] for i in range(4)]
    return result + R

# S-DES Encryption and Decryption Functions
def sdes_encrypt(plain_text, key1, key2):
    # Initial permutation
    data = initial_permutation(plain_text)

    # Split into two halves
    L, R = data[:4], data[4:]

    # First round of fk
    result = fk(L, R, key1)

    # Swap halves
    L, R = result[4:], result[:4]

    # Second round of fk
    result = fk(L, R, key2)

    # Inverse permutation
    cipher_text = inverse_initial_permutation(result)
    return cipher_text

def sdes_decrypt(cipher_text, key1, key2):
    # Initial permutation
    data = initial_permutation(cipher_text)

    # Split into two halves
    L, R = data[:4], data[4:]

    # First round of fk
    result = fk(L, R, key2)

    # Swap halves
    L, R = result[4:], result[:4]

    # Second round of fk
    result = fk(L, R, key1)

    # Inverse permutation
    plain_text = inverse_initial_permutation(result)
    return plain_text

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(12, 10))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights[0], label='Layer 1 -> Ih Layer', color='red')
    plt.plot(weights[1], label='Layer 2 -> Ih Layer', color='blue')
    plt.plot(weights[2], label='Ih Layer -> Layer 2', color='green')
    plt.xlabel('Simulation Step')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    os.makedirs("SNNresults", exist_ok=True)
    plt.savefig("SNNresults/SNNwithSTDP_DES_raster_plot.png")
    plt.show()

# Key Generation Function
def generate_keys(seed):
    np.random.seed(seed)
    key = np.random.randint(0, 1024)  # Generate a random 10-bit key
    key_bin = int_to_binary_list(key, 10)

    # Split the 10-bit key into two 8-bit subkeys
    sub_key1 = key_bin[:8]
    sub_key2 = key_bin[2:10] + [0] * 2  # Adjust to 8 bits

    return sub_key1, sub_key2, key_bin

# Normalize ASCII values to a suitable current range
def normalize_ascii_values(ascii_values, target_min=300.0, target_max=600.0):
    min_val = min(ascii_values)
    max_val = max(ascii_values)
    normalized_values = [
        target_min + (value - min_val) * (target_max - target_min) / (max_val - min_val)
        for value in ascii_values
    ]
    return normalized_values

def simulate_neural_network_with_input(spike_train, ascii_values, num_steps=10, simulation_duration=10.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons = len(spike_train)

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", num_neurons),
        'layer2': nest.Create("iaf_psc_alpha", num_neurons),
        'layer3': nest.Create("iaf_psc_alpha", num_neurons),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", num_neurons)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1lih = {"synapse_model": "stdp_synapse", "weight": -1200.0}
    syn_spec_l2l3 = {"weight": 1200.0}
    syn_spec_lnl2 = {"weight": 1200.0}
    syn_spec_l2ih = {"synapse_model": "stdp_synapse", "weight": 1200.0}
    syn_spec_ihl2 = {"synapse_model": "stdp_synapse", "weight": -600.0}

    # Connect neurons between layers
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Define the connections for lateral inhibition to layer 3
    for i in range(num_neurons):
        for j in range(num_neurons):
            nest.Connect(neuron_layers['layer1'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l1lih)
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l2ih)
            nest.Connect(neuron_layers['lateral_ih'][i], neuron_layers['layer3'][j], syn_spec=syn_spec_ihl2)

    # Connect noise layer to layer 2
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron],
                         syn_spec=syn_spec_lnl2)

    # Normalize ASCII values to a suitable current range for neuron stimulation
    normalized_ascii_values = normalize_ascii_values(ascii_values)

    # Define synaptic weight recording for STDP connections
    stdp_synapse_weights_l1lih = []
    stdp_synapse_weights_l2ih = []
    stdp_synapse_weights_ihl2 = []

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Inject currents into neurons in layer 1 based on normalized ASCII values
        for i, current in enumerate(normalized_ascii_values):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": float(current)})

        # Simulate the network for the given duration
        nest.Simulate(simulation_duration)

        # Record synaptic weights after each simulation step
        stdp_synapse_weights_l1lih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer1'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_l2ih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_ihl2.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['lateral_ih'], neuron_layers['layer3']), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]

    # Convert lists to numpy arrays for easier manipulation
    stdp_synapse_weights_l1lih = np.array(stdp_synapse_weights_l1lih)
    stdp_synapse_weights_l2ih = np.array(stdp_synapse_weights_l2ih)
    stdp_synapse_weights_ihl2 = np.array(stdp_synapse_weights_ihl2)

    # Print or visualize the recorded synaptic weights
    print("Synaptic weights from layer 1 to lateral inhibition layer (l1 to lih):")
    print(stdp_synapse_weights_l1lih)

    print("Synaptic weights from layer 2 to lateral inhibition layer (l2 to lih):")
    print(stdp_synapse_weights_l2ih)

    print("Synaptic weights from lateral inhibition layer to layer 3 (lih to l3):")
    print(stdp_synapse_weights_ihl2)

    # Save the final weights
    save_weights(stdp_synapse_weights_l2ih[-1], "weights_l2ih.npy")

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    weights = [stdp_synapse_weights_l1lih, stdp_synapse_weights_l2ih, stdp_synapse_weights_ihl2]
    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights)

    return ts_layer1, ts_layer2, ts_layer3
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
# Purpura distance function
def purpura_distance(spiketrains, cost):
    """
    Calculates the Victor-Purpura distance for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                if len(st1) == 0 or len(st2) == 0:
                    continue  # Skip empty spike trains

                idx1, idx2 = 0, 0
                len_st1, len_st2 = len(st1), len(st2)

                while idx1 < len_st1 and idx2 < len_st2:
                    time_diff = st1[idx1] - st2[idx2]

                    if time_diff > 0:
                        distance += np.exp(-time_diff / cost)
                        idx2 += 1
                    elif time_diff < 0:
                        distance += np.exp(time_diff / cost)
                        idx1 += 1
                    else:
                        idx1 += 1
                        idx2 += 1

                # Add remaining spikes from longer spike train
                while idx1 < len_st1:
                    if len_st2 > 0:
                        distance += np.exp(-(st1[idx1] - st2[-1]) / cost)
                    else:
                        distance += np.exp(-(st1[idx1] - st1[-1]) / cost)
                    idx1 += 1

                while idx2 < len_st2:
                    if len_st1 > 0:
                        distance += np.exp(-(st1[-1] - st2[idx2]) / cost)
                    else:
                        distance += np.exp(-(st2[idx2] - st2[-1]) / cost)
                    idx2 += 1

    return distance

# van Rossum metric function
def rossum_metric(*spiketrains, tau):
    """
    Calculates the van Rossum metric for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                for spike_time1 in st1:
                    for spike_time2 in st2:
                        distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance

# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)

# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate S-DES sub-keys
key = 42  # Example key for XOR
sub_key1, sub_key2, key_bin = generate_keys(key)

# Print the keys
print("Generated Key (10 bits):", key_bin)
print("Sub Key 1 (8 bits):", sub_key1)
print("Sub Key 2 (8 bits):", sub_key2)

# Encrypt and decrypt the message using S-DES for the binary values
cipher_text_combined = [sdes_encrypt(value, sub_key1, sub_key2) for value in binary_values_combined]
decrypted_text_combined = [sdes_decrypt(value, sub_key1, sub_key2) for value in cipher_text_combined]

decrypted_ascii_values_combined = [binary_list_to_int(value) for value in decrypted_text_combined]

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key)

# Simulate the neural network with the spike train as input
ts_layer1, ts_layer2, ts_layer3 = simulate_neural_network_with_input(spike_train_combined, combined_ascii_values)

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key)

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes :", decoded_text_from_spikes_combined)

# Calculate Purpura distance
cost = 1.0  # Example cost parameter
purpura_dist = purpura_distance([ts_layer1, ts_layer2, ts_layer3], cost)
print(f"Purpura Distance: {purpura_dist}")

# Calculate van Rossum metric
tau = 2.0  # Example tau parameter
rossum_dist = rossum_metric(ts_layer1, ts_layer2, ts_layer3, tau=tau)
print(f"van Rossum Metric: {rossum_dist}")


# # SNN_S DES-STDP and RESUME

# In[117]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']

# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)

speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)

# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)

# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)

loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)

# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]

def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)

# Initial Permutation (IP) and Inverse Permutation (IP-1) Functions
def initial_permutation(data):
    permutation = [1, 5, 2, 0, 3, 7, 4, 6]
    return [data[i] for i in permutation]

def inverse_initial_permutation(data):
    permutation = [3, 0, 2, 4, 6, 1, 7, 5]
    return [data[i] for i in permutation]

# S-Box and Permutation Functions
S0 = [[1, 0, 3, 2],
      [3, 2, 1, 0],
      [0, 2, 1, 3],
      [3, 1, 3, 2]]

S1 = [[0, 1, 2, 3],
      [2, 0, 1, 3],
      [3, 0, 1, 0],
      [2, 1, 0, 3]]

def s_box_lookup(s_box, input_bits):
    row = int(f"{input_bits[0]}{input_bits[3]}", 2)
    col = int(f"{input_bits[1]}{input_bits[2]}", 2)
    return list(map(int, format(s_box[row][col], '02b')))

def fk(L, R, sub_key):
    # Expansion and permutation
    E_P = [3, 0, 1, 2, 1, 2, 3, 0]
    E_P_R = [R[i] for i in E_P]

    # XOR with sub_key
    xor_result = [E_P_R[i] ^ sub_key[i] for i in range(8)]

    # Split and lookup in S-boxes
    left = xor_result[:4]
    right = xor_result[4:]
    S_output = s_box_lookup(S0, left) + s_box_lookup(S1, right)

    # Permutation P4
    P4 = [S_output[1], S_output[3], S_output[2], S_output[0]]

    # XOR with L
    result = [L[i] ^ P4[i] for i in range(4)]
    return result + R

# S-DES Encryption and Decryption Functions
def sdes_encrypt(plain_text, key1, key2):
    # Initial permutation
    data = initial_permutation(plain_text)

    # Split into two halves
    L, R = data[:4], data[4:]

    # First round of fk
    result = fk(L, R, key1)

    # Swap halves
    L, R = result[4:], result[:4]

    # Second round of fk
    result = fk(L, R, key2)

    # Inverse permutation
    cipher_text = inverse_initial_permutation(result)
    return cipher_text

def sdes_decrypt(cipher_text, key1, key2):
    # Initial permutation
    data = initial_permutation(cipher_text)

    # Split into two halves
    L, R = data[:4], data[4:]

    # First round of fk
    result = fk(L, R, key2)

    # Swap halves
    L, R = result[4:], result[:4]

    # Second round of fk
    result = fk(L, R, key1)

    # Inverse permutation
    plain_text = inverse_initial_permutation(result)
    return plain_text

# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)

# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))

# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)

# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values

# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,
                senders_teaching_layer, ts_teaching_layer):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # Teaching Layer
    plt.subplot(3, 2, 6)
    plt.title('Spike Raster Plot - Teaching Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_teaching_layer, ts_teaching_layer):
        plt.vlines(spike_time, sender, sender + 1, color='pink')

    plt.tight_layout()
    plt.savefig("SNNresults/SNN_S-DES STDP and Resume_raster_plot.png")
    plt.show()

# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)

# Key Generation Function
def generate_keys(seed):
    np.random.seed(seed)
    key = np.random.randint(0, 1024)  # Generate a random 10-bit key
    key_bin = int_to_binary_list(key, 10)

    # Split the 10-bit key into two 8-bit subkeys
    sub_key1 = key_bin[:8]
    sub_key2 = key_bin[2:10] + [0] * 2  # Adjust to 8 bits

    return sub_key1, sub_key2, key_bin

# Simulate the neural network with spike train input
def simulate_neural_network_with_input(spike_train, num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons = len(spike_train)

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", 10),
        'layer2': nest.Create("iaf_psc_alpha", 16),
        'layer3': nest.Create("iaf_psc_alpha", 10),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18),
        'teaching': nest.Create("iaf_psc_alpha", 10)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "stdp_triplet_synapse", "weight": 400.0, "Wmax": 900}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_tll3 = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -100.0}
    syn_spec_l3ih = {"weight": 350.0}

    # Connect layer 1 to layer 2
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)

    # Connect layer 2 to layer 3
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Connect teaching layer to layer 3
    for i in range(len(neuron_layers['teaching'])):
        nest.Connect(neuron_layers['teaching'][i], neuron_layers['layer3'][i], syn_spec=syn_spec_tll3)

    # Connect layer 3 to lateral inhibition layer
    nest.Connect(neuron_layers['layer3'], neuron_layers['lateral_ih'], syn_spec=syn_spec_l3ih)

    # Connect lateral inhibition layer to layer 2
    nest.Connect(neuron_layers['lateral_ih'], neuron_layers['layer2'], syn_spec=syn_spec_ihl2)

    # Connect noise layer to layer 2
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron], syn_spec=syn_spec_lnl2)

    # Simulation steps
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=2)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
            nest.SetStatus(neuron_layers['teaching'][i], {"I_e": current})

        # Simulate the network
        nest.Simulate(simulation_duration)
    print("Number of neurons in Layer 1:", len(neuron_layers['layer1']))
    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorders['teaching'], "events")[0]

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
    # Save final weights
    print("Saving weights ...")
    save_weights(np.asarray(nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['layer3']), "weight")), "weights_l2l3.npy")

    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,
                senders_teaching_layer, ts_teaching_layer)

# Use the combined ASCII values you already generated
simulate_neural_network_with_input(combined_ascii_values, num_steps=200, simulation_duration=50.0)

# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate S-DES sub-keys
key = 42  # Example key for XOR
sub_key1, sub_key2, key_bin = generate_keys(key)

# Print the keys
print("Generated Key (10 bits):", key_bin)
print("Sub Key 1 (8 bits):", sub_key1)
print("Sub Key 2 (8 bits):", sub_key2)

# Encrypt and decrypt the message using S-DES for the binary values
cipher_text_combined = [sdes_encrypt(value, sub_key1, sub_key2) for value in binary_values_combined]
decrypted_text_combined = [sdes_decrypt(value, sub_key1, sub_key2) for value in cipher_text_combined]

decrypted_ascii_values_combined = [binary_list_to_int(value) for value in decrypted_text_combined]

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key)

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key)

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)


# # SNN_S-DES with Lateral Inhibition- Training using resume with noise

# In[116]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']


# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)


speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)


# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)


save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)


# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)


loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)


# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]


def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)


# Initial Permutation (IP) and Inverse Permutation (IP-1) Functions
def initial_permutation(data):
    permutation = [1, 5, 2, 0, 3, 7, 4, 6]
    return [data[i] for i in permutation]


def inverse_initial_permutation(data):
    permutation = [3, 0, 2, 4, 6, 1, 7, 5]
    return [data[i] for i in permutation]


# S-Box and Permutation Functions
S0 = [[1, 0, 3, 2],
      [3, 2, 1, 0],
      [0, 2, 1, 3],
      [3, 1, 3, 2]]

S1 = [[0, 1, 2, 3],
      [2, 0, 1, 3],
      [3, 0, 1, 0],
      [2, 1, 0, 3]]


def s_box_lookup(s_box, input_bits):
    row = int(f"{input_bits[0]}{input_bits[3]}", 2)
    col = int(f"{input_bits[1]}{input_bits[2]}", 2)
    return list(map(int, format(s_box[row][col], '02b')))


def fk(L, R, sub_key):
    # Expansion and permutation
    E_P = [3, 0, 1, 2, 1, 2, 3, 0]
    E_P_R = [R[i] for i in E_P]

    # XOR with sub_key
    xor_result = [E_P_R[i] ^ sub_key[i] for i in range(8)]

    # Split and lookup in S-boxes
    left = xor_result[:4]
    right = xor_result[4:]
    S_output = s_box_lookup(S0, left) + s_box_lookup(S1, right)

    # Permutation P4
    P4 = [S_output[1], S_output[3], S_output[2], S_output[0]]

    # XOR with L
    result = [L[i] ^ P4[i] for i in range(4)]
    return result + R


# S-DES Encryption and Decryption Functions
def sdes_encrypt(plain_text, key1, key2):
    # Initial permutation
    data = initial_permutation(plain_text)

    # Split into two halves
    L, R = data[:4], data[4:]

    # First round of fk
    result = fk(L, R, key1)

    # Swap halves
    L, R = result[4:], result[:4]

    # Second round of fk
    result = fk(L, R, key2)

    # Inverse permutation
    cipher_text = inverse_initial_permutation(result)
    return cipher_text


def sdes_decrypt(cipher_text, key1, key2):
    # Initial permutation
    data = initial_permutation(cipher_text)

    # Split into two halves
    L, R = data[:4], data[4:]

    # First round of fk
    result = fk(L, R, key2)

    # Swap halves
    L, R = result[4:], result[:4]

    # Second round of fk
    result = fk(L, R, key1)

    # Inverse permutation
    plain_text = inverse_initial_permutation(result)
    return plain_text


# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)


# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))


# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)


# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values


# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.scatter(range(len(weights)), weights, label='Layer 2 -> Layer 3', color='red')
    plt.xlabel('Layer 2 -> synapse index')
    plt.ylabel('Synaptic Weight')
    plt.title('Pre-trained Synaptic Weights')
    plt.grid()

    plt.tight_layout()
    plt.savefig("SNNresults/SNN Training using Resume_Noise_raster_plot.png")
    plt.show()


def load_weights(filename):
    """
    Load weights from an NPY file.

    Parameters:
        filename (str): The name of the NPY file containing the weights.

    Returns:
        ndarray: The loaded weights.
    """
    # Construct the file path
    file_path = os.path.join("final_weights", filename)

    # Load the weights from the NPY file
    try:
        weights = np.load(file_path, allow_pickle=True)
        return weights
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None


def purpura_distance(spiketrain1, spiketrain2, cost):
    """
    spiketrain1 and spiketrain2 are vectors containing the spike times of two spike trains.
    cost is a parameter that determines the sensitivity of the distance calculation to the time differences between
    spikes.
    The function calculates the Victor-Purpura distance by iterating over the spike times of the two spike trains and
    summing exponential terms based on the time differences.
    The result is the Victor-Purpura distance, a measure of the dissimilarity between the spike trains.

    Parameters:
        spiketrain1 (array-like): Spike times of the first spike train.
        spiketrain2 (array-like): Spike times of the second spike train.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance between the two spike trains.
    """
    # Ensure spiketrains are sorted
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    i = 0
    j = 0
    while i < len(spiketrain1) and j < len(spiketrain2):
        time_diff = spiketrain1[i] - spiketrain2[j]
        if time_diff > 0:
            distance += np.exp(-time_diff / cost)
            j += 1
        elif time_diff < 0:
            distance += np.exp(time_diff / cost)
            i += 1
        else:
            i += 1
            j += 1

    # Add remaining spikes from longer spike train
    while i < len(spiketrain1):
        distance += np.exp(-(spiketrain1[i] - spiketrain2[-1]) / cost)
        i += 1
    while j < len(spiketrain2):
        distance += np.exp(-(spiketrain1[-1] - spiketrain2[j]) / cost)
        j += 1

    return distance


def rossum_metric(spiketrain1, spiketrain2, tau):
    """
    spiketrain1 and spiketrain2 are vectors containing the spike times of two spike trains.
    tau is a parameter that represents the time constant for the kernel function, which determines the weight of each
    spike time difference.
    The function calculates the van Rossum metric by iterating over all pairs of spike times from the two spike trains
    and summing the exponential decay of their absolute time differences.
    The result is the van Rossum metric, a measure of the similarity between the spike trains.

    Parameters:
        spiketrain1 (array-like): Spike times of the first spike train.
        spiketrain2 (array-like): Spike times of the second spike train.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric between the two spike trains.
    """
    # Ensure spiketrains are sorted
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for spike_time1 in spiketrain1:
        for spike_time2 in spiketrain2:
            distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance


def generate_keys(seed):
    np.random.seed(seed)
    key = np.random.randint(0, 1024)  # Generate a random 10-bit key
    key_bin = int_to_binary_list(key, 10)

    # Split the 10-bit key into two 8-bit subkeys
    sub_key1 = key_bin[:8]
    sub_key2 = key_bin[2:10] + [0] * 2  # Adjust to 8 bits

    return sub_key1, sub_key2, key_bin

def simulate_neural_network_with_input(ascii_values, num_steps=20, simulation_duration=50.0, min_current=300.0,
                                       max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    # Create neurons for each layer
    num_neurons_layer1 = 16  # Adjusted to match the expected number of inputs
    num_neurons_layer2 = 16
    num_neurons_layer3 = 10

    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", num_neurons_layer1),
        'layer2': nest.Create("iaf_psc_alpha", num_neurons_layer2),
        'layer3': nest.Create("iaf_psc_alpha", num_neurons_layer3),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18),
        'teaching': nest.Create("iaf_psc_alpha", num_neurons_layer3)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "stdp_triplet_synapse", "weight": 400.0, "Wmax": 900}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_tll3 = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -100.0}
    syn_spec_l3ih = {"weight": 350.0}

    # Connect layer 1 to layer 2
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)

    # Load weights
    print("Loading weights ...")
    weights_l2l3 = load_weights("weights_l2l3.npy")

    # Check if weights_l2l3 has the expected size
    expected_size = len(neuron_layers['layer2']) * len(neuron_layers['layer3'])
    if weights_l2l3 is None or len(weights_l2l3) != expected_size:
        raise ValueError(
            f"Expected weights of size {expected_size}, but got {None if weights_l2l3 is None else len(weights_l2l3)}")

    # Connect layer 2 to layer 3 with loaded weights
    for i in range(len(neuron_layers['layer2'])):
        for j in range(len(neuron_layers['layer3'])):
            weight_index = i * len(neuron_layers['layer3']) + j
            weight = weights_l2l3[weight_index]
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['layer3'][j], syn_spec={"weight": weight})

    # Connect teaching layer to layer 3
    for i in range(len(neuron_layers['teaching'])):
        nest.Connect(neuron_layers['teaching'][i], neuron_layers['layer3'][i], syn_spec=syn_spec_tll3)

    # Connect layer 3 to lateral inhibition layer
    nest.Connect(neuron_layers['layer3'], neuron_layers['lateral_ih'], syn_spec=syn_spec_l3ih)

    # Connect lateral inhibition layer to layer 2
    nest.Connect(neuron_layers['lateral_ih'], neuron_layers['layer2'], syn_spec=syn_spec_ihl2)

    # Connect noise layer to layer 2
    connection_mapping = {0: [1, 4], 1: [7, 10]}
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(neuron_layers['noise'][noise_neuron], neuron_layers['layer2'][target_neuron],
                         syn_spec=syn_spec_lnl2)

    # Scale ASCII values to the desired current range
    scaled_currents = np.interp(ascii_values, (0, 255), (min_current, max_current))

    # Simulation steps
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Apply scaled currents to neurons in layer 1
        for i, current in enumerate(scaled_currents[:num_neurons_layer1]):  # Ensure it fits within layer1 neurons
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
            if i < len(neuron_layers['teaching']):  # Ensure it fits within teaching layer neurons
                nest.SetStatus(neuron_layers['teaching'][i], {"I_e": current})

        # Simulate the network
        nest.Simulate(simulation_duration)

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorders['teaching'], "events")[0]

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]

    # Compute the purpura_distance and rossum_metric
    cost = 1.0  # You can adjust this parameter as needed
    tau = 1.0  # You can adjust this parameter as needed

    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights_l2l3)
def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
# Generate S-DES sub-keys
key = 42  # Example key for XOR
sub_key1, sub_key2, key_bin = generate_keys(key)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_json.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_json.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Encrypt and decrypt the message using S-DES for the binary values
cipher_text_combined = [sdes_encrypt(value, sub_key1, sub_key2) for value in binary_values_combined]
decrypted_text_combined = [sdes_decrypt(value, sub_key1, sub_key2) for value in cipher_text_combined]

decrypted_ascii_values_combined = [binary_list_to_int(value) for value in decrypted_text_combined]
decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key)

# Simulate the neural network with the spike train as input
simulate_neural_network_with_input(combined_ascii_values, num_steps=20, simulation_duration=50.0, min_current=300.0,
                                   max_current=600.0)

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key)

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)


# In[ ]:





# In[3]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Define a simple set of categories and cities
categories = {
    'Man': {'count': 1, 'age_range': (21, 70)},
    'Woman': {'count': 1, 'age_range': (17, 59)},
    'Boy': {'count': 1, 'age_range': (6, 14)},
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']


# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)


speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)


# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)


save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)


# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)


loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)


# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]


def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)


# AES Encryption and Decryption Functions
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plain_text, AES.block_size))
    return cipher.iv + ct_bytes


def aes_decrypt(cipher_text, key):
    iv = cipher_text[:16]
    ct = cipher_text[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)


# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)


# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))


# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)  # Create a 256-length array, representing possible spike positions
        encoded_value = char ^ key  # XOR the ASCII value with the key
        spikes[encoded_value] = 1  # Set the spike at the position determined by the encoded value
        spike_train.append(spikes)

    return np.array(spike_train)  # Return as a numpy array for easier processing



# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values


# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(12, 10))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights[0], label='Layer 1 -> Ih Layer', color='red')
    plt.plot(weights[1], label='Layer 2 -> Ih Layer', color='blue')
    plt.plot(weights[2], label='Ih Layer -> Layer 2', color='green')
    plt.xlabel('Simulation Step')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    os.makedirs("SNNresults", exist_ok=True)
    plt.savefig("SNNresults/SNNwithSTDP_AES_raster_plot.png")
    plt.show()


def simulate_neural_network_with_input(spike_train, num_steps=10, simulation_duration=10.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    
    # Define neuron counts for each layer
    num_neurons_layer1 = 10
    num_neurons_layer2 = 16
    num_neurons_layer3 = 10

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", num_neurons_layer1),
        'layer2': nest.Create("iaf_psc_alpha", num_neurons_layer2),
        'layer3': nest.Create("iaf_psc_alpha", num_neurons_layer3),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18)
    }
    
    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}
    
    # Connect neurons to spike recorders
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity and synapses
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1lih = {"synapse_model": "stdp_synapse", "weight": -1200.0}
    syn_spec_l2l3 = {"weight": 400.0}

    # Connect the layers
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Initialize spike trains and currents
    for step in range(num_steps):
        random_currents = np.random.uniform(min_current, max_current, size=num_neurons_layer1)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})

        # Simulate the network for the specified duration
        nest.Simulate(simulation_duration)

    # Retrieve spike data
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]

    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    return ts_layer1, ts_layer2, ts_layer3, senders_layer3


def decode_layer3(senders_layer3, key):
    decoded_ascii_values = []
    for sender in senders_layer3:
        if isinstance(sender, (int, np.integer)):
            decoded_value = int(sender) ^ key  # XOR with the key to decode
            decoded_ascii_values.append(decoded_value)
        else:
            raise ValueError(f"Unexpected data type for sender: {type(sender)}")
    
    decoded_text = ascii_to_text(decoded_ascii_values)
    return decoded_text



# Simulate the neural network with the spike train as input
ts_layer1, ts_layer2, ts_layer3, senders_layer3 = simulate_neural_network_with_input(spike_train_combined)

# Decode the output from Layer 3 using the first byte of the key
decoded_text = decode_layer3(senders_layer3, key[0])
print("Decoded Text from Spikes:", decoded_text)

# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)


# Purpura distance function
def purpura_distance(spiketrains, cost):
    """
    Calculates the Victor-Purpura distance for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                if len(st1) == 0 or len(st2) == 0:
                    continue  # Skip empty spike trains

                idx1, idx2 = 0, 0
                len_st1, len_st2 = len(st1), len(st2)

                while idx1 < len_st1 and idx2 < len_st2:
                    time_diff = st1[idx1] - st2[idx2]

                    if time_diff > 0:
                        distance += np.exp(-time_diff / cost)
                        idx2 += 1
                    elif time_diff < 0:
                        distance += np.exp(time_diff / cost)
                        idx1 += 1
                    else:
                        idx1 += 1
                        idx2 += 1

                # Add remaining spikes from longer spike train
                while idx1 < len_st1:
                    if len_st2 > 0:
                        distance += np.exp(-(st1[idx1] - st2[-1]) / cost)
                    else:
                        distance += np.exp(-(st1[idx1] - st1[-1]) / cost)
                    idx1 += 1

                while idx2 < len_st2:
                    if len_st1 > 0:
                        distance += np.exp(-(st1[-1] - st2[idx2]) / cost)
                    else:
                        distance += np.exp(-(st2[idx2] - st2[-1]) / cost)
                    idx2 += 1

    return distance


# van Rossum metric function
def rossum_metric(*spiketrains, tau):
    """
    Calculates the van Rossum metric for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                for spike_time1 in st1:
                    for spike_time2 in st2:
                        distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance


# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate AES key
key = get_random_bytes(16)  # AES-128 key

# Print the generated key in hexadecimal format
print("Generated AES Key:", key.hex())

# Encrypt and decrypt the message using AES for the binary values
plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = aes_encrypt(plain_text_combined, key)
decrypted_text_combined = aes_decrypt(cipher_text_combined, key)

decrypted_ascii_values_combined = list(decrypted_text_combined)

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key[0])  # Using the first byte of the AES key

# Simulate the neural network with the spike train as input
ts_layer3 = simulate_neural_network_with_input(spike_train_combined)
decoded_text = decode_layer3(senders_layer3, key[0])
print("Decoded Text from Spikes:", decoded_text)
# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key[0])

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)

decoded_ascii_values = []

# Calculate Purpura distance
cost = 1.0  # Example cost parameter
purpura_dist = purpura_distance([ts_layer1, ts_layer2, ts_layer3], cost)
print(f"Purpura Distance: {purpura_dist}")

# Calculate van Rossum metric
tau = 2.0  # Example tau parameter
rossum_dist = rossum_metric(ts_layer1, ts_layer2, ts_layer3, tau=tau)
print(f"van Rossum Metric: {rossum_dist}")


# In[4]:


import pandas as pd
import json
import random
import numpy as np
import nest
import matplotlib.pyplot as plt
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Define a simple set of categories and cities
categories = {
    'Girl': {'count': 1, 'age_range': (8, 15)}
}
cities = ['NewYork', 'Chicago', 'Phoenix']


# Generate speaker information
def generate_speakers(categories, cities):
    speakers = []
    speaker_id = 1
    for category, info in categories.items():
        count = info['count']
        age_range = info['age_range']
        for _ in range(count):
            age = random.randint(*age_range)
            city = random.choice(cities)
            speakers.append({
                'ID': speaker_id,
                'Category': category,
                'Age': age,
                'City': city
            })
            speaker_id += 1
    return pd.DataFrame(speakers)


speakers_df = generate_speakers(categories, cities)

# Display the dataset
print(speakers_df)

# Save the dataset to a CSV file
speakers_df.to_csv('simple_synthetic_snn_dataset.csv', index=False)


# Save the dataset to a JSON file
def save_synthetic_data_to_json(filename, data_df):
    data_dict = data_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)


save_synthetic_data_to_json('simple_synthetic_snn_dataset.json', speakers_df)


# Load the dataset from a JSON file
def load_simple_synthetic_data_from_json(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict)


loaded_df_json = load_simple_synthetic_data_from_json('simple_synthetic_snn_dataset.json')
print("Loaded Data from JSON:")
print(loaded_df_json)


# Text to ASCII and ASCII to Text conversion
def text_to_ascii(text):
    return [ord(char) for char in text]


def ascii_to_text(ascii_values):
    return ''.join(chr(int(value)) for value in ascii_values)


# AES Encryption and Decryption Functions
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plain_text, AES.block_size))
    return cipher.iv + ct_bytes


def aes_decrypt(cipher_text, key):
    iv = cipher_text[:16]
    ct = cipher_text[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)


# Convert binary list to integer
def binary_list_to_int(binary_list):
    return int(''.join(map(str, binary_list)), 2)


# Convert integer to binary list of given size
def int_to_binary_list(value, size):
    return list(map(int, format(value, f'0{size}b')))


# Efficient Spike Train Encoding using Population Coding
def ascii_to_spike_train(ascii_values, key):
    spike_train = []

    for char in ascii_values:
        spikes = np.zeros(256)
        encoded_value = char ^ key
        spikes[encoded_value] = 1
        spike_train.append(spikes)

    return np.array(spike_train)


# Spike Train to ASCII Conversion
def spike_train_to_ascii(spike_train, key):
    ascii_values = []

    for spikes in spike_train:
        encoded_value = np.argmax(spikes)
        decoded_value = encoded_value ^ key
        ascii_values.append(decoded_value)

    return ascii_values


# Raster Plot Function
def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights):
    plt.figure(figsize=(12, 10))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights[0], label='Layer 1 -> Ih Layer', color='red')
    plt.plot(weights[1], label='Layer 2 -> Ih Layer', color='blue')
    plt.plot(weights[2], label='Ih Layer -> Layer 2', color='green')
    plt.xlabel('Simulation Step')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    os.makedirs("SNNresults", exist_ok=True)
    plt.savefig("SNNresults/SNNwithSTDP_AES_raster_plot.png")
    plt.show()


# Simulate the neural network with spike train input
def simulate_neural_network_with_input(spike_train, num_steps=10,
                                       simulation_duration=10.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)
    nest.SetKernelStatus({'print_time': False})
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})

    num_neurons_layer1 = 10
    num_neurons_layer2 = 16
    num_neurons_layer3 = 10

    # Create neurons for each layer
    neuron_layers = {
        'layer1': nest.Create("iaf_psc_alpha", num_neurons_layer1),
        'layer2': nest.Create("iaf_psc_alpha", num_neurons_layer2),
        'layer3': nest.Create("iaf_psc_alpha", num_neurons_layer3),
        'noise': nest.Create("poisson_generator", 2),
        'lateral_ih': nest.Create("iaf_psc_alpha", 18)
    }
    nest.SetStatus(neuron_layers['noise'], {"rate": 10.0})

    # Create spike recorders for each layer
    spike_recorders = {layer: nest.Create("spike_recorder") for layer in neuron_layers}
    print("Number of neurons in Layer 1:", len(neuron_layers['layer1']))

    # Connect spike recorders to neurons
    for layer, neurons in neuron_layers.items():
        nest.Connect(neurons, spike_recorders[layer])

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1lih = {"synapse_model": "stdp_synapse", "weight": -1200.0}
    syn_spec_l2l3 = {"weight": 400.0}
    syn_spec_lnl2 = {"weight": 800.0}
    syn_spec_l2ih = {"synapse_model": "stdp_synapse", "weight": 1200.0}
    syn_spec_ihl2 = {"synapse_model": "stdp_synapse", "weight": -600.0}

    # Connect neurons between layers
    nest.Connect(neuron_layers['layer1'], neuron_layers['layer2'], syn_spec=syn_spec_l1l2)
    nest.Connect(neuron_layers['layer2'], neuron_layers['layer3'], syn_spec=syn_spec_l2l3)

    # Define the connections for lateral inhibition all-to-all
    for i in range(num_neurons_layer1):
        for j in range(18):
            nest.Connect(neuron_layers['layer1'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l1lih)
    for i in range(num_neurons_layer2):
        for j in range(18):
            nest.Connect(neuron_layers['layer2'][i], neuron_layers['lateral_ih'][j], syn_spec=syn_spec_l2ih)
            nest.Connect(neuron_layers['lateral_ih'][j], neuron_layers['layer2'][i], syn_spec=syn_spec_ihl2)

    # Define synaptic weight recording for STDP connections
    stdp_synapse_weights_l1lih = []
    stdp_synapse_weights_l2ih = []
    stdp_synapse_weights_ihl2 = []

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=num_neurons_layer1)
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layers['layer1'][i], {"I_e": current})
        nest.Simulate(simulation_duration)

        # Record synaptic weights after each simulation step
        stdp_synapse_weights_l1lih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer1'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_l2ih.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['layer2'], neuron_layers['lateral_ih']), "weight"))
        stdp_synapse_weights_ihl2.append(
            nest.GetStatus(nest.GetConnections(neuron_layers['lateral_ih'], neuron_layers['layer2']), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorders['layer1'], "events")[0]
    events_layer2 = nest.GetStatus(spike_recorders['layer2'], "events")[0]
    events_layer3 = nest.GetStatus(spike_recorders['layer3'], "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorders['noise'], "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorders['lateral_ih'], "events")[0]

    # Convert lists to numpy arrays for easier manipulation
    stdp_synapse_weights_l1lih = np.array(stdp_synapse_weights_l1lih)
    stdp_synapse_weights_l2ih = np.array(stdp_synapse_weights_l2ih)
    stdp_synapse_weights_ihl2 = np.array(stdp_synapse_weights_ihl2)

    # Print or visualize the recorded synaptic weights
    print("Synaptic weights from layer 1 to lateral inhibition layer (l1 to lih):")
    print(stdp_synapse_weights_l1lih)

    print("Synaptic weights from layer 2 to lateral inhibition layer (l2 to lih):")
    print(stdp_synapse_weights_l2ih)

    print("Synaptic weights from lateral inhibition layer to layer 2 (lih to l2):")
    print(stdp_synapse_weights_ihl2)

    # Save the final weights
    save_weights(stdp_synapse_weights_l2ih[-1], "weights_l2ih.npy")

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    weights = [stdp_synapse_weights_l1lih, stdp_synapse_weights_l2ih, stdp_synapse_weights_ihl2]
    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer, weights)

    return ts_layer1, ts_layer2, ts_layer3

def decode_layer3(spike_train_layer3, key):
    decoded_ascii_values = []
    for spike_pattern in spike_train_layer3:
        decoded_value = np.argmax(spike_pattern)
        decoded_ascii_values.append(decoded_value)

    decoded_text = ascii_to_text([value ^ key for value in decoded_ascii_values])
    return decoded_text
# Save weights function
def save_weights(weights, filename):
    os.makedirs("final_weights", exist_ok=True)
    np.save(os.path.join("final_weights", filename), weights)
    np.savetxt(os.path.join("final_weights", filename + ".txt"), weights)


# Purpura distance function
def purpura_distance(spiketrains, cost):
    """
    Calculates the Victor-Purpura distance for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                if len(st1) == 0 or len(st2) == 0:
                    continue  # Skip empty spike trains

                idx1, idx2 = 0, 0
                len_st1, len_st2 = len(st1), len(st2)

                while idx1 < len_st1 and idx2 < len_st2:
                    time_diff = st1[idx1] - st2[idx2]

                    if time_diff > 0:
                        distance += np.exp(-time_diff / cost)
                        idx2 += 1
                    elif time_diff < 0:
                        distance += np.exp(time_diff / cost)
                        idx1 += 1
                    else:
                        idx1 += 1
                        idx2 += 1

                # Add remaining spikes from longer spike train
                while idx1 < len_st1:
                    if len_st2 > 0:
                        distance += np.exp(-(st1[idx1] - st2[-1]) / cost)
                    else:
                        distance += np.exp(-(st1[idx1] - st1[-1]) / cost)
                    idx1 += 1

                while idx2 < len_st2:
                    if len_st1 > 0:
                        distance += np.exp(-(st1[-1] - st2[idx2]) / cost)
                    else:
                        distance += np.exp(-(st2[idx2] - st2[-1]) / cost)
                    idx2 += 1

    return distance


# van Rossum metric function
def rossum_metric(*spiketrains, tau):
    """
    Calculates the van Rossum metric for a list of spike trains.

    Parameters:
        spiketrains (list of array-like): List of spike trains.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric among the spike trains.
    """
    # Ensure spiketrains are sorted
    sorted_spiketrains = [np.sort(st) for st in spiketrains]

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for i, st1 in enumerate(sorted_spiketrains):
        for j, st2 in enumerate(sorted_spiketrains):
            if i != j:  # Avoid self-comparison
                for spike_time1 in st1:
                    for spike_time2 in st2:
                        distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance


# Load synthetic data from CSV
synthetic_csv_filename = 'simple_synthetic_snn_dataset.csv'
loaded_df_csv = pd.read_csv(synthetic_csv_filename)

# Load synthetic data from JSON
synthetic_json_filename = 'simple_synthetic_snn_dataset.json'
loaded_df_json = load_simple_synthetic_data_from_json(synthetic_json_filename)
print("Loaded Data from JSON:")
print(loaded_df_json)

# Prepare data for channel 1 and channel 2
text_data_channel1 = loaded_df_csv.apply(lambda x: f"{x['ID']} {x['Category']}", axis=1).tolist()
text_data_channel2 = loaded_df_csv.apply(lambda x: f"{x['Age']} {x['City']}", axis=1).tolist()

ascii_values_channel1 = [text_to_ascii(text) for text in text_data_channel1 if text]
ascii_values_channel2 = [text_to_ascii(text) for text in text_data_channel2 if text]

# Flatten ASCII values
flat_ascii_values_channel1 = [value for sublist in ascii_values_channel1 for value in sublist]
flat_ascii_values_channel2 = [value for sublist in ascii_values_channel2 for value in sublist]

# Save ASCII values to a file
ascii_filename_channel1 = 'ascii_values_channel1.csv'
ascii_filename_channel2 = 'ascii_values_channel2.csv'
pd.DataFrame(flat_ascii_values_channel1, columns=['ASCII_Value']).to_csv(ascii_filename_channel1, index=False)
pd.DataFrame(flat_ascii_values_channel2, columns=['ASCII_Value']).to_csv(ascii_filename_channel2, index=False)

# Combine flat ASCII values from both channels
combined_ascii_values = flat_ascii_values_channel1 + flat_ascii_values_channel2

# Convert ASCII values to binary format
binary_values_combined = [int_to_binary_list(value, 8) for value in combined_ascii_values]

# Generate AES key
key = get_random_bytes(16)  # AES-128 key

# Print the generated key in hexadecimal format
print("Generated AES Key:", key.hex())

# Encrypt and decrypt the message using AES for the binary values
plain_text_combined = bytes(combined_ascii_values)
cipher_text_combined = aes_encrypt(plain_text_combined, key)
decrypted_text_combined = aes_decrypt(cipher_text_combined, key)

decrypted_ascii_values_combined = list(decrypted_text_combined)

decrypted_text_combined_final = ascii_to_text(decrypted_ascii_values_combined)
print("Ciphertext Combined:", cipher_text_combined)
print("Decrypted Combined Text:", decrypted_text_combined_final)

# Encode ASCII Values to Spike Train using Key
spike_train_combined = ascii_to_spike_train(combined_ascii_values, key[0])  # Using the first byte of the AES key

# Simulate the neural network with the spike train as input
ts_layer1, ts_layer2, ts_layer3 = simulate_neural_network_with_input(spike_train_combined)

# Decode Spike Train back to ASCII using Key
decoded_ascii_values_from_spikes_combined = spike_train_to_ascii(spike_train_combined, key[0])

decoded_text_from_spikes_combined = ascii_to_text(decoded_ascii_values_from_spikes_combined)
print("Decoded Text from Spikes:", decoded_text_from_spikes_combined)

# Calculate Purpura distance
cost = 1.0  # Example cost parameter
purpura_dist = purpura_distance([ts_layer1, ts_layer2, ts_layer3], cost)
print(f"Purpura Distance: {purpura_dist}")

# Calculate van Rossum metric
tau = 2.0  # Example tau parameter
rossum_dist = rossum_metric(ts_layer1, ts_layer2, ts_layer3, tau=tau)
print(f"van Rossum Metric: {rossum_dist}")


# In[ ]:




