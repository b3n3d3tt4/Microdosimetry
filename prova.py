import numpy as np
import matplotlib.pyplot as plt
import importlib
import basicfunc as bf  # Assumo che tu abbia già definito bf.linear e bf.retta

importlib.reload(bf)

# === Calibrazione canali ===
ch1_x = np.loadtxt('lab_infn\picchi_ch1.txt', usecols=1)
ch1_y = np.loadtxt('lab_infn\picchi_ch1.txt', usecols=0)
ch2_x = np.loadtxt('lab_infn\picchi_ch2.txt', usecols=1)
ch2_y = np.loadtxt('lab_infn\picchi_ch2.txt', usecols=0)

calibration_ch1 = bf.linear(ch1_x, ch1_y * 0.128, sx=1/ch1_x, xlabel='ADC channels', ylabel='Voltage [mV]', titolo='Calibration channel 1', plot=True, save=True)
calibration_ch2 = bf.linear(ch2_x, ch2_y * 0.128, sx=1/ch2_x, xlabel='ADC channels', ylabel='Voltage [mV]', titolo='Calibration channel 2', plot=True, save=True)

# === Caricamento spettro ===
data1 = np.loadtxt('lab_infn\Mini-TEPC\gamma_820V_finale_ch1 copy.Spe')
data2 = np.loadtxt('lab_infn\Mini-TEPC\gamma_820V_finale_ch2 copy.Spe')

# === Calibrazione asse x ===
arr1 = np.arange(len(data1)) * calibration_ch1[0][0] + calibration_ch1[0][1]
arr2 = np.arange(len(data2)) * calibration_ch2[0][0] + calibration_ch2[0][1]

# === Larghezza bin ===
wid1 = arr1[1] - arr1[0]
wid2 = arr2[1] - arr2[0]

# === Normalizzazione f(h) = n(h) / (N * Δh) ===
n1 = np.sum(data1)
n2 = np.sum(data2)
data1_fh = data1 / (n1 * calibration_ch1[0][0])
data2_fh = data2 / (n2 * calibration_ch2[0][0])

# === Bordi bin ===
edges1 = np.append(arr1, arr1[-1] + wid1)
edges2 = np.append(arr2, arr2[-1] + wid2)

# === Rebinning logaritmico ===
def get_log_bins(start_exp, stop_exp, N=60):
    total_bins = int((stop_exp - start_exp) * N)
    x = np.linspace(start_exp, stop_exp, total_bins + 1)
    return 10 ** x

vmin = max(np.min(arr1[arr1 > 0]), np.min(arr2[arr2 > 0]))
vmax = max(np.max(arr1), np.max(arr2))
start_exp = np.floor(np.log10(vmin))
stop_exp = np.ceil(np.log10(vmax))
log_edges = get_log_bins(start_exp, stop_exp, N=60)
log_centers = (log_edges[:-1] + log_edges[1:]) / 2
log_widths = log_edges[1:] - log_edges[:-1]

# === Rebinning di f(h) ===
rebinned1_fh, _ = np.histogram(arr1, bins=log_edges, weights=data1_fh)
rebinned2_fh, _ = np.histogram(arr2, bins=log_edges, weights=data2_fh)

# === Calcolo h̄_F ===
mean_hF_1 = np.sum(log_centers * rebinned1_fh * log_widths)
mean_hF_2 = np.sum(log_centers * rebinned2_fh * log_widths)

# === Calcolo d(h) = (h / h̄_F) * f(h) ===
d1 = (log_centers / mean_hF_1) * rebinned1_fh
d2 = (log_centers / mean_hF_2) * rebinned2_fh

# === Plot f(h) ===
plt.figure(figsize=(8, 6))
plt.step(log_centers, rebinned1_fh, where='mid', label='f(h) Ch1', color='blue')
plt.step(log_centers, rebinned2_fh, where='mid', label='f(h) Ch2', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Voltage [mV]')
plt.ylabel('f(h)')
plt.title('Densità di probabilità f(h)')
plt.grid(True, which='both', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("grafici/fh_distribution.pdf", dpi=300)
plt.show()

# === Plot d(h) ===
plt.figure(figsize=(8, 6))
plt.step(log_centers, d1, where='mid', label='d(h) Ch1', color='blue')
plt.step(log_centers, d2, where='mid', label='d(h) Ch2', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Voltage [mV]')
plt.ylabel('d(h)')
plt.title('Distribuzione normalizzata d(h) = (h / h̄F) · f(h)')
plt.grid(True, which='both', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("grafici/dh_distribution.pdf", dpi=300)
plt.show()

# === Verifica aree (debug opzionale) ===
area_f1 = np.sum(rebinned1_fh * log_widths)
area_d1 = np.sum(d1 * log_widths)
print("∫f(h)dh =", area_f1)  # Deve essere ~1
print("∫d(h)dh =", area_d1)  # Deve essere ~1
