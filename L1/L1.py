import numpy as np
import matplotlib.pyplot as plt


def amp_spectrum(x, n=None):
    return np.abs(clean(np.fft.fft(x, n) / len(x)))


def phase_spectrum(x, n=None):
    return np.angle(clean(np.fft.fft(x, n) / len(x)))


def power_density_spectrum(x, n=None):
    return np.abs(clean(np.fft.rfft(x, n) / len(x)))


def clean(x):
    for i in range(len(x)):
        if np.abs(x[i]) < 10 ** -6:
            x[i] = 0
    return np.array(x)


def amp_power(x):
    return sum(amp_spectrum(x)**2)


def data_power(x):
    return sum(np.abs(x)**2) / N


def convolution(s1, s2):
    conv = []
    for n in range(0, N):
        val = 0
        for m in range(0, N):
            val += s1[m] * s2[(n - m)]
        conv.append(val)
    return conv


def fourier_convolution(s1, s2):
    return np.abs(np.fft.ifft(np.fft.fft(s1) * np.fft.fft(s2)))


def parsevals_theorem(x):
    print(f"Tw. Parsevala dla {x}:")
    print(f"Z danych : {data_power(x)}")
    print(f"Z FFT: {amp_power(x)}")
    print(f"Poprawność: {data_power(x) == amp_power(x)}\n")


def generate_double_plot(x, n=None, xlabel=None, ylabel=None, title=None):
    fig, axs = plt.subplots(2)
    axs[0].stem(amp_spectrum(x, n), use_line_collection=True)
    axs[0].set_ylabel("Widmo amplitudowe", fontsize=10)
    axs[1].stem(phase_spectrum(x, n), use_line_collection=True)

    if xlabel:
        axs[0].set_xlabel(xlabel[0], fontsize=10)
        axs[1].set_xlabel(xlabel[1], fontsize=10)

    if ylabel:
        axs[0].set_ylabel(ylabel[0], fontsize=10)
        axs[1].set_ylabel(ylabel[1], fontsize=10)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def harmonic(A, f, x):
    return A * np.sin(2 * np.pi * f * x)


def s_2(A, N, x):
    return harmonic(A, 1/N, x)


def s_3(A, n, N):
    return A * (1 - ((n % N) / N))


def s_4(A, f, x):
    return sum([harmonic(A[i], f[i], x) for i in range(len(A))])


#%% Zad 1 %%#
s1 = np.array([2, 3, 1, 0])
s2 = np.array([0, 3, 1, 0])

generate_double_plot(s1, ylabel=["Widmo amplitudowe", "Widmo fazowe"], xlabel=["n", "n"], title=f"Widma dla s={s1}")
generate_double_plot(s2, ylabel=["Widmo amplitudowe", "Widmo fazowe"], xlabel=["n", "n"], title=f"Widma dla s={s2}")

parsevals_theorem(s1)
parsevals_theorem(s2)

print(f"Splot policzony \"ręcznie\": {convolution(s1, s2)}")
print(f"Splot policzony Fourierem: {fourier_convolution(s1, s2)}")

#%% Zad 2 %%#
A = 2
N = 88

for n0 in [0, N/4, N/2, 3*N/4]:
    x = [s_2(A, N, n-n0) for n in range(N)]
    generate_double_plot(x, ylabel=["Widmo amplitudowe", "Widmo fazowe"], xlabel=["n", "n"], title=f"Przesunięcie o {n0}")

#%% Zad 3 %%#
A = 4
N = 12
N0 = [0, N, 4*N, 9*N]

for n0 in N0:
    t = [s_3(A, n, N) for n in range(N)]
    generate_double_plot(t, N + n0, ylabel=["Widmo amplitudowe", "Widmo fazowe"], xlabel=["n", "n"], title=f"Dopełnienie {n0} zerami")

#%% Zad 4 %%#
A = [0.1, 0.7, 0.9]
f = [3000, 8000, 11000]
N1 = 2048
N2 = 3 * N1 // 2

f0 = 48000

for N in [N1, N2]:
    s = [s_4(A, f, i) for i in np.arange(0, 1 / f0 * N, 1 / f0)]
    power = power_density_spectrum(s)
    plt.plot(np.arange(0, f0 / 2 + 1, f0 / N), power)
    plt.xticks(np.arange(0, f0 / 2 + 1, 3000))
    plt.title(f"Widmo gęstości mocy dla N = {N}")
    plt.xlabel("f")
    plt.ylabel("Gęstość mocy")
    plt.show()
