import numpy as np
import matplotlib.pyplot as plt
import math


def widmo_amp(x, n=None):
    return np.abs(np.fft.fft(x, n) / N)


def widmo_faz(x, n=None):
    return np.angle(np.fft.fft(x, n) / N)


def moc_z_amp(x):
    return sum(widmo_amp(x)**2)


def moc_z_danych(x):
    return sum(np.abs(x)**2) / N


def tw_parsevala(x):
    print(f"Tw. Parsevala dla {x}:\nZ danych : {moc_z_danych(x)}\nZ FFT: {moc_z_amp(x)}\nPoprawność: {moc_z_danych(x)==moc_z_amp(x)}\n")


def generate_plot(x, n=None, title=None):
    fig, axs = plt.subplots(2)
    axs[0].stem(widmo_amp(x, n), use_line_collection=True)
    axs[0].set_title("Widmo amplitudowe", fontsize=10)
    axs[1].stem(widmo_faz(x, n), use_line_collection=True)
    axs[1].set_title("Widmo fazowe", fontsize=10)
    if title is not None:
        plt.suptitle(title)
    plt.subplots_adjust(top=0.9)
    plt.show()


def s2(x):
    return A * np.sin(2 * math.pi * x / N)


def s3(A, n, N):
    return A * (1 - ((n % N) / N))


def s4(A1, f1, A2, f2, A3, f3, t):
    return A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + A3 * np.sin(2 * np.pi * f3 * t)


#%% Zad 1
N = 4
s1 = np.array([2, 3, 1, 0])
s2 = np.array([0, 3, 1, 0])

generate_plot(s1)
generate_plot(s2)

tw_parsevala(s1)
tw_parsevala(s2)

splot = []
for n in range(0, N):
    val = 0
    for m in range(0, N):
        val += s1[m] * s2[(n - m)]
    splot.append(val)

print(f"Splot policzony ręcznie: {splot}")
print(f"Splot policzony Fourierem: {np.abs(np.fft.ifft(np.fft.fft(s1) * np.fft.fft(s2)))}")

#%% Zad 2
A = 2
N = 88

# for n0 in [0, N/4, N/2, 3*N/4]:
#     x = []
#     for n in range(N):
#         x.append(s(n-n0))
#     generate_plot(x)

#%% Zad 3
A = 4
N = 12
N0 = [0, N, 4*N, 9*N]

for n0 in N0:
    t = []
    for n in range(N):
        t.append(s3(A, n, N))

    generate_plot(t, N+n0, f"Dopełnienie {n0} zerami")

#%% Zad 4
A1 = 0.1
A2 = 0.7
A3 = 0.9
f1 = 3000
f2 = 8000
f3 = 11000
N1 = 2048

# temp = []
# for i in range(N1):
#     temp.append(s4(A1, f1, A2, f2, A3, f3, i))
# plt.plot(temp)
# plt.show()

moc = []
for k in range(N1):
    temp = []
    for i in range(N1):
        temp.append(np.abs(s4(A1, f1, A2, f2, A3, f3, i) * np.exp(-1j * 2 * np.pi * i / N1)))
    moc.append(np.sum(temp)**2 / N1)
plt.plot(moc)
plt.show()
