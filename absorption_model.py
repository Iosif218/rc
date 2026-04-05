import os
import math
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


root_window = None
pulse_type_var = None
freq_entry = None
vel_entry = None
path_entry = None
q_factor_entry = None

warning_var = None
result_amp_var = None
result_freq_var = None

fig = None
ax1 = None
ax2 = None
canvas = None

last_result = None


def ricker(frequency: float, length: float, dt: float) -> np.ndarray:
    t = np.arange(-length / 2, length / 2, dt)
    x = np.pi * frequency * t
    return (1 - 2 * x * x) * np.exp(-(x * x))


def berlage(frequency: float, length: float, dt: float, n: float = 2.0, alpha: float = None) -> np.ndarray:

    if alpha is None:
        alpha = frequency / 3.0

    t = np.arange(0, length, dt)
    s = (t ** n) * np.exp(-alpha * t) * np.sin(2 * np.pi * frequency * t)

    # центрируем по времени, чтобы график выглядел похоже на остальные
    s = s - np.mean(s)
    max_abs = np.max(np.abs(s))
    if max_abs > 0:
        s = s / max_abs
    return s


def puzyrev(frequency: float, length: float, dt: float, damping: float = 0.12) -> np.ndarray:

    t = np.arange(-length / 2, length / 2, dt)
    env = np.exp(-np.abs(t) * frequency * damping)
    s = env * np.sin(2 * np.pi * frequency * t)

    max_abs = np.max(np.abs(s))
    if max_abs > 0:
        s = s / max_abs
    return s


def build_source_signal(pulse_type: str, frequency: float, duration: float, dt: float) -> np.ndarray:
    if pulse_type == "Рикер":
        return ricker(frequency, duration, dt)
    if pulse_type == "Берлаге":
        return berlage(frequency, duration, dt)
    if pulse_type == "Пузырев":
        return puzyrev(frequency, duration, dt)
    raise ValueError(f"Неизвестный тип импульса: {pulse_type}")


# ---------- calculations ----------
def central_frequency(freqs: np.ndarray, spectrum_abs: np.ndarray) -> float:

    power = np.square(spectrum_abs)
    denom = np.sum(power)
    if denom <= 0:
        return 0.0
    return float(np.sum(freqs * power) / denom)


def validate_inputs(frequency: float, velocity: float, path: float, quality_factor: float):
    if frequency <= 0 or velocity <= 0 or path <= 0 or quality_factor <= 0:
        raise ValueError("Нельзя задавать нулевые или отрицательные параметры.")

    if not (10 <= frequency <= 5000):
        raise ValueError("Центральная частота должна быть в диапазоне от 10 Гц до 5000 Гц.")

    if not (100 <= velocity <= 5000):
        raise ValueError("Скорость должна быть в диапазоне от 100 м/с до 5000 м/с.")

    if not (1 <= path <= 5000):
        raise ValueError("Путь должен быть в диапазоне от 1 м до 5000 м.")

    if not (5 <= quality_factor <= 1000):
        raise ValueError("Добротность Q должна быть в диапазоне от 5 до 1000.")


def signals(pulse_type: str, frequency: float, velocity: float, path: float, quality_factor: float):
    validate_inputs(frequency, velocity, path, quality_factor)

    dt = 0.0001
    duration = max(20 / frequency, 0.02)
    nsmp = int(duration / dt)

    signal_before = build_source_signal(pulse_type, frequency, duration, dt)

    # подгоняем длину массива к nsmp
    if len(signal_before) > nsmp:
        signal_before = signal_before[:nsmp]
    elif len(signal_before) < nsmp:
        pad = nsmp - len(signal_before)
        signal_before = np.pad(signal_before, (0, pad))

    time = np.linspace(-duration / 2, duration / 2, len(signal_before), endpoint=False)

    amplitude_spectrum = fft.rfft(signal_before)
    frequency_spectrum = fft.rfftfreq(len(signal_before), dt)

    n_f = np.pi * frequency_spectrum / (quality_factor * velocity)
    absorption_factor = np.exp(-n_f * path)
    absorbed_spectrum = amplitude_spectrum * absorption_factor
    signal_after = fft.irfft(absorbed_spectrum, n=len(signal_before))

    amp_before = float(np.max(np.abs(signal_before)))
    amp_after = float(np.max(np.abs(signal_after)))
    amp_drop = amp_before - amp_after
    amp_drop_percent = (amp_drop / amp_before * 100.0) if amp_before > 0 else 0.0

    spec_before_abs = np.abs(amplitude_spectrum)
    spec_after_abs = np.abs(absorbed_spectrum)

    cf_before = central_frequency(frequency_spectrum, spec_before_abs)
    cf_after = central_frequency(frequency_spectrum, spec_after_abs)
    cf_shift = cf_after - cf_before

    wavelength = velocity / frequency
    far_zone_ok = path >= wavelength

    return {
        "time": time,
        "signal_before": signal_before,
        "signal_after": signal_after,
        "spectrum_before": spec_before_abs,
        "spectrum_after": spec_after_abs,
        "frequency_spectrum": frequency_spectrum,
        "amp_before": amp_before,
        "amp_after": amp_after,
        "amp_drop": amp_drop,
        "amp_drop_percent": amp_drop_percent,
        "cf_before": cf_before,
        "cf_after": cf_after,
        "cf_shift": cf_shift,
        "wavelength": wavelength,
        "far_zone_ok": far_zone_ok,
        "pulse_type": pulse_type,
        "frequency": frequency,
        "velocity": velocity,
        "path": path,
        "quality_factor": quality_factor,
    }


def update_warning_text(result: dict):
    wavelength = result["wavelength"]
    path = result["path"]

    if result["far_zone_ok"]:
        warning_var.set(
            f"Предупреждение по дальней зоне: условие выполняется "
            f"(путь = {path:.2f} м, длина волны = {wavelength:.2f} м)."
        )
    else:
        warning_var.set(
            f"ВНИМАНИЕ: путь меньше преобладающей длины волны "
            f"(путь = {path:.2f} м, длина волны = {wavelength:.2f} м). "
            f"Модель экспоненциального затухания в этой области может быть некорректной."
        )


def update_result_labels(result: dict):
    result_amp_var.set(
        f"Amax: {result['amp_before']:.6f} → {result['amp_after']:.6f}   "
        f"ΔA = -{result['amp_drop']:.6f} ({result['amp_drop_percent']:.2f}%)"
    )
    result_freq_var.set(
        f"fц: {result['cf_before']:.2f} Гц → {result['cf_after']:.2f} Гц   "
        f"Δfц = {result['cf_shift']:.2f} Гц"
    )


def calculate():
    global last_result

    try:
        pulse_type = pulse_type_var.get()
        frequency = float(freq_entry.get())
        velocity = float(vel_entry.get())
        path = float(path_entry.get())
        quality_factor = float(q_factor_entry.get())

        result = signals(pulse_type, frequency, velocity, path, quality_factor)
        last_result = result

        time = result["time"]
        signal_before = result["signal_before"]
        signal_after = result["signal_after"]
        spectrum_before = result["spectrum_before"]
        spectrum_after = result["spectrum_after"]
        frequency_spectrum = result["frequency_spectrum"]

        ax1.clear()
        ax2.clear()

        ax1.plot(time, signal_before, linewidth=2, label="До поглощения")
        ax1.plot(time, signal_after, linewidth=2, label="После поглощения")
        ax1.set_title(f"Сигнал до и после поглощения ({pulse_type})")
        ax1.set_xlabel("Время, с")
        ax1.set_ylabel("Амплитуда")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        max_t = min(1 / frequency, max(abs(time.min()), abs(time.max())))
        ax1.set_xlim(-max_t, max_t)

        ax2.plot(frequency_spectrum, spectrum_before, linewidth=2, label="До поглощения")
        ax2.plot(frequency_spectrum, spectrum_after, linewidth=2, label="После поглощения")
        ax2.set_title("Спектр до и после поглощения")
        ax2.set_xlabel("Частота, Гц")
        ax2.set_ylabel("Амплитуда")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        max_freq = min(frequency * 3, frequency_spectrum.max())
        ax2.set_xlim(0, max_freq)

        update_result_labels(result)
        update_warning_text(result)
        canvas.draw()

    except ValueError as e:
        messagebox.showerror("Ошибка ввода", str(e))
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


def save_figure():
    if last_result is None:
        messagebox.showwarning("Нет данных", "Сначала выполните расчёт.")
        return

    default_name = "absorption_model.png"
    path = filedialog.asksaveasfilename(
        title="Сохранить график",
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
    )
    if not path:
        return

    meta_text = (
        f"Импульс: {last_result['pulse_type']}\n"
        f"Частота: {last_result['frequency']} Гц\n"
        f"Скорость: {last_result['velocity']} м/с\n"
        f"Путь: {last_result['path']} м\n"
        f"Q: {last_result['quality_factor']}\n"
        f"Amax до: {last_result['amp_before']:.6f}\n"
        f"Amax после: {last_result['amp_after']:.6f}\n"
        f"ΔA: -{last_result['amp_drop']:.6f} ({last_result['amp_drop_percent']:.2f}%)\n"
        f"fц до: {last_result['cf_before']:.2f} Гц\n"
        f"fц после: {last_result['cf_after']:.2f} Гц\n"
        f"Δfц: {last_result['cf_shift']:.2f} Гц\n"
        f"Длина волны: {last_result['wavelength']:.2f} м\n"
    )

    fig.text(
        0.01, 0.01, meta_text,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
    )

    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        messagebox.showinfo("Успех", "График сохранён.")
    except Exception as e:
        messagebox.showerror("Ошибка сохранения", str(e))
    finally:
        if fig.texts:
            fig.texts[-1].remove()
        canvas.draw()


def _bind_numeric_validation(entry: ttk.Entry):
    def on_key(event):
        allowed = "0123456789.,"
        control_keys = {
            "BackSpace", "Delete", "Left", "Right", "Home", "End", "Tab"
        }
        if event.keysym in control_keys:
            return
        if event.char and event.char not in allowed:
            return "break"

    entry.bind("<KeyPress>", on_key)


def run(root):
    global root_window, pulse_type_var, freq_entry, vel_entry, path_entry, q_factor_entry
    global warning_var, result_amp_var, result_freq_var
    global fig, ax1, ax2, canvas

    root_window = root

    for widget in root.winfo_children():
        widget.destroy()

    root.title("Моделирование сейсмического поглощения")
    root.geometry("1280x760")

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(main_frame, padding=10)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    right_frame = ttk.Frame(main_frame, padding=10)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    ttk.Label(left_frame, text="Параметры модели", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(30, 15)
    )

    ttk.Label(left_frame, text="Форма импульса:").grid(row=1, column=0, sticky="w", pady=5)
    pulse_type_var = tk.StringVar(value="Рикер")
    pulse_box = ttk.Combobox(
        left_frame,
        textvariable=pulse_type_var,
        state="readonly",
        values=["Рикер", "Берлаге", "Пузырев"],
        width=18
    )
    pulse_box.grid(row=1, column=1, pady=5, sticky="ew")

    ttk.Label(left_frame, text="Центральная частота (Гц):").grid(row=2, column=0, sticky="w", pady=5)
    freq_entry = ttk.Entry(left_frame, width=20)
    freq_entry.grid(row=2, column=1, pady=5, sticky="ew")
    freq_entry.insert(0, "25")

    ttk.Label(left_frame, text="Скорость (м/с):").grid(row=3, column=0, sticky="w", pady=5)
    vel_entry = ttk.Entry(left_frame, width=20)
    vel_entry.grid(row=3, column=1, pady=5, sticky="ew")
    vel_entry.insert(0, "1500")

    ttk.Label(left_frame, text="Путь (м):").grid(row=4, column=0, sticky="w", pady=5)
    path_entry = ttk.Entry(left_frame, width=20)
    path_entry.grid(row=4, column=1, pady=5, sticky="ew")
    path_entry.insert(0, "100")

    ttk.Label(left_frame, text="Добротность Q:").grid(row=5, column=0, sticky="w", pady=5)
    q_factor_entry = ttk.Entry(left_frame, width=20)
    q_factor_entry.grid(row=5, column=1, pady=5, sticky="ew")
    q_factor_entry.insert(0, "30")

    for entry in (freq_entry, vel_entry, path_entry, q_factor_entry):
        _bind_numeric_validation(entry)

    ttk.Label(
        left_frame,
        text="Ограничения:\n"
             "f: 10..5000 Гц\n"
             "V: 100..5000 м/с\n"
             "L: 1..5000 м\n"
             "Q: 5..1000",
        justify="left"
    ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(12, 12))

    ttk.Button(left_frame, text="Посчитать", command=calculate).grid(
        row=7, column=0, columnspan=2, sticky="ew", pady=(5, 8)
    )
    ttk.Button(left_frame, text="Сохранить картинку", command=save_figure).grid(
        row=8, column=0, columnspan=2, sticky="ew", pady=(0, 10)
    )

    result_amp_var = tk.StringVar(value="Amax: —")
    result_freq_var = tk.StringVar(value="fц: —")
    warning_var = tk.StringVar(value="")

    ttk.Label(
        left_frame,
        textvariable=result_amp_var,
        wraplength=320,
        justify="left"
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 6))

    ttk.Label(
        left_frame,
        textvariable=result_freq_var,
        wraplength=320,
        justify="left"
    ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(0, 6))

    ttk.Label(
        left_frame,
        textvariable=warning_var,
        wraplength=320,
        justify="left",
        foreground="darkred"
    ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 6))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 6.6))
    fig.tight_layout(pad=3.2)

    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    calculate()