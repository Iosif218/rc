import os
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import zoeppritz as z
from common import aki_rpp, shuye_rpp, liquids_rpp, rsh


WARN_BG = "#ffe7a3"
DANGER_BG = "#ffb3b3"


def poisson_from_vp_vs(vp: float, vs: float) -> float:
    if vp <= 0 or vs <= 0:
        return float("nan")
    denom = 2.0 * (vp * vp - vs * vs)
    if denom == 0:
        return float("nan")
    return (vp * vp - 2.0 * vs * vs) / denom


def shuey_intercept_gradient(vp1, vs1, rho1, vp2, vs2, rho2):
    vp = 0.5 * (vp1 + vp2)
    vs = 0.5 * (vs1 + vs2)
    rho = 0.5 * (rho1 + rho2)

    if vp == 0 or vs == 0 or rho == 0:
        return float("nan"), float("nan")

    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1

    a = 0.5 * (dvp / vp + drho / rho)
    b = 0.5 * (dvp / vp) - 2.0 * (vs * vs / (vp * vp)) * (drho / rho + 2.0 * dvs / vs)
    return a, b


class ReflectionsApp(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Reflections")
        self.minsize(1200, 720)

        self.output_dir = tk.StringVar(value=os.getcwd())
        self.file_name = tk.StringVar(value="reflections")
        self.file_ext = tk.StringVar(value=".png")

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ---------- Left: controls ----------
        ctrl = ttk.Frame(self, padding=8)
        ctrl.grid(row=0, column=0, sticky="ns")
        ctrl.columnconfigure(0, weight=1)

        # ---------- Right: plot ----------
        plot_frame = ttk.Frame(self, padding=8)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8.0, 6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Model
        model_box = ttk.LabelFrame(ctrl, text="Model", padding=8)
        model_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(model_box, text="Approximation").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.model = tk.StringVar(value="Zoeppritz")
        self.model_cb = ttk.Combobox(
            model_box,
            textvariable=self.model,
            state="readonly",
            width=18,
            values=["Zoeppritz", "Aki-Richards", "Shuey", "Liquids (PP)", "SH (Rsh)"]
        )
        self.model_cb.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(model_box, text="Wave type").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.wave = tk.StringVar(value="PP")
        self.wave_cb = ttk.Combobox(
            model_box,
            textvariable=self.wave,
            state="readonly",
            width=18,
            values=["PP", "PS", "SP", "SS"]
        )
        self.wave_cb.grid(row=1, column=1, sticky="ew", pady=4)

        # Layer parameters
        layers = ttk.LabelFrame(ctrl, text="Layers parameters", padding=8)
        layers.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(layers, text="").grid(row=0, column=0, padx=3)
        ttk.Label(layers, text="Layer 1").grid(row=0, column=1, padx=3)
        ttk.Label(layers, text="Layer 2").grid(row=0, column=2, padx=3)
        ttk.Label(layers, text="").grid(row=0, column=3, padx=3)

        def make_spin(parent, row, label, v1, v2, unit, from_, to, inc):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 5), pady=3)

            var1 = tk.StringVar(value=str(v1))
            s1 = tk.Spinbox(
                parent, width=8, justify="right",
                from_=from_, to=to, increment=50,
                textvariable=var1, command=self._on_spin_change,
                bg="white"
            )
            s1.grid(row=row, column=1, padx=3, pady=3)

            var2 = tk.StringVar(value=str(v2))
            s2 = tk.Spinbox(
                parent, width=8, justify="right",
                from_=from_, to=to, increment=50,
                textvariable=var2, command=self._on_spin_change,
                bg="white"
            )
            s2.grid(row=row, column=2, padx=3, pady=3)

            ttk.Label(parent, text=unit).grid(row=row, column=3, sticky="w", padx=(4, 0), pady=3)
            return s1, s2

        self.vp1_e, self.vp2_e = make_spin(layers, 1, "Vp", 1600, 2400, "m/s", 100, 8000, 10)
        self.vs1_e, self.vs2_e = make_spin(layers, 2, "Vs", 500, 1100, "m/s", 50, 5000, 10)
        self.rho1_e, self.rho2_e = make_spin(layers, 3, "Density", 1.8, 2.3, "g/cm³", 0.5, 5.0, 0.05)

        ttk.Label(layers, text="Poisson").grid(row=4, column=0, sticky="w", padx=(0, 5), pady=3)
        self.nu1_var = tk.StringVar(value="—")
        self.nu2_var = tk.StringVar(value="—")
        ttk.Entry(layers, textvariable=self.nu1_var, width=8, state="readonly").grid(row=4, column=1, padx=3, pady=3)
        ttk.Entry(layers, textvariable=self.nu2_var, width=8, state="readonly").grid(row=4, column=2, padx=3, pady=3)

        self.warn_var = tk.StringVar(value="")
        self.warn_lbl = ttk.Label(layers, textvariable=self.warn_var, foreground="red", wraplength=310)
        self.warn_lbl.grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # Calculation
        calc_box = ttk.LabelFrame(ctrl, text="Calculation", padding=8)
        calc_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(calc_box, text="R₀").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.r0_var = tk.StringVar(value="—")
        ttk.Entry(calc_box, textvariable=self.r0_var, width=10, state="readonly").grid(row=0, column=1, pady=3)

        ttk.Label(calc_box, text="G").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.g_var = tk.StringVar(value="—")
        ttk.Entry(calc_box, textvariable=self.g_var, width=10, state="readonly").grid(row=1, column=1, pady=3)

        ttk.Button(calc_box, text="Calculate", command=self.calculate).grid(row=0, column=2, rowspan=2, padx=(12, 0), sticky="nsew")

        self.auto_recalc = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            calc_box,
            text="Recalculate automatically",
            variable=self.auto_recalc
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))

        # Visual parameters
        visual_box = ttk.LabelFrame(ctrl, text="Visual parameters", padding=8)
        visual_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        visual_box.columnconfigure(0, weight=1)
        visual_box.columnconfigure(1, weight=1)

        draw_box = ttk.LabelFrame(visual_box, text="Part to draw", padding=6)
        draw_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.show_re = tk.BooleanVar(value=True)
        self.show_im = tk.BooleanVar(value=True)
        self.show_abs = tk.BooleanVar(value=True)

        ttk.Checkbutton(draw_box, text="real", variable=self.show_re, command=self._auto_calc).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(draw_box, text="imaginary", variable=self.show_im, command=self._auto_calc).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(draw_box, text="absolute", variable=self.show_abs, command=self._auto_calc).grid(row=2, column=0, sticky="w")

        bounds_box = ttk.LabelFrame(visual_box, text="Boundaries", padding=6)
        bounds_box.grid(row=0, column=1, sticky="nsew")

        ttk.Label(bounds_box, text="").grid(row=0, column=0)
        ttk.Label(bounds_box, text="min").grid(row=0, column=1)
        ttk.Label(bounds_box, text="max").grid(row=0, column=2)
        ttk.Label(bounds_box, text="auto").grid(row=0, column=3)

        ttk.Label(bounds_box, text="Angles").grid(row=1, column=0, sticky="w")
        self.ang_from = ttk.Entry(bounds_box, width=6)
        self.ang_from.grid(row=1, column=1, padx=2)
        self.ang_from.insert(0, "0")
        self.ang_to = ttk.Entry(bounds_box, width=6)
        self.ang_to.grid(row=1, column=2, padx=2)
        self.ang_to.insert(0, "60")
        self.auto_ang = tk.BooleanVar(value=True)
        ttk.Checkbutton(bounds_box, variable=self.auto_ang).grid(row=1, column=3)

        ttk.Label(bounds_box, text="R").grid(row=2, column=0, sticky="w")
        self.rmin = ttk.Entry(bounds_box, width=6)
        self.rmin.grid(row=2, column=1, padx=2)
        self.rmin.insert(0, "-1.00")
        self.rmax = ttk.Entry(bounds_box, width=6)
        self.rmax.grid(row=2, column=2, padx=2)
        self.rmax.insert(0, "1.00")
        self.auto_r = tk.BooleanVar(value=True)
        ttk.Checkbutton(bounds_box, variable=self.auto_r).grid(row=2, column=3)

        # Export
        export_box = ttk.LabelFrame(ctrl, text="Export", padding=8)
        export_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        export_box.columnconfigure(1, weight=1)

        ttk.Button(export_box, text="Save graph", command=self.save_graph).grid(row=0, column=0, padx=(0, 6), pady=4, sticky="ew")
        ttk.Button(export_box, text="Save to table", command=self.export_csv).grid(row=0, column=1, padx=(0, 0), pady=4, sticky="ew")

        ttk.Button(export_box, text="Change folder", command=self.pick_output_dir).grid(row=1, column=0, padx=(0, 6), pady=4, sticky="ew")
        ttk.Entry(export_box, textvariable=self.output_dir, state="readonly").grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(export_box, text="File name").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(export_box, textvariable=self.file_name).grid(row=2, column=1, sticky="ew", pady=4)

        ext_box = ttk.Combobox(export_box, textvariable=self.file_ext, state="readonly", width=8,
                               values=[".png", ".pdf", ".svg", ".csv"])
        ext_box.grid(row=2, column=2, padx=(6, 0), pady=4)

        # Функции Zoeppritz по типам волн
        self.ZOEPPRITZ_FUNCS = {
            "PP": getattr(z, "calc_rpp", None),
            "PS": getattr(z, "calc_rps", None),
            "SP": getattr(z, "calc_rsp", None),
            "SS": getattr(z, "calc_rss", None),
        }

        # Binds
        for w in (
            self.vp1_e, self.vp2_e, self.vs1_e, self.vs2_e,
            self.rho1_e, self.rho2_e, self.ang_from, self.ang_to
        ):
            w.bind("<KeyRelease>", self._on_input_change)
            w.bind("<ButtonRelease-1>", self._on_input_change)
            w.bind("<FocusOut>", self._on_input_change)

        self.model_cb.bind("<<ComboboxSelected>>", self._on_model_selected)
        self.wave_cb.bind("<<ComboboxSelected>>", lambda e: self._auto_calc())

        self.last_angles = None
        self.last_r = None

        self._update_wave_options()
        self.update_derived()
        self.calculate()

    def _on_spin_change(self):
        self._on_input_change()

    def _on_input_change(self, _event=None):
        self.update_derived()
        self._auto_calc()

    def _on_model_selected(self, _event=None):
        self._update_wave_options()
        self.update_derived()
        self._auto_calc()

    def _update_wave_options(self):
        model = self.model.get()

        if model == "Zoeppritz":
            available = [w for w, fn in self.ZOEPPRITZ_FUNCS.items() if fn is not None]
            if not available:
                available = ["PP"]
        elif model in ("Aki-Richards", "Shuey", "Liquids (PP)"):
            available = ["PP"]
        elif model == "SH (Rsh)":
            available = ["SS"]
        else:
            available = ["PP"]

        self.wave_cb["values"] = available
        if self.wave.get() not in available:
            self.wave.set(available[0])

    def _auto_calc(self):
        if self.auto_recalc.get():
            self.calculate()

    def read_layer_values(self):
        vp1 = float(self.vp1_e.get())
        vs1 = float(self.vs1_e.get())
        rho1 = float(str(self.rho1_e.get()).replace(",", "."))

        vp2 = float(self.vp2_e.get())
        vs2 = float(self.vs2_e.get())
        rho2 = float(str(self.rho2_e.get()).replace(",", "."))

        return vp1, vs1, rho1, vp2, vs2, rho2

    def _set_widget_bg(self, widget, color):
        try:
            widget.configure(bg=color)
        except Exception:
            pass

    def reset_warning_colors(self):
        for w in (self.vp1_e, self.vp2_e, self.vs1_e, self.vs2_e):
            self._set_widget_bg(w, "white")

    def update_warnings(self, vp1, vs1, vp2, vs2):
        self.reset_warning_colors()
        msgs = []

        if vp1 > 0 and vs1 > 0 and vs1 > vp1 / math.sqrt(2):
            self._set_widget_bg(self.vs1_e, DANGER_BG)
            msgs.append("В 1-м слое скорости заданы некорректно")

        if vp2 > 0 and vs2 > 0 and vs2 > vp2 / math.sqrt(2):
            self._set_widget_bg(self.vs2_e, DANGER_BG)
            msgs.append("Во 2-м слое скорости заданы некорректно")

        def mark_diff(w1, w2, a, b, name):
            if a <= 0 or b <= 0:
                return
            rel = abs(a - b) / max(a, b)
            if rel > 0.70:
                self._set_widget_bg(w1, WARN_BG)
                self._set_widget_bg(w2, WARN_BG)
                msgs.append("Слои слишком разные")

        mark_diff(self.vp1_e, self.vp2_e, vp1, vp2, "Vp")
        #mark_diff(self.vs1_e, self.vs2_e, vs1, vs2, "Vs")

        self.warn_var.set("; ".join(msgs))

    def update_derived(self):
        try:
            vp1, vs1, rho1, vp2, vs2, rho2 = self.read_layer_values()

            nu1 = poisson_from_vp_vs(vp1, vs1)
            nu2 = poisson_from_vp_vs(vp2, vs2)
            self.nu1_var.set(f"{nu1:.3f}" if np.isfinite(nu1) else "—")
            self.nu2_var.set(f"{nu2:.3f}" if np.isfinite(nu2) else "—")

            a, b = shuey_intercept_gradient(vp1, vs1, rho1, vp2, vs2, rho2)
            self.r0_var.set(f"{a:.3f}" if np.isfinite(a) else "—")
            self.g_var.set(f"{b:.3f}" if np.isfinite(b) else "—")

            self.update_warnings(vp1, vs1, vp2, vs2)

        except Exception:
            self.nu1_var.set("—")
            self.nu2_var.set("—")
            self.r0_var.set("—")
            self.g_var.set("—")
            self.warn_var.set("")
            self.reset_warning_colors()

    def read_inputs(self):
        vp1, vs1, rho1, vp2, vs2, rho2 = self.read_layer_values()

        a0 = float(self.ang_from.get())
        a1 = float(self.ang_to.get())

        if a1 < a0:
            a0, a1 = a1, a0

        angles = np.arange(a0, a1 + 1e-9, 1.0)
        return vp1, vs1, rho1, vp2, vs2, rho2, angles

    def compute_r(self, vp1, vs1, rho1, vp2, vs2, rho2, angles):
        model = self.model.get()
        wave = self.wave.get()

        if model == "Zoeppritz":
            fn = self.ZOEPPRITZ_FUNCS.get(wave)
            if fn is None:
                raise ValueError(f"Для типа волны {wave} функция в zoeppritz.py не найдена")
            return np.array([fn(vp1, vs1, rho1, vp2, vs2, rho2, float(a)) for a in angles], dtype=complex)

        if model == "Aki-Richards":
            if wave != "PP":
                raise ValueError("Модель Aki-Richards поддерживает только PP")
            return np.array([aki_rpp(vp1, vs1, rho1, vp2, vs2, rho2, float(a)) for a in angles], dtype=complex)

        if model == "Shuey":
            if wave != "PP":
                raise ValueError("Модель Shuey поддерживает только PP")
            return np.array([shuye_rpp(vp1, vs1, rho1, vp2, vs2, rho2, float(a)) for a in angles], dtype=complex)

        if model == "Liquids (PP)":
            if wave != "PP":
                raise ValueError("Модель Liquids поддерживает только PP")
            return np.array([liquids_rpp(vp1, rho1, vp2, rho2, float(a)) for a in angles], dtype=complex)

        if model == "SH (Rsh)":
            if wave != "SS":
                raise ValueError("Модель SH (Rsh) поддерживает только SS")
            return np.array([rsh(vs1, rho1, vs2, rho2, float(a)) for a in angles], dtype=complex)

        raise ValueError(f"Unknown model: {model}")

    def calculate(self):
        try:
            vp1, vs1, rho1, vp2, vs2, rho2, angles = self.read_inputs()
            r = self.compute_r(vp1, vs1, rho1, vp2, vs2, rho2, angles)

            self.last_angles = angles
            self.last_r = r

            self.ax.clear()

            title = (
                f"Vp1={vp1:.0f}, Vs1={vs1:.0f}, Dn1={rho1:.2f}, "
                f"Vp2={vp2:.0f}, Vs2={vs2:.0f}, Dn2={rho2:.2f}"
            )
            self.ax.set_title(title, fontsize=10)

            if self.show_re.get():
                self.ax.plot(angles, np.real(r), color="red", label="real")
            if self.show_im.get():
                self.ax.plot(angles, np.imag(r), color="blue", label="imaginary")
            if self.show_abs.get():
                self.ax.plot(angles, np.abs(r), color="black", label="absolute")

            self.ax.set_xlabel(f"Angles ({self.wave.get()}-wave)")
            self.ax.set_ylabel("R")
            self.ax.grid(True, linestyle=":", alpha=0.7)

            if not self.auto_r.get():
                try:
                    ymin = float(self.rmin.get())
                    ymax = float(self.rmax.get())
                    self.ax.set_ylim(ymin, ymax)
                except Exception:
                    pass

            self.ax.set_xlim(float(angles.min()), float(angles.max()))
            self.ax.tick_params(top=True, right=True, labeltop=True, labelright=True)
            self.ax.legend(loc="best")

            self.canvas.draw()
            self.update_derived()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _build_save_path(self, ext):
        name = self.file_name.get().strip() or "reflections"
        return os.path.join(self.output_dir.get(), name + ext)

    def pick_output_dir(self):
        d = filedialog.askdirectory(initialdir=self.output_dir.get(), title="Select output directory")
        if d:
            self.output_dir.set(d)

    def save_graph(self):
        if self.last_angles is None or self.last_r is None:
            messagebox.showwarning("No data", "Nothing to save. Click Calculate first.")
            return

        ext = self.file_ext.get()
        if ext == ".csv":
            ext = ".png"

        path = filedialog.asksaveasfilename(
            title="Save graph",
            initialfile=os.path.basename(self._build_save_path(ext)),
            initialdir=os.path.dirname(self._build_save_path(ext)),
            defaultextension=ext,
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if not path:
            return

        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def export_csv(self):
        if self.last_angles is None or self.last_r is None:
            messagebox.showwarning("No data", "Nothing to export. Click Calculate first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export CSV",
            initialfile=os.path.basename(self._build_save_path(".csv")),
            initialdir=os.path.dirname(self._build_save_path(".csv")),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["angle_deg", "real", "imag", "abs"])
                for a, val in zip(self.last_angles, self.last_r):
                    w.writerow([float(a), float(np.real(val)), float(np.imag(val)), float(np.abs(val))])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")


if __name__ == "__main__":
    app = ReflectionsApp()
    app.mainloop()