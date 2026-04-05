import tkinter as tk
from tkinter import ttk

import absorption_model
import test_reflection


def return_to_menu(child_window, parent_window):
    child_window.destroy()
    parent_window.deiconify()


def open_absorption_model(parent_window):
    parent_window.withdraw()

    absorption_window = tk.Toplevel(parent_window)
    absorption_window.title("Моделирование сейсмического поглощения")
    absorption_window.geometry("1200x700")

    absorption_window.protocol(
        "WM_DELETE_WINDOW",
        lambda: return_to_menu(absorption_window, parent_window)
    )

    absorption_model.run(absorption_window)

    back_btn = ttk.Button(
        absorption_window,
        text="← Назад в меню",
        command=lambda: return_to_menu(absorption_window, parent_window)
    )
    back_btn.place(x=10, y=10)


def open_reflection_model(parent_window):
    parent_window.withdraw()

    reflection_window = test_reflection.ReflectionsApp(parent_window)

    reflection_window.protocol(
        "WM_DELETE_WINDOW",
        lambda: return_to_menu(reflection_window, parent_window)
    )

    back_btn = ttk.Button(
        reflection_window,
        text="← Назад в меню",
        command=lambda: return_to_menu(reflection_window, parent_window)
    )
    back_btn.place(x=10, y=10)


def create_main_menu():
    root = tk.Tk()
    root.title("Геофизические модели")
    root.geometry("500x400")

    style = ttk.Style()
    style.theme_use("clam")

    menubar = tk.Menu(root, bg="#2c3e50", fg="white", font=("Arial", 10))
    root.config(menu=menubar)

    help_menu = tk.Menu(menubar, tearoff=0, bg="#34495e", fg="white")
    help_menu.add_command(label="О программе")
    menubar.add_cascade(label="Справка", menu=help_menu)

    models_menu = tk.Menu(menubar, tearoff=0, bg="#34495e", fg="white")
    models_menu.add_command(
        label="Моделирование сейсмического поглощения",
        command=lambda: open_absorption_model(root)
    )
    models_menu.add_command(
        label="Моделирование коэффициентов отражения",
        command=lambda: open_reflection_model(root)
    )
    menubar.add_cascade(label="Модели", menu=models_menu)

    tests_menu = tk.Menu(menubar, tearoff=0, bg="#34495e", fg="white")
    tests_menu.add_command(label="Тесты")
    menubar.add_cascade(label="Тесты", menu=tests_menu)

    menubar.add_command(label="Выход", command=root.quit)

    main_frame = ttk.Frame(root, padding="30")
    main_frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        main_frame,
        text="Геофизические модели",
        font=("Arial", 18, "bold"),
        fg="#2c3e50"
    )
    title_label.pack(pady=(0, 20))

    subtitle_label = tk.Label(
        main_frame,
        text="Выберите модель из меню 'Модели'",
        font=("Arial", 11),
        fg="#7f8c8d"
    )
    subtitle_label.pack(pady=(0, 30))

    absorption_btn = ttk.Button(
        main_frame,
        text="Открыть модель поглощения",
        command=lambda: open_absorption_model(root),
        width=30
    )
    absorption_btn.pack(pady=10)

    reflection_btn = ttk.Button(
        main_frame,
        text="Открыть модель отражений",
        command=lambda: open_reflection_model(root),
        width=30
    )
    reflection_btn.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_main_menu()