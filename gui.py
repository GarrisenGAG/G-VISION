import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

# тема
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("G-Vision")
        self.geometry("800x600")
        self.overrideredirect(True)

        self.config(bg='#010101')
        self.attributes('-transparentcolor', '#010101')

        self.pad = ctk.CTkFrame(self,
                                fg_color="#0a133b",
                                corner_radius=30,
                                bg_color='#010101')
        self.pad.pack(fill="both", expand=True, padx=15, pady=15)

        self.left = ctk.CTkFrame(self.pad,
                                 fg_color="#24306e",
                                 width=200,
                                 corner_radius=20)
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        self.right = ctk.CTkFrame(self.pad,
                                  fg_color="#08103b",
                                  corner_radius=20)
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.btn_load = ctk.CTkButton(self.left, text="Загрузить фото",
                                      font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                      command=self.load_image, fg_color="#08103b")
        self.btn_load.pack(pady=10, padx=10)

        self.btn_run = ctk.CTkButton(self.left, text="Распознать",
                                     font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                     command=self.recognize, fg_color="#08103b")
        self.btn_run.pack(pady=5, padx=10)

        self.btn_clear = ctk.CTkButton(self.left, text="Очистить",
                                       font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                       command=self.clear, fg_color="#08103b")
        self.btn_clear.pack(pady=5, padx=10)

        self.status = ctk.CTkLabel(self.left, text="Жду картинку",
                                   font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                   wraplength=180)
        self.status.pack(pady=10, padx=10)

        self.image_label = ctk.CTkLabel(self.right, text="*тут будет фото*",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"))
        self.image_label.pack(pady=5)

        self.result_box = ctk.CTkTextbox(self.right, height=200, font=ctk.CTkFont(family="Arial", size=12, weight="bold"), corner_radius=20, fg_color="#21263d")
        self.result_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_path = None

        self.bind('<Escape>', lambda e: self.destroy())

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("изображения", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.image_path = path
        img = ctk.CTkImage(Image.open(path), size=(400, 300))
        self.image_label.configure(image=img, text="")
        self.image_label.image = img
        self.status.configure(text="Фото загружено")

    def recognize(self):
        if not self.image_path:
            self.status.configure(text="Сначала загрузи фото")
            return
        self.status.configure(text="распознаю...")
        # from train_garrisen import GVisionOCR
        # ocr = GVisionOCR("g-vision-final.pt", "g-vision-config.json")
        # result = ocr.recognize(self.image_path)
        # text = result["text"]
        text = "*тут будет распознанный текст*"
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", text)
        self.status.configure(text="готово")

    def clear(self):
        self.image_label.configure(image=None, text="*тут будет фото*",
                                   font=ctk.CTkFont(family="Arial", size=16, weight="bold"))
        self.result_box.delete("1.0", "end")
        self.image_path = None
        self.status.configure(text="Очищено")


if __name__ == "__main__":
    app = App()
    app.mainloop()
