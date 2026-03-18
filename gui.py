import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

# тема
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("G-Vision OCR")
        self.geometry("800x600")
        
        self.left = ctk.CTkFrame(self, width=200)
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_load = ctk.CTkButton(self.left, text="загрузить фото", command=self.load_image)
        self.btn_load.pack(pady=10, padx=10)

        self.btn_run = ctk.CTkButton(self.left, text="распознать", command=self.recognize)
        self.btn_run.pack(pady=5, padx=10)

        self.btn_clear = ctk.CTkButton(self.left, text="очистить", command=self.clear)
        self.btn_clear.pack(pady=5, padx=10)

        self.status = ctk.CTkLabel(self.left, text="готов к работе", wraplength=180)
        self.status.pack(pady=10, padx=10)

        self.right = ctk.CTkFrame(self)
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.right, text="тут будет фото")
        self.image_label.pack(pady=5)

        self.result_box = ctk.CTkTextbox(self.right, height=200)
        self.result_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_path = None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("изображения", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.image_path = path
        img = ctk.CTkImage(Image.open(path), size=(400, 300))
        self.image_label.configure(image=img, text="")
        self.image_label.image = img
        self.status.configure(text="фото загружено")

    def recognize(self):
        if not self.image_path:
            self.status.configure(text="сначала загрузи фото")
            return
        self.status.configure(text="распознаю...")
        # тут подключается модель
        # from train_garrisen import GVisionOCR
        # ocr = GVisionOCR("g-vision-final.pt", "g-vision-config.json")
        # result = ocr.recognize(self.image_path)
        # text = result["text"]
        text = "тут будет распознанный текст"
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", text)
        self.status.configure(text="готово")

    def clear(self):
        self.image_label.configure(image=None, text="тут будет фото")
        self.result_box.delete("1.0", "end")
        self.image_path = None
        self.status.configure(text="очищено")


if __name__ == "__main__":
    app = App()
    app.mainloop()
