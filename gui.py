import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import threading
import json
import os
from pathlib import Path

# theme
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

        # Animation variables
        self.is_animating = False
        self.animation_thread = None
        self.video_path = Path(__file__).parent / "wait_anim.mkv"

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

        self.btn_load = ctk.CTkButton(self.left, text="Load photo",
                                      font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                      command=self.load_image, fg_color="#08103b")
        self.btn_load.pack(pady=10, padx=10)

        self.btn_run = ctk.CTkButton(self.left, text="Recognize",
                                     font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                     command=self.recognize, fg_color="#08103b")
        self.btn_run.pack(pady=5, padx=10)

        self.btn_clear = ctk.CTkButton(self.left, text="Clear",
                                       font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                       command=self.clear, fg_color="#08103b")
        self.btn_clear.pack(pady=5, padx=10)

        self.status = ctk.CTkLabel(self.left, text="Waiting for image",
                                   font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                   wraplength=180)
        self.status.pack(pady=10, padx=10)

        self.image_label = ctk.CTkLabel(self.right, text="*Image preview*",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"))
        self.image_label.pack(pady=5)

        self.result_box = ctk.CTkTextbox(self.right, height=200, font=ctk.CTkFont(family="Arial", size=12, weight="bold"), corner_radius=20, fg_color="#21263d")
        self.result_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_path = None
        self.ocr = None
        self._init_ocr()

        self.bind('<Escape>', lambda e: self.destroy())

    def play_animation(self):
        """Play animation in a separate thread"""
        def animate():
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.status.configure(text="Animation load error")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1 / fps if fps > 0 else 0.033
            
            while self.is_animating:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to label size
                frame = cv2.resize(frame, (400, 300))
                
                # Convert to PhotoImage
                img = Image.fromarray(frame)
                photo = ctk.CTkImage(img, size=(400, 300))
                
                # Update label in main thread
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                self.image_label.update()
                
                # Wait before next frame
                self.after(int(frame_delay * 1000), lambda: None)
            
            cap.release()
        
        if not self.is_animating:
            self.is_animating = True
            self.animation_thread = threading.Thread(target=animate, daemon=True)
            self.animation_thread.start()

    def stop_animation(self):
        """Stop animation"""
        self.is_animating = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1)

    def _init_ocr(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(root_dir, "best.pt")
        config_path = os.path.join(root_dir, "g-vision-config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                model_from_cfg = cfg.get("model")
                if model_from_cfg:
                    candidate = model_from_cfg if os.path.isabs(model_from_cfg) else os.path.join(root_dir, model_from_cfg)
                    if os.path.exists(candidate):
                        model_path = candidate
            except Exception as ex:
                print(f"Failed to load OCR config: {ex}")

        if not os.path.exists(model_path):
            fallback_paths = [
                os.path.join(root_dir, "best_model.pt"),
                os.path.join(root_dir, "G-Vision.pt"),
                os.path.join(root_dir, "G-VISION.pt"),
                os.path.join(root_dir, "best.pt"),
                os.path.join(root_dir, "runs", "ocr_final", "best_model.pt")
            ]
            for p in fallback_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        if os.path.exists(model_path):
            from train import GVisionOCR
            try:
                self.ocr = GVisionOCR(model_path, config_path if os.path.exists(config_path) else None)
                self.status.configure(text="OCR model loaded")
            except Exception as exc:
                print(f"Failed to initialize OCR: {exc}")
                self.ocr = None
        else:
            self.status.configure(text="OCR model not found")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.image_path = path
        img = ctk.CTkImage(Image.open(path), size=(400, 300))
        self.image_label.configure(image=img, text="")
        self.image_label.image = img
        self.status.configure(text="Image loaded")

    def recognize(self):
        if not self.image_path:
            self.status.configure(text="Load image first")
            return
        
        if self.ocr is None:
            self.status.configure(text="OCR model not loaded")
            return

        self.status.configure(text="Recognizing...")
        self.play_animation()
        
        def process():
            try:
                result = self.ocr.recognize(self.image_path)
                text = result.get("text", "")
                if not text:
                    text = "No text found. Check the image quality and model."
                self.result_box.delete("1.0", "end")
                self.result_box.insert("1.0", text)
            except Exception as exc:
                self.result_box.delete("1.0", "end")
                self.result_box.insert("1.0", f"Recognition error: {exc}")
            finally:
                self.stop_animation()
                self.status.configure(text="Ready")

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def clear(self):
        self.stop_animation()
        self.image_label.configure(image=None, text="*Image preview*",
                                   font=ctk.CTkFont(family="Arial", size=16, weight="bold"))
        self.result_box.delete("1.0", "end")
        self.image_path = None
        self.status.configure(text="Cleared")


if __name__ == "__main__":
    app = App()
    app.mainloop()
