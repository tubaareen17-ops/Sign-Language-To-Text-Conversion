# --- 1. IMPORTS ---
import numpy as np
import cv2
import os
import operator
from string import ascii_uppercase
import time
from collections import deque 
import threading                # New: For background processing
from queue import Queue, Empty  # New: To pass data between threads

# --- Modern UI (ttkbootstrap) ---
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
except ImportError:
    print("ERROR: ttkbootstrap not found. Please run 'pip install ttkbootstrap'")
    import tkinter as tk
    from tkinter import ttk as ttk_fallback
    class DummyWindow(tk.Tk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    ttk = ttk_fallback
    ttk.Window = DummyWindow
    PRIMARY, SECONDARY, DANGER, INVERSE, LIGHT, DARK = "primary", "secondary", "danger", "inverse", "light", "dark"

from PIL import Image, ImageTk

# --- Spell Checker ---
try:
    from spellchecker import SpellChecker
except ImportError:
    print("ERROR: pyspellchecker not found. Please run 'pip install pyspellchecker'")
    SpellChecker = None

# --- TensorFlow / Keras ---
try:
    from tensorflow.keras.models import model_from_json
except ImportError:
    print("ERROR: TensorFlow/Keras is not installed or configured correctly.")
    model_from_json = None

# ===================================================================
# --- 2. THE MAIN APPLICATION CLASS ---
# ===================================================================

class Application:

    def __init__(self):
        # --- Model and Dictionary Initialization ---
        self.spell_checker = SpellChecker() if SpellChecker else None
        if not self.spell_checker:
            print("Warning: SpellChecker not initialized.")

        # --- Video Capture Setup ---
        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            print("CRITICAL ERROR: Cannot open webcam (index 0).")
            self.models_loaded_ok = False
            return

        # --- Model Loading ---
        self.models_loaded_ok = self._load_all_models()
        if not self.models_loaded_ok:
            print("CRITICAL ERROR: Failed to load models. Exiting.")
            self.vs.release()
            return
            
        # --- Stabilization & Logic Variables ---
        
        # ========================= FIX 1 =========================
        # Increased maxlen from 40 to 60 (2 seconds @ 30fps).
        # This requires the sign to be held stable for much longer.
        self.prediction_history = deque(maxlen=60) 
        # =========================================================

        self.raw_symbol = "---"     # NEW: The raw, flickering symbol from the thread
        self.stable_symbol = "---"  # The stabilized symbol
        self.last_added_symbol = "" 

        # --- State Variables ---
        self.str = ""  # The sentence
        self.word = "" # The current word being formed

        # --- NEW: Threading & Queue Setup ---
        self.stop_event = threading.Event() # To signal thread to stop
        self.frame_queue = Queue(maxsize=1) 
        self.result_queue = Queue(maxsize=1)
        
        self.prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        self.prediction_thread.start()

        # --- Modern GUI Setup (ttkbootstrap) ---
        self.root = ttk.Window(themename="superhero") 
        self.root.title("Real-Time Sign Language Translator")
        self.root.geometry("1100x900")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        # --- Responsive Grid Layout ---
        self.root.grid_columnconfigure(0, weight=1) 
        self.root.grid_columnconfigure(1, weight=1) 
        self.root.grid_rowconfigure(0, weight=0) 
        self.root.grid_rowconfigure(1, weight=1) 
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)

        # --- 1. Title ---
        self.T = ttk.Label(self.root, text="Sign Language To Text Conversion", 
                           font=("Courier", 30, "bold"), bootstyle=LIGHT)
        self.T.grid(row=0, column=0, columnspan=2, pady=20)

        # --- 2. Left Frame (Video) ---
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.panel = ttk.Label(self.video_frame, bootstyle=SECONDARY)
        self.panel.pack(fill=BOTH, expand=True)

        # --- 3. Right Frame (ROI & Output) ---
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=0) # ROI Title
        self.right_frame.grid_rowconfigure(1, weight=1) # ROI Image
        self.right_frame.grid_rowconfigure(2, weight=0) # NEW: Raw Feed Title
        self.right_frame.grid_rowconfigure(3, weight=0) # NEW: Raw Feed Label
        self.right_frame.grid_rowconfigure(4, weight=0) # Stable Title
        self.right_frame.grid_rowconfigure(5, weight=0) # Stable Label

        self.panel2_label = ttk.Label(self.right_frame, text="Processed ROI", 
                                      font=("Courier", 16, "bold"), bootstyle=LIGHT)
        self.panel2_label.grid(row=0, column=0, pady=(0, 10))
        
        self.panel2 = ttk.Label(self.right_frame, bootstyle=INVERSE)
        self.panel2.grid(row=1, column=0, sticky="nsew", padx=20)
        
        # --- NEW: Clearer UI for Raw vs Stable ---
        self.raw_feed_title = ttk.Label(self.right_frame, text="Raw Prediction", 
                                        font=("Courier", 16, "bold"), bootstyle=LIGHT)
        self.raw_feed_title.grid(row=2, column=0, pady=(15, 0))
        
        self.raw_feed_label = ttk.Label(self.right_frame, text="---", 
                                        font=("Courier", 20, "bold"), bootstyle=PRIMARY)
        self.raw_feed_label.grid(row=3, column=0, pady=(0, 10))

        self.T1 = ttk.Label(self.right_frame, text="STABLE CHARACTER", 
                            font=("Courier", 24, "bold"), bootstyle=LIGHT)
        self.T1.grid(row=4, column=0, pady=(10, 0))
        
        self.panel3 = ttk.Label(self.right_frame, text="---", 
                                font=("Courier", 48, "bold"), bootstyle=DANGER)
        self.panel3.grid(row=5, column=0)
        # --- End of UI Change ---

        # --- 4. Bottom Frame (Text Output) ---
        self.output_frame = ttk.Frame(self.root)
        self.output_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(1, weight=5)

        self.T2 = ttk.Label(self.output_frame, text="WORD:", 
                            font=("Courier", 24, "bold"), bootstyle=LIGHT)
        self.T2.grid(row=0, column=0, sticky="e", padx=10)
        
        self.panel4 = ttk.Label(self.output_frame, text="", 
                                font=("Courier", 24), bootstyle=PRIMARY, anchor="w")
        self.panel4.grid(row=0, column=1, sticky="ew")

        self.T3 = ttk.Label(self.output_frame, text="SENTENCE:", 
                            font=("Courier", 24, "bold"), bootstyle=LIGHT)
        self.T3.grid(row=1, column=0, sticky="ne", padx=10, pady=10)
        
        self.panel5 = ttk.Label(self.output_frame, text="", wraplength=800, 
                                font=("Courier", 24), bootstyle=SECONDARY, anchor="nw")
        self.panel5.grid(row=1, column=1, sticky="ew", pady=10)

        # --- 5. Suggestion Frame ---
        self.suggestion_frame = ttk.Frame(self.root)
        self.suggestion_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.T4 = ttk.Label(self.suggestion_frame, text="SUGGESTIONS", 
                            font=("Courier", 20, "bold"), bootstyle=LIGHT)
        self.T4.pack(pady=(0, 10))

        self.button_frame = ttk.Frame(self.suggestion_frame)
        self.button_frame.pack()

        button_style = {"width": 20, "bootstyle": (PRIMARY, OUTLINE)}
        self.bt1 = ttk.Button(self.button_frame, text="", command=self.action1, **button_style)
        self.bt1.grid(row=0, column=0, padx=10)
        self.bt2 = ttk.Button(self.button_frame, text="", command=self.action2, **button_style)
        self.bt2.grid(row=0, column=1, padx=10)
        self.bt3 = ttk.Button(self.button_frame, text="", command=self.action3, **button_style)
        self.bt3.grid(row=0, column=2, padx=10)

        # Start the video processing loop
        print("Initialization complete. Starting video loop...")
        self.video_loop()

    def _load_all_models(self):
        """Loads all models. Returns True on success, False on failure."""
        if model_from_json is None:
            print("CRITICAL: model_from_json is None. TensorFlow not imported correctly.")
            return False
        try:
            # Use absolute path for robustness
            script_dir = os.path.abspath(os.path.dirname(__file__))
            self.model_dir = os.path.join(script_dir, "Models")
            print(f"Loading models from: {self.model_dir}")

            self.loaded_model = self._load_model_core(
                self.model_dir, "model_new.json", "model_new.h5", "Main")
            if not self.loaded_model:
                return False # Critical failure

            # Load secondary models (non-critical, can be None)
            self.loaded_model_dru = self._load_model_core(
                self.model_dir, "model-bw_dru.json", "model-bw_dru.weights.h5", "DRU")
            self.loaded_model_tkdi = self._load_model_core(
                self.model_dir, "model-bw_tkdi.json", "model-bw_tkdi.weights.h5", "TKDI")
            self.loaded_model_smn = self._load_model_core(
                self.model_dir, "model-bw_smn.json", "model-bw_smn.weights.h5", "SMN")

            print("All models loaded successfully.")
            return True
        except Exception as e:
            print(f"--- CRITICAL ERROR during model loading ---")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model_core(self, model_dir, json_filename, weights_filename, model_name):
        """Helper to load a single model."""
        try:
            json_path = os.path.join(model_dir, json_filename)
            weights_path = os.path.join(model_dir, weights_filename)

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"{model_name} model JSON not found at {json_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"{model_name} model weights not found at {weights_path}")

            with open(json_path, "r") as json_file:
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)
            if loaded_model is None:
                raise ValueError(f"model_from_json returned None for {model_name}.")

            loaded_model.load_weights(weights_path)
            print(f"Loaded {model_name} model successfully.")
            return loaded_model
        except Exception as e:
            print(f"Warning: Could not load {model_name} model. Set to None. Error: {e}")
            return None

    # --- NEW: BACKGROUND PREDICTION THREAD ---
    def _prediction_worker(self):
        """
        Runs on a separate thread.
        Continuously waits for frames, predicts them, and puts results in the queue.
        """
        print("Prediction thread started...")
        while not self.stop_event.is_set():
            try:
                # Wait for a frame to arrive from the main thread
                frame_to_process = self.frame_queue.get(timeout=1.0) 
                
                # A "None" frame is our signal to shut down
                if frame_to_process is None:
                    break 

                # --- Run the SLOW prediction ---
                symbol = self.predict(frame_to_process)
                # --- Prediction is done ---

                # Put the result in the queue for the main thread to pick up
                try:
                    self.result_queue.put_nowait(symbol)
                except:
                    pass # Result queue is full, drop this result

                self.frame_queue.task_done()
                
            except:
                # Timeout, just loop again
                continue
        print("Prediction thread stopped.")

    def video_loop(self):
        """
        Main UI thread loop.
        Does FAST operations only: Grab frame, update UI, send frame to background thread.
        """
        if not self.root or not hasattr(self.root, 'winfo_exists') or not self.root.winfo_exists():
            print("GUI window closed. Stopping video loop.")
            self.destructor()
            return

        ok, frame = self.vs.read()
        if not ok:
            print("Failed to grab frame from camera.")
            self.root.after(100, self.video_loop) # Try again
            return

        try:
            # --- 1. Fast CV & UI updates ---
            cv2image = cv2.flip(frame, 1)
            height, width, _ = cv2image.shape
            x1, y1 = int(width * 0.05), int(height * 0.15)
            x2, y2 = int(width * 0.60), int(height * 0.85)

            cv2.rectangle(cv2image, (x1, y1), (x2, y2), (39, 174, 96), 3)
            
            # --- Robustness Check ---
            panel_w, panel_h = self.panel.winfo_width(), self.panel.winfo_height()
            if panel_w > 1 and panel_h > 1:
                cv2image_tk = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
                self.current_image = Image.fromarray(cv2image_tk)
                imgtk = ImageTk.PhotoImage(image=self.current_image.resize((panel_w, panel_h)))
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

            # --- 2. Fast Image Processing ---
            roi = cv2image[y1:y2, x1:x2]
            if roi.size == 0:
                self.root.after(30, self.video_loop) # Schedule next loop
                return
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # --- 3. Send to Background Thread (Fast) ---
            try:
                self.frame_queue.put_nowait(res)
            except:
                pass # Queue is full, frame is dropped (which is fine)

            # --- 4. Check for Results from Thread (Fast) ---
            try:
                # Check if the background thread has a new result
                self.raw_symbol = self.result_queue.get_nowait()
                
                # We have a new symbol, update the logic
                self.update_logic(self.raw_symbol) 
                
                # Update the new "Raw" UI label
                self.raw_feed_label.config(text=self.raw_symbol)
                
                self.result_queue.task_done()
            except Empty:
                pass # No new result, just keep looping

            # --- 5. Update UI with current state (Fast) ---
            panel2_w, panel2_h = self.panel2.winfo_width(), self.panel2.winfo_height()
            if panel2_w > 1 and panel2_h > 1:
                self.current_image2 = Image.fromarray(res)
                imgtk2 = ImageTk.PhotoImage(image=self.current_image2.resize((panel2_w, panel2_h)))
                self.panel2.imgtk = imgtk2
                self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.stable_symbol)
            self.panel4.config(text=self.word)
            self.panel5.config(text=self.str)

        except Exception as e:
            print(f"Error in video_loop: {e}")
            import traceback
            traceback.print_exc()

        # Schedule the next call to video_loop
        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        """
        Handles prediction *only*. Returns the predicted symbol as a string.
        This is the SLOW function that runs on the background thread.
        """
        if not self.models_loaded_ok:
            return "MODEL ERROR"

        try:
            # 1. Resize, Normalize, Reshape
            test_image = cv2.resize(test_image, (128, 128))
            normalized_image = test_image.astype('float32') / 255.0
            input_data = normalized_image.reshape(1, 128, 128, 1)

            # Layer 1 Prediction
            result = self.loaded_model.predict(input_data, verbose=0)
            
            class_indices = {i: chr(64 + i) for i in range(1, 27)} # A=1, ... Z=26
            class_indices[0] = 'blank'
            
            predicted_index = np.argmax(result[0])
            symbol = class_indices.get(predicted_index, 'blank')

            # --- Layer 2 Cascaded Predictions ---
            if symbol in ['D', 'R', 'U'] and self.loaded_model_dru:
                result_dru = self.loaded_model_dru.predict(input_data, verbose=0)
                symbol = ['D', 'R', 'U'][np.argmax(result_dru[0])]

            if symbol in ['D', 'I', 'K', 'T'] and self.loaded_model_tkdi:
                result_tkdi = self.loaded_model_tkdi.predict(input_data, verbose=0)
                symbol = ['D', 'I', 'K', 'T'][np.argmax(result_tkdi[0])]

            if symbol in ['M', 'N', 'S'] and self.loaded_model_smn:
                result_smn = self.loaded_model_smn.predict(input_data, verbose=0)
                symbol = ['M', 'N', 'S'][np.argmax(result_smn[0])]

            return symbol

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "PRED_ERR"
            
    def update_logic(self, symbol):
        """
        Handles the stabilization logic using the prediction history.
        This is FAST and runs on the main UI thread.
        """
        
        # ========================= FIX 2 =========================
        # Increased threshold from 0.95 to 1.0.
        # This requires 60/60 frames to match (the strictest possible).
        STABILITY_THRESHOLD = 1.0
        # =========================================================
        
        self.prediction_history.append(symbol)
        
        try:
            most_common_symbol = max(set(self.prediction_history), 
                                     key=list(self.prediction_history).count)
        except ValueError:
            most_common_symbol = "---" 
            
        count = list(self.prediction_history).count(most_common_symbol)
        
        # Check if the threshold is met
        if (count / self.prediction_history.maxlen) >= STABILITY_THRESHOLD:
            # We have a STABLE symbol
            if most_common_symbol != self.stable_symbol:
                self.stable_symbol = most_common_symbol
                
                if self.stable_symbol == "blank":
                    if self.word: # Only finalize if word exists
                        self.finalize_word()
                
                elif self.stable_symbol != "---":
                    # Only add if it's a new letter
                    if self.stable_symbol != self.last_added_symbol:
                        self.word += self.stable_symbol
                        self.last_added_symbol = self.stable_symbol
                        self.clear_suggestions()
        else:
            # Not stable, reset to "---"
            self.stable_symbol = "---"

    def finalize_word(self):
        """Called when a 'blank' is detected."""
        print(f"Finalizing word: {self.word}")
        
        if not self.str or self.str.strip().endswith(('.', '!', '?')):
            formatted_word = self.word.title()
        else:
            formatted_word = self.word.lower()
            
        self.str += formatted_word + " "
        self.update_suggestions(self.word)
        self.word = ""
        self.last_added_symbol = ""
        
    def update_suggestions(self, word):
        """Gets suggestions and updates the buttons."""
        if not self.spell_checker or not word:
            self.clear_suggestions()
            return
            
        word_lower = word.lower()
        candidates = self.spell_checker.candidates(word_lower)
        
        if not candidates or (word_lower in self.spell_checker and len(candidates) <= 1):
             self.clear_suggestions()
             return

        suggestions = sorted(list(candidates), key=lambda w: self.spell_checker.correction(w) == w, reverse=True)
        suggestions = [s for s in suggestions if s != word_lower]
        suggestions.extend(["", "", ""])
        
        self.bt1.config(text = suggestions[0].upper())
        self.bt2.config(text = suggestions[1].upper())
        self.bt3.config(text = suggestions[2].upper())

    def clear_suggestions(self):
        self.bt1.config(text="")
        self.bt2.config(text="")
        self.bt3.config(text="")

    def action1(self): self._apply_suggestion(self.bt1.cget("text"))
    def action2(self): self._apply_suggestion(self.bt2.cget("text"))
    def action3(self): self._apply_suggestion(self.bt3.cget("text"))

    def _apply_suggestion(self, suggestion_text):
        """Replaces the *last* word in the sentence with the suggestion."""
        if not suggestion_text or not self.str:
            return
            
        suggestion = suggestion_text.lower()
        words = self.str.strip().split(' ')
        if not words:
            return
            
        last_word = words[-1]
        if last_word.istitle():
            suggestion = suggestion.title()
            
        words[-1] = suggestion # Replace
        self.str = ' '.join(words) + ' '
        
        self.word = ""
        self.last_added_symbol = ""
        self.clear_suggestions()

    def destructor(self):
        """Cleans up resources upon closing."""
        print("Closing Application... Signaling thread to stop.")
        
        # --- NEW: Clean Thread Shutdown ---
        self.stop_event.set()
        try:
            # Send a "None" to unblock the queue.get()
            self.frame_queue.put_nowait(None)
        except:
            pass
        
        # Wait for the thread to finish
        if hasattr(self, 'prediction_thread'):
            self.prediction_thread.join(timeout=1.0) 
        
        # --- Original Cleanup ---
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
        if hasattr(self, 'vs') and self.vs and self.vs.isOpened():
            self.vs.release()
        cv2.destroyAllWindows()
        print("Application closed.")

# ===================================================================
# --- 3. RUN THE APPLICATION ---
# ===================================================================

if __name__ == '__main__':
    print("Starting Application...")
    try:
        app = Application()
        if hasattr(app, 'root') and app.root and app.models_loaded_ok:
            app.root.mainloop()
        else:
            print("\nApplication GUI failed to initialize. Please check console errors.")
            if 'app' in locals() and hasattr(app, 'vs') and app.vs and app.vs.isOpened():
                app.vs.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"CRITICAL ERROR during Application startup: {e}")
        import traceback
        traceback.print_exc()
        if 'app' in locals() and hasattr(app, 'vs') and app.vs and app.vs.isOpened():
            app.vs.release()
        cv2.destroyAllWindows()
