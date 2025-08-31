import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading

class ChatUI:
    def __init__(self, client):
        self.client = client
        self.root = tk.Tk()
        self.root.title("Clean Chat")

        self.username_set = False

        tk.Label(self.root, text="user name:").pack()
        self.username_entry = tk.Entry(self.root)
        self.username_entry.pack()
        self.username_entry.focus_set()

        # One-time "Set username" button
        self.set_username_button = tk.Button(self.root, text="Set username", command=self.set_username)
        self.set_username_button.pack(pady=(2, 6))
        self.username_entry.bind("<Return>", lambda e: self.set_username())

        self.chat_display = ScrolledText(self.root, state='disabled')
        self.chat_display.pack(fill='both', expand=True)

        # Define color tags
        self.chat_display.tag_config("user", foreground="blue")
        self.chat_display.tag_config("system", foreground="black")
        self.chat_display.tag_config("warning", foreground="red")

        # Message entry starts disabled until username is set
        self.message_entry = tk.Entry(self.root, state='disabled')
        self.message_entry.pack(fill='x')
        self.message_entry.bind("<Return>", self.send_message)

        # Send button exists but is NOT packed until username is set
        self.send_button = tk.Button(self.root, text="שלח", command=self.send_message)

        self.root.after(100, self.check_messages)

    def start(self):
        self.root.mainloop()

    def check_messages(self):
        while not self.client.message_queue.empty():
            msg = self.client.message_queue.get()
            self.display_message(msg)
        self.root.after(100, self.check_messages)

    def set_username(self):
        username = self.username_entry.get().strip()
        if not username:
            self.display_message("[SYSTEM] Please enter a non-empty username.")
            return



        if not self.username_set:
            # Header "1" means 'set name'
            msg_type = 1
            # First time setup
            self.username_set = True
            self.message_entry.config(state='normal')
            self.send_button.pack(pady=6)
            self.message_entry.focus_set()
            # self.display_message(f"[SYSTEM] Username set as '{username}'. You can now chat.")

            # Change button text to "Change username"
            self.set_username_button.config(text="Change username", state="normal")
            self.username_entry.config(state='normal')  # keep editable
        else:
            # Header "2" means 'change name'
            msg_type = 2
        self.client.send(username,msg_type)

    def send_message(self, event=None):
        if not self.username_set:
            self.display_message("[SYSTEM] Set your username first. It will be sent as your first message.")
            return

        msg = self.message_entry.get().strip()
        if msg:
            self.client.send(msg,0)
            self.message_entry.delete(0, tk.END)

    def display_message(self, msg):
        self.chat_display.config(state='normal')

        if msg.startswith("[SYSTEM]"):
            tag = "system"
        elif msg.startswith("[WARNING]"):
            tag = "warning"
        else:
            tag = "user"

        self.chat_display.insert(tk.END, msg + "\n", tag)
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)
