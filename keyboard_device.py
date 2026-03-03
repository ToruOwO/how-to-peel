from pynput import keyboard


# Keep track of currently pressed keys
class KeyboardUI:
    def __init__(self):
        self.pressed_keys = set()
        self.listener = None

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def join(self):
        self.listener.join()

    def on_press(self, key):
        if not hasattr(key, "char"):
            return

        try:
            # Add the pressed key to the set
            if key.char and key.char.isalpha():
                self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        if not hasattr(key, "char"):
            return

        try:
            # Remove the released key from the set
            if key.char and key.char.isalpha():
                self.pressed_keys.remove(key.char)
        except AttributeError:
            pass
        # Stop listener if 'esc' is released
        if key == keyboard.Key.esc:
            return False


if __name__ == "__main__":
    keyboard_ui = KeyboardUI()
    keyboard_ui.start()
    while True:
        print(keyboard_ui.pressed_keys)
    keyboard_ui.join()
