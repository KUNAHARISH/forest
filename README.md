# Raspberry Pi 4 Setup Guide for Smart Forest Monitor

To successfully deploy your Smart Forest Monitoring AI from Windows to your Raspberry Pi 4 (8GB), follow these exact steps:

### Step 1: Copy Files to the Pi
You need to transfer your entire `ai project` folder from your Windows Desktop to the Pi (using a USB Flash Drive, WinSCP, or Google Drive). 
Make sure you transfer:
- `main.py`
- `yolov8n.pt`
- `dataset/known_faces/` (along with all your 6 face photos inside it)
- `runs/detect/train/weights/best.pt` (your custom trained Lion model)

### Step 2: Wire the Hardware
Connect your **IR Sensor (PIR Sensor)** to the Raspberry Pi's GPIO pins:
- **VCC pin** ➡️ goes to **5V Power** (Pin 2 or 4)
- **GND pin** ➡️ goes to **Ground** (Pin 6)
- **OUT / DATA pin** ➡️ goes directly to **GPIO 17** (Pin 11)

*Plug your **USB Webcam** into any of the blue USB 3.0 ports on the Pi.*

### Step 3: Install Required Packages
Open the terminal on your Raspberry Pi and run these commands to download all the AI and Camera packages:

```bash
sudo apt update
sudo apt install python3-opencv python3-pip cmake -y

# Modern Raspberry Pi OS requires running python in an environment:
python3 -m venv forest_env
source forest_env/bin/activate

# Install the machine learning dependencies
pip install ultralytics face_recognition requests RPi.GPIO
```

### Step 4: Run the System!
In the terminal, make sure you are inside your uploaded folder and run the code:
```bash
python main.py
```

### Step 5: Make it Run Automatically on Boot (Headless Setup)
To make your Raspberry Pi act as a standalone drone-camera that completely auto-starts `main.py` whenever you plug it into the wall, use `crontab`:

1. Open your Raspberry Pi terminal.
2. Type `crontab -e` and hit Enter. (If it asks you to pick an editor, type `1` for `nano`).
3. Scroll to the very bottom of the file using your arrow keys, and paste this single line:
   ```bash
   @reboot cd "/home/pi/Desktop/ai project" && python "/home/pi/Desktop/ai project/main.py" > /home/pi/forest_camera.log 2>&1 &
   ```
   *(Note: This assumes you paste your `ai project` folder directly onto the Pi's Desktop)*
4. Press **CTRL+X**, then **Y**, then **Enter** to save.

**That's it!** You can completely unplug your mouse, keyboard, and screen. Whenever you provide power to the Raspberry Pi, it will boot up, run your AI code quietly in the background, and start pinging your Telegram instantly when people walk by!
