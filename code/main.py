import sounddevice as sd
import json
import subprocess
import numpy as np
import time
import os
import random
from queue import Queue
from vosk import Model, KaldiRecognizer
from gpiozero import LED
import lgpio as GPIO


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "/home/abc/voice_assistant/models/vosk-model-small-hi-0.22"
PIPER_BIN = "/home/abc/piper/piper"
PIPER_MODEL = "/home/abc/piper/voices/hi_IN-priyamvada-medium.onnx"
WAV_OUT = "/home/abc/voice_assistant/out.wav"

MIC_RATE = 48000
VOSK_RATE = 16000
DEVICE = 0

WAKE_WORDS = ["सुनो", "सुन", "सुनो जी"]

SILENCE_TIMEOUT = 1.5
ACTIVE_TIMEOUT = 30

USER_DATA_FILE = "user_data.json"
LED_PIN = 17

team = "Idea Igniters"


# =========================================================
# RTC SETUP (DS1302)
# =========================================================

RTC_CLK = 11
RTC_DAT = 10
RTC_RST = 8

chip = GPIO.gpiochip_open(0)

GPIO.gpio_claim_output(chip, RTC_CLK)
GPIO.gpio_claim_output(chip, RTC_DAT)
GPIO.gpio_claim_output(chip, RTC_RST)


def bcd_to_dec(bcd):
    return (bcd >> 4) * 10 + (bcd & 0x0F)


def rtc_write_byte(data):
    GPIO.gpio_claim_output(chip, RTC_DAT)
    for i in range(8):
        GPIO.gpio_write(chip, RTC_DAT, (data >> i) & 1)
        GPIO.gpio_write(chip, RTC_CLK, 1)
        GPIO.gpio_write(chip, RTC_CLK, 0)


def rtc_read_byte():
    byte = 0
    GPIO.gpio_claim_input(chip, RTC_DAT)

    for i in range(8):
        bit = GPIO.gpio_read(chip, RTC_DAT)
        byte |= (bit << i)
        GPIO.gpio_write(chip, RTC_CLK, 1)
        GPIO.gpio_write(chip, RTC_CLK, 0)

    GPIO.gpio_claim_output(chip, RTC_DAT)
    return byte


def rtc_read_time():
    GPIO.gpio_write(chip, RTC_RST, 1)

    rtc_write_byte(0x81)
    sec = rtc_read_byte()

    rtc_write_byte(0x83)
    minute = rtc_read_byte()

    rtc_write_byte(0x85)
    hour = rtc_read_byte()

    rtc_write_byte(0x87)
    date = rtc_read_byte()

    rtc_write_byte(0x89)
    month = rtc_read_byte()

    rtc_write_byte(0x8D)
    year = rtc_read_byte()

    GPIO.gpio_write(chip, RTC_RST, 0)

    return (
        bcd_to_dec(hour),
        bcd_to_dec(minute),
        bcd_to_dec(sec),
        bcd_to_dec(date),
        bcd_to_dec(month),
        2000 + bcd_to_dec(year)
    )


# =========================================================
# GPIO SETUP
# =========================================================

led = LED(LED_PIN)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading Vosk model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, VOSK_RATE)

audio_queue = Queue()

awake = False
last_speech_time = None
last_interaction_time = None
command_text = ""
user_name = "Lakshman"
is_speaking = False

joke_mode = False
current_joke = None

# Calculation state
calc_mode = False
calc_step = 0
num1 = None
num2 = None


# =========================================================
# LOAD / SAVE NAME
# =========================================================

def load_user():
    global user_name
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
            if data.get("name"):
                user_name = data.get("name")


def save_user():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({"name": user_name}, f)


load_user()


# =========================================================
# TEXT TO SPEECH
# =========================================================

def speak(text):
    global is_speaking
    is_speaking = True

    print("Assistant:", text)

    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", WAV_OUT],
        input=text.encode()
    )

    subprocess.run(["aplay", WAV_OUT])

    while not audio_queue.empty():
        audio_queue.get()

    is_speaking = False


# =========================================================
# AUDIO RESAMPLING
# =========================================================

def downsample(audio):
    audio_np = np.frombuffer(audio, dtype=np.int16)
    resampled = np.interp(
        np.linspace(0, len(audio_np),
                    int(len(audio_np) * VOSK_RATE / MIC_RATE)),
        np.arange(len(audio_np)),
        audio_np
    )
    return resampled.astype(np.int16).tobytes()


# =========================================================
# INTENT DETECTION
# =========================================================

def get_intent(text):

    if "मेरा नाम क्या" in text:
        return "GET_NAME"

    if "मेरा नाम" in text and "है" in text:
        return "SET_NAME"

    if any(word in text for word in ["टीम", "टीम का नाम"]):
        return "TEAM_NAME"

    if any(word in text for word in ["गणना", "कैलकुलेट"]):
        return "CALCULATE"

    if any(word in text for word in ["समय", "टाइम", "वक्त"]):
        return "TIME"

    if any(word in text for word in ["तारीख", "दिनांक", "आज"]):
        return "DATE"

    if any(word in text for word in ["नमस्ते", "हेलो"]):
        return "GREETING"

    if any(word in text for word in ["कैसे", "ठीक"]):
        return "STATUS"

    if any(word in text for word in ["कौन", "तुम"]):
        return "ABOUT"

    if any(word in text for word in ["मजाक", "जोक"]):
        return "JOKE"

    if any(word in text for word in ["धन्यवाद", "शुक्रिया"]):
        return "THANKS"

    if any(word in text for word in ["मदद"]):
        return "HELP"

    if "लाइट" in text and ("चालू" in text or "ऑन" in text):
        return "LIGHT_ON"

    if "लाइट" in text and ("बंद" in text or "ऑफ" in text):
        return "LIGHT_OFF"

    if any(word in text for word in ["नेटवर्क", "वाईफाई"]):
        return "NETWORK_STATUS"

    if any(word in text for word in ["सिस्टम स्टेटस"]):
        return "SYSTEM_STATUS"

    if any(word in text for word in ["अलविदा", "रुको", "स्टॉप"]):
        return "EXIT"

    return "UNKNOWN"


# =========================================================
# HANDLE INTENT
# =========================================================

# Hindi numbers 1–50
hindi_numbers = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45,
    "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50
}

def handle_intent(intent, text):
    global user_name, joke_mode, current_joke
    global calc_mode, calc_step, num1, num2

    if intent == "TIME":
        try:
            h, m, s, d, mo, y = rtc_read_time()
            return f"अभी समय है {h} बजकर {m} मिनट {s} सेकंड"
        except:
            return "समय पढ़ने में समस्या आ रही है"

    if intent == "DATE":
        try:
            h, m, s, d, mo, y = rtc_read_time()
            return f"आज की तारीख है {d}-{mo}-{y}"
        except:
            return "तारीख पढ़ने में समस्या आ रही है"

    if intent == "TEAM_NAME":
        return f"हमारी टीम का नाम {team} है"

    if intent == "SET_NAME":
        words = text.split()
        if len(words) >= 4:
            user_name = words[2]
            save_user()
            return f"ठीक है, आपका नाम {user_name} सेट कर दिया गया है"
        return "मैं आपका नाम समझ नहीं पाया"

    if intent == "GET_NAME":
        return f"आपका नाम {user_name} है"

    if intent == "GREETING":
        return f"नमस्ते {user_name}"

    if intent == "STATUS":
        return "मैं ठीक हूँ धन्यवाद"

    if intent == "ABOUT":
        return "मैं आपका ऑफलाइन हिंदी वॉइस असिस्टेंट हूँ"

    if intent == "THANKS":
        return "आपका स्वागत है"

    if intent == "HELP":
        return "आप समय, तारीख, लाइट कंट्रोल, नाम सेट, टीम नाम, सिस्टम स्टेटस, मजाक या गणना कर सकते हैं"

    if intent == "JOKE":
        joke_mode = True
        jokes = [
            {"question": "कंप्यूटर डॉक्टर के पास क्यों गया",
             "answer": "क्योंकि उसे वायरस हो गया था"},
            {"question": "प्रोग्रामर को कॉफी क्यों पसंद है",
             "answer": "क्योंकि वह डिबग करता है"}
        ]
        current_joke = random.choice(jokes)
        return current_joke["question"]

    if intent == "CALCULATE":
        calc_mode = True
        calc_step = 1
        return "पहली संख्या बताइए"

    if calc_mode:
        numbers = []
        for word in text.split():
            if word.isdigit():
                numbers.append(int(word))
            elif word in hindi_numbers:
                numbers.append(hindi_numbers[word])

        # Step 1: First number
        if calc_step == 1 and numbers:
            num1 = numbers[0]
            calc_step = 2
            return "दूसरी संख्या बताइए"

        # Step 2: Second number
        if calc_step == 2 and numbers:
            num2 = numbers[0]
            calc_step = 3
            return "कौन सा ऑपरेशन करना है"

        # Step 3: Operation
        if calc_step == 3:

            if "जोड़" in text:
                result = num1 + num2

            elif "घट" in text:
                result = num1 - num2

            elif "गुणा" in text:
                result = num1 * num2

            elif "भाग" in text:
                if num2 == 0:
                    calc_mode = False
                    calc_step = 0
                    return "शून्य से भाग नहीं कर सकते"
                result = num1 / num2

            elif "मॉड" in text:
                result = num1 % num2

            else:
                return "ऑपरेशन समझ नहीं आया"

            calc_mode = False
            calc_step = 0
            return f"उत्तर है {result}"

    if intent == "LIGHT_ON":
        led.on()
        return "लाइट चालू कर दी गई है"

    if intent == "LIGHT_OFF":
        led.off()
        return "लाइट बंद कर दी गई है"

    if intent == "EXIT":
        return "ठीक है, मैं सो रहा हूँ"

    return "मुझे समझ नहीं आया"
# =========================================================
# AUDIO CALLBACK
# =========================================================

def callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


# =========================================================
# TIME BASED GREETING (PURE HINDI)
# =========================================================

def get_time_greeting():
    try:
        h, m, s, d, mo, y = rtc_read_time()
    except:
        return "नमस्ते"

    if 5 <= h < 12:
        return "सुप्रभात"
    elif 12 <= h < 17:
        return "शुभ दोपहर"
    else:
        return "शुभ संध्या"
# =========================================================
# MAIN LOOP
# =========================================================

print("Assistant Ready")

try:
    with sd.RawInputStream(
        samplerate=MIC_RATE,
        blocksize=19200,
        device=DEVICE,
        dtype="int16",
        channels=1,
        callback=callback
    ):

        while True:

            raw_audio = audio_queue.get()

            if is_speaking:
                continue

            data = downsample(raw_audio)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                if not text:
                    continue

                print("Heard:", text)
                current_time = time.time()

                if not awake:
                    if any(w in text for w in WAKE_WORDS):
                        awake = True
                        last_interaction_time = current_time
                        greeting = get_time_greeting()
        	        speak(f"{greeting} {user_name}, बोलिए")
                    continue

                command_text = text
                last_speech_time = current_time
                last_interaction_time = current_time

            if awake and last_speech_time:
                if time.time() - last_speech_time > SILENCE_TIMEOUT:

                    if joke_mode and command_text:
                        speak("अच्छा प्रयास, सही जवाब सुनिए")
                        speak(current_joke["answer"])
                        joke_mode = False
                        command_text = ""
                        last_speech_time = None
                        recognizer.Reset()
                        continue

                    intent = get_intent(command_text)
                    response = handle_intent(intent, command_text)
                    speak(response)

                    recognizer.Reset()
                    command_text = ""
                    last_speech_time = None
                    last_interaction_time = time.time()

                    if intent == "EXIT":
                        awake = False
                        last_interaction_time = None

            if awake and last_interaction_time:
                if time.time() - last_interaction_time > ACTIVE_TIMEOUT:
                    print("Sleeping...")
                    awake = False
                    last_interaction_time = None

except KeyboardInterrupt:
    print("Assistant stopped")

finally:
    led.off()
    GPIO.gpiochip_close(chip)














///////rtc_test


import RPi.GPIO as GPIO
import time

CLK = 11
DAT = 10
RST = 8

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(CLK, GPIO.OUT)
GPIO.setup(DAT, GPIO.OUT)
GPIO.setup(RST, GPIO.OUT)

def bcd_to_dec(bcd):
    return (bcd >> 4) * 10 + (bcd & 0x0F)

def read_byte():
    byte = 0
    GPIO.setup(DAT, GPIO.IN)
    for i in range(8):
        bit = GPIO.input(DAT)
        byte |= (bit << i)
        GPIO.output(CLK, 1)
        GPIO.output(CLK, 0)
    return byte

def write_byte(data):
    GPIO.setup(DAT, GPIO.OUT)
    for i in range(8):
        GPIO.output(DAT, (data >> i) & 1)
        GPIO.output(CLK, 1)
        GPIO.output(CLK, 0)

def rtc_read(address):
    GPIO.output(RST, 1)
    write_byte(address)
    data = read_byte()
    GPIO.output(RST, 0)
    return data

def read_time():
    sec = bcd_to_dec(rtc_read(0x81))
    minute = bcd_to_dec(rtc_read(0x83))
    hour = bcd_to_dec(rtc_read(0x85))
    date = bcd_to_dec(rtc_read(0x87))
    month = bcd_to_dec(rtc_read(0x89))
    year = bcd_to_dec(rtc_read(0x8D)) + 2000

    return hour, minute, sec, date, month, year

try:
    while True:
        h, m, s, d, mo, y = read_time()
        print(f"{d:02d}-{mo:02d}-{y}  {h:02d}:{m:02d}:{s:02d}")
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Stopped")



///////////////////rtc_set

import RPi.GPIO as GPIO
import time
from datetime import datetime

# BCM pin numbers
CLK = 11
DAT = 10
RST = 8

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(CLK, GPIO.OUT)
GPIO.setup(DAT, GPIO.OUT)
GPIO.setup(RST, GPIO.OUT)

def dec_to_bcd(val):
    return (val // 10 << 4) + (val % 10)

def write_byte(data):
    GPIO.setup(DAT, GPIO.OUT)
    for i in range(8):
        GPIO.output(DAT, (data >> i) & 1)
        GPIO.output(CLK, 1)
        GPIO.output(CLK, 0)

def rtc_write(address, data):
    GPIO.output(RST, 1)
    write_byte(address)
    write_byte(data)
    GPIO.output(RST, 0)

def set_time():
    now = datetime.now()

    rtc_write(0x8E, 0x00)  # Disable write protection

    rtc_write(0x80, dec_to_bcd(now.second))
    rtc_write(0x82, dec_to_bcd(now.minute))
    rtc_write(0x84, dec_to_bcd(now.hour))
    rtc_write(0x86, dec_to_bcd(now.day))
    rtc_write(0x88, dec_to_bcd(now.month))
    rtc_write(0x8C, dec_to_bcd(now.year % 100))

    rtc_write(0x8E, 0x80)  # Enable write protection

    print("RTC Time Set Successfully")

try:
    set_time()
finally:
    GPIO.cleanup()
