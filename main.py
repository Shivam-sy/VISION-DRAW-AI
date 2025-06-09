import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import webbrowser
import time
import subprocess
import pywhatkit
import pyautogui
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio

# Cooldown & flags
last_triggered = {
    "whatsapp": 0,
    "notion": 0,
    "gmail": 0,
    "mute": 0,
    "3d_toggle": 0,
    "dashboard_toggle": 0
}
cooldown_seconds = 5

enable3DMode = False
enableDashboardMode = False

# Setup Open3D window
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh.compute_vertex_normals()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Control Window", width=640, height=480)
vis.add_geometry(mesh)
view_control = vis.get_view_control()

def show_sample_graph():
    fig = go.Figure(data=go.Scatter(y=[10, 12, 9, 13, 15]))
    fig.update_layout(title=" Sample Data", xaxis_title="Time", yaxis_title="Value")
    pio.write_image(fig, "temp_graph.png")
    img = cv2.imread("temp_graph.png")
    cv2.namedWindow("Dashboard Chart", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dashboard Chart", 800, 600)
    cv2.imshow("Dashboard Chart", img)
    cv2.waitKey(10000)
    cv2.destroyWindow("Dashboard Chart")

def handle3DGestures(fingers, hand_type):
    if hand_type == "Right":
        if fingers == [0, 1, 0, 0, 0]:
            view_control.translate(0.05, 0)
        elif fingers == [0, 0, 1, 0, 0]:
            view_control.translate(-0.05, 0)
        elif fingers == [1, 0, 0, 0, 0]:
            view_control.scale(1.1)
        elif fingers == [0, 0, 0, 0, 1]:
            view_control.scale(0.9)
        elif fingers == [0, 1, 1, 0, 0]:
            view_control.rotate(5.0, 0.0)
    vis.poll_events()
    vis.update_renderer()

def handleDashboardGestures(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        pyautogui.scroll(-300)
    elif fingers == [0, 0, 0, 0, 1]:
        pyautogui.scroll(300)
    elif fingers == [0, 1, 1, 0, 0]:
        pyautogui.hotkey('ctrl', 'tab')

genai.configure(api_key="AIzaSyCNNHchzXeSH4kc383ixfdg4qyxVqcQjtI")
model = genai.GenerativeModel("gemini-1.5-flash")

cap = cv2.VideoCapture(0)
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.3)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    hand_data = []
    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            hand_type = hand["type"]
            hand_data.append((fingers, lmList, hand_type))
    return hand_data

def draw(fingers, lmList, prev_pos, canvas):
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, color=(235, 206, 135), thickness=5)
    elif fingers == [0, 1, 1, 0, 0]:
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [0, 1, 0, 0, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Find the side of the triangle", pil_image])
        print("\nThe answer of the problem:")
        print(response.text)
        return True
    return False

def send_whatsapp_message(phone_number, message):
    try:
        pywhatkit.sendwhatmsg_instantly(phone_no=phone_number, message=message, wait_time=0, tab_close=True)
        print(f" Message sent to {phone_number}")
    except Exception as e:
        print(f" Failed to send message: {e}")

def performActionOnGesture(fingers):
    current_time = time.time()
    if fingers == [0, 1, 1, 0, 0]:
        screenshot = pyautogui.screenshot()
        screenshot_path = "screenshot.png"
        screenshot.save(screenshot_path)
        print("Screenshot taken!")
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        cv2.imshow("Screenshot", screenshot_cv)
        cv2.waitKey(0)
        cv2.destroyWindow("Screenshot")
        return True
    elif fingers == [1, 0, 0, 0, 0]:
        pyautogui.press("volumeup")
        print(" Volume up")
        return True
    elif fingers == [0, 1, 1, 1, 1]:
        pyautogui.press("volumedown")
        print(" Volume down")
        return True
    elif fingers == [1, 1, 1, 1, 1] and current_time - last_triggered["mute"] > cooldown_seconds:
        pyautogui.press("volumemute")
        print(" Mute")
        last_triggered["mute"] = current_time
        return True
    elif fingers == [0, 1, 1, 1, 1] and current_time - last_triggered["whatsapp"] > cooldown_seconds:
        send_whatsapp_message("+919719834746", "Hello! This is a gesture-triggered message.")
        last_triggered["whatsapp"] = current_time
        return True
    elif fingers == [0, 1, 1, 0, 1] and current_time - last_triggered["whatsapp"] > cooldown_seconds:
        subprocess.Popen([r"C:\\Users\\shiva\\OneDrive\\Desktop\\WhatsApp.lnk"], shell=True)
        print(" Opening WhatsApp Desktop...")
        last_triggered["whatsapp"] = current_time
        return True
    elif fingers == [0, 1, 1, 0, 0] and current_time - last_triggered["notion"] > cooldown_seconds:
        subprocess.Popen([r"C:\\Users\\shiva\\AppData\\Local\\Programs\\Notion\\Notion.exe"])
        print(" Opening Notion Desktop...")
        last_triggered["notion"] = current_time
        return True
    elif fingers == [0, 1, 1, 1, 0] and current_time - last_triggered["gmail"] > cooldown_seconds:
        webbrowser.open("https://mail.google.com/mail/u/3/#inbox")
        print(" Opening Gmail...")
        last_triggered["gmail"] = current_time
        return True
    return False

prev_pos = None
canvas = None

while True:
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)

    if canvas is None:
        canvas = np.zeros_like(img)

    hand_data = getHandInfo(img)
    gesture_handled = False

    if hand_data:
        current_time = time.time()

        for fingers, lmList, hand_type in hand_data:
            if gesture_handled:
                break

            prev_pos, canvas = draw(fingers, lmList, prev_pos, canvas)

            if len(hand_data) == 2:
                fingers1, _, _ = hand_data[0]
                fingers2, _, _ = hand_data[1]

                if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1] and current_time - last_triggered["3d_toggle"] > cooldown_seconds:
                    enable3DMode = not enable3DMode
                    enableDashboardMode = False
                    last_triggered["3d_toggle"] = current_time
                    print(" 3D Interaction Mode Toggled:", enable3DMode)
                    gesture_handled = True
                    break

                elif fingers1 == [0, 1, 1, 1, 0] and fingers2 == [0, 1, 1, 1, 0] and current_time - last_triggered["dashboard_toggle"] > cooldown_seconds:
                    enableDashboardMode = not enableDashboardMode
                    enable3DMode = False
                    last_triggered["dashboard_toggle"] = current_time
                    print(" Dashboard Mode Toggled:", enableDashboardMode)
                    show_sample_graph()
                    gesture_handled = True
                    break

            if enable3DMode:
                handle3DGestures(fingers, hand_type)
                gesture_handled = True
                break

            elif enableDashboardMode:
                handleDashboardGestures(fingers)
                gesture_handled = True
                break

            else:
                if performActionOnGesture(fingers):
                    gesture_handled = True
                    break

                if sendToAI(model, canvas, fingers):
                    cap.release()
                    cv2.destroyAllWindows()
                    vis.destroy_window()
                    exit()

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.25, 0)

    if not success:
        print("Failed to capture image")
        break

    cv2.imshow("Image Combined", image_combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()