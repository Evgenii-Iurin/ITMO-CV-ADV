# Virtual Canvas

# ОБУЧЕНИЕ СОБСТВЕННОГО БЭЙЗЛАЙНА


# ПРИЛОЖЕНИЕ С ОПЕНСОРС-МОДЕЛЬЮ

demo:


![Keypoint Detection Demo](./assets/keypoint_detection_demo.gif)

# How to use

```
poetry install
poetry shell
```

### If you are working in WSL2 (Windows) / if you're not - skip and go to "Run application"


1. Check if camera is available, run the following command:
    ```
    ls /dev/video*
    ```
    It should return some number of directories, for example:

    ```
    /dev/video0  /dev/video1
    ```
2. If you don't get directories, don't worry, just read section `Attach USB device to WSL (only for WSL2 users)`
3. Once your devices is connected to the WSL, let's check if it works:
    ```
    ffplay -f v4l2 -input_format mjpeg -video_size 640x480 -i /dev/video0 # <-- Adjust directory
    ```
    It should open window from camera

4. If you see the video from camera - that's nice. Add this direcotry directly to the script `app.py`
    ```
    DEV = "/dev/video0" # overwrite with your device's directory
    ```

### Run application
    python src/hw_7/app.py


### How to draw?
1. **Launch the script** and make sure your webcam shows you.
2. **Hold up just your index finger.** The cursor appears at your fingertip - this is *draw mode*.
3. **Move your hand** to sketch lines on the virtual canvas.
4. **Show an open palm** (all fingers extended) to *pause* drawing and reset the last point.
5. **Clear the canvas** at any time with the **`c`** key.
6. Exit with **`q`** or **Esc**.


### Attach USB device to WSL (only for WSL2 users)

if you are working on the WSL2 (Windows), you need to set up the USD device first. In this section, we will describe the process of connecting USD Camera Device

> Note

1. Open Windows terminal as an administrator
2. First, check the WSL version : `uname -r` (inside the WSL terminal)
    It will show smth like that:
    ```
    5.15.167.4-microsoft-standard-WSL2
    ```
    According [to that article](https://learn.microsoft.com/en-us/windows/wsl/connect-usb), if your WSL version is older than `5.10.60.1`, but you it's HIGHLY recommended to install newest WSL version `6.6.*`. The solution was tested for `6.6.87.1-microsoft-standard-WSL2` version. Run following commands in the Windows terminal.
        ```
        wsl --update --pre-release
        wsl --shutdown
        # Re-open code editor (VSCode)
        ```
3. All instructions you can find [here](), we will just go through the commands.
    In the Windows terminal, type:
    ```
    winget install --interactive --exact dorssel.usbipd-win
    ```

4. List all of the USB devices connected to Windows by opening PowerShell in administrator mode and entering the following command

    ```
    usbipd list
    ```
    For example, in my case I have:

    ```
    Connected:
    BUSID  VID:PID    DEVICE                                                        STATE
    1-6    04f2:b7b6  Integrated Camera, Camera DFU Device                          Shared
    1-14   0bda:5852  Realtek Bluetooth Adapter                                     Not shared
    ```

    You can see the device wth BUDID 1-6. You should check your BUSID, we will use it in the next steps

5. Before attaching the USB device, the command usbipd bind must be used to share the device, allowing it to be attached to WSL. This requires administrator privileges. Select the bus ID of the device you would like to use in WSL and run the following command. After running the command, verify that the device is shared using the command usbipd list again.
    ```
    usbipd bind --busid 1-6
    ```

6. To attach the USB device, run the following command. (You no longer need to use an elevated administrator prompt.) Ensure that a WSL command prompt is open in order to keep the WSL 2 lightweight VM active. Note that as long as the USB device is attached to WSL, it cannot be used by Windows. Once attached to WSL, the USB device can be used by any distribution running as WSL 2. Verify that the device is attached using usbipd list. From the WSL prompt, run lsusb to verify that the USB device is listed and can be interacted with using Linux tools.
    ```
    usbipd attach --wsl --busid 1-6
    ```

7. Open Ubuntu (or your preferred WSL command line) and list the attached USB devices using the command:

    ```
    lsusb
    ```

    if lsusb is not installed, do it:

    ```
    sudo apt install usbutils
    ```

8. Chekc if it works:
    ```
    ls /dev/video*
    ```

    it should list device derictories

    ```
    (itmo-cv-adv-py3.11) ➜  ITMO-CV-ADV git:(dev) ✗ ls /dev/video*
    >> /dev/video0  /dev/video1
    ```

    if it doesn't work, please, make you update WSL Version

### Detach USB Device
Once you are done using the device in WSL, you can either physically disconnect the USB device or run this command from PowerShell:
    ```
    usbipd detach --busid <busid>
    ```
