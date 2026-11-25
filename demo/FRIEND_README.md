# Indus Epigraphy AI: demo package

This folder runs the project's live demo, which takes about 3 minutes.
Setup is needed once and takes 5 to 10 minutes, mostly downloading PyTorch (about 1 GB).
Commands that work on both systems are written once; where Mac and Windows differ, both are given.

---

## 1. Install Python (skip if already installed)

You need **Python 3.10, 3.11 or 3.12**. Check by opening a terminal:
- **Mac:** open the **Terminal** app.
- **Windows:** open **PowerShell** from the Start menu.

Then type:

| Mac | Windows |
|---|---|
| `python3 --version` | `py --version` |

If you see "command not found", or a version older than 3.10, install it:
- **Mac:** download "Python 3.12 macOS installer" from https://www.python.org/downloads/ and run it.
- **Windows:** download "Python 3.12 Windows installer (64-bit)" from the same page. On the first screen, **tick "Add python.exe to PATH"** before clicking Install.

Close and reopen the terminal after installing.

## 2. Unzip and go into the folder

Unzip `indus_demo.zip` onto the **Desktop**. You will get a folder called `indus_demo`.

| Mac | Windows |
|---|---|
| `cd ~/Desktop/indus_demo` | `cd $HOME\Desktop\indus_demo` |

If your Windows Desktop is inside OneDrive, use `cd $HOME\OneDrive\Desktop\indus_demo` instead.

## 3. One-time setup

This creates a private Python environment inside the folder and installs the libraries.

| Mac | Windows |
|---|---|
| `bash demo/setup_mac.sh` | `demo\setup_windows.bat` |

It should end with `setup OK - torch 2.x.x`.

If Windows says scripts are disabled, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 4. Turn on the environment

Do this **every time** you open a new terminal window. Your prompt will then start with `(.venv)`.

| Mac | Windows |
|---|---|
| `source .venv/bin/activate` | `.venv\Scripts\activate` |

## 5. Practice run (no pauses, nothing opens)

```
python demo/demo.py --no-pause
```

This takes about 30 seconds. Check that you see:
- `STEP 1` to `STEP 7`, then `done.`
- In STEP 5: **every sign read correctly**.
- In STEP 6: **flagged: noise-like**.

If you do, everything works.

## 6. Recording the video

**1. Arrange the screen.**
- Make the terminal text large: **Cmd +** on Mac, **Ctrl +** on Windows, pressed 3 or 4 times.
- Put the terminal on the **left half** of the screen and a browser on the **right half**.

**2. Start recording.**
- **Mac:** press **Cmd + Shift + 5**, choose "Record Entire Screen", click Record. Stop it with the ■ button in the top menu bar; the video is saved to the Desktop.
- **Windows 11:** press **Win + Shift + R** (Snipping Tool), select the area, click Start. Alternatively, use **Win + G** (Game Bar) and press the record button. Videos go to Videos\Captures.

**3. Run the demo.**
```
python demo/demo.py --open
```
- Each step prints its result in the terminal and opens a visual report as a **new browser tab**.
- Wait about 5 seconds, click back on the terminal, and press **Enter** for the next step.
- After STEP 7 it prints `done.` Then stop recording.

## 7. The voice-over

The script to read is in **`demo/VOICEOVER.md`**. It has one paragraph per step, each marked with the time it starts (for example "[0:30–0:55] Step 2"). It is about 370 words, roughly 2 minutes 40 seconds at a calm pace.

There are two ways to add it:
- **Read it yourself (best).** Record the screen first. Then play the video back and read each paragraph as its step appears, using QuickTime → File → New Audio Recording, or the Windows Sound Recorder. Combine the two in iMovie or Clipchamp (Windows).
- **Computer voice (Mac only).** This saves the narration as `voiceover.aiff`:
  ```
  say -v Samantha -r 150 -f demo/VOICEOVER.txt -o voiceover.aiff
  ```

`demo/README.md` (in the project repository) has the shot list and the on-screen captions.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No such file or directory: demo/demo.py` | You are not inside the `indus_demo` folder. Repeat step 2. |
| `ModuleNotFoundError` | The environment is not on. Repeat step 4. If it still fails, repeat step 3. |
| `pip` fails installing `torch` | Your Python is too new or too old. Install 3.12 (step 1), delete the `.venv` folder, and redo step 3. |
| Odd symbols like `[1m` in the Windows output | Use **Windows Terminal** or PowerShell, not the old "Command Prompt". |
| Browser tabs do not open | Run without `--open` and open `reports/demo/<step>/index.html` by double-clicking it. |
| Want different example inscriptions in steps 3, 4 and 7 | Add `--seed 7` (any number). Do a practice run first: the voice-over numbers are written for the default. |

Note that nothing is uploaded anywhere: the demo runs entirely on this laptop.
