UNSLOTH TRAINER GUI

Distribution notes:
- A single EXE can run by itself only if everything it needs is bundled.
- In practice, ML apps are usually distributed as a ZIP/installer because models, caches, and dependencies are large.
- This project supports both: one-file EXE and optional installer with license page.

Run GUI from source:
1) Activate environment:
   .\unsloth_env\Scripts\Activate.ps1
2) Start GUI:
   python .\training\gui_app.py

What the GUI does:
- Setup button:
  Creates required folders and installs Python dependencies.
- Mode selector:
  Train for Talking or Train for Rephrasing/Style.
- Theme selector:
  Toggle Dark and Light mode.
- License gate:
  On first launch, user must accept the license agreement to continue.
- Upload Files:
  Copies selected .txt/.docx/.pdf files into the right data folder.
- Train:
  Runs the mode-specific training script.
- Chat:
  Runs mode-specific chat script.
  Chat is disabled until a trained adapter exists.
- Build EXE:
  Uses PyInstaller to build a Windows executable.

Build EXE via batch script:
- Run: .\training\build_gui_exe.bat
- Result: .\training\dist\UnslothTrainerGUI.exe

Build installer with agreement page (Inno Setup):
- Install Inno Setup 6+ (ISCC in PATH)
- Run: .\training\build_installer.bat
- Result: UnslothTrainerStudioSetup.exe
- Installer shows the license agreement from LICENSE_MIT.txt

Folders used (all relative to project root):
- trained material
- style material
- my_custom_model
- my_style_model

Scripts used:
- training\training_for_chat\train_me.py
- training\training_for_chat\chat.py
- training\training_for_style\train_style.py
- training\training_for_style\chat_style.py
