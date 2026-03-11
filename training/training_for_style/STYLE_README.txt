STYLE PIPELINE QUICK START

This workspace now has two separate model flows:

1) School material Q&A flow (existing)
- Train: train_me.py
- Chat: chat.py
- Data folder: trained material
- Output model: my_custom_model

2) Personal writing style rewrite flow (new)
- Train: train_style.py
- Chat: chat_style.py
- Data folder: style material
- Output model: my_style_model

How to use style flow:
1. Put your own writing files into "style material" (.txt, .docx, .pdf).
2. Run: D:/AI-stuff/training/unsloth/school-version/unsloth_env/Scripts/python.exe train_style.py
3. Run: D:/AI-stuff/training/unsloth/school-version/unsloth_env/Scripts/python.exe chat_style.py
4. Paste a short voice sample once, then paste AI-generated text to rewrite.

Tip:
- More personal writing samples (5+ pages) gives better voice matching.
- Keep writing samples in the exact tone you want copied.
