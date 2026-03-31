# Meeko 🎵
*Your empathetic mood-detecting music companion*

Meeko goes beyond facial expressions to understand your true emotional state by analyzing:
- **Voice tone & energy** (highest priority)
- **Heart rate data** (when available)
- **Text sentiment** (medium priority)  
- **Facial expressions** (lowest priority - people can fake smiles!)

## Quick Start
```bash
pip install -r requirements.txt
python meeko.py
```

## How Meeko Works
1. **Multi-signal capture**: Face, voice, text, and optional physiological data
2. **Intelligent fusion**: Weighs signals by reliability (voice > heart rate > text > face)
3. **Conflict resolution**: Asks for clarification when signals don't align
4. **Empathetic recommendations**: Matches music to your true emotional needs

Meeko knows that a forced smile doesn't mean you're happy - let your voice and heart tell the real story! 💙