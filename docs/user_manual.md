# User Manual & Practitioner Guide

**Project Name:** Project Medhā (मेधा) — Sanskrit & Prakrit Live Recitation Verifier  
**Version:** 1.0.0  
**Target Audience:** Students, Practitioners, Scholars, and Reciters of Sacred Texts  

---

## 1. Introduction to Project Medhā

Welcome to **Project Medhā (मेधा)**, your personal AI-powered preceptor for mastering traditional oral recitation of Sanskrit and Prakrit scriptures (such as the *Ṇamōkāra Mahāmantra*, *Uvasaggaharaṁ Stōtra*, *Bhaktāmara Stōtra*, and Vedic hymns).

Medhā listens to your live voice through your computer or phone microphone, verifies your pronunciation syllable-by-syllable (*akshara*), tracks your rhythm, and highlights any hesitations or mispronunciations in real time.

---

## 2. Navigating the Interface

The application interface is intentionally simple, calm, and distraction-free:

```
+----------------------------------------------------------------------------------+
|  🕉️ Project Medhā (मेधा)          [📖 User Manual]   [⚙️ MLOps Pipeline]   ● Ready |
+----------------------------------------------------------------------------------+
|  [ 🎙️ Live Recitation ]      [ 📁 Audio File Verification ]                       |
|                                                                                  |
|  Select Sacred Manuscript: [ Ṇamōkāra Mahāmantra (णमोकार महामंत्र)   ▼ ]           |
|  [ ] Hide Text for Blind Memorization Test                                       |
|                                                                                  |
|  +----------------------------------------------------------------------------+  |
|  | णमो अरिहंताणं                                                              |  |
|  | णमो सिद्धाणं                                                               |  |
|  | णमो आयरियाणं                                                               |  |
|  +----------------------------------------------------------------------------+  |
|                                                                                  |
|  [ 🎙️ Start Live Recitation ]                     Hesitation Threshold: [ 1.5s ] |
|                                                                                  |
|  Overall Accuracy: 100%   |   Hesitation Pauses: 0   |   Latency: 312 ms         |
+----------------------------------------------------------------------------------+
```

### Top Navigation Bar
* **📖 User Manual:** Opens quick on-screen instructions at any time.
* **⚙️ MLOps Pipeline:** Opens the technical evaluation and pipeline architecture modal (intended for course instructors, evaluators, and system administrators).
* **System Status Badge:** Shows `● Live Ready` when the backend inference engine and model server are active.

---

## 3. How to Practice: Step-by-Step

### 3.1 Mode 1: Live Continuous Recitation (Recommended)

1. **Select a Scripture:**
   - Choose a preset scripture from the dropdown menu (e.g., *Ṇamōkāra Mahāmantra*, *Uvasaggaharaṁ Stōtra*, *Khamemi Savve Jive*, *Gayatri Mantra*), or paste custom Devanagari text into the reference box.
2. **Select Practice Mode:**
   - **Guided Mode (Default):** The text remains visible on screen as you recite.
   - **Blind Memorization Test:** Check the box **"Hide Text for Blind Memorization Test"** to test your memory without visual prompts. The text will be veiled with an auspicious golden motif and will reveal itself line-by-line as you chant correctly!
3. **Start Reciting:**
   - Click the green **"🎙️ Start Live Recitation"** button and allow microphone access in your browser.
   - Begin chanting at your natural speed.
4. **Natural Breath Pauses:**
   - Pause naturally for about 0.5 to 1.0 seconds at the end of each line or breath phrase.
   - Medhā will immediately transcribe the line and color-code your pronunciation!
5. **Real-Time Color Codes:**
   - <span style="color: #22c55e; font-weight: bold;">Green (Correct):</span> Syllable pronounced accurately.
   - <span style="color: #ef4444; font-weight: bold;">Red (Mispronounced):</span> Syllable substituted or mispronounced.
   - <span style="color: #f59e0b; font-weight: bold;">Amber / Yellow (Omitted):</span> Syllable missed or skipped.
   - **⚠️ Hesitation Warning:** If you pause for longer than 1.5 seconds, a gentle hesitation indicator appears, helping you build continuous, confident recall.
6. **Finish Session:**
   - Click **"⏹️ Finish Session"** when done. A session score card will summarize your total accuracy percentage, syllables recited, and cadence consistency.

---

### 3.2 Mode 2: Audio File Verification

If you have pre-recorded your recitation or received an audio recording from a student:

1. Click the **"📁 Audio File Verification"** tab.
2. Enter or paste the reference manuscript text.
3. Click or drag-and-drop your audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac`).
4. Click **"Verify Recitation"**.
5. Medhā will display:
   - Overall pronunciation similarity score (%).
   - Complete syllable diff comparison.
   - Detailed timestamped list of any pauses exceeding 1.5 seconds.
   - Total model inference latency.

---

## 4. Collaborative Voice Contribution (Data Collector)

Medhā improves its accuracy through community voice contributions across different accents and speaking styles:

* **Access the Voice Studio:** Navigate to the data collection portal at `http://localhost:8501/collector.html` or via the sidebar link.
* **Recording:** Read the short 2–10 second verse shown on screen, click **"Record"**, recite clearly, and click **"Upload"**.
* Every verified recording is automatically sanitized using Ray distributed audio processing and included in continuous model training cycles.

---

## 5. Frequently Asked Questions (FAQ)

### Q1: Why does Medhā tolerate missing spaces?
In Sanskrit and Prakrit, words often blend together through *sandhi* (e.g. *namo arihantāṇaṁ* can be written or spoken as a single euphonic breath). Medhā's space-agnostic phonetic aligner evaluates the actual sound (*shabda*) rather than typing spaces.

### Q2: How can I change the hesitation warning sensitivity?
Before clicking "Start Live Recitation", adjust the **Hesitation Alert Threshold** slider (default is 1.5 seconds). For beginners, 2.0s or 2.5s is recommended; for advanced memorization tests, 1.2s to 1.5s ensures mastery.

### Q3: What microphone should I use?
Your standard laptop microphone or mobile headset works well. Ensure you are in a quiet room with minimal background echo or ambient fan noise.
