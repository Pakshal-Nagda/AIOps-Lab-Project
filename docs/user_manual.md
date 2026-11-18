# User Manual & Practitioner Guide

**Project Name:** Project Medhā (मेधा) — Sanskrit & Prakrit Live Recitation Verifier  
**Version:** 1.0.0  
**Target Audience:** Students, Practitioners, Scholars, and Reciters of Sacred Texts  

---

## 1. Introduction to Project Medhā

Welcome to **Project Medhā (मेधा)**, an AI-powered recitation assistant for mastering traditional oral recitation of Sanskrit and Prakrit scriptures (such as the *Ṇamōkāra Mahāmantra*, *Khamēmi Savva Jīvē*, *Uvasaggaharaṁ Stōtra*, *Bhaktāmara Stōtra*, and Vedic hymns).

Medhā listens to your live voice through your computer or phone microphone, verifies pronunciation syllable-by-syllable (*akshara*), tracks rhythmic cadence, and highlights hesitations or mispronunciations in real time.

---

## 2. Navigating the Interface

The application interface is clean, dark-themed, and distraction-free:

```
+----------------------------------------------------------------------------------+
|  मे Project Medhā [मेधा]       [● Online (MLflow)]  [📖 User Manual]  [🎙️ Contribute Audio] |
|     Sanskrit & Prakrit Recitation Verifier                                       |
+----------------------------------------------------------------------------------+
|  [ 🎙️ Live Recitation (Microphone) ]    [ 📁 Audio File Verification ]            |
|                                                                                  |
|  1. Enter Verse / Shloka to Recite     [ 👁️ Hide Text for Blind Memorization Test ]|
|  +----------------------------------------------------------------------------+  |
|  | Paste your Sanskrit or Prakrit verse here (single or multi-line)...         |  |
|  | e.g. णमो अरिहंताणं णमो सिद्धाणं णमो आयरियाणं णमो उवज्झायाणं णमो लोए सव्वसाहूणं |  |
|  +----------------------------------------------------------------------------+  |
|  ✨ Tip: Spaces and typographic punctuation are visual conventions and never penalized. |
|                                                                                  |
|  [ 🎙️ Begin Recitation ]                                        Status: [ Ready ]|
|  +----------------------------------------------------------------------------+  |
|  | Pronunciation Accuracy |   Cadence Hesitations   |     Lines Completed     |  |
|  |         -- %           |            0            |          0 / 0          |  |
|  +----------------------------------------------------------------------------+  |
|                                                                                  |
|  📜 Live Recitation Canvas                                      [ Clear Canvas ] |
|  +----------------------------------------------------------------------------+  |
|  | Your recitation will stream onto this canvas in real-time as you speak...  |  |
|  +----------------------------------------------------------------------------+  |
|  ● Correct  ● Mispronounced  ● Skipped Akshara  ● Added Sound  [⏸️ 1.8s] Hesitation|
+----------------------------------------------------------------------------------+
|  Project Medhā                                                                   |
+----------------------------------------------------------------------------------+
```

### Top Navigation Bar
* **मे Project Medhā:** App branding with Sanskrit/Prakrit subtitle.
* **System Status Badge:** Displays `● Online (MLflow Serving)` when backend inference and model serving are operational, or `● Offline` if disconnected.
* **📖 User Manual:** Opens the in-app quick guide modal.
* **🎙️ Contribute Audio:** Direct navigation to the collaborative voice contributor portal (`collector.html`).

---

## 3. How to Practice: Step-by-Step

### 3.1 Mode 1: Live Continuous Recitation (Microphone)

1. **Enter Scripture Verse:**
   - Type or paste your Sanskrit or Prakrit verse, shloka, or sutra into the reference text box.
   - You can enter a single line or multi-line verses (e.g. the 5-line *Ṇamōkāra Mahāmantra*).
   - *Note:* Text containing no Devanagari characters will be prompted for correction.
2. **Blind Memorization Practice (Optional):**
   - Click **"👁️ Hide Text for Blind Memorization Test"** to test your memory without reading from the screen.
   - The reference textarea collapses into a blind memorization card. As you recite, verified lines materialize onto the canvas below line-by-line!
   - Click **"🙈 Show Text (Disable Blind Mode)"** at any point to restore the reference text.
3. **Start Reciting:**
   - Click the orange **"🎙️ Begin Recitation"** button and grant microphone permission in your browser.
   - The button switches to **"⏹️ Finish Recitation"** with an active recording glow, and the audio waveform indicator activates.
   - Speak naturally at your normal chanting tempo.
4. **Natural Rhythm & Pauses:**
   - Medhā continuously captures speech chunks (~128ms framing) via WebSockets.
   - Natural breath pauses (0.5s–1.0s) between lines or breath phrases trigger automatic line transcription and alignment.
   - Unbroken recitation across verse boundaries is fully supported—you do not need to pause artificially.
5. **Real-Time Visual Feedback & Color Legend:**
   - <span style="color: #34d399; font-weight: bold;">Green (Correct):</span> Syllable pronounced accurately.
   - <span style="color: #f87171; font-weight: bold;">Red Underline (Mispronounced):</span> Pronunciation discrepancy. Hover to view expected vs heard akshara.
   - <span style="color: #fbbf24; font-weight: bold;">Yellow Dashed (Skipped Akshara):</span> Syllable omitted in recitation.
   - <span style="color: #c084fc; font-weight: bold;">Purple (Added Sound):</span> Extra sound or unexpected syllable chanted.
   - <span style="color: #fbbf24; font-family: monospace;">[⏸️ 1.8s] Cadence Hesitation:</span> Pauses exceeding 1.5 seconds are annotated inline with an amber duration badge, signaling a rhythm hesitation without disruptive alert modals.
6. **Live Metrics HUD:**
   - **Pronunciation Accuracy:** Running percentage score of phonetically matched aksharas.
   - **Cadence Hesitations:** Cumulative count of pauses $> 1.5$ seconds.
   - **Lines Completed:** Real-time progress tracker (e.g., `3 / 5` lines).
7. **Complete Session:**
   - Click **"⏹️ Finish Recitation"**.
   - A completion summary banner appears with total lines completed, overall pronunciation accuracy percentage, and cadence hesitations.

---

### 3.2 Mode 2: Audio File Verification

For pre-recorded recitations or evaluating recorded audio submissions:

1. Click the **"📁 Audio File Verification"** tab.
2. Enter the expected reference text in Devanagari script into **Expected Reference Text**.
3. Select an audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac`) via **Select Audio Recording**.
4. Click **"Analyze Recording"**.
5. Medhā displays:
   - Match accuracy score percentage (`Match Accuracy: X%`).
   - Syllable-by-syllable phonetic diff highlighting correct, mispronounced, and skipped aksharas.

---

## 4. Collaborative Speech Collector Portal (`collector.html`)

Help improve the Sanskrit/Prakrit acoustic model by contributing recordings across different speaker styles and accents:

* **Access:** Click **"Contribute Audio"** in the top header or visit `http://localhost:8501/collector.html`.
* **Speaker Name:** Enter your name in the **Speaker / Contributor Name** field (defaults to placeholder `"Enter your name here"`).
* **Record Prompt:** Review the verse prompt displayed in large Devanagari script, click **"🔴 Start Recording"**, chant the prompt (recommended 2–10 seconds), and click **"⏹️ Stop Recording"**.
* **Review & Submit:** Listen to your audio playback preview, then click **"✅ Save & Next Prompt"** to upload the clip and advance to the next prompt.
* **File Upload Alternative:** If browser microphone access is restricted (e.g., on certain mobile devices over HTTP), expand the **"📁 Alternatively, upload pre-recorded audio file for this prompt"** dropdown to upload a voice memo directly.

---

## 5. Frequently Asked Questions (FAQ)

### Q1: Why does Medhā tolerate missing spaces?
In Sanskrit and Prakrit, word boundaries often merge euphonically through *sandhi* (e.g., *namō arihantāṇaṁ* and *namōarihantāṇaṁ* produce identical acoustic phonemes). Medhā's space-agnostic phonetic aligner evaluates sound units (*shabda*) rather than typography, preserving original manuscript spacing in the visualization.

### Q2: What is the cadence hesitation threshold?
Medhā uses a calibrated **1.5-second pause threshold** implemented in the streaming Voice Activity Detector (VAD). Natural breath pauses between lines (0.5s–1.0s) are treated as line boundaries, whereas silences longer than 1.5 seconds are flagged with an inline `[⏸️ Xs]` badge to track recitation fluency.

### Q3: What audio hardware is recommended?
Any standard built-in laptop microphone, smartphone microphone, or headset is suitable. The audio pipeline performs automatic 16kHz resampling, noise gate thresholding, and peak normalization.
