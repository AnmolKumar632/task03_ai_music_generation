// NeuralHarmony Web Audio Engine, Multi-Instrument Synthesizer, FX Rack & Melodic Analytics
document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const modelStatusBadge = document.getElementById("modelStatusBadge");
    const modelStatusText = document.getElementById("modelStatusText");
    const deviceBadge = document.getElementById("deviceBadge");
    const vocabSizeVal = document.getElementById("vocabSizeVal");

    const presetButtons = document.querySelectorAll(".preset-btn");
    const instrumentSelect = document.getElementById("instrumentSelect");
    const tempInput = document.getElementById("temperatureInput");
    const tempDisplay = document.getElementById("tempValueDisplay");
    const lengthInput = document.getElementById("lengthInput");
    const lengthDisplay = document.getElementById("lengthValueDisplay");
    const tempoSelect = document.getElementById("tempoSelect");
    const generateBtn = document.getElementById("generateBtn");

    // FX Rack elements
    const reverbSpaceSelect = document.getElementById("reverbSpaceSelect");
    const delayFeedbackInput = document.getElementById("delayFeedbackInput");

    // Analytics elements
    const toggleAnalyticsBtn = document.getElementById("toggleAnalyticsBtn");
    const analyticsDrawer = document.getElementById("analyticsDrawer");
    const detectedKeyBadge = document.getElementById("detectedKeyBadge");
    const metricTotalNotes = document.getElementById("metricTotalNotes");
    const metricChordRatio = document.getElementById("metricChordRatio");
    const metricDensity = document.getElementById("metricDensity");
    const metricDuration = document.getElementById("metricDuration");
    const pitchBarsRow = document.getElementById("pitchBarsRow");

    const virtualKeyboard = document.getElementById("virtualKeyboard");
    const promptTagsContainer = document.getElementById("promptTagsContainer");
    const clearSeedBtn = document.getElementById("clearSeedBtn");

    const trackTitle = document.getElementById("trackTitle");
    const trackMeta = document.getElementById("trackMeta");
    const downloadMidiBtn = document.getElementById("downloadMidiBtn");
    const downloadWavBtn = document.getElementById("downloadWavBtn");
    const canvasPlaceholder = document.getElementById("canvasPlaceholder");

    const canvas = document.getElementById("pianoRollCanvas");
    const ctx = canvas.getContext("2d");

    const playPauseBtn = document.getElementById("playPauseBtn");
    const playIcon = document.getElementById("playIcon");
    const stopBtn = document.getElementById("stopBtn");
    const loopBtn = document.getElementById("loopBtn");
    const volumeSlider = document.getElementById("volumeSlider");

    const progressBarContainer = document.getElementById("progressBarContainer");
    const progressBarFill = document.getElementById("progressBarFill");
    const scrubHandle = document.getElementById("scrubHandle");
    const currentTimeLabel = document.getElementById("currentTimeLabel");
    const totalTimeLabel = document.getElementById("totalTimeLabel");
    const trackHistoryList = document.getElementById("trackHistoryList");

    // Audio & State
    let audioCtx = null;
    let masterDryGain = null;
    let masterOutGain = null;
    let convolverReverb = null;
    let reverbWetGain = null;
    let delayNode = null;
    let delayFeedbackGain = null;
    let delayWetGain = null;

    let isPlaying = false;
    let isLooping = false;
    let currentPlaybackTime = 0.0;
    let totalTrackDuration = 0.0;
    let activeNotes = [];
    let animationFrameId = null;
    let playbackStartTime = 0;
    let currentCompositionFilename = "composition.wav";

    let customSeedNotes = [];

    const GENRE_PRESETS = {
        classical: { temp: 0.75, length: 60, tempo: "0.4", instrument: "piano", reverb: "cathedral", delay: 0.1 },
        lofi: { temp: 0.65, length: 40, tempo: "0.6", instrument: "electric_piano", reverb: "room", delay: 0.3 },
        cyberpunk: { temp: 1.15, length: 80, tempo: "0.25", instrument: "synth_lead", reverb: "cosmic", delay: 0.45 },
        ambient: { temp: 0.85, length: 50, tempo: "0.6", instrument: "strings", reverb: "cosmic", delay: 0.35 }
    };

    const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    // 1. Synthetic Impulse Response Generator for Reverb
    function buildImpulseResponse(ctx, durationSeconds, decayRate) {
        const rate = ctx.sampleRate;
        const length = rate * durationSeconds;
        const impulse = ctx.createBuffer(2, length, rate);
        const left = impulse.getChannelData(0);
        const right = impulse.getChannelData(1);

        for (let i = 0; i < length; i++) {
            const decay = Math.pow(1 - i / length, decayRate);
            left[i] = (Math.random() * 2 - 1) * decay;
            right[i] = (Math.random() * 2 - 1) * decay;
        }
        return impulse;
    }

    // 2. Audio Graph Setup with Reverb & Stereo Delay
    function initAudio() {
        const AudioContext = window.AudioContext || window.webkitAudioContext;

        if (!audioCtx || audioCtx.state === "closed") {
            audioCtx = new AudioContext();
            masterOutGain = null;
            masterDryGain = null;
            convolverReverb = null;
            reverbWetGain = null;
            delayNode = null;
            delayFeedbackGain = null;
            delayWetGain = null;

            // Master Volume
            masterOutGain = audioCtx.createGain();
            masterOutGain.gain.value = parseFloat(volumeSlider.value);
            masterOutGain.connect(audioCtx.destination);

            // Dry Path
            masterDryGain = audioCtx.createGain();
            masterDryGain.gain.value = 0.85;
            masterDryGain.connect(masterOutGain);

            // Reverb Subsystem
            convolverReverb = audioCtx.createConvolver();
            updateReverbSpace();

            reverbWetGain = audioCtx.createGain();
            reverbWetGain.gain.value = reverbSpaceSelect.value === "off" ? 0.0 : 0.35;
            convolverReverb.connect(reverbWetGain);
            reverbWetGain.connect(masterOutGain);

            // Stereo Delay Subsystem
            delayNode = audioCtx.createDelay(1.0);
            delayNode.delayTime.value = 0.28; // ~120 BPM 1/8th note delay

            delayFeedbackGain = audioCtx.createGain();
            delayFeedbackGain.gain.value = parseFloat(delayFeedbackInput.value);

            delayWetGain = audioCtx.createGain();
            delayWetGain.gain.value = parseFloat(delayFeedbackInput.value) * 0.7;

            delayNode.connect(delayFeedbackGain);
            delayFeedbackGain.connect(delayNode);
            delayNode.connect(delayWetGain);
            delayWetGain.connect(masterOutGain);
        }

        if (audioCtx.state === "suspended") {
            audioCtx.resume();
        }
    }

    function updateReverbSpace() {
        if (!audioCtx || !convolverReverb) return;
        const space = reverbSpaceSelect.value;
        if (space === "off") {
            if (reverbWetGain) reverbWetGain.gain.value = 0.0;
        } else if (space === "room") {
            convolverReverb.buffer = buildImpulseResponse(audioCtx, 1.2, 3.5);
            if (reverbWetGain) reverbWetGain.gain.value = 0.25;
        } else if (space === "cathedral") {
            convolverReverb.buffer = buildImpulseResponse(audioCtx, 2.8, 1.8);
            if (reverbWetGain) reverbWetGain.gain.value = 0.4;
        } else if (space === "cosmic") {
            convolverReverb.buffer = buildImpulseResponse(audioCtx, 4.5, 1.1);
            if (reverbWetGain) reverbWetGain.gain.value = 0.55;
        }
    }

    function midiToFreq(midiPitch) {
        return 440 * Math.pow(2, (midiPitch - 69) / 12);
    }

    // 3. Multi-Instrument Polyphonic Synthesizer
    function playTone(freq, startTime, duration, instrument = "piano", targetCtx = audioCtx, targetDest = null) {
        if (!targetCtx) return;

        // Route through FX rack if in main context
        const noteGain = targetCtx.createGain();

        if (targetDest) {
            noteGain.connect(targetDest);
        } else {
            noteGain.connect(masterDryGain);
            if (convolverReverb) noteGain.connect(convolverReverb);
            if (delayNode) noteGain.connect(delayNode);
        }

        if (instrument === "electric_piano") {
            const osc1 = targetCtx.createOscillator();
            const osc2 = targetCtx.createOscillator();
            const bellGain = targetCtx.createGain();

            osc1.type = "sine";
            osc1.frequency.setValueAtTime(freq, startTime);

            osc2.type = "sine";
            osc2.frequency.setValueAtTime(freq * 3.98, startTime);

            bellGain.gain.setValueAtTime(0.3, startTime);
            bellGain.gain.exponentialRampToValueAtTime(0.001, startTime + Math.min(0.2, duration));

            noteGain.gain.setValueAtTime(0.001, startTime);
            noteGain.gain.exponentialRampToValueAtTime(0.2, startTime + 0.015);
            noteGain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

            osc1.connect(noteGain);
            osc2.connect(bellGain);
            bellGain.connect(noteGain);

            osc1.start(startTime);
            osc2.start(startTime);
            osc1.stop(startTime + duration);
            osc2.stop(startTime + duration);
        } else if (instrument === "synth_lead") {
            const osc1 = targetCtx.createOscillator();
            const osc2 = targetCtx.createOscillator();
            const filter = targetCtx.createBiquadFilter();

            osc1.type = "sawtooth";
            osc2.type = "sawtooth";
            osc1.frequency.setValueAtTime(freq, startTime);
            osc2.frequency.setValueAtTime(freq * 1.008, startTime);

            filter.type = "lowpass";
            filter.frequency.setValueAtTime(freq * 4, startTime);
            filter.frequency.exponentialRampToValueAtTime(freq * 1.5, startTime + duration);
            filter.Q.value = 4.0;

            noteGain.gain.setValueAtTime(0.001, startTime);
            noteGain.gain.exponentialRampToValueAtTime(0.18, startTime + 0.02);
            noteGain.gain.setValueAtTime(0.14, startTime + duration - 0.05);
            noteGain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

            osc1.connect(filter);
            osc2.connect(filter);
            filter.connect(noteGain);

            osc1.start(startTime);
            osc2.start(startTime);
            osc1.stop(startTime + duration);
            osc2.stop(startTime + duration);
        } else if (instrument === "strings") {
            const osc1 = targetCtx.createOscillator();
            const osc2 = targetCtx.createOscillator();
            osc1.type = "triangle";
            osc2.type = "sawtooth";
            osc1.frequency.setValueAtTime(freq, startTime);
            osc2.frequency.setValueAtTime(freq * 0.995, startTime);

            const attack = Math.min(0.15, duration * 0.4);
            noteGain.gain.setValueAtTime(0.0001, startTime);
            noteGain.gain.linearRampToValueAtTime(0.18, startTime + attack);
            noteGain.gain.setValueAtTime(0.14, startTime + duration - 0.1);
            noteGain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

            osc1.connect(noteGain);
            osc2.connect(noteGain);

            osc1.start(startTime);
            osc2.start(startTime);
            osc1.stop(startTime + duration);
            osc2.stop(startTime + duration);
        } else {
            const osc = targetCtx.createOscillator();
            const subOsc = targetCtx.createOscillator();

            osc.type = "triangle";
            osc.frequency.setValueAtTime(freq, startTime);

            subOsc.type = "sine";
            subOsc.frequency.setValueAtTime(freq * 0.5, startTime);

            const attack = 0.02;
            const release = 0.08;
            const peakGain = 0.22;

            noteGain.gain.setValueAtTime(0.0001, startTime);
            noteGain.gain.exponentialRampToValueAtTime(peakGain, startTime + attack);
            noteGain.gain.setValueAtTime(peakGain * 0.75, startTime + duration - release);
            noteGain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

            osc.connect(noteGain);
            subOsc.connect(noteGain);

            osc.start(startTime);
            subOsc.start(startTime);
            osc.stop(startTime + duration);
            subOsc.stop(startTime + duration);
        }
    }

    // 4. Playback Transport
    function startPlayback(fromTime = 0) {
        if (!activeNotes.length) return;
        initAudio();

        stopScheduledPlayback();
        if (audioCtx && audioCtx.state === "suspended") {
            audioCtx.resume();
        }

        isPlaying = true;
        playIcon.textContent = "⏸";

        currentPlaybackTime = fromTime;
        playbackStartTime = audioCtx.currentTime - fromTime;
        const currentInstrument = instrumentSelect.value;

        activeNotes.forEach(note => {
            if (note.time + note.duration >= fromTime) {
                const noteStartInAudioCtx = playbackStartTime + note.time;
                if (noteStartInAudioCtx >= audioCtx.currentTime) {
                    note.pitches.forEach(pitch => {
                        playTone(midiToFreq(pitch), noteStartInAudioCtx, note.duration, currentInstrument);
                    });
                }
            }
        });

        renderLoop();
    }

    function pausePlayback() {
        if (!isPlaying) return;
        stopScheduledPlayback();
        if (audioCtx && audioCtx.state === "running") {
            audioCtx.suspend();
        }
        isPlaying = false;
        playIcon.textContent = "▶";
    }

    function stopPlayback() {
        stopScheduledPlayback();
        if (audioCtx && audioCtx.state === "running") {
            audioCtx.suspend();
        }
        isPlaying = false;
        playIcon.textContent = "▶";
        currentPlaybackTime = 0;
        updateProgressUI(0);
        renderPianoRoll();
    }

    function stopScheduledPlayback() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }

    function renderLoop() {
        if (!isPlaying || !audioCtx) return;

        currentPlaybackTime = audioCtx.currentTime - playbackStartTime;

        if (currentPlaybackTime >= totalTrackDuration) {
            if (isLooping) {
                startPlayback(0);
                return;
            } else {
                stopPlayback();
                return;
            }
        }

        updateProgressUI(currentPlaybackTime);
        renderPianoRoll();
        animationFrameId = requestAnimationFrame(renderLoop);
    }

    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    function updateProgressUI(time) {
        const pct = totalTrackDuration > 0 ? (time / totalTrackDuration) * 100 : 0;
        progressBarFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
        scrubHandle.style.left = `${Math.min(100, Math.max(0, pct))}%`;
        currentTimeLabel.textContent = formatTime(time);
    }

    // 5. Piano Roll Visualizer
    function renderPianoRoll() {
        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = "#060911";
        ctx.fillRect(0, 0, width, height);

        const minPitch = 48; // C3
        const maxPitch = 84; // C6
        const pitchRange = maxPitch - minPitch;
        const rowHeight = height / pitchRange;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
        ctx.lineWidth = 1;
        for (let p = 0; p <= pitchRange; p++) {
            const y = height - (p * rowHeight);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        if (!activeNotes.length) return;

        const duration = totalTrackDuration || 1;
        const pixelsPerSecond = width / duration;

        activeNotes.forEach(noteItem => {
            const x = noteItem.time * pixelsPerSecond;
            const w = Math.max(6, noteItem.duration * pixelsPerSecond - 2);

            noteItem.pitches.forEach(pitch => {
                const normalizedPitch = Math.max(minPitch, Math.min(maxPitch, pitch));
                const y = height - ((normalizedPitch - minPitch + 1) * rowHeight);

                const isCurrentlyActive = isPlaying &&
                    currentPlaybackTime >= noteItem.time &&
                    currentPlaybackTime < (noteItem.time + noteItem.duration);

                if (isCurrentlyActive) {
                    ctx.fillStyle = "#38bdf8";
                    ctx.shadowColor = "#06b6d4";
                    ctx.shadowBlur = 15;
                } else if (noteItem.is_chord) {
                    ctx.fillStyle = "#ec4899";
                    ctx.shadowColor = "#d946ef";
                    ctx.shadowBlur = 8;
                } else {
                    ctx.fillStyle = "#a855f7";
                    ctx.shadowColor = "#8b5cf6";
                    ctx.shadowBlur = 8;
                }

                ctx.beginPath();
                ctx.roundRect(x, y + 1, w, rowHeight - 2, 4);
                ctx.fill();
            });
        });

        ctx.shadowBlur = 0;

        if (totalTrackDuration > 0) {
            const cursorX = (currentPlaybackTime / totalTrackDuration) * width;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cursorX, 0);
            ctx.lineTo(cursorX, height);
            ctx.stroke();
        }
    }

    // 6. Melodic Analytics Computation
    function computeAndRenderAnalytics(notes, duration) {
        if (!notes || !notes.length) return;

        const pitchCounts = new Array(12).fill(0);
        let chordCount = 0;
        let totalIndividualPitches = 0;

        notes.forEach(item => {
            if (item.is_chord) chordCount++;
            item.pitches.forEach(p => {
                const pitchClass = p % 12;
                pitchCounts[pitchClass]++;
                totalIndividualPitches++;
            });
        });

        metricTotalNotes.textContent = totalIndividualPitches;
        metricChordRatio.textContent = `${Math.round((chordCount / notes.length) * 100)}%`;
        metricDensity.textContent = (totalIndividualPitches / Math.max(1, duration)).toFixed(1);
        metricDuration.textContent = `${Math.round(duration)}s`;

        // Estimate Root Key: pitch class with highest frequency
        let maxCount = -1;
        let dominantPitchClass = 0;
        pitchCounts.forEach((c, idx) => {
            if (c > maxCount) {
                maxCount = c;
                dominantPitchClass = idx;
            }
        });
        const tonicName = NOTE_NAMES[dominantPitchClass];
        detectedKeyBadge.textContent = `Key: ~${tonicName} Harmonic`;

        // Render Pitch Class Bars
        const maxPitchFreq = Math.max(1, ...pitchCounts);
        pitchBarsRow.innerHTML = NOTE_NAMES.map((name, idx) => {
            const count = pitchCounts[idx];
            const pct = Math.max(4, Math.round((count / maxPitchFreq) * 100));
            return `
                <div class="pitch-bar-col" title="${name}: ${count} notes">
                    <div class="pitch-bar-fill" style="height: ${pct}%;"></div>
                    <span class="pitch-bar-label">${name}</span>
                </div>
            `;
        }).join("");
    }

    toggleAnalyticsBtn.addEventListener("click", () => {
        analyticsDrawer.classList.toggle("hidden");
        toggleAnalyticsBtn.classList.toggle("active", !analyticsDrawer.classList.contains("hidden"));
    });

    // 7. Virtual Keyboard Trigger
    function triggerVirtualKey(pitch, noteLabel) {
        initAudio();
        const freq = midiToFreq(pitch);
        playTone(freq, audioCtx.currentTime, 0.4, instrumentSelect.value);

        if (customSeedNotes.length >= 8) {
            customSeedNotes.shift();
        }
        customSeedNotes.push({ pitch, noteLabel });
        renderPromptChips();
    }

    function renderPromptChips() {
        if (!customSeedNotes.length) {
            promptTagsContainer.innerHTML = '<span class="prompt-placeholder">No seed notes played (Click keys or press A-K)</span>';
            return;
        }
        promptTagsContainer.innerHTML = customSeedNotes.map(item =>
            `<span class="seed-chip">${item.noteLabel}</span>`
        ).join("");
    }

    virtualKeyboard.querySelectorAll(".key").forEach(key => {
        key.addEventListener("mousedown", () => {
            const pitch = parseInt(key.dataset.pitch, 10);
            const note = key.dataset.note;
            key.classList.add("active");
            triggerVirtualKey(pitch, note);
        });
        key.addEventListener("mouseup", () => key.classList.remove("active"));
        key.addEventListener("mouseleave", () => key.classList.remove("active"));
    });

    const KEY_MAP = {
        'a': 60, 'w': 61, 's': 62, 'e': 63, 'd': 64,
        'f': 65, 't': 66, 'g': 67, 'y': 68, 'h': 69,
        'u': 70, 'j': 71, 'k': 72
    };

    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        const key = e.key.toLowerCase();
        if (KEY_MAP[key]) {
            const pitch = KEY_MAP[key];
            const keyEl = virtualKeyboard.querySelector(`[data-pitch="${pitch}"]`);
            if (keyEl && !keyEl.classList.contains("active")) {
                keyEl.classList.add("active");
                triggerVirtualKey(pitch, keyEl.dataset.note);
            }
        }
    });

    window.addEventListener("keyup", (e) => {
        const key = e.key.toLowerCase();
        if (KEY_MAP[key]) {
            const keyEl = virtualKeyboard.querySelector(`[data-pitch="${KEY_MAP[key]}"]`);
            if (keyEl) keyEl.classList.remove("active");
        }
    });

    clearSeedBtn.addEventListener("click", () => {
        customSeedNotes = [];
        renderPromptChips();
    });

    // 8. Genre Presets
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            presetButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const preset = GENRE_PRESETS[btn.dataset.preset];
            if (preset) {
                tempInput.value = preset.temp;
                tempDisplay.textContent = preset.temp;
                lengthInput.value = preset.length;
                lengthDisplay.textContent = `${preset.length} notes`;
                tempoSelect.value = preset.tempo;
                instrumentSelect.value = preset.instrument;
                reverbSpaceSelect.value = preset.reverb;
                delayFeedbackInput.value = preset.delay;
                updateReverbSpace();
                if (delayFeedbackGain) delayFeedbackGain.gain.value = preset.delay;
            }
        });
    });

    reverbSpaceSelect.addEventListener("change", updateReverbSpace);
    delayFeedbackInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        if (delayFeedbackGain) delayFeedbackGain.gain.value = val;
        if (delayWetGain) delayWetGain.gain.value = val * 0.7;
    });

    // 9. API Status Check
    async function checkModelStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            if (data.status === "ready") {
                modelStatusBadge.classList.add("ready");
                modelStatusText.textContent = "Model Online (Ready)";
                deviceBadge.textContent = data.device.toUpperCase();
                vocabSizeVal.textContent = `${data.vocab_size} tokens`;
                generateBtn.disabled = false;
            } else {
                modelStatusText.textContent = "Model Not Trained";
                generateBtn.disabled = true;
            }
        } catch (e) {
            modelStatusText.textContent = "Backend Offline";
            generateBtn.disabled = true;
        }
    }

    // 10. Generate Music Request
    async function handleGenerate() {
        initAudio();
        stopPlayback();

        const numNotes = parseInt(lengthInput.value, 10);
        const temperature = parseFloat(tempInput.value);
        const stepDuration = parseFloat(tempoSelect.value);
        const instrument = instrumentSelect.value;
        const seedNotes = customSeedNotes.length > 0 ? customSeedNotes.map(n => String(n.pitch)) : null;

        generateBtn.disabled = true;
        generateBtn.querySelector(".btn-text").textContent = "Synthesizing Composition...";

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    num_notes: numNotes,
                    temperature: temperature,
                    seed_notes: seedNotes,
                    step_duration: stepDuration,
                    instrument: instrument
                })
            });

            const data = await res.json();
            if (data.success) {
                activeNotes = data.notes;
                totalTrackDuration = data.total_duration;
                currentCompositionFilename = data.filename.replace(".mid", ".wav");

                trackTitle.textContent = data.filename;
                trackMeta.textContent = `${data.total_notes} Events &bull; ${data.total_duration}s &bull; ${instrument.replace('_', ' ').toUpperCase()}`;

                downloadMidiBtn.href = data.download_url;
                downloadMidiBtn.classList.remove("disabled");

                downloadWavBtn.classList.remove("disabled");

                canvasPlaceholder.classList.add("hidden");
                totalTimeLabel.textContent = formatTime(totalTrackDuration);

                computeAndRenderAnalytics(activeNotes, totalTrackDuration);

                updateProgressUI(0);
                renderPianoRoll();

                setTimeout(() => startPlayback(0), 100);
                loadHistory();
            } else {
                alert("Generation Error: " + (data.error || "Unknown failure"));
            }
        } catch (err) {
            alert("Network error while communicating with AI generation server.");
        } finally {
            generateBtn.disabled = false;
            generateBtn.querySelector(".btn-text").textContent = "Generate Composition";
        }
    }

    // 11. Offline WAV Synthesis and Export
    async function exportToWav() {
        if (!activeNotes.length || !totalTrackDuration) return;

        downloadWavBtn.textContent = "Rendering WAV...";
        downloadWavBtn.disabled = true;

        const sampleRate = 44100;
        const totalSamples = Math.ceil((totalTrackDuration + 1.5) * sampleRate);
        const offlineCtx = new OfflineAudioContext(1, totalSamples, sampleRate);
        const inst = instrumentSelect.value;

        activeNotes.forEach(note => {
            note.pitches.forEach(pitch => {
                playTone(midiToFreq(pitch), note.time, note.duration, inst, offlineCtx, offlineCtx.destination);
            });
        });

        const renderedBuffer = await offlineCtx.startRendering();
        const wavBlob = audioBufferToWavBlob(renderedBuffer);

        const url = URL.createObjectURL(wavBlob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = currentCompositionFilename;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);

        downloadWavBtn.textContent = "🔊 Audio (.wav)";
        downloadWavBtn.disabled = false;
    }

    function audioBufferToWavBlob(buffer) {
        const numOfChan = buffer.numberOfChannels;
        const length = buffer.length * numOfChan * 2 + 44;
        const out = new DataView(new ArrayBuffer(length));
        const channels = [];
        let offset = 0;
        let pos = 0;

        function setUint16(data) {
            out.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data) {
            out.setUint32(pos, data, true);
            pos += 4;
        }

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8);
        setUint32(0x45564157); // "WAVE"
        setUint32(0x20746d66); // "fmt "
        setUint32(16);
        setUint16(1);
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan);
        setUint16(numOfChan * 2);
        setUint16(16);
        setUint32(0x61746164); // "data"
        setUint32(length - pos - 4);

        for (let i = 0; i < buffer.numberOfChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }

        while (offset < buffer.length) {
            for (let i = 0; i < numOfChan; i++) {
                let sample = Math.max(-1, Math.min(1, channels[i][offset]));
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
                out.setInt16(pos, sample, true);
                pos += 2;
            }
            offset++;
        }

        return new Blob([out.buffer], { type: "audio/wav" });
    }

    // 12. History List
    async function loadHistory() {
        try {
            const res = await fetch("/api/history");
            const data = await res.json();
            if (data.history && data.history.length > 0) {
                trackHistoryList.innerHTML = data.history.map(item => `
                    <div class="track-item">
                        <span class="track-name">♬ ${item.filename}</span>
                        <a href="${item.download_url}" class="track-link" download>Download (${item.size_kb} KB)</a>
                    </div>
                `).join("");
            }
        } catch (e) {}
    }

    tempInput.addEventListener("input", (e) => {
        tempDisplay.textContent = e.target.value;
    });

    lengthInput.addEventListener("input", (e) => {
        lengthDisplay.textContent = `${e.target.value} notes`;
    });

    generateBtn.addEventListener("click", handleGenerate);
    downloadWavBtn.addEventListener("click", exportToWav);

    playPauseBtn.addEventListener("click", () => {
        if (isPlaying) {
            pausePlayback();
        } else {
            startPlayback(currentPlaybackTime);
        }
    });

    stopBtn.addEventListener("click", stopPlayback);

    loopBtn.addEventListener("click", () => {
        isLooping = !isLooping;
        loopBtn.classList.toggle("active", isLooping);
    });

    volumeSlider.addEventListener("input", (e) => {
        if (masterOutGain) {
            masterOutGain.gain.value = parseFloat(e.target.value);
        }
    });

    progressBarContainer.addEventListener("click", (e) => {
        if (!totalTrackDuration) return;
        const rect = progressBarContainer.getBoundingClientRect();
        const clickRatio = (e.clientX - rect.left) / rect.width;
        const targetTime = clickRatio * totalTrackDuration;
        if (isPlaying) {
            startPlayback(targetTime);
        } else {
            currentPlaybackTime = targetTime;
            updateProgressUI(targetTime);
            renderPianoRoll();
        }
    });

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        renderPianoRoll();
    }
    window.addEventListener("resize", resizeCanvas);

    resizeCanvas();
    checkModelStatus();
    loadHistory();
});
