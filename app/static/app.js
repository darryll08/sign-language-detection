const video = document.getElementById("video");
const cameraStatus = document.getElementById("camera-status");

const detectionMode = document.getElementById("detection-mode");
const handStatus = document.getElementById("hand-status");
const landmarkStatus = document.getElementById("landmark-status");

const captureBtn = document.getElementById("capture-btn");

const capturePredictionLabel = document.getElementById("capture-prediction-label");
const captureConfidenceText = document.getElementById("capture-confidence-text");
const captureConfidenceFill = document.getElementById("capture-confidence-fill");
const captureResultNote = document.getElementById("capture-result-note");
const captureTop3List = document.getElementById("capture-top3-list");

const startSpellBtn = document.getElementById("start-spell-btn");
const stopSpellBtn = document.getElementById("stop-spell-btn");
const backspaceBtn = document.getElementById("backspace-btn");
const clearTextBtn = document.getElementById("clear-text-btn");

const spellOutput = document.getElementById("spell-output");
const spellModeStatus = document.getElementById("spell-mode-status");
const spellRawLabel = document.getElementById("spell-raw-label");
const spellRawConfidence = document.getElementById("spell-raw-confidence");
const spellLastCommitted = document.getElementById("spell-last-committed");
const spellNote = document.getElementById("spell-note");

// hidden compatibility ids
const spellSystemState = document.getElementById("spell-system-state");
const spellCandidateLabel = document.getElementById("spell-candidate-label");
const spellCandidateVotes = document.getElementById("spell-candidate-votes");
const spellReleaseCount = document.getElementById("spell-release-count");

let stream = null;

// Mode 2 settings
const LIVE_INTERVAL_MS = 450;
const CONFIDENCE_THRESHOLD = 0.80;
const STABLE_WINDOW = 8;
const MIN_STABLE_COUNT = 6;
const RELEASE_REQUIRED_COUNT = 4;

let spellModeRunning = false;
let spellIntervalId = null;
let spellRequestInFlight = false;

let spellState = "WAITING_LETTER";
let stableHistory = [];
let releaseCounter = 0;
let spelledText = "";
let lastCommittedLabel = "-";

function setCameraStatus(text) {
    cameraStatus.innerText = text;
}

function setDetectionPanel(handDetected, landmarksActive = true) {
    detectionMode.innerText = "landmark_mlp";
    handStatus.innerText = handDetected ? "hand detected" : "waiting";
    landmarkStatus.innerText = landmarksActive ? "active" : "inactive";
}

async function buildCaptureBlobFromVideo() {
    const canvas = document.createElement("canvas");
    canvas.width = 448;
    canvas.height = 448;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.95);
    });
}

async function requestPrediction(endpoint, blob) {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const response = await fetch(endpoint, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        let detail = "Request failed.";
        try {
            const data = await response.json();
            detail = data.detail || detail;
        } catch (e) {
            // ignore
        }
        throw new Error(detail);
    }

    return await response.json();
}

async function startCamera() {
    if (stream) {
        setCameraStatus("camera already running");
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        setCameraStatus("camera ready");
        setDetectionPanel(false, true);
    } catch (error) {
        console.error(error);
        setCameraStatus("failed to access camera");
        setDetectionPanel(false, false);
    }
}

function stopCamera() {
    stopSpellMode();

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    setCameraStatus("camera stopped");
    setDetectionPanel(false, false);
}

function updateCaptureResult(label, confidence) {
    const percent = confidence * 100;

    capturePredictionLabel.innerText = label;
    captureConfidenceText.innerText = `${percent.toFixed(1)}%`;
    captureConfidenceFill.style.width = `${percent.toFixed(1)}%`;

    if (label === "nothing") {
        captureResultNote.innerText = "Tidak ada tangan yang terdeteksi jelas, atau sistem belum melihat gesture yang valid.";
        return;
    }

    if (percent >= 95) {
        captureResultNote.innerText = "Model sangat yakin dengan hasil ini.";
    } else if (percent >= 80) {
        captureResultNote.innerText = "Hasil cukup meyakinkan.";
    } else if (percent >= 60) {
        captureResultNote.innerText = "Hasil masih cukup ragu. Coba rapikan pose tangan lalu capture lagi.";
    } else {
        captureResultNote.innerText = "Confidence rendah. Coba posisikan tangan lebih jelas dan satu tangan saja.";
    }
}

function setCaptureEmptyState(message) {
    capturePredictionLabel.innerText = "-";
    captureConfidenceText.innerText = "0.0%";
    captureConfidenceFill.style.width = "0%";
    captureTop3List.innerHTML = `<div class="top3-empty">Belum ada hasil prediksi.</div>`;
    captureResultNote.innerText = message;
}

function renderCaptureTop3(top3) {
    if (!top3 || top3.length === 0) {
        captureTop3List.innerHTML = `<div class="top3-empty">Belum ada hasil prediksi.</div>`;
        return;
    }

    captureTop3List.innerHTML = "";

    top3.forEach((item, index) => {
        const percent = (item.confidence * 100).toFixed(1);

        const row = document.createElement("div");
        row.className = "top3-row";
        row.innerHTML = `
            <div class="top3-rank">#${index + 1}</div>
            <div class="top3-label">${item.label}</div>
            <div class="top3-bar-track">
                <div class="top3-bar-fill" style="width: ${percent}%"></div>
            </div>
            <div class="top3-percent">${percent}%</div>
        `;
        captureTop3List.appendChild(row);
    });
}

async function captureFrame() {
    if (!stream) {
        setCaptureEmptyState("Start camera terlebih dahulu.");
        return;
    }

    if (spellModeRunning) {
        setCaptureEmptyState("Stop spell mode dulu sebelum capture.");
        return;
    }

    try {
        captureBtn.disabled = true;
        captureResultNote.innerText = "Reading hand landmarks...";

        const blob = await buildCaptureBlobFromVideo();
        if (!blob) {
            throw new Error("Gagal membuat capture image.");
        }

        const data = await requestPrediction("/predict-capture", blob);

        setDetectionPanel(data.label !== "nothing", data.used_landmarks);

        if (!data.ok) {
            setCaptureEmptyState(data.message || "Prediction failed.");
            return;
        }

        updateCaptureResult(data.label, data.confidence);
        renderCaptureTop3(data.top3);

    } catch (error) {
        console.error(error);
        setCaptureEmptyState(error.message);
    } finally {
        captureBtn.disabled = false;
    }
}

function resetSpellStateUI() {
    spellSystemState.innerText = "WAITING_LETTER";
    spellCandidateLabel.innerText = "-";
    spellCandidateVotes.innerText = `0 / ${STABLE_WINDOW}`;
    spellReleaseCount.innerText = `0 / ${RELEASE_REQUIRED_COUNT}`;
    spellLastCommitted.innerText = lastCommittedLabel;
    spellRawLabel.innerText = "-";
    spellRawConfidence.innerText = "-";
}

function updateSpellOutput() {
    spellOutput.innerText = spelledText.length > 0 ? spelledText : "(kosong)";
}

function clearSpellText() {
    spelledText = "";
    lastCommittedLabel = "-";
    spellLastCommitted.innerText = "-";
    updateSpellOutput();
    spellNote.innerText = "...";
}

function backspaceSpellText() {
    if (spelledText.length === 0) {
        spellNote.innerText = "Tidak ada karakter untuk dihapus.";
        return;
    }

    spelledText = spelledText.slice(0, -1);
    updateSpellOutput();
    spellNote.innerText = "Karakter terakhir dihapus.";
}

function getMajorityInfo(labels) {
    if (!labels || labels.length === 0) {
        return { label: null, count: 0 };
    }

    const counts = {};
    let bestLabel = null;
    let bestCount = 0;

    for (const label of labels) {
        counts[label] = (counts[label] || 0) + 1;
        if (counts[label] > bestCount) {
            bestCount = counts[label];
            bestLabel = label;
        }
    }

    return { label: bestLabel, count: bestCount };
}

function commitLabel(label) {
    lastCommittedLabel = label;
    spellLastCommitted.innerText = label;

    if (label === "space") {
        if (spelledText.length > 0 && !spelledText.endsWith(" ")) {
            spelledText += " ";
        }
    } else if (label === "del") {
        spelledText = spelledText.slice(0, -1);
    } else if (label !== "nothing") {
        spelledText += label;
    }

    updateSpellOutput();
}

function startSpellMode() {
    if (!stream) {
        spellNote.innerText = "Start camera terlebih dahulu.";
        return;
    }

    if (spellModeRunning) {
        spellNote.innerText = "Spell mode sudah berjalan.";
        return;
    }

    spellModeRunning = true;
    spellState = "WAITING_LETTER";
    stableHistory = [];
    releaseCounter = 0;

    spellModeStatus.innerText = "running";
    spellNote.innerText = "Tahan satu gesture sampai stabil. Setelah tersimpan, lepaskan tangan sebentar.";

    startSpellBtn.disabled = true;
    stopSpellBtn.disabled = false;
    captureBtn.disabled = true;

    if (spellIntervalId) {
        clearInterval(spellIntervalId);
    }

    spellIntervalId = setInterval(processSpellFrame, LIVE_INTERVAL_MS);
}

function stopSpellMode() {
    spellModeRunning = false;

    if (spellIntervalId) {
        clearInterval(spellIntervalId);
        spellIntervalId = null;
    }

    spellRequestInFlight = false;
    spellState = "WAITING_LETTER";
    stableHistory = [];
    releaseCounter = 0;

    spellModeStatus.innerText = "idle";
    startSpellBtn.disabled = false;
    stopSpellBtn.disabled = true;
    captureBtn.disabled = false;
}

async function processSpellFrame() {
    if (!spellModeRunning || !stream || spellRequestInFlight) {
        return;
    }

    try {
        spellRequestInFlight = true;

        const blob = await buildCaptureBlobFromVideo();
        if (!blob) {
            throw new Error("Gagal membuat live frame.");
        }

        const data = await requestPrediction("/predict-live", blob);

        setDetectionPanel(data.label !== "nothing", data.used_landmarks);
        handleSpellResult(data);

    } catch (error) {
        console.error(error);
        spellModeStatus.innerText = "error";
        spellNote.innerText = error.message;
    } finally {
        spellRequestInFlight = false;
    }
}

function handleNoHand(message) {
    spellRawLabel.innerText = "nothing";
    spellRawConfidence.innerText = "100.0%";
    stableHistory = [];

    if (spellState === "WAITING_RELEASE") {
        releaseCounter += 1;

        if (releaseCounter >= RELEASE_REQUIRED_COUNT) {
            spellState = "WAITING_LETTER";
            releaseCounter = 0;
            spellNote.innerText = "Release berhasil. Silakan tampilkan huruf berikutnya.";
        } else {
            spellNote.innerText = message || "No valid hand/gesture detected.";
        }
    } else {
        releaseCounter = 0;
        spellNote.innerText = message || "No valid hand/gesture detected.";
    }
}

function handleSpellPrediction(rawLabel, confidence) {
    const percent = confidence * 100;
    const reliable = confidence >= CONFIDENCE_THRESHOLD;

    spellRawLabel.innerText = rawLabel;
    spellRawConfidence.innerText = `${percent.toFixed(1)}%`;

    if (spellState === "WAITING_LETTER") {
        if (reliable && rawLabel !== "nothing") {
            stableHistory.push(rawLabel);

            if (stableHistory.length > STABLE_WINDOW) {
                stableHistory.shift();
            }

            const majority = getMajorityInfo(stableHistory);

            if (majority.label && majority.count >= MIN_STABLE_COUNT) {
                commitLabel(majority.label);

                spellState = "WAITING_RELEASE";
                stableHistory = [];
                releaseCounter = 0;

                spellNote.innerText = `Huruf "${majority.label}" sudah disimpan. Lepaskan tangan sebentar.`;
            } else {
                spellNote.innerText = "Membaca gesture...";
            }
        } else {
            stableHistory = [];

            if (rawLabel === "nothing") {
                spellNote.innerText = "Silakan tampilkan gesture berikutnya.";
            } else {
                spellNote.innerText = "Confidence belum cukup. Tahan gesture lebih stabil.";
            }
        }
    } else if (spellState === "WAITING_RELEASE") {
        const isReleaseFrame = !reliable || rawLabel === "nothing";

        if (isReleaseFrame) {
            releaseCounter += 1;

            if (releaseCounter >= RELEASE_REQUIRED_COUNT) {
                spellState = "WAITING_LETTER";
                releaseCounter = 0;
                spellNote.innerText = "Siap membaca huruf berikutnya.";
            } else {
                spellNote.innerText = "Menunggu jeda sebelum huruf berikutnya.";
            }
        } else {
            releaseCounter = 0;
            spellNote.innerText = "Huruf sudah tersimpan. Lepaskan tangan sebentar.";
        }
    }
}

function handleSpellResult(data) {
    if (!data.ok) {
        handleNoHand(data.message || "No valid hand.");
        return;
    }

    if (data.label === "nothing") {
        handleNoHand("No valid hand/gesture detected.");
        return;
    }

    handleSpellPrediction(data.label, data.confidence);
}

// initial state
updateSpellOutput();
resetSpellStateUI();
setCameraStatus("idle");
setDetectionPanel(false, false);