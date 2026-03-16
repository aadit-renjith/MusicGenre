/* ───────────────────────────────────────────────
   Music Genre Classifier — Frontend Logic
   ─────────────────────────────────────────────── */

// ── Configuration ──────────────────────────────
// For local development, use http://localhost:8000
// For production, replace with your Render.com URL
const API_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "https://music-genre-api.onrender.com";  // <-- Replace with your Render URL after deployment

// ── Genre emoji / color map ────────────────────
const GENRE_META = {
    blues:     { emoji: "🎷", hue: 220 },
    classical: { emoji: "🎻", hue: 45  },
    country:   { emoji: "🤠", hue: 30  },
    disco:     { emoji: "🪩", hue: 300 },
    hiphop:    { emoji: "🎤", hue: 350 },
    jazz:      { emoji: "🎺", hue: 40  },
    metal:     { emoji: "🤘", hue: 0   },
    pop:       { emoji: "🎧", hue: 280 },
    reggae:    { emoji: "🌴", hue: 120 },
    rock:      { emoji: "🎸", hue: 15  },
};

// ── DOM Elements ───────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const fileInfo       = document.getElementById("file-info");
const fileName       = document.getElementById("file-name");
const fileSize       = document.getElementById("file-size");
const btnRemove      = document.getElementById("btn-remove");
const audioPlayer    = document.getElementById("audio-player");
const audioElement   = document.getElementById("audio-element");
const btnPredict     = document.getElementById("btn-predict");
const btnText        = document.querySelector(".btn-text");
const btnLoader      = document.getElementById("btn-loader");
const resultsSection = document.getElementById("results-section");
const genreEmoji     = document.getElementById("genre-emoji");
const genreName      = document.getElementById("genre-name");
const genreConfidence = document.getElementById("genre-confidence");
const confidenceBars = document.getElementById("confidence-bars");
const btnRetry       = document.getElementById("btn-retry");
const errorToast     = document.getElementById("error-toast");
const errorMessage   = document.getElementById("error-message");
const dropZoneContent = document.getElementById("drop-zone-content");

let selectedFile = null;

// ── File Selection ─────────────────────────────

// Click to browse
dropZone.addEventListener("click", () => fileInput.click());

// File input change
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    const allowed = [".wav", ".mp3", ".ogg", ".flac"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showError("Unsupported format. Please upload .wav, .mp3, .ogg, or .flac");
        return;
    }

    selectedFile = file;

    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.classList.remove("hidden");

    // Show audio player
    const url = URL.createObjectURL(file);
    audioElement.src = url;
    audioPlayer.classList.remove("hidden");

    // Show predict button
    btnPredict.classList.remove("hidden");

    // Hide results if showing
    resultsSection.classList.add("hidden");

    // Hide drop zone visuals slightly
    dropZoneContent.style.opacity = "0.4";
    dropZoneContent.style.pointerEvents = "none";
}

// Remove file
btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    audioPlayer.classList.add("hidden");
    audioElement.src = "";
    btnPredict.classList.add("hidden");
    resultsSection.classList.add("hidden");
    dropZoneContent.style.opacity = "1";
    dropZoneContent.style.pointerEvents = "auto";
}

// ── Prediction ─────────────────────────────────

btnPredict.addEventListener("click", async () => {
    if (!selectedFile) return;

    // UI: loading state
    btnPredict.disabled = true;
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    hideError();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        showResults(data);

    } catch (err) {
        console.error("Prediction error:", err);
        showError(err.message || "Failed to connect to the API. Is the backend running?");
    } finally {
        btnPredict.disabled = false;
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
});

// ── Render Results ─────────────────────────────

function showResults(data) {
    const genre = data.predicted_genre.toLowerCase();
    const meta = GENRE_META[genre] || { emoji: "🎵", hue: 260 };

    // Primary result
    genreEmoji.textContent = meta.emoji;
    genreName.textContent = capitalize(data.predicted_genre);

    // Top confidence
    const topConf = data.confidence[data.predicted_genre];
    genreConfidence.textContent = `${(topConf * 100).toFixed(1)}% confidence`;

    // Confidence bars — sort descending
    const sorted = Object.entries(data.confidence)
        .sort((a, b) => b[1] - a[1]);

    confidenceBars.innerHTML = "";

    sorted.forEach(([label, prob], i) => {
        const pct = (prob * 100).toFixed(1);
        const isTop = label === data.predicted_genre;

        const row = document.createElement("div");
        row.className = "bar-row";
        row.style.animationDelay = `${i * 0.06}s`;

        row.innerHTML = `
            <span class="bar-label">${capitalize(label)}</span>
            <div class="bar-track">
                <div class="bar-fill ${isTop ? 'top' : ''}" style="--target-width: ${pct}%"></div>
            </div>
            <span class="bar-value">${pct}%</span>
        `;

        confidenceBars.appendChild(row);
    });

    // Animate bars after render
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll(".bar-fill").forEach((bar) => {
                bar.style.width = bar.style.getPropertyValue("--target-width");
            });
        }, 100);
    });

    // Show section
    resultsSection.classList.remove("hidden");

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ── Retry ──────────────────────────────────────

btnRetry.addEventListener("click", () => {
    resetUpload();
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// ── Error Toast ────────────────────────────────

function showError(msg) {
    errorMessage.textContent = msg;
    errorToast.classList.remove("hidden");
    setTimeout(hideError, 6000);
}

function hideError() {
    errorToast.classList.add("hidden");
}

// ── Helpers ────────────────────────────────────

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
