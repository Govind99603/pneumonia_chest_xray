
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const analyzeBtn = document.getElementById("analyzeBtn");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const fileName = document.getElementById("fileName");
const removeFile = document.getElementById("removeFile");
const errorBox = document.getElementById("errorBox");
const spinner = document.getElementById("spinner");
const btnText = document.getElementById("btnText");

const emptyResult = document.getElementById("emptyResult");
const prediction = document.getElementById("prediction");
const predictionLabel = document.getElementById("predictionLabel");
const confidence = document.getElementById("confidence");
const normalProbability = document.getElementById("normalProbability");
const pneumoniaProbability = document.getElementById("pneumoniaProbability");
const pneumoniaBar = document.getElementById("pneumoniaBar");

let selectedFile = null;

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
}

function setFile(file) {
    clearError();

    if (!file) return;

    const allowed = ["image/png", "image/jpeg", "image/webp"];

    if (!allowed.includes(file.type)) {
        showError("Please select a PNG, JPG, JPEG or WEBP image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("File is too large. Maximum size is 10 MB.");
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = (event) => {
        preview.src = event.target.result;
        fileName.textContent = file.name;
        previewWrap.classList.remove("hidden");
        dropzone.classList.add("hidden");
        analyzeBtn.disabled = false;
    };

    reader.readAsDataURL(file);
}

fileInput.addEventListener("change", (event) => {
    setFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
    });
});

dropzone.addEventListener("drop", (event) => {
    setFile(event.dataTransfer.files[0]);
});

removeFile.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    preview.src = "";
    previewWrap.classList.add("hidden");
    dropzone.classList.remove("hidden");
    analyzeBtn.disabled = true;
    clearError();
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    clearError();

    analyzeBtn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        emptyResult.classList.add("hidden");
        prediction.classList.remove("hidden");

        predictionLabel.textContent = data.prediction;
        confidence.textContent =
            `${(data.confidence * 100).toFixed(1)}%`;

        normalProbability.textContent =
            `${(data.normal_probability * 100).toFixed(1)}%`;

        pneumoniaProbability.textContent =
            `${(data.pneumonia_probability * 100).toFixed(1)}%`;

        pneumoniaBar.style.width =
            `${data.pneumonia_probability * 100}%`;

        predictionLabel.style.color =
            data.prediction === "PNEUMONIA"
                ? "#ff9c9c"
                : "#55e3c1";

        document.getElementById("prediction")
            .scrollIntoView({ behavior: "smooth", block: "nearest" });

    } catch (error) {
        showError(error.message);
    } finally {
        analyzeBtn.disabled = false;
        spinner.classList.add("hidden");
        btnText.textContent = "Analyze X-ray";
    }
});
