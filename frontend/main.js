const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const previewImage = document.getElementById('previewImage');
const placeholder = document.getElementById('placeholder');

const backendValue = document.getElementById('backendValue');
const predictionValue = document.getElementById('predictionValue');
const scoreValue = document.getElementById('scoreValue');
const thresholdValue = document.getElementById('thresholdValue');
const deviceValue = document.getElementById('deviceValue');
const messageValue = document.getElementById('messageValue');

let selectedFile = null;

function setMessage(text, type = 'ok') {
  messageValue.textContent = text;
  messageValue.classList.remove('ok', 'bad');
  messageValue.classList.add(type);
}

async function loadBackendStatus() {
  backendValue.textContent = 'Checking...';
  backendValue.classList.remove('ok', 'bad');

  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    const data = await response.json();

    if (!response.ok || !data.ready) {
      backendValue.textContent = 'Offline';
      backendValue.classList.add('bad');
      setMessage(data.error || 'Model backend is not ready', 'bad');
      return;
    }

    backendValue.textContent = 'Online';
    backendValue.classList.add('ok');
    thresholdValue.textContent = Number(data.threshold).toFixed(4);
    deviceValue.textContent = data.device;
    setMessage('Backend loaded. Upload an image and run detection.', 'ok');
  } catch (error) {
    backendValue.textContent = 'Offline';
    backendValue.classList.add('bad');
    setMessage('Cannot reach backend server', 'bad');
  }
}

function renderPreview(file) {
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.classList.add('visible');
  previewImage.hidden = false;
  placeholder.hidden = true;
}

imageInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    selectedFile = null;
    return;
  }

  selectedFile = file;
  renderPreview(file);
  setMessage('Image ready. Click run detection.', 'ok');
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!selectedFile) {
    setMessage('Choose an image first.', 'bad');
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  setMessage('Model is running inference...', 'ok');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      setMessage(data.error || 'Prediction failed', 'bad');
      return;
    }

    predictionValue.textContent = data.prediction;
    predictionValue.classList.remove('ok', 'bad');
    predictionValue.classList.add(data.prediction === 'REAL' ? 'ok' : 'bad');

    scoreValue.textContent = Number(data.score).toFixed(4);
    thresholdValue.textContent = Number(data.threshold).toFixed(4);
    deviceValue.textContent = data.device;
    setMessage('Inference complete.', 'ok');
  } catch (error) {
    setMessage('Request failed. Is backend running?', 'bad');
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run Detection';
  }
});

loadBackendStatus();
