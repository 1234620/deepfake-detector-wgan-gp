const revealCards = Array.from(document.querySelectorAll('.reveal-shell'));

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry, index) => {
      if (!entry.isIntersecting) return;
      const delay = index * 110;
      setTimeout(() => {
        entry.target.classList.add('is-visible');
      }, delay);
      observer.unobserve(entry.target);
    });
  },
  { threshold: 0.2 }
);

revealCards.forEach((card) => observer.observe(card));

function setStatus(el, ok, pending = false) {
  if (!el) return;
  el.classList.remove('status-ok', 'status-bad', 'status-pending');
  if (pending) {
    el.classList.add('status-pending');
    return;
  }
  el.classList.add(ok ? 'status-ok' : 'status-bad');
}

async function checkTextFile(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.text();
}

function checkImageFile(path) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = `${path}?v=${Date.now()}`;
  });
}

async function runRuntimeChecks() {
  const pipelineStatus = document.getElementById('pipelineStatus');
  const thresholdValue = document.getElementById('thresholdValue');
  const thresholdFileStatus = document.getElementById('thresholdFileStatus');
  const cmStatus = document.getElementById('cmStatus');
  const rocStatus = document.getElementById('rocStatus');
  const lossStatus = document.getElementById('lossStatus');

  const thresholdArtifact = document.getElementById('thresholdArtifact');
  const cmArtifact = document.getElementById('cmArtifact');
  const rocArtifact = document.getElementById('rocArtifact');
  const lossArtifact = document.getElementById('lossArtifact');

  setStatus(pipelineStatus, false, true);
  setStatus(thresholdFileStatus, false, true);
  setStatus(cmStatus, false, true);
  setStatus(rocStatus, false, true);
  setStatus(lossStatus, false, true);

  let okCount = 0;
  const totalChecks = 4;

  try {
    const rawThreshold = await checkTextFile('/checkpoints/threshold.txt');
    const threshold = Number.parseFloat(rawThreshold.trim());
    if (Number.isFinite(threshold) && thresholdValue) {
      thresholdValue.textContent = threshold.toFixed(4);
    } else if (thresholdValue) {
      thresholdValue.textContent = 'Invalid value';
    }
    if (thresholdFileStatus) thresholdFileStatus.textContent = 'OK';
    if (thresholdArtifact) thresholdArtifact.textContent = 'checkpoints/threshold.txt: OK';
    setStatus(thresholdFileStatus, true);
    okCount += 1;
  } catch (error) {
    if (thresholdValue) thresholdValue.textContent = 'Unavailable';
    if (thresholdFileStatus) thresholdFileStatus.textContent = 'Missing';
    if (thresholdArtifact) thresholdArtifact.textContent = 'checkpoints/threshold.txt: Missing';
    setStatus(thresholdFileStatus, false);
  }

  const hasCM = await checkImageFile('/results/confusion_matrix.png');
  if (cmStatus) cmStatus.textContent = hasCM ? 'OK' : 'Missing';
  if (cmArtifact) cmArtifact.textContent = `results/confusion_matrix.png: ${hasCM ? 'OK' : 'Missing'}`;
  setStatus(cmStatus, hasCM);
  okCount += hasCM ? 1 : 0;

  const hasROC = await checkImageFile('/results/roc_curve.png');
  if (rocStatus) rocStatus.textContent = hasROC ? 'OK' : 'Missing';
  if (rocArtifact) rocArtifact.textContent = `results/roc_curve.png: ${hasROC ? 'OK' : 'Missing'}`;
  setStatus(rocStatus, hasROC);
  okCount += hasROC ? 1 : 0;

  const hasLoss = await checkImageFile('/results/training_losses.png');
  if (lossStatus) lossStatus.textContent = hasLoss ? 'OK' : 'Missing';
  if (lossArtifact) lossArtifact.textContent = `results/training_losses.png: ${hasLoss ? 'OK' : 'Missing'}`;
  setStatus(lossStatus, hasLoss);
  okCount += hasLoss ? 1 : 0;

  if (pipelineStatus) pipelineStatus.textContent = `${okCount}/${totalChecks} Online`;
  setStatus(pipelineStatus, okCount === totalChecks);
}

runRuntimeChecks();

const generatedAt = document.getElementById('generatedAt');
if (generatedAt) {
  const now = new Date();
  generatedAt.textContent = `Generated: ${now.toISOString().slice(0, 10)} (local runtime stamp)`;
}
