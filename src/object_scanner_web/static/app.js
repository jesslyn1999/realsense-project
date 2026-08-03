import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const viewer = document.querySelector("#viewer");
const placeholder = document.querySelector("#viewer-placeholder");
const themeToggle = document.querySelector("#theme-toggle");
const stateBadge = document.querySelector("#state-badge");
const sessionNameInput = document.querySelector("#session-name");
const message = document.querySelector("#message");
const databasePath = document.querySelector("#database-path");
const totalPoints = document.querySelector("#total-points");
const displayedPoints = document.querySelector("#displayed-points");
const fitButton = document.querySelector("#fit-button");
const resetButton = document.querySelector("#reset-button");
const transformationName = document.querySelector("#transformation-name");
const transformationPosition = document.querySelector(
  "#transformation-position",
);
const transformationParentFrame = document.querySelector(
  "#transformation-parent-frame",
);
const transformationMatrix = document.querySelector("#transformation-matrix");
const publishTransformationButton = document.querySelector(
  "#publish-transformation-button",
);
const previousTransformationButton = document.querySelector(
  "#previous-transformation-button",
);
const nextTransformationButton = document.querySelector(
  "#next-transformation-button",
);
const savedSessionSelect = document.querySelector("#saved-session");
const replayDock = document.querySelector("#replay-dock");
const replaySessionName = document.querySelector("#replay-session-name");
const replayPosition = document.querySelector("#replay-position");
const replayPreviousButton = document.querySelector("#replay-previous-button");
const replayNextButton = document.querySelector("#replay-next-button");
const replayExitButton = document.querySelector("#replay-exit-button");
const cameraOverlayCheckbox = document.querySelector(
  "#camera-overlay-checkbox",
);
const referenceColor = document.querySelector("#reference-color");
const referenceHex = document.querySelector("#reference-hex");
const referenceRgb = document.querySelector("#reference-rgb");
const applyColorButton = document.querySelector("#apply-color-button");
const cameraColorButton = document.querySelector("#camera-color-button");
const cameraDialog = document.querySelector("#camera-color-dialog");
const cameraPreview = document.querySelector("#camera-preview");
const cameraInstruction = document.querySelector("#camera-instruction");
const pixelLoupe = document.querySelector("#pixel-loupe");
const sampleSwatch = document.querySelector("#sample-swatch");
const sampleRgb = document.querySelector("#sample-rgb");
const samplePosition = document.querySelector("#sample-position");
const closeCameraButton = document.querySelector("#close-camera-button");
const cancelCameraButton = document.querySelector("#cancel-camera-button");
const refreshCameraButton = document.querySelector("#refresh-camera-button");
const useCameraColorButton = document.querySelector(
  "#use-camera-color-button",
);
const commandButtons = {
  start: document.querySelector("#start-button"),
  pause: document.querySelector("#pause-button"),
  resume: document.querySelector("#resume-button"),
  stop: document.querySelector("#stop-button"),
};
const SESSION_NAME_PATTERN = /^[A-Za-z0-9_-]+$/;
const THEME_STORAGE_KEY = "object-scanner-theme";
const MAX_REPLAY_POINTS = 250_000;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
scene.add(new THREE.GridHelper(2, 20, 0x5d8fdc, 0x3b4554));
const axesHelper = new THREE.AxesHelper(0.25);
axesHelper.setColors(0xe66b70, 0x5d8fdc, 0x9a7bdc);
scene.add(axesHelper);

let currentState = "stopped";
let currentDatabasePath = null;
let transformBurstActive = false;
let pointCloud = null;
let cameraRgb = null;
let cameraWidth = 0;
let cameraHeight = 0;
let selectedPixel = null;
let selectedCameraRgb = null;
let replayFrames = [];
let replayIndex = 0;
let replayGeneration = 0;
let replayMode = false;
let replayClouds = [];
let replayDisplayedPoints = 0;
let replayTotalPoints = 0;
let replayCameraOverlay = null;

const sourceCanvas = document.createElement("canvas");
const sourceContext = sourceCanvas.getContext("2d");
const previewContext = cameraPreview.getContext("2d");
const loupeContext = pixelLoupe.getContext("2d");

function resetView() {
  camera.position.set(1.2, 1.0, 1.2);
  camera.near = 0.001;
  camera.far = 1000;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setTheme(theme, persist = false) {
  const isLight = theme === "light";
  document.documentElement.dataset.theme = isLight ? "light" : "dark";
  themeToggle.textContent = isLight ? "Dark" : "Light";
  themeToggle.setAttribute(
    "aria-label",
    `Switch to ${isLight ? "dark" : "light"} theme`,
  );
  themeToggle.setAttribute("aria-pressed", String(isLight));

  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, isLight ? "light" : "dark");
    } catch {
      // Keep the selected theme for this page when storage is unavailable.
    }
  }

  if (cameraRgb !== null) {
    drawPreview();
    const position = selectedPixel || {
      x: Math.floor(cameraWidth / 2),
      y: Math.floor(cameraHeight / 2),
    };
    drawLoupe(position.x, position.y);
  }
}

function rgbToHex(rgb) {
  return `#${rgb
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")}`;
}

function hexToRgb(hex) {
  return [1, 3, 5].map((offset) =>
    Number.parseInt(hex.slice(offset, offset + 2), 16),
  );
}

function showReferenceColor(rgb) {
  const hex = rgbToHex(rgb);
  referenceColor.value = hex;
  referenceHex.textContent = hex.toUpperCase();
  referenceRgb.textContent = `RGB ${rgb.join(", ")}`;
}

function showTransformation(transformation) {
  transformationName.textContent = transformation.name;
  transformationParentFrame.textContent =
    `Parent frame: ${transformation.parent_frame_id}`;
  transformationMatrix.textContent = transformation.matrix
    .map((row) => `[${row.join(", ")}]`)
    .join("\n");
}

function isSessionNameValid() {
  return SESSION_NAME_PATTERN.test(sessionNameInput.value);
}

function updateControls(busy = false) {
  commandButtons.start.disabled =
    busy || currentState !== "stopped" || !isSessionNameValid();
  commandButtons.pause.disabled = busy || currentState !== "recording";
  commandButtons.resume.disabled = busy || currentState !== "paused";
  commandButtons.stop.disabled =
    busy || !["recording", "paused"].includes(currentState);
  fitButton.disabled = pointCloud === null && replayClouds.length === 0;
  const colorDisabled = busy || currentState !== "stopped";
  referenceColor.disabled = colorDisabled;
  applyColorButton.disabled = colorDisabled;
  cameraColorButton.disabled = colorDisabled;
  sessionNameInput.disabled = busy || currentState !== "stopped";
  publishTransformationButton.disabled = busy || transformBurstActive;
  previousTransformationButton.disabled = busy || transformBurstActive;
  nextTransformationButton.disabled = busy || transformBurstActive;
  savedSessionSelect.disabled =
    busy ||
    currentState !== "stopped" ||
    savedSessionSelect.options.length < 2;
  replayPreviousButton.disabled = busy || !replayMode || replayIndex === 0;
  replayNextButton.disabled =
    busy || !replayMode || replayIndex >= replayFrames.length - 1;
  replayExitButton.disabled = busy || !replayMode;
}

function applyStatus(status) {
  currentState = status.state;
  currentDatabasePath = status.database_path;
  stateBadge.textContent =
    currentState.charAt(0).toUpperCase() + currentState.slice(1);
  stateBadge.className = `state-badge ${currentState}`;
  databasePath.textContent = currentDatabasePath || "No recording selected";
  transformBurstActive = Boolean(status.transform_burst_active);
  if (currentState !== "stopped" && replayMode) {
    exitReplay(false);
  }
  if (
    typeof status.session_name === "string" &&
    status.session_name &&
    (currentState !== "stopped" || sessionNameInput.value === "")
  ) {
    sessionNameInput.value = status.session_name;
  }
  if (Array.isArray(status.target_rgb)) {
    showReferenceColor(status.target_rgb);
  }
  if (status.transformation) {
    showTransformation(status.transformation);
  }
  if (
    Number.isInteger(status.transformation_index) &&
    Number.isInteger(status.transformation_total)
  ) {
    transformationPosition.textContent =
      `Matrix ${status.transformation_index} / ${status.transformation_total}`;
  }
  updateControls();
}

function disposeCloud(cloud) {
  scene.remove(cloud);
  cloud.geometry.dispose();
  cloud.material.dispose();
}

function clearReplayClouds() {
  for (const replayCloud of replayClouds) {
    if (replayCloud.cloud !== null) {
      disposeCloud(replayCloud.cloud);
    }
  }
  replayClouds = [];
  replayDisplayedPoints = 0;
  replayTotalPoints = 0;
}

function clearPointCloud() {
  if (pointCloud !== null) {
    disposeCloud(pointCloud);
    pointCloud = null;
  }
  clearReplayClouds();
  totalPoints.textContent = "—";
  displayedPoints.textContent = "—";
  placeholder.classList.remove("hidden");
  updateControls();
}

function fitCloud() {
  let sphere = null;
  if (pointCloud !== null) {
    pointCloud.geometry.computeBoundingSphere();
    sphere = pointCloud.geometry.boundingSphere;
  } else if (replayClouds.length > 0) {
    const bounds = new THREE.Box3();
    for (const replayCloud of replayClouds) {
      if (replayCloud.cloud !== null) {
        bounds.expandByObject(replayCloud.cloud);
      }
    }
    if (!bounds.isEmpty()) {
      sphere = bounds.getBoundingSphere(new THREE.Sphere());
    }
  }
  if (sphere === null) {
    return;
  }
  const radius = Math.max(sphere.radius, 0.01);
  const distance =
    (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.15;
  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() === 0) {
    direction.set(1, 1, 1);
  }
  direction.normalize();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, distance);
  camera.near = Math.max(radius / 1000, 0.0001);
  camera.far = Math.max(distance + radius * 20, 10);
  camera.updateProjectionMatrix();
  controls.update();
}

function decodePointPayload(buffer) {
  if (buffer.byteLength < 12) {
    throw new Error("Point payload is incomplete");
  }
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "PCD1") {
    throw new Error("Point payload has an unsupported format");
  }

  const view = new DataView(buffer);
  const displayed = view.getUint32(4, true);
  const total = view.getUint32(8, true);
  const positionBytes = displayed * 3 * Float32Array.BYTES_PER_ELEMENT;
  const expectedBytes = 12 + positionBytes + displayed * 3;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error("Point payload size does not match its header");
  }

  const positions = new Float32Array(buffer, 12, displayed * 3);
  const colors = new Uint8Array(buffer, 12 + positionBytes, displayed * 3);
  return { positions, colors, displayed, total };
}

function createPointCloud(payload) {
  if (payload.displayed === 0) {
    return null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(payload.positions, 3),
  );
  geometry.setAttribute(
    "color",
    new THREE.BufferAttribute(payload.colors, 3, true),
  );
  const material = new THREE.PointsMaterial({
    size: 0.004,
    sizeAttenuation: true,
    vertexColors: true,
  });
  return new THREE.Points(geometry, material);
}

function clearReplayCameraOverlay() {
  if (replayCameraOverlay === null) {
    return;
  }
  scene.remove(replayCameraOverlay);
  replayCameraOverlay.traverse((object) => {
    object.geometry?.dispose();
    if (Array.isArray(object.material)) {
      object.material.forEach((material) => material.dispose());
    } else {
      object.material?.dispose();
    }
  });
  replayCameraOverlay = null;
}

function updateReplayCameraOverlay() {
  clearReplayCameraOverlay();
  if (!replayMode || !cameraOverlayCheckbox.checked) {
    return;
  }

  const matrix = replayFrames[replayIndex]?.matrix;
  if (!Array.isArray(matrix) || matrix.length !== 4) {
    return;
  }

  const rotation = new THREE.Matrix4().set(
    matrix[0][0],
    matrix[0][1],
    matrix[0][2],
    0,
    matrix[1][0],
    matrix[1][1],
    matrix[1][2],
    0,
    matrix[2][0],
    matrix[2][1],
    matrix[2][2],
    0,
    0,
    0,
    0,
    1,
  );
  const overlay = new THREE.Group();
  overlay.position.set(matrix[0][3], matrix[1][3], matrix[2][3]);
  overlay.setRotationFromMatrix(rotation);

  const source = new THREE.Mesh(
    new THREE.SphereGeometry(0.008, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  );
  overlay.add(source);
  overlay.add(new THREE.AxesHelper(0.1));
  overlay.add(
    new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(),
      0.25,
      0xf4cf57,
      0.025,
      0.012,
    ),
  );

  replayCameraOverlay = overlay;
  scene.add(replayCameraOverlay);
}

function renderPayload(buffer) {
  const payload = decodePointPayload(buffer);
  clearPointCloud();
  pointCloud = createPointCloud(payload);
  if (pointCloud !== null) {
    scene.add(pointCloud);
    placeholder.classList.add("hidden");
    fitCloud();
  }
  totalPoints.textContent = payload.total.toLocaleString();
  displayedPoints.textContent = payload.displayed.toLocaleString();
  updateControls();
}

async function loadPoints() {
  setMessage("Loading recorded points…");
  const response = await fetch("/api/points", { cache: "no-store" });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || "Cannot load recorded points");
  }
  renderPayload(await response.arrayBuffer());
  setMessage("Showing all committed points recorded so far.");
}

function resetReplay(clearSelection = true) {
  replayGeneration += 1;
  replayMode = false;
  replayFrames = [];
  replayIndex = 0;
  replayDock.hidden = true;
  clearReplayCameraOverlay();
  clearReplayClouds();
  if (clearSelection) {
    savedSessionSelect.value = "";
  }
  if (pointCloud === null) {
    totalPoints.textContent = "—";
    displayedPoints.textContent = "—";
    placeholder.classList.remove("hidden");
  }
  updateControls();
}

async function exitReplay(restoreCurrentRecording = true) {
  const sessionName = savedSessionSelect.value;
  resetReplay();
  if (
    restoreCurrentRecording &&
    currentDatabasePath &&
    currentState === "stopped"
  ) {
    await loadPoints();
  } else {
    setMessage(`Exited replay '${sessionName}'.`);
  }
}

async function loadSavedSessions() {
  const selectedSession = savedSessionSelect.value;
  const response = await fetch("/api/sessions", { cache: "no-store" });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.message || "Cannot list saved sessions");
  }

  savedSessionSelect.replaceChildren(
    new Option(
      result.sessions.length ? "Select a saved session" : "No saved sessions",
      "",
    ),
  );
  for (const session of result.sessions) {
    savedSessionSelect.add(new Option(session.name, session.name));
  }
  if (result.sessions.some((session) => session.name === selectedSession)) {
    savedSessionSelect.value = selectedSession;
  } else if (selectedSession) {
    resetReplay();
  }
  updateControls();
}

function replayFramePointLimit(index) {
  const minimum = Math.floor(MAX_REPLAY_POINTS / replayFrames.length);
  const remainder = MAX_REPLAY_POINTS % replayFrames.length;
  return minimum + (index < remainder ? 1 : 0);
}

function updateReplayDisplay() {
  replayPosition.textContent =
    `Frame ${replayIndex + 1} / ${replayFrames.length}`;
  totalPoints.textContent = replayTotalPoints.toLocaleString();
  displayedPoints.textContent = replayDisplayedPoints.toLocaleString();
  replayDock.hidden = !replayMode;
  updateReplayCameraOverlay();
  updateControls();
}

async function addReplayFrame(index, generation = replayGeneration) {
  const sessionName = savedSessionSelect.value;
  const frame = replayFrames[index];
  if (!sessionName || frame === undefined) {
    return false;
  }

  const maxPoints = replayFramePointLimit(index);
  let payload;
  if (maxPoints === 0) {
    payload = {
      positions: new Float32Array(),
      colors: new Uint8Array(),
      displayed: 0,
      total: frame.point_count,
    };
  } else {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(sessionName)}/frames/${frame.id}` +
        `?max_points=${maxPoints}`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || "Cannot load replay frame");
    }
    payload = decodePointPayload(await response.arrayBuffer());
  }
  if (generation !== replayGeneration) {
    return false;
  }

  const cloud = createPointCloud(payload);
  if (cloud !== null) {
    scene.add(cloud);
    placeholder.classList.add("hidden");
  }
  replayClouds.push({
    cloud,
    displayed: payload.displayed,
    total: payload.total,
  });
  replayDisplayedPoints += payload.displayed;
  replayTotalPoints += payload.total;
  replayIndex = index;
  updateReplayDisplay();
  fitCloud();
  return true;
}

async function loadReplaySession() {
  const sessionName = savedSessionSelect.value;
  resetReplay(false);
  clearPointCloud();
  if (!sessionName) {
    return;
  }

  const generation = replayGeneration;
  setMessage(`Loading replay '${sessionName}'…`);
  try {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(sessionName)}/frames`,
      { cache: "no-store" },
    );
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "Cannot load replay");
    }
    if (generation !== replayGeneration) {
      return;
    }
    replayFrames = result.frames;
    if (replayFrames.length === 0) {
      setMessage(`Replay '${sessionName}' has no recorded frames.`);
      return;
    }
    replayMode = true;
    replaySessionName.textContent = sessionName;
    replayDock.hidden = false;
    await addReplayFrame(0, generation);
    setMessage(
      `Replay '${sessionName}' is ready. Use Next to add another frame.`,
    );
  } catch (error) {
    if (generation === replayGeneration) {
      resetReplay(false);
      setMessage(error.message, true);
    }
  } finally {
    updateControls();
  }
}

async function nextReplayFrame() {
  if (!replayMode || replayIndex >= replayFrames.length - 1) {
    return;
  }
  updateControls(true);
  try {
    const nextIndex = replayIndex + 1;
    if (await addReplayFrame(nextIndex)) {
      setMessage(
        `Added frame ${nextIndex + 1} of ${replayFrames.length} to the replay.`,
      );
    }
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    updateControls();
  }
}

function previousReplayFrame() {
  if (!replayMode || replayIndex === 0) {
    return;
  }
  const removed = replayClouds.pop();
  if (removed.cloud !== null) {
    disposeCloud(removed.cloud);
  }
  replayDisplayedPoints -= removed.displayed;
  replayTotalPoints -= removed.total;
  replayIndex -= 1;
  if (replayDisplayedPoints === 0) {
    placeholder.classList.remove("hidden");
  }
  updateReplayDisplay();
  fitCloud();
  setMessage(
    `Removed frame ${replayIndex + 2}; showing cumulative frames 1–${
      replayIndex + 1
    }.`,
  );
}

async function applyReferenceColor(rgb) {
  updateControls(true);
  setMessage("Updating reference color…");
  try {
    const response = await fetch("/api/reference-color", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rgb }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "Cannot update reference color");
    }
    applyStatus(result);
    setMessage(`Reference color set to RGB ${rgb.join(", ")}.`);
    return true;
  } catch (error) {
    setMessage(error.message, true);
    return false;
  } finally {
    updateControls();
  }
}

async function publishTransformation() {
  updateControls(true);
  setMessage("Publishing the displayed matrix for the next point cloud…");
  try {
    const response = await fetch("/api/transformation/publish", {
      method: "POST",
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "Cannot publish transformation");
    }
    applyStatus(result);
    setMessage(`${result.message}. Next: ${result.transformation.name}.`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    updateControls();
  }
}

async function stepTransformation(delta) {
  updateControls(true);
  try {
    const response = await fetch("/api/transformation/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delta }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "Cannot change transformation");
    }
    applyStatus(result);
    setMessage(`Selected transformation '${result.transformation.name}'.`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    updateControls();
  }
}

function drawPreview() {
  if (cameraRgb === null) {
    return;
  }
  previewContext.clearRect(0, 0, cameraWidth, cameraHeight);
  previewContext.drawImage(sourceCanvas, 0, 0);
  if (selectedPixel === null) {
    return;
  }

  const lineWidth = Math.max(1, cameraWidth / 700);
  previewContext.lineWidth = lineWidth;
  previewContext.strokeStyle = "#f4cf57";
  previewContext.strokeRect(
    selectedPixel.x - 2.5,
    selectedPixel.y - 2.5,
    5,
    5,
  );
  previewContext.strokeStyle = "#ffffff";
  previewContext.strokeRect(
    selectedPixel.x - 0.5,
    selectedPixel.y - 0.5,
    1,
    1,
  );
}

function drawLoupe(x, y) {
  if (cameraRgb === null) {
    return;
  }
  const sampleSize = 15;
  const radius = Math.floor(sampleSize / 2);
  const scale = pixelLoupe.width / sampleSize;
  const requestedX = x - radius;
  const requestedY = y - radius;
  const sourceX = Math.max(0, requestedX);
  const sourceY = Math.max(0, requestedY);
  const sourceRight = Math.min(cameraWidth, x + radius + 1);
  const sourceBottom = Math.min(cameraHeight, y + radius + 1);
  const sourceWidth = sourceRight - sourceX;
  const sourceHeight = sourceBottom - sourceY;
  const destinationX = (sourceX - requestedX) * scale;
  const destinationY = (sourceY - requestedY) * scale;

  loupeContext.fillStyle = getComputedStyle(
    document.documentElement,
  ).getPropertyValue("--surface-deep");
  loupeContext.fillRect(0, 0, pixelLoupe.width, pixelLoupe.height);
  loupeContext.imageSmoothingEnabled = false;
  loupeContext.drawImage(
    sourceCanvas,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    destinationX,
    destinationY,
    sourceWidth * scale,
    sourceHeight * scale,
  );

  loupeContext.lineWidth = 1;
  loupeContext.strokeStyle = "rgba(255, 255, 255, 0.2)";
  for (let index = 0; index <= sampleSize; index += 1) {
    const position = Math.round(index * scale) + 0.5;
    loupeContext.beginPath();
    loupeContext.moveTo(position, 0);
    loupeContext.lineTo(position, pixelLoupe.height);
    loupeContext.stroke();
    loupeContext.beginPath();
    loupeContext.moveTo(0, position);
    loupeContext.lineTo(pixelLoupe.width, position);
    loupeContext.stroke();
  }

  const center = radius * scale;
  loupeContext.lineWidth = 3;
  loupeContext.strokeStyle = "#ffffff";
  loupeContext.strokeRect(center, center, scale, scale);
  loupeContext.lineWidth = 1;
  loupeContext.strokeStyle = "#f4cf57";
  loupeContext.strokeRect(
    center - 2 * scale,
    center - 2 * scale,
    5 * scale,
    5 * scale,
  );
}

function cameraCoordinates(event) {
  const bounds = cameraPreview.getBoundingClientRect();
  return {
    x: Math.min(
      cameraWidth - 1,
      Math.max(
        0,
        Math.floor(((event.clientX - bounds.left) / bounds.width) * cameraWidth),
      ),
    ),
    y: Math.min(
      cameraHeight - 1,
      Math.max(
        0,
        Math.floor(
          ((event.clientY - bounds.top) / bounds.height) * cameraHeight,
        ),
      ),
    ),
  };
}

function averageCameraColor(x, y) {
  const sum = [0, 0, 0];
  let count = 0;
  for (
    let sampleY = Math.max(0, y - 2);
    sampleY <= Math.min(cameraHeight - 1, y + 2);
    sampleY += 1
  ) {
    for (
      let sampleX = Math.max(0, x - 2);
      sampleX <= Math.min(cameraWidth - 1, x + 2);
      sampleX += 1
    ) {
      const offset = (sampleY * cameraWidth + sampleX) * 3;
      sum[0] += cameraRgb[offset];
      sum[1] += cameraRgb[offset + 1];
      sum[2] += cameraRgb[offset + 2];
      count += 1;
    }
  }
  return sum.map((channel) => Math.round(channel / count));
}

function decodeCameraFrame(buffer) {
  if (buffer.byteLength < 12) {
    throw new Error("Camera frame is incomplete");
  }
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RGB1") {
    throw new Error("Camera frame has an unsupported format");
  }
  const header = new DataView(buffer);
  cameraWidth = header.getUint32(4, true);
  cameraHeight = header.getUint32(8, true);
  const expectedBytes = 12 + cameraWidth * cameraHeight * 3;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error("Camera frame size does not match its header");
  }

  cameraRgb = new Uint8Array(buffer, 12);
  const rgba = new Uint8ClampedArray(cameraWidth * cameraHeight * 4);
  for (
    let sourceOffset = 0, destinationOffset = 0;
    sourceOffset < cameraRgb.length;
    sourceOffset += 3, destinationOffset += 4
  ) {
    rgba[destinationOffset] = cameraRgb[sourceOffset];
    rgba[destinationOffset + 1] = cameraRgb[sourceOffset + 1];
    rgba[destinationOffset + 2] = cameraRgb[sourceOffset + 2];
    rgba[destinationOffset + 3] = 255;
  }

  sourceCanvas.width = cameraWidth;
  sourceCanvas.height = cameraHeight;
  cameraPreview.width = cameraWidth;
  cameraPreview.height = cameraHeight;
  sourceContext.putImageData(
    new ImageData(rgba, cameraWidth, cameraHeight),
    0,
    0,
  );
  selectedPixel = null;
  selectedCameraRgb = null;
  useCameraColorButton.disabled = true;
  sampleRgb.textContent = "No pixel selected";
  samplePosition.textContent = "5×5 average";
  cameraInstruction.textContent =
    "Move over the image to magnify pixels, then click one.";
  drawPreview();
  drawLoupe(Math.floor(cameraWidth / 2), Math.floor(cameraHeight / 2));
}

async function captureCameraFrame() {
  refreshCameraButton.disabled = true;
  useCameraColorButton.disabled = true;
  cameraInstruction.textContent = "Waiting for the next RealSense RGB frame…";
  try {
    const response = await fetch("/api/camera-frame", { cache: "no-store" });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || "Cannot capture camera frame");
    }
    decodeCameraFrame(await response.arrayBuffer());
  } catch (error) {
    cameraInstruction.textContent = error.message;
    setMessage(error.message, true);
  } finally {
    refreshCameraButton.disabled = false;
  }
}

async function sendCommand(command) {
  updateControls(true);
  setMessage(`${command.charAt(0).toUpperCase() + command.slice(1)} requested…`);
  try {
    const options = { method: "POST" };
    if (command === "start") {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify({
        session_name: sessionNameInput.value,
      });
    }
    const response = await fetch(`/api/recording/${command}`, options);
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || `Cannot ${command} recording`);
    }

    applyStatus(result);
    if (command === "start") {
      resetReplay();
      clearPointCloud();
      setMessage(
        `Recording '${result.session_name}' filtered world-frame points.`,
      );
    } else if (command === "pause" || command === "stop") {
      await loadPoints();
      if (command === "stop") {
        await loadSavedSessions();
      }
    } else {
      setMessage("Recording resumed in the same SQLite session.");
    }
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    updateControls();
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Cannot read scanner status");
    }
    const status = await response.json();
    applyStatus(status);
    setMessage("Scanner controls are ready.");
    await loadSavedSessions();
    if (
      currentDatabasePath &&
      ["paused", "stopped"].includes(currentState)
    ) {
      await loadPoints();
    }
  } catch (error) {
    setMessage(error.message, true);
  }
}

for (const [command, button] of Object.entries(commandButtons)) {
  button.addEventListener("click", () => sendCommand(command));
}
themeToggle.addEventListener("click", () => {
  const nextTheme =
    document.documentElement.dataset.theme === "light" ? "dark" : "light";
  setTheme(nextTheme, true);
});
sessionNameInput.addEventListener("input", () => updateControls());
publishTransformationButton.addEventListener("click", publishTransformation);
previousTransformationButton.addEventListener("click", () =>
  stepTransformation(-1),
);
nextTransformationButton.addEventListener("click", () =>
  stepTransformation(1),
);
savedSessionSelect.addEventListener("change", loadReplaySession);
replayPreviousButton.addEventListener("click", previousReplayFrame);
replayNextButton.addEventListener("click", nextReplayFrame);
replayExitButton.addEventListener("click", () => {
  exitReplay().catch((error) => setMessage(error.message, true));
});
cameraOverlayCheckbox.addEventListener("change", updateReplayCameraOverlay);
referenceColor.addEventListener("input", () => {
  showReferenceColor(hexToRgb(referenceColor.value));
});
applyColorButton.addEventListener("click", () => {
  applyReferenceColor(hexToRgb(referenceColor.value));
});
cameraColorButton.addEventListener("click", () => {
  cameraDialog.showModal();
  captureCameraFrame();
});
refreshCameraButton.addEventListener("click", captureCameraFrame);
closeCameraButton.addEventListener("click", () => cameraDialog.close());
cancelCameraButton.addEventListener("click", () => cameraDialog.close());
useCameraColorButton.addEventListener("click", async () => {
  if (
    selectedCameraRgb !== null &&
    (await applyReferenceColor(selectedCameraRgb))
  ) {
    cameraDialog.close();
  }
});
cameraPreview.addEventListener("pointermove", (event) => {
  if (cameraRgb === null) {
    return;
  }
  const position = cameraCoordinates(event);
  drawLoupe(position.x, position.y);
});
cameraPreview.addEventListener("click", (event) => {
  if (cameraRgb === null) {
    return;
  }
  selectedPixel = cameraCoordinates(event);
  selectedCameraRgb = averageCameraColor(selectedPixel.x, selectedPixel.y);
  sampleSwatch.style.backgroundColor = rgbToHex(selectedCameraRgb);
  sampleRgb.textContent = `RGB ${selectedCameraRgb.join(", ")}`;
  samplePosition.textContent =
    `Pixel ${selectedPixel.x}, ${selectedPixel.y} · 5×5 average`;
  cameraInstruction.textContent =
    "The white outline is the selected pixel; yellow is the averaged area.";
  useCameraColorButton.disabled = false;
  drawPreview();
  drawLoupe(selectedPixel.x, selectedPixel.y);
});
fitButton.addEventListener("click", fitCloud);
resetButton.addEventListener("click", resetView);

const resizeObserver = new ResizeObserver(() => {
  const width = viewer.clientWidth;
  const height = viewer.clientHeight;
  if (width === 0 || height === 0) {
    return;
  }
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
});
resizeObserver.observe(viewer);

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

setTheme(document.documentElement.dataset.theme);
resetView();
animate();
loadStatus();
