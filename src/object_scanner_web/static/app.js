import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const viewer = document.querySelector("#viewer");
const placeholder = document.querySelector("#viewer-placeholder");
const remoteLoadingOverlay = document.querySelector("#remote-loading-overlay");
const orientationGizmo = document.querySelector("#orientation-gizmo");
const pointCoordinateTooltip = document.querySelector(
  "#point-coordinate-tooltip",
);
const analyzeButton = document.querySelector("#analyze-button");
const adjustRepairButton = document.querySelector("#adjust-repair-button");
const analysisDock = document.querySelector("#analysis-dock");
const analysisStatus = document.querySelector("#analysis-status");
const analysisEditControls = document.querySelector("#analysis-edit-controls");
const analysisRepairVisible = document.querySelector(
  "#analysis-repair-visible",
);
const analysisSegmentVisible = document.querySelector(
  "#analysis-segment-visible",
);
const analysisApplyButton = document.querySelector("#analysis-apply-button");
const analysisCancelButton = document.querySelector("#analysis-cancel-button");
const analysisResetButton = document.querySelector("#analysis-reset-button");
const analysisModeButtons = {
  translate: document.querySelector("#analysis-move-button"),
  rotate: document.querySelector("#analysis-rotate-button"),
  scale: document.querySelector("#analysis-scale-button"),
};
const scanHelpButton = document.querySelector("#scan-help-button");
const scanHelpDialog = document.querySelector("#scan-help-dialog");
const closeScanHelpButton = document.querySelector(
  "#close-scan-help-button",
);
const dismissScanHelpButton = document.querySelector(
  "#dismiss-scan-help-button",
);
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
const transformationModeSelect = document.querySelector(
  "#transformation-mode",
);
const jsonTransformationPanel = document.querySelector(
  "#json-transformation-panel",
);
const charucoTransformationPanel = document.querySelector(
  "#charuco-transformation-panel",
);
const charucoPreview = document.querySelector("#charuco-preview");
const charucoPreviewStatus = document.querySelector(
  "#charuco-preview-status",
);
const charucoCornerCount = document.querySelector("#charuco-corner-count");
const charucoRmse = document.querySelector("#charuco-rmse");
const charucoMatrix = document.querySelector("#charuco-matrix");
const charucoCaptureButton = document.querySelector(
  "#charuco-capture-button",
);
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
const replayStageButtons = {
  raw: document.querySelector("#replay-raw-button"),
  filtered: document.querySelector("#replay-filtered-button"),
  aligned: document.querySelector("#replay-aligned-button"),
};
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
const REPAIR_SESSION = "demo5";
const REPLAY_STAGE_LABELS = {
  raw: "Raw",
  filtered: "Outliers removed",
  aligned: "ICP aligned",
};
const POINT_PICK_THRESHOLD_M = 0.008;
const POINT_TOOLTIP_OFFSET_PX = 12;
const CHARUCO_PREVIEW_INTERVAL_MS = 200;
const REMOTE_CONTROL_INTERVAL_MS = 300;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewer.appendChild(renderer.domElement);

const gizmoScene = new THREE.Scene();
const gizmoCamera = new THREE.OrthographicCamera(
  -1.05,
  1.05,
  1.05,
  -1.05,
  0.1,
  10,
);
const gizmoRenderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
});
gizmoRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
gizmoRenderer.setSize(104, 104, false);
gizmoRenderer.outputColorSpace = THREE.SRGBColorSpace;
orientationGizmo.appendChild(gizmoRenderer.domElement);

function createAxisLabel(text, color, position) {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.fillStyle = color;
  context.font = "700 42px sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 32, 34);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const label = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );
  label.position.copy(position);
  label.scale.setScalar(0.3);
  gizmoScene.add(label);
}

const gizmoOrigin = new THREE.Vector3();
const gizmoAxes = [
  ["X", new THREE.Vector3(1, 0, 0), 0xe45b5b, "#e45b5b"],
  ["Y", new THREE.Vector3(0, 1, 0), 0x45c878, "#45c878"],
  ["Z", new THREE.Vector3(0, 0, 1), 0x4f8fe8, "#4f8fe8"],
];
for (const [label, direction, color, cssColor] of gizmoAxes) {
  gizmoScene.add(
    new THREE.ArrowHelper(
      direction,
      gizmoOrigin,
      0.68,
      color,
      0.18,
      0.11,
    ),
  );
  createAxisLabel(label, cssColor, direction.clone().multiplyScalar(0.87));
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
const repairTransformControls = new TransformControls(
  camera,
  renderer.domElement,
);
const repairTransformHelper = repairTransformControls.getHelper();
repairTransformHelper.visible = false;
scene.add(repairTransformHelper);
const pointRaycaster = new THREE.Raycaster();
pointRaycaster.params.Points.threshold = POINT_PICK_THRESHOLD_M;
const pointPointer = new THREE.Vector2();
const hoveredPoint = new THREE.Vector3();
const grid = new THREE.GridHelper(2, 20, 0x5d8fdc, 0x3b4554);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
const axesHelper = new THREE.AxesHelper(0.25);
axesHelper.setColors(0xe66b70, 0x5d8fdc, 0x9a7bdc);
scene.add(axesHelper);

let currentState = "stopped";
let currentDatabasePath = null;
let transformBurstActive = false;
let transformationMode = "charuco";
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
let replayStage = "raw";
let replayAvailableStages = new Set(["raw"]);
let replayStageLoading = false;
let analysisBusy = false;
let repairPointCloud = null;
let repairMesh = null;
let repairInitialTransform = null;
let repairTransform = null;
let repairEditing = false;
let remoteLoadingVisible = false;
let remoteCommandRunning = false;
let remoteControlInitialized = false;
let pendingPointHover = null;
let charucoPreviewGeneration = 0;
let charucoPreviewRunning = false;

const sourceCanvas = document.createElement("canvas");
const sourceContext = sourceCanvas.getContext("2d");
const previewContext = cameraPreview.getContext("2d");
const loupeContext = pixelLoupe.getContext("2d");
const charucoPreviewContext = charucoPreview.getContext("2d");

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

function setRemoteLoadingVisible(visible) {
  remoteLoadingVisible = visible;
  remoteLoadingOverlay.hidden = !visible;
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
  transformationMatrix.textContent = formatMatrix(transformation.matrix);
}

function formatMatrix(matrix) {
  return matrix
    .map((row) => `[${row.join(", ")}]`)
    .join("\n");
}

function showCharucoResult(result) {
  if (!result) {
    return;
  }
  charucoCornerCount.textContent = Number.isInteger(result.corner_count)
    ? result.corner_count.toString()
    : "—";
  charucoRmse.textContent = Number.isFinite(result.reprojection_rmse_px)
    ? `${result.reprojection_rmse_px.toFixed(3)} px`
    : "—";
  if (Array.isArray(result.matrix)) {
    charucoMatrix.textContent = formatMatrix(result.matrix);
  }
}

function showTransformationMode() {
  transformationModeSelect.value = transformationMode;
  jsonTransformationPanel.hidden = transformationMode !== "json";
  charucoTransformationPanel.hidden = transformationMode !== "charuco";
  updateCharucoPreviewLoop();
}

function isSessionNameValid() {
  return SESSION_NAME_PATTERN.test(sessionNameInput.value);
}

function hasRepairSession() {
  return Array.from(savedSessionSelect.options).some(
    (option) => option.value === REPAIR_SESSION,
  );
}

function updateControls(busy = false) {
  busy = busy || replayStageLoading || analysisBusy;
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
  transformationModeSelect.disabled =
    busy || currentState !== "stopped" || transformBurstActive;
  const jsonDisabled =
    busy || transformationMode !== "json" || transformBurstActive;
  publishTransformationButton.disabled = jsonDisabled;
  previousTransformationButton.disabled = jsonDisabled;
  nextTransformationButton.disabled = jsonDisabled;
  charucoCaptureButton.disabled =
    busy ||
    transformationMode !== "charuco" ||
    currentState !== "recording" ||
    transformBurstActive;
  savedSessionSelect.disabled =
    busy ||
    currentState !== "stopped" ||
    savedSessionSelect.options.length < 2;
  replayPreviousButton.disabled = busy || !replayMode || replayIndex === 0;
  replayNextButton.disabled =
    busy || !replayMode || replayIndex >= replayFrames.length - 1;
  replayExitButton.disabled = busy || !replayMode;
  for (const [stage, button] of Object.entries(replayStageButtons)) {
    button.disabled =
      busy || !replayMode || !replayAvailableStages.has(stage);
    button.setAttribute("aria-pressed", String(stage === replayStage));
  }
  analyzeButton.disabled =
    busy ||
    currentState !== "stopped" ||
    !hasRepairSession();
  adjustRepairButton.disabled =
    busy || currentState !== "stopped" || repairTransform === null;
  const analysisDisabled = busy || repairMesh === null;
  analysisApplyButton.disabled = analysisDisabled;
  analysisCancelButton.disabled = analysisDisabled;
  analysisResetButton.disabled =
    analysisDisabled || repairInitialTransform === null;
  for (const button of Object.values(analysisModeButtons)) {
    button.disabled = analysisDisabled;
  }
  analysisRepairVisible.disabled =
    busy || repairMesh === null || repairEditing;
  analysisSegmentVisible.disabled = busy || repairPointCloud === null;
}

function applyStatus(status) {
  currentState = status.state;
  currentDatabasePath = status.database_path;
  stateBadge.textContent =
    currentState.charAt(0).toUpperCase() + currentState.slice(1);
  stateBadge.className = `state-badge ${currentState}`;
  databasePath.textContent = currentDatabasePath || "No recording selected";
  transformBurstActive = Boolean(status.transform_burst_active);
  if (["json", "charuco"].includes(status.transformation_mode)) {
    transformationMode = status.transformation_mode;
    showTransformationMode();
  }
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
  if (status.last_charuco) {
    showCharucoResult(status.last_charuco);
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
  hidePointTooltip();
  scene.remove(cloud);
  cloud.geometry.dispose();
  cloud.material.dispose();
}

function setAnalysisMode(mode) {
  repairTransformControls.setMode(mode);
  for (const [buttonMode, button] of Object.entries(analysisModeButtons)) {
    button.setAttribute("aria-pressed", String(buttonMode === mode));
  }
}

function matrixFromRows(rows) {
  if (
    !Array.isArray(rows) ||
    rows.length !== 4 ||
    rows.some((row) => !Array.isArray(row) || row.length !== 4)
  ) {
    throw new Error("Repair analysis returned an invalid transform");
  }
  return new THREE.Matrix4().set(...rows.flat());
}

function repairMatrixRows() {
  repairMesh.updateMatrix();
  const elements = repairMesh.matrix.elements;
  return [
    [elements[0], elements[4], elements[8], elements[12]],
    [elements[1], elements[5], elements[9], elements[13]],
    [elements[2], elements[6], elements[10], elements[14]],
    [elements[3], elements[7], elements[11], elements[15]],
  ];
}

function hideRepairEditor() {
  repairTransformControls.detach();
  repairTransformHelper.visible = false;
  controls.enabled = true;
  repairEditing = false;
  analysisEditControls.hidden = true;
  if (repairMesh !== null) {
    repairMesh.visible = analysisRepairVisible.checked;
  }
  updateControls();
}

function clearRepairAnalysis() {
  hideRepairEditor();
  if (repairPointCloud !== null) {
    disposeCloud(repairPointCloud);
    repairPointCloud = null;
  }
  if (repairMesh !== null) {
    scene.remove(repairMesh);
    repairMesh.geometry.dispose();
    repairMesh.material.dispose();
    repairMesh = null;
  }
  repairInitialTransform = null;
  repairTransform = null;
  analysisRepairVisible.checked = false;
  analysisSegmentVisible.checked = true;
  analysisDock.hidden = true;
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
  clearRepairAnalysis();
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

function visiblePointClouds() {
  const clouds =
    pointCloud !== null
      ? [pointCloud]
      : replayClouds
          .map((replayCloud) => replayCloud.cloud)
          .filter((cloud) => cloud !== null);
  if (repairPointCloud !== null) {
    clouds.push(repairPointCloud);
  }
  return clouds;
}

function hidePointTooltip() {
  pendingPointHover = null;
  pointCoordinateTooltip.hidden = true;
}

function updatePointTooltip(clientX, clientY) {
  const canvasBounds = renderer.domElement.getBoundingClientRect();
  const clouds = visiblePointClouds();
  if (
    clouds.length === 0 ||
    clientX < canvasBounds.left ||
    clientX > canvasBounds.right ||
    clientY < canvasBounds.top ||
    clientY > canvasBounds.bottom
  ) {
    hidePointTooltip();
    return;
  }

  pointPointer.set(
    ((clientX - canvasBounds.left) / canvasBounds.width) * 2 - 1,
    -((clientY - canvasBounds.top) / canvasBounds.height) * 2 + 1,
  );
  pointRaycaster.setFromCamera(pointPointer, camera);
  const intersection = pointRaycaster.intersectObjects(clouds, false)[0];
  if (intersection === undefined || !Number.isInteger(intersection.index)) {
    hidePointTooltip();
    return;
  }

  hoveredPoint.fromBufferAttribute(
    intersection.object.geometry.getAttribute("position"),
    intersection.index,
  );
  intersection.object.localToWorld(hoveredPoint);
  pointCoordinateTooltip.textContent =
    `World · X ${(hoveredPoint.x * 100).toFixed(2)} cm · ` +
    `Y ${(hoveredPoint.y * 100).toFixed(2)} cm · ` +
    `Z ${(hoveredPoint.z * 100).toFixed(2)} cm`;
  pointCoordinateTooltip.hidden = false;

  const viewerBounds = viewer.getBoundingClientRect();
  const left = Math.min(
    clientX - viewerBounds.left + POINT_TOOLTIP_OFFSET_PX,
    viewer.clientWidth - pointCoordinateTooltip.offsetWidth - 8,
  );
  const top = Math.min(
    clientY - viewerBounds.top + POINT_TOOLTIP_OFFSET_PX,
    viewer.clientHeight - pointCoordinateTooltip.offsetHeight - 8,
  );
  pointCoordinateTooltip.style.left = `${Math.max(8, left)}px`;
  pointCoordinateTooltip.style.top = `${Math.max(8, top)}px`;
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

  const frame = replayFrames[replayIndex];
  const matrix =
    replayStage === "aligned" && Array.isArray(frame?.optimized_matrix)
      ? frame.optimized_matrix
      : frame?.matrix;
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

async function loadRawPoints(alignmentError) {
  try {
    const response = await fetch("/api/points?source=raw", {
      cache: "no-store",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || "Cannot load raw points");
    }
    renderPayload(await response.arrayBuffer());
    setMessage(
      `${alignmentError} Showing a combined raw preview using captured poses; ` +
        "it is not refinement-aligned.",
      true,
    );
  } catch (error) {
    clearPointCloud();
    setMessage(
      `${alignmentError} Raw preview also failed: ${error.message}`,
      true,
    );
  }
}

async function loadPoints() {
  setMessage("Loading saved aligned point clouds…");
  const response = await fetch("/api/points", { cache: "no-store" });
  if (!response.ok) {
    const error = await response.json();
    await loadRawPoints(error.message || "Cannot load aligned points");
    return;
  }
  renderPayload(await response.arrayBuffer());
  const acceptedEdges = response.headers.get("X-Accepted-Edges");
  const rejectedEdges = response.headers.get("X-Rejected-Edges");
  const charucoFrames = Number(response.headers.get("X-Charuco-Frames"));
  const charucoMax = Number(
    response.headers.get("X-Charuco-Max-Reprojection-Px"),
  );
  const cloudFraction = Number(
    response.headers.get("X-Cloud-3mm-Fraction"),
  );
  const charucoMetrics =
    charucoFrames > 0 && Number.isFinite(charucoMax) &&
    Number.isFinite(cloudFraction)
      ? ` ChArUco max ${charucoMax.toFixed(3)} px; ` +
        `${(cloudFraction * 100).toFixed(3)}% within 3 mm.`
      : "";
  const processingWarning = response.headers.get("X-Processing-Warning");
  setMessage(
    (processingWarning ? `${processingWarning} ` : "") +
      `Showing cleaned, registered preview (${acceptedEdges} edges accepted, ` +
      `${rejectedEdges} rejected).${charucoMetrics}`,
    Boolean(processingWarning),
  );
}

function resetReplay(clearSelection = true) {
  clearRepairAnalysis();
  replayGeneration += 1;
  replayMode = false;
  replayFrames = [];
  replayIndex = 0;
  replayStage = "raw";
  replayAvailableStages = new Set(["raw"]);
  replayStageLoading = false;
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
    `Frame ${replayIndex + 1} / ${replayFrames.length} · ` +
    REPLAY_STAGE_LABELS[replayStage];
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
        `?max_points=${maxPoints}&stage=${encodeURIComponent(replayStage)}`,
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

async function loadReplaySession(
  sessionName = savedSessionSelect.value,
  preferredStage = "raw",
) {
  savedSessionSelect.value = sessionName;
  resetReplay(false);
  clearPointCloud();
  if (!sessionName) {
    return false;
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
      return false;
    }
    replayFrames = result.frames;
    replayAvailableStages = new Set(result.stages || ["raw"]);
    if (replayFrames.length === 0) {
      setMessage(`Replay '${sessionName}' has no recorded frames.`);
      return false;
    }
    replayStage = replayAvailableStages.has(preferredStage)
      ? preferredStage
      : "raw";
    replayMode = true;
    replaySessionName.textContent = sessionName;
    replayDock.hidden = false;
    if (!(await addReplayFrame(0, generation))) {
      return false;
    }
    setMessage(
      `Replay '${sessionName}' is ready. Use Next to add another frame.`,
    );
    return true;
  } catch (error) {
    if (generation === replayGeneration) {
      resetReplay(false);
      setMessage(error.message, true);
    }
    return false;
  } finally {
    updateControls();
  }
}

async function setReplayStage(stage) {
  if (
    !replayMode ||
    replayStageLoading ||
    stage === replayStage ||
    !replayAvailableStages.has(stage)
  ) {
    return;
  }

  clearRepairAnalysis();
  const lastIndex = replayIndex;
  replayGeneration += 1;
  const generation = replayGeneration;
  replayStage = stage;
  replayStageLoading = true;
  clearReplayClouds();
  replayIndex = 0;
  updateReplayDisplay();
  setMessage(`Loading ${REPLAY_STAGE_LABELS[stage].toLowerCase()} replay…`);
  try {
    for (let index = 0; index <= lastIndex; index += 1) {
      if (!(await addReplayFrame(index, generation))) {
        return;
      }
    }
    setMessage(
      `Showing cumulative frames 1–${lastIndex + 1}: ` +
        `${REPLAY_STAGE_LABELS[stage]}.`,
    );
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    replayStageLoading = false;
    updateControls();
  }
}

function setRepairPointOverlay(points) {
  if (
    !Array.isArray(points) ||
    points.length % 3 !== 0 ||
    points.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Repair analysis returned invalid points");
  }
  if (repairPointCloud !== null) {
    disposeCloud(repairPointCloud);
    repairPointCloud = null;
  }
  if (points.length === 0) {
    return;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(points), 3),
  );
  const material = new THREE.PointsMaterial({
    color: 0xff5b35,
    size: 0.007,
    sizeAttenuation: true,
    depthTest: false,
  });
  repairPointCloud = new THREE.Points(geometry, material);
  repairPointCloud.renderOrder = 2;
  repairPointCloud.visible = analysisSegmentVisible.checked;
  scene.add(repairPointCloud);
}

async function loadRepairMesh() {
  if (repairMesh !== null) {
    return;
  }
  const geometry = await new STLLoader().loadAsync("/api/repair/reference");
  const material = new THREE.MeshBasicMaterial({
    color: 0xff8a3d,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.78,
  });
  repairMesh = new THREE.Mesh(geometry, material);
  repairMesh.visible = false;
  scene.add(repairMesh);
}

function applyRepairMeshTransform(transform) {
  matrixFromRows(transform).decompose(
    repairMesh.position,
    repairMesh.quaternion,
    repairMesh.scale,
  );
  repairMesh.updateMatrixWorld(true);
}

async function placeRepairMesh(transform) {
  await loadRepairMesh();
  applyRepairMeshTransform(transform);
}

async function showRepairEditor(transform) {
  await placeRepairMesh(transform);
  repairEditing = true;
  repairMesh.visible = true;
  repairTransformControls.attach(repairMesh);
  repairTransformHelper.visible = true;
  setAnalysisMode("translate");
  analysisDock.hidden = false;
  analysisEditControls.hidden = false;
  analysisStatus.textContent =
    "Move, rotate, or uniformly scale the repair, then apply.";
  updateControls();
}

async function requestRepairAnalysis(transform) {
  const sessionName = savedSessionSelect.value;
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionName)}/repair-analysis`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transform }),
    },
  );
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.message || "Cannot analyze the repair region");
  }
  matrixFromRows(result.transform);
  if (!Number.isInteger(result.point_count)) {
    throw new Error("Repair analysis returned an invalid point count");
  }
  return result;
}

async function analyzeRepair() {
  analysisBusy = true;
  updateControls();
  setMessage("Loading demo5 and its original repair segmentation…");
  try {
    if (!replayMode || savedSessionSelect.value !== REPAIR_SESSION) {
      if (!(await loadReplaySession(REPAIR_SESSION, "aligned"))) {
        throw new Error(`Cannot load replay '${REPAIR_SESSION}'.`);
      }
    } else if (replayStage !== "aligned") {
      await setReplayStage("aligned");
    }
    if (!replayAvailableStages.has("aligned")) {
      throw new Error("demo5 has no aligned recording for repair analysis");
    }

    const result = await requestRepairAnalysis(null);
    analysisRepairVisible.checked = false;
    analysisSegmentVisible.checked = true;
    setRepairPointOverlay(result.points);
    repairInitialTransform = result.transform.map((row) => [...row]);
    repairTransform = result.transform.map((row) => [...row]);
    await placeRepairMesh(repairTransform);
    repairMesh.visible = false;
    repairEditing = false;
    analysisEditControls.hidden = true;
    analysisDock.hidden = false;
    analysisStatus.textContent =
      `${result.point_count.toLocaleString()} original repair-region points.`;
    setMessage(
      `Showing ${result.point_count.toLocaleString()} original segmented ` +
        "repair-region points.",
    );
  } catch (error) {
    clearRepairAnalysis();
    setMessage(error.message, true);
  } finally {
    analysisBusy = false;
    updateControls();
  }
}

async function adjustRepair() {
  if (repairTransform === null) {
    return;
  }
  await showRepairEditor(repairTransform);
}

async function applyRepairPlacement() {
  if (repairMesh === null) {
    return;
  }
  analysisBusy = true;
  updateControls();
  setMessage("Updating the repair segmentation from the adjusted placement…");
  try {
    const result = await requestRepairAnalysis(repairMatrixRows());
    setRepairPointOverlay(result.points);
    repairTransform = result.transform.map((row) => [...row]);
    hideRepairEditor();
    analysisStatus.textContent =
      `${result.point_count.toLocaleString()} adjusted repair-region points.`;
    setMessage(
      `Repair placement applied; ` +
        `${result.point_count.toLocaleString()} scan points selected.`,
    );
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    analysisBusy = false;
    updateControls();
  }
}

function resetRepairPlacement() {
  if (repairMesh === null || repairInitialTransform === null) {
    return;
  }
  applyRepairMeshTransform(repairInitialTransform);
}

async function cancelRepairAdjustment() {
  if (repairTransform !== null) {
    await placeRepairMesh(repairTransform);
  }
  hideRepairEditor();
  if (repairPointCloud !== null) {
    analysisStatus.textContent =
      `${repairPointCloud.geometry.attributes.position.count.toLocaleString()} ` +
      "repair-region points.";
  }
}

async function nextReplayFrame() {
  if (!replayMode || replayIndex >= replayFrames.length - 1) {
    return false;
  }
  updateControls(true);
  try {
    const nextIndex = replayIndex + 1;
    if (await addReplayFrame(nextIndex)) {
      setMessage(
        `Added frame ${nextIndex + 1} of ${replayFrames.length} to the replay.`,
      );
      return true;
    }
    return false;
  } catch (error) {
    setMessage(error.message, true);
    return false;
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

async function setTransformationMode() {
  const selectedMode = transformationModeSelect.value;
  updateControls(true);
  setMessage(`Switching to ${selectedMode} transformation mode…`);
  try {
    const response = await fetch("/api/transformation/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: selectedMode }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "Cannot change transformation mode");
    }
    applyStatus(result);
    setMessage(result.message);
  } catch (error) {
    transformationModeSelect.value = transformationMode;
    setMessage(error.message, true);
  } finally {
    updateControls();
  }
}

async function captureCharuco() {
  updateControls(true);
  setMessage("Detecting ChArUco board for the next point cloud…");
  try {
    const response = await fetch("/api/charuco/capture", {
      method: "POST",
    });
    const result = await response.json();
    showCharucoResult(result.charuco_capture || result);
    if (!response.ok) {
      throw new Error(result.message || "Cannot capture with ChArUco");
    }
    applyStatus(result);
    setMessage(
      `${result.message}: ${result.charuco_capture.corner_count} corners, ` +
        `${result.charuco_capture.valid_depth_corner_count} valid depth, ` +
        `${result.charuco_capture.reprojection_rmse_px.toFixed(3)} px RMSE.`,
    );
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

function decodeRgbPayload(buffer) {
  if (buffer.byteLength < 12) {
    throw new Error("Camera frame is incomplete");
  }
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RGB1") {
    throw new Error("Camera frame has an unsupported format");
  }
  const header = new DataView(buffer);
  const width = header.getUint32(4, true);
  const height = header.getUint32(8, true);
  const expectedBytes = 12 + width * height * 3;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error("Camera frame size does not match its header");
  }

  const rgb = new Uint8Array(buffer, 12);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (
    let sourceOffset = 0, destinationOffset = 0;
    sourceOffset < rgb.length;
    sourceOffset += 3, destinationOffset += 4
  ) {
    rgba[destinationOffset] = rgb[sourceOffset];
    rgba[destinationOffset + 1] = rgb[sourceOffset + 1];
    rgba[destinationOffset + 2] = rgb[sourceOffset + 2];
    rgba[destinationOffset + 3] = 255;
  }
  return {
    width,
    height,
    rgb,
    imageData: new ImageData(rgba, width, height),
  };
}

function decodeCameraFrame(buffer) {
  const frame = decodeRgbPayload(buffer);
  cameraWidth = frame.width;
  cameraHeight = frame.height;
  cameraRgb = frame.rgb;

  sourceCanvas.width = cameraWidth;
  sourceCanvas.height = cameraHeight;
  cameraPreview.width = cameraWidth;
  cameraPreview.height = cameraHeight;
  sourceContext.putImageData(frame.imageData, 0, 0);
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

async function runCharucoPreview(generation) {
  while (
    charucoPreviewRunning &&
    charucoPreviewGeneration === generation
  ) {
    const started = performance.now();
    try {
      const response = await fetch("/api/charuco/preview", {
        cache: "no-store",
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || "Cannot load ChArUco preview");
      }
      const frame = decodeRgbPayload(await response.arrayBuffer());
      if (
        !charucoPreviewRunning ||
        charucoPreviewGeneration !== generation
      ) {
        return;
      }
      charucoPreview.width = frame.width;
      charucoPreview.height = frame.height;
      charucoPreviewContext.putImageData(frame.imageData, 0, 0);
      charucoPreviewStatus.textContent = "Live detection · up to 5 FPS";
    } catch (error) {
      if (
        !charucoPreviewRunning ||
        charucoPreviewGeneration !== generation
      ) {
        return;
      }
      charucoPreviewStatus.textContent = error.message;
    }

    const remaining =
      CHARUCO_PREVIEW_INTERVAL_MS - (performance.now() - started);
    if (remaining > 0) {
      await new Promise((resolve) => window.setTimeout(resolve, remaining));
    }
  }
}

function updateCharucoPreviewLoop() {
  const shouldRun = transformationMode === "charuco";
  if (shouldRun === charucoPreviewRunning) {
    return;
  }
  charucoPreviewRunning = shouldRun;
  charucoPreviewGeneration += 1;
  if (shouldRun) {
    charucoPreviewStatus.textContent = "Waiting for camera preview…";
    runCharucoPreview(charucoPreviewGeneration);
  } else {
    charucoPreviewStatus.textContent = "Preview inactive in JSON mode.";
  }
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
  const commandLabel = command.charAt(0).toUpperCase() + command.slice(1);
  setMessage(
    ["pause", "stop"].includes(command)
      ? `${commandLabel} requested; aligning and saving point clouds…`
      : `${commandLabel} requested…`,
  );
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
    if (typeof result.state === "string") {
      applyStatus(result);
    }
    if (!response.ok) {
      const alignmentFailedAfterTransition =
        (command === "pause" && result.state === "paused") ||
        (command === "stop" && result.state === "stopped");
      if (alignmentFailedAfterTransition) {
        if (command === "stop") {
          await loadSavedSessions();
        }
        await loadRawPoints(
          result.message || `Cannot ${command} and align recording`,
        );
        return;
      }
      if (["pause", "stop"].includes(command)) {
        clearPointCloud();
      }
      throw new Error(result.message || `Cannot ${command} recording`);
    }

    if (command === "start") {
      resetReplay();
      clearPointCloud();
      setMessage(
        `Recording '${result.session_name}' filtered world-frame points.`,
      );
    } else if (command === "pause" || command === "stop") {
      if (command === "stop") {
        await loadSavedSessions();
      }
      await loadPoints();
    } else {
      clearPointCloud();
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

function remoteViewerReport(
  completedCommandId = null,
  commandError = null,
  commandBusy = remoteCommandRunning,
) {
  const report = {
    replay_mode: replayMode,
    replay_index: replayIndex,
    replay_total: replayFrames.length,
    can_next:
      replayMode &&
      !replayStageLoading &&
      replayIndex < replayFrames.length - 1,
    loading_visible: remoteLoadingVisible,
    busy: replayStageLoading || commandBusy,
    stage: replayStage,
  };
  if (completedCommandId !== null) {
    report.completed_command_id = completedCommandId;
    report.error = commandError;
  }
  return report;
}

async function reportRemoteViewer(
  completedCommandId = null,
  commandError = null,
  commandBusy = remoteCommandRunning,
) {
  const response = await fetch("/api/remote/viewer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      remoteViewerReport(completedCommandId, commandError, commandBusy),
    ),
  });
  if (!response.ok) {
    const result = await response.json();
    throw new Error(result.message || "Cannot report remote viewer state");
  }
}

async function executeRemoteCommand(command, demoSession) {
  switch (command) {
    case "start_replay": {
      const preferredStage = replayMode ? replayStage : "raw";
      if (!(await loadReplaySession(demoSession, preferredStage))) {
        throw new Error(`Cannot start replay '${demoSession}'.`);
      }
      break;
    }
    case "next":
      if (!(await nextReplayFrame())) {
        throw new Error("There is no next replay frame.");
      }
      break;
    case "show_loading":
      setRemoteLoadingVisible(true);
      break;
    case "stop_loading":
      setRemoteLoadingVisible(false);
      break;
    default:
      throw new Error(`Unsupported remote command '${command}'.`);
  }
}

async function pollRemoteControl() {
  if (remoteCommandRunning) {
    return;
  }

  let result;
  try {
    const response = await fetch("/api/remote/command", {
      cache: "no-store",
    });
    if (!response.ok) {
      return;
    }
    result = await response.json();
  } catch {
    return;
  }

  if (!remoteControlInitialized) {
    setRemoteLoadingVisible(Boolean(result.loading_visible));
    remoteControlInitialized = true;
  }

  if (result.command === null) {
    reportRemoteViewer().catch(() => {});
    return;
  }

  remoteCommandRunning = true;
  reportRemoteViewer().catch(() => {});
  let commandError = null;
  try {
    await executeRemoteCommand(
      result.command.command,
      result.demo_session,
    );
  } catch (error) {
    commandError = error.message;
    setMessage(commandError, true);
  }

  try {
    await reportRemoteViewer(
      result.command.id,
      commandError,
      false,
    );
  } catch {
    // Keep the main viewer usable if the remote status endpoint is unavailable.
  } finally {
    remoteCommandRunning = false;
  }
}

repairTransformControls.addEventListener("dragging-changed", (event) => {
  controls.enabled = !event.value;
});
repairTransformControls.addEventListener("objectChange", () => {
  if (
    repairMesh === null ||
    repairTransformControls.getMode() !== "scale"
  ) {
    return;
  }
  const uniformScale = Math.max(
    Math.cbrt(
      Math.abs(repairMesh.scale.x * repairMesh.scale.y * repairMesh.scale.z),
    ),
    1e-6,
  );
  repairMesh.scale.setScalar(uniformScale);
});

for (const [command, button] of Object.entries(commandButtons)) {
  button.addEventListener("click", () => sendCommand(command));
}
analyzeButton.addEventListener("click", () => {
  analyzeRepair().catch((error) => setMessage(error.message, true));
});
adjustRepairButton.addEventListener("click", () => {
  adjustRepair().catch((error) => setMessage(error.message, true));
});
analysisApplyButton.addEventListener("click", applyRepairPlacement);
analysisCancelButton.addEventListener("click", () => {
  cancelRepairAdjustment().catch((error) => setMessage(error.message, true));
});
analysisResetButton.addEventListener("click", resetRepairPlacement);
analysisRepairVisible.addEventListener("change", () => {
  if (repairMesh !== null && !repairEditing) {
    repairMesh.visible = analysisRepairVisible.checked;
  }
});
analysisSegmentVisible.addEventListener("change", () => {
  if (repairPointCloud !== null) {
    repairPointCloud.visible = analysisSegmentVisible.checked;
  }
});
for (const [mode, button] of Object.entries(analysisModeButtons)) {
  button.addEventListener("click", () => setAnalysisMode(mode));
}
scanHelpButton.addEventListener("click", () => scanHelpDialog.showModal());
closeScanHelpButton.addEventListener("click", () => scanHelpDialog.close());
dismissScanHelpButton.addEventListener("click", () => scanHelpDialog.close());
themeToggle.addEventListener("click", () => {
  const nextTheme =
    document.documentElement.dataset.theme === "light" ? "dark" : "light";
  setTheme(nextTheme, true);
});
sessionNameInput.addEventListener("input", () => updateControls());
transformationModeSelect.addEventListener("change", setTransformationMode);
publishTransformationButton.addEventListener("click", publishTransformation);
charucoCaptureButton.addEventListener("click", captureCharuco);
previousTransformationButton.addEventListener("click", () =>
  stepTransformation(-1),
);
nextTransformationButton.addEventListener("click", () =>
  stepTransformation(1),
);
savedSessionSelect.addEventListener("change", () => loadReplaySession());
replayPreviousButton.addEventListener("click", previousReplayFrame);
replayNextButton.addEventListener("click", nextReplayFrame);
replayExitButton.addEventListener("click", () => {
  exitReplay().catch((error) => setMessage(error.message, true));
});
for (const [stage, button] of Object.entries(replayStageButtons)) {
  button.addEventListener("click", () => setReplayStage(stage));
}
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
renderer.domElement.addEventListener("pointermove", (event) => {
  pendingPointHover = {
    clientX: event.clientX,
    clientY: event.clientY,
  };
});
renderer.domElement.addEventListener("pointerleave", hidePointTooltip);
renderer.domElement.addEventListener("pointerdown", hidePointTooltip);

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
  if (pendingPointHover !== null) {
    const { clientX, clientY } = pendingPointHover;
    pendingPointHover = null;
    updatePointTooltip(clientX, clientY);
  }
  controls.update();
  renderer.render(scene, camera);
  gizmoCamera.position
    .copy(camera.position)
    .sub(controls.target)
    .normalize()
    .multiplyScalar(3);
  gizmoCamera.up.copy(camera.up);
  gizmoCamera.lookAt(gizmoOrigin);
  gizmoRenderer.render(gizmoScene, gizmoCamera);
  requestAnimationFrame(animate);
}

setTheme(document.documentElement.dataset.theme);
showTransformationMode();
resetView();
animate();
loadStatus().then(() => {
  pollRemoteControl();
  window.setInterval(pollRemoteControl, REMOTE_CONTROL_INTERVAL_MS);
});
