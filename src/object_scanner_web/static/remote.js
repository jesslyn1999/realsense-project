const statusText = document.querySelector("#remote-status");
const buttons = {
  start_replay: document.querySelector("#remote-start-replay"),
  next: document.querySelector("#remote-next"),
  show_loading: document.querySelector("#remote-show-loading"),
  stop_loading: document.querySelector("#remote-stop-loading"),
};
const STAGE_LABELS = {
  raw: "Raw",
  filtered: "Outliers removed",
  aligned: "ICP aligned",
};
const STATUS_INTERVAL_MS = 300;

let sending = false;
let readingStatus = false;

function showStatus(status) {
  const unavailable =
    sending ||
    !status.connected ||
    status.pending_command !== null ||
    status.busy;

  buttons.start_replay.disabled = unavailable;
  buttons.next.disabled = unavailable || !status.can_next;
  buttons.show_loading.disabled = unavailable || status.loading_visible;
  buttons.stop_loading.disabled = unavailable || !status.loading_visible;

  if (!status.connected) {
    statusText.textContent = "Main viewer is not connected.";
    statusText.classList.add("error");
  } else if (status.pending_command !== null || status.busy) {
    statusText.textContent = "Waiting for the main viewer…";
    statusText.classList.remove("error");
  } else if (status.last_error) {
    statusText.textContent = status.last_error;
    statusText.classList.add("error");
  } else if (status.replay_mode) {
    statusText.textContent =
      `Replay ${status.demo_session}: frame ${status.replay_index + 1} / ` +
      `${status.replay_total} · ${STAGE_LABELS[status.stage]}`;
    statusText.classList.remove("error");
  } else {
    statusText.textContent = status.loading_visible
      ? "Main display is showing Loading."
      : "Main viewer is ready.";
    statusText.classList.remove("error");
  }
}

async function refreshStatus() {
  if (readingStatus) {
    return;
  }
  readingStatus = true;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Cannot read remote status");
    }
    showStatus(await response.json());
  } catch (error) {
    statusText.textContent = error.message;
    statusText.classList.add("error");
    for (const button of Object.values(buttons)) {
      button.disabled = true;
    }
  } finally {
    readingStatus = false;
  }
}

async function sendCommand(command) {
  if (sending) {
    return;
  }
  sending = true;
  for (const button of Object.values(buttons)) {
    button.disabled = true;
  }
  statusText.textContent = "Sending command…";
  statusText.classList.remove("error");
  try {
    const response = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "Command failed");
    }
  } catch (error) {
    statusText.textContent = error.message;
    statusText.classList.add("error");
  } finally {
    sending = false;
    await refreshStatus();
  }
}

for (const [command, button] of Object.entries(buttons)) {
  button.addEventListener("click", () => sendCommand(command));
}

refreshStatus();
window.setInterval(refreshStatus, STATUS_INTERVAL_MS);
