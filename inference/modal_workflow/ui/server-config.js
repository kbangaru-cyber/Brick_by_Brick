(function () {
  const KEY = "brickagent.modal.server_url";
  const params = new URLSearchParams(window.location.search);
  const incoming = (params.get("server") || "").trim().replace(/\/+$/, "");
  const stored = (localStorage.getItem(KEY) || "").trim().replace(/\/+$/, "");
  const fallback = "https://kbangaru--brickagent-modal-serve.modal.run";

  const serverUrl = incoming || stored || fallback;
  window.BRICKAGENT_MODAL_SERVER_URL = serverUrl;

  if (incoming) {
    localStorage.setItem(KEY, incoming);
  }

  window.addEventListener("DOMContentLoaded", () => {
    const hint = document.getElementById("modal-server-hint");
    if (hint) {
      hint.textContent = `Modal: ${serverUrl}`;
    }
  });
})();
