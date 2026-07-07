(function () {
  const LOCAL_SERVER_URL = "http://localhost:8080";
  const target = (window.BRICKAGENT_MODAL_SERVER_URL || "").replace(/\/+$/, "");
  const realFetch = window.fetch.bind(window);

  if (!target) {
    throw new Error("BRICKAGENT_MODAL_SERVER_URL is not configured.");
  }

  window.fetch = function (input, init) {
    if (typeof input === "string" && input.startsWith(LOCAL_SERVER_URL)) {
      input = target + input.slice(LOCAL_SERVER_URL.length);
    }
    return realFetch(input, init);
  };
})();
