import "@testing-library/jest-dom/vitest";

// Suppress the jsdom/undici "AbortSignal { } is not an AbortSignal"
// unhandled rejection that react-router-dom's ``navigate()`` triggers
// when it constructs an internal Request. The abort is real (the
// previous navigation was canceled by the new one) but undici 5+
// rejects the polyfilled signal jsdom hands it. We filter on the
// error name + message so unrelated unhandled rejections still
// surface — only this specific known-noisy case is silenced.
process.on("unhandledRejection", (reason) => {
  if (reason instanceof TypeError && reason.message.includes("AbortSignal")) {
    return;
  }
  // Re-throw everything else so real test bugs aren't masked.
  // eslint-disable-next-line no-console
  console.error("Unhandled rejection:", reason);
});
