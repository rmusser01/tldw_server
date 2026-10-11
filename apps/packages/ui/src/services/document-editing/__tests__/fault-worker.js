// Test-only real worker faults: never bundled by a production editor.
if (self.name === "initstall")
  while (true) {
    self.e4Spin = (self.e4Spin || 0) + 1;
  }
self.onmessage = () => {
  if (self.name === "stall")
    while (true) {
      self.e4Spin = (self.e4Spin || 0) + 1;
    }
  if (self.name === "crash") throw new Error("Synthetic worker crash");
  self.postMessage(null);
};
