/**
 * AudioCapture
 *
 * Cross-platform audio capture for voice assistant input.
 * Handles microphone access, audio processing, and streaming.
 */

import { EventEmitter } from 'eventemitter3';
import type { AudioConfig } from '../types';

export interface AudioCaptureEvents {
  start: () => void;
  stop: () => void;
  data: (audioData: Float32Array) => void;
  level: (level: number) => void;
  error: (error: Error) => void;
}

const DEFAULT_CONFIG: AudioConfig = {
  sampleRate: 16000,
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

export class AudioCapture extends EventEmitter<AudioCaptureEvents> {
  private config: AudioConfig;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private analyzerNode: AnalyserNode | null = null;
  private isCapturing = false;
  private levelInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: Partial<AudioConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Start capturing audio from the microphone.
   */
  async start(): Promise<void> {
    if (this.isCapturing) {
      return;
    }

    try {
      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channelCount,
          echoCancellation: this.config.echoCancellation,
          noiseSuppression: this.config.noiseSuppression,
          autoGainControl: this.config.autoGainControl,
        },
      });

      // Create audio context
      this.audioContext = new AudioContext({
        sampleRate: this.config.sampleRate,
      });

      // Create source node
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Create analyzer for level metering
      this.analyzerNode = this.audioContext.createAnalyser();
      this.analyzerNode.fftSize = 256;
      this.sourceNode.connect(this.analyzerNode);

      // Try to use AudioWorklet for low-latency capture
      try {
        await this.setupWorklet();
      } catch {
        // Fallback to ScriptProcessor (deprecated but more widely supported)
        this.setupScriptProcessor();
      }

      // Start level monitoring
      this.startLevelMonitoring();

      this.isCapturing = true;
      this.emit('start');
    } catch (error) {
      this.emit('error', error instanceof Error ? error : new Error(String(error)));
      throw error;
    }
  }

  /**
   * Stop capturing audio.
   */
  stop(): void {
    if (!this.isCapturing) {
      return;
    }

    this.stopLevelMonitoring();

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.analyzerNode) {
      this.analyzerNode.disconnect();
      this.analyzerNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    this.isCapturing = false;
    this.emit('stop');
  }

  /**
   * Check if currently capturing.
   */
  isActive(): boolean {
    return this.isCapturing;
  }

  /**
   * Get the current audio level (0.0 - 1.0).
   */
  getLevel(): number {
    if (!this.analyzerNode) {
      return 0;
    }

    const dataArray = new Uint8Array(this.analyzerNode.frequencyBinCount);
    this.analyzerNode.getByteFrequencyData(dataArray);

    // Calculate RMS
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / dataArray.length);

    // Normalize to 0-1
    return Math.min(1, rms / 128);
  }

  // Private methods

  private async setupWorklet(): Promise<void> {
    if (!this.audioContext) {
      throw new Error('Audio context not initialized');
    }

    // Define the worklet processor inline
    const workletCode = `
      class AudioCaptureProcessor extends AudioWorkletProcessor {
        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input && input.length > 0) {
            // Send mono audio data
            this.port.postMessage({
              audioData: input[0]
            });
          }
          return true;
        }
      }
      registerProcessor('audio-capture-processor', AudioCaptureProcessor);
    `;

    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    await this.audioContext.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-capture-processor');

    this.workletNode.port.onmessage = (event) => {
      const { audioData } = event.data;
      if (audioData) {
        this.emit('data', new Float32Array(audioData));
      }
    };

    this.sourceNode?.connect(this.workletNode);
  }

  private setupScriptProcessor(): void {
    if (!this.audioContext || !this.sourceNode) {
      throw new Error('Audio context not initialized');
    }

    // Use ScriptProcessorNode as fallback (deprecated but widely supported)
    const bufferSize = 4096;
    const scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptNode.onaudioprocess = (event: AudioProcessingEvent) => {
      const inputData = event.inputBuffer.getChannelData(0);
      this.emit('data', new Float32Array(inputData));
    };

    this.sourceNode.connect(scriptNode);
    scriptNode.connect(this.audioContext.destination);
  }

  private startLevelMonitoring(): void {
    // Emit level every 100ms
    this.levelInterval = setInterval(() => {
      const level = this.getLevel();
      this.emit('level', level);
    }, 100);
  }

  private stopLevelMonitoring(): void {
    if (this.levelInterval) {
      clearInterval(this.levelInterval);
      this.levelInterval = null;
    }
  }
}
