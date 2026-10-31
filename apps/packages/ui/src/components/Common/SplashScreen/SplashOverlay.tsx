import React, { useCallback, useEffect, useRef, useState } from "react";
import SplashCanvas from "./SplashCanvas";
import AsciiArtOverlay from "./AsciiArtOverlay";
import type { SplashCard } from "./engine/types";

interface SplashOverlayProps {
  card: SplashCard;
  message: string;
  onDismiss: () => void;
}

/**
 * Full-screen splash overlay. Renders canvas effect + ASCII art + message.
 * Dismissible by click, keypress, or timeout.
 */
const SplashOverlay: React.FC<SplashOverlayProps> = ({ card, message, onDismiss }) => {
  const [fading, setFading] = useState(false);
  const dismissStartedRef = useRef(false);
  const prefersReducedMotion =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const handleDismiss = useCallback(() => {
    if (dismissStartedRef.current) return;
    dismissStartedRef.current = true;
    setFading(true);
    window.setTimeout(onDismiss, 300); // match fade-out duration
  }, [onDismiss]);

  // Auto-dismiss after card duration (routes through fade-out)
  useEffect(() => {
    const duration = card.duration ?? 2500;
    const timer = window.setTimeout(handleDismiss, duration);
    return () => clearTimeout(timer);
  }, [card.duration, handleDismiss]);

  // Keyboard dismiss
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Don't dismiss on modifier-only presses
      if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return;
      handleDismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleDismiss]);

  return (
    <div
      onClick={handleDismiss}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        backgroundColor: "rgb(var(--color-bg) / 0.96)",
        backgroundImage:
          "radial-gradient(circle at 20% 20%, rgb(var(--color-primary) / 0.16), transparent 45%), radial-gradient(circle at 80% 80%, rgb(var(--color-accent) / 0.12), transparent 50%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        pointerEvents: fading ? "none" : "auto",
        opacity: fading ? 0 : 1,
        transition: "opacity 0.3s ease-out",
      }}
      role="dialog"
      aria-label="Splash screen"
    >
      {/* Inline keyframe styles */}
      <style>{`
        @keyframes splashFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes splashPulse {
          0%, 100% { opacity: 0.35; }
          50% { opacity: 0.7; }
        }
      `}</style>

      {!prefersReducedMotion && card.effect && (
        <SplashCanvas
          effectName={card.effect}
          effectConfig={card.effectConfig}
          active={!fading}
        />
      )}

      <AsciiArtOverlay
        artKey={card.asciiArt}
        title={card.title}
        message={message}
        reducedMotion={prefersReducedMotion}
      />
    </div>
  );
};

export default SplashOverlay;
