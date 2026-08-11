import { useState } from 'react';
import { GameScreen } from './ui/GameScreen';
import { ReplayScreen } from './ui/ReplayScreen';
import { SetupScreen } from './ui/SetupScreen';
import type { SetupResult } from './ui/useGameWorker';

export function App() {
  const [setup, setSetup] = useState<SetupResult | null>(null);
  const [showReplays, setShowReplays] = useState(false);

  if (showReplays) {
    return (
      <div className="app">
        <ReplayScreen onExit={() => setShowReplays(false)} />
      </div>
    );
  }
  return (
    <div className="app">
      {setup === null ? (
        <SetupScreen onStart={setSetup} onReplays={() => setShowReplays(true)} />
      ) : (
        <GameScreen setup={setup} onExit={() => setSetup(null)} />
      )}
    </div>
  );
}
