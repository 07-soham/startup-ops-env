import React from 'react';
import { Scenario } from '../lib/api';
import { FolderOpen } from 'lucide-react';

interface ScenarioSelectorProps {
  scenarios: Scenario[];
  selectedScenario: string | null;
  onScenarioChange: (scenario: string) => void;
  disabled?: boolean;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  scenarios,
  selectedScenario,
  onScenarioChange,
  disabled = false,
}) => {
  return (
    <div className="space-y-2">
      <label className="label flex items-center gap-2">
        <FolderOpen size={16} />
        Select Scenario
      </label>
      <select
        value={selectedScenario || ''}
        onChange={(e) => onScenarioChange(e.target.value)}
        disabled={disabled}
        className="input"
      >
        <option value="">-- Select a scenario --</option>
        {scenarios.map((scenario) => (
          <option key={scenario.name} value={scenario.name}>
            {scenario.name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
          </option>
        ))}
      </select>
      {selectedScenario && (
        <div className="mt-2 text-sm text-gray-600">
          {scenarios.find((s) => s.name === selectedScenario)?.description}
        </div>
      )}
    </div>
  );
};
