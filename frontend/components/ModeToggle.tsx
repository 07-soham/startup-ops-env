import React from 'react';
import { Play, Hand } from 'lucide-react';

interface ModeToggleProps {
  mode: 'auto' | 'manual';
  onModeChange: (mode: 'auto' | 'manual') => void;
}

export const ModeToggle: React.FC<ModeToggleProps> = ({ mode, onModeChange }) => {
  return (
    <div className="flex bg-gray-100 rounded-lg p-1">
      <button
        onClick={() => onModeChange('auto')}
        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
          mode === 'auto'
            ? 'bg-white shadow-sm text-blue-600'
            : 'text-gray-600 hover:text-gray-800'
        }`}
      >
        <Play size={18} />
        <span className="font-medium">Auto</span>
      </button>
      <button
        onClick={() => onModeChange('manual')}
        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
          mode === 'manual'
            ? 'bg-white shadow-sm text-blue-600'
            : 'text-gray-600 hover:text-gray-800'
        }`}
      >
        <Hand size={18} />
        <span className="font-medium">Manual</span>
      </button>
    </div>
  );
};
