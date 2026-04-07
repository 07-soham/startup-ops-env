import React from 'react';
import { TaskInput as TaskInputType } from '../lib/api';
import { Plus, Trash2, CheckSquare } from 'lucide-react';

interface TaskInputProps {
  tasks: TaskInputType[];
  onTasksChange: (tasks: TaskInputType[]) => void;
}

export const TaskInput: React.FC<TaskInputProps> = ({ tasks, onTasksChange }) => {
  const addTask = () => {
    onTasksChange([
      ...tasks,
      {
        name: `Task ${tasks.length + 1}`,
        hours_required: 8,
        deadline: 5,
        priority: 'medium',
        effort: 3,
        impact: 1.0,
      },
    ]);
  };

  const removeTask = (index: number) => {
    const newTasks = tasks.filter((_, i) => i !== index);
    onTasksChange(newTasks);
  };

  const updateTask = (index: number, field: keyof TaskInputType, value: string | number) => {
    const newTasks = [...tasks];
    newTasks[index] = { ...newTasks[index], [field]: value };
    onTasksChange(newTasks);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="label flex items-center gap-2">
          <CheckSquare size={16} />
          Tasks
        </label>
        <button
          onClick={addTask}
          className="btn-secondary flex items-center gap-1 text-sm"
        >
          <Plus size={16} />
          Add Task
        </button>
      </div>

      {tasks.length === 0 && (
        <div className="text-gray-500 text-sm italic">No tasks added yet.</div>
      )}

      {tasks.map((task, index) => (
        <div key={index} className="card border border-gray-200 p-4">
          <div className="flex justify-between items-start mb-3">
            <span className="text-sm font-medium text-gray-500">Task #{index + 1}</span>
            <button
              onClick={() => removeTask(index)}
              className="text-red-500 hover:text-red-700"
            >
              <Trash2 size={16} />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-2">
              <label className="text-xs text-gray-500">Name</label>
              <input
                type="text"
                value={task.name}
                onChange={(e) => updateTask(index, 'name', e.target.value)}
                className="input text-sm"
              />
            </div>

            <div>
              <label className="text-xs text-gray-500">Hours</label>
              <input
                type="number"
                value={task.hours_required}
                onChange={(e) => updateTask(index, 'hours_required', parseFloat(e.target.value))}
                className="input text-sm"
                min={1}
                max={40}
                step={0.5}
              />
            </div>

            <div>
              <label className="text-xs text-gray-500">Deadline</label>
              <input
                type="number"
                value={task.deadline}
                onChange={(e) => updateTask(index, 'deadline', parseInt(e.target.value))}
                className="input text-sm"
                min={1}
                max={20}
              />
            </div>

            <div>
              <label className="text-xs text-gray-500">Priority</label>
              <select
                value={task.priority}
                onChange={(e) => updateTask(index, 'priority', e.target.value)}
                className="input text-sm"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-500">Impact</label>
              <input
                type="number"
                value={task.impact}
                onChange={(e) => updateTask(index, 'impact', parseFloat(e.target.value))}
                className="input text-sm"
                min={0.5}
                max={3}
                step={0.1}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
