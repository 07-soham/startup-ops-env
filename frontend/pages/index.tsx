import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { ModeToggle } from '../components/ModeToggle';
import { ScenarioSelector } from '../components/ScenarioSelector';
import { EmailInput } from '../components/EmailInput';
import { TaskInput } from '../components/TaskInput';
import { ResultsDisplay } from '../components/ResultsDisplay';
import {
  getScenarios,
  runSimulation,
  Scenario,
  EmailInput as EmailInputType,
  TaskInput as TaskInputType,
  SimulationResponse,
} from '../lib/api';
import { Play, RotateCcw, Rocket, Github } from 'lucide-react';

export default function Home() {
  const [mode, setMode] = useState<'auto' | 'manual'>('auto');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [emails, setEmails] = useState<EmailInputType[]>([]);
  const [tasks, setTasks] = useState<TaskInputType[]>([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SimulationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadScenarios();
  }, []);

  const loadScenarios = async () => {
    try {
      const data = await getScenarios();
      setScenarios(data);
    } catch (err) {
      console.error('Failed to load scenarios:', err);
    }
  };

  const handleRunSimulation = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const request = {
        mode,
        ...(mode === 'auto'
          ? { scenario: selectedScenario }
          : { emails, tasks }),
        seed: 42,
        max_steps: 50,
      };

      const response = await runSimulation(request);
      setResults(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to run simulation');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
    setEmails([]);
    setTasks([]);
    setSelectedScenario(null);
  };

  const canRun =
    mode === 'auto'
      ? selectedScenario !== null
      : emails.length > 0 || tasks.length > 0;

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>StartupOps AI Simulator</title>
        <meta name="description" content="AI-powered startup operations simulation" />
      </Head>

      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Rocket className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">StartupOps AI Simulator</h1>
                <p className="text-sm text-gray-600">AI vs Human Decision Making</p>
              </div>
            </div>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
            >
              <Github size={20} />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Input */}
          <div className="space-y-6">
            <div className="card">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold">Configuration</h2>
                <ModeToggle mode={mode} onModeChange={setMode} />
              </div>

              {mode === 'auto' ? (
                <div className="space-y-4">
                  <ScenarioSelector
                    scenarios={scenarios}
                    selectedScenario={selectedScenario}
                    onScenarioChange={setSelectedScenario}
                  />

                  {selectedScenario && (
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">Scenario Details</h4>
                      {scenarios
                        .find((s) => s.name === selectedScenario)
                        ?.email_count > 0 && (
                        <div className="text-sm text-blue-700">
                          📧 {scenarios.find((s) => s.name === selectedScenario)?.email_count} emails
                        </div>
                      )}
                      {scenarios
                        .find((s) => s.name === selectedScenario)
                        ?.task_count > 0 && (
                        <div className="text-sm text-blue-700">
                          📋 {scenarios.find((s) => s.name === selectedScenario)?.task_count} tasks
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-6">
                  <EmailInput emails={emails} onEmailsChange={setEmails} />
                  <TaskInput tasks={tasks} onTasksChange={setTasks} />
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                  {error}
                </div>
              )}

              <div className="flex gap-3 pt-4 border-t">
                <button
                  onClick={handleRunSimulation}
                  disabled={!canRun || loading}
                  className="btn-primary flex-1 flex items-center justify-center gap-2"
                >
                  <Play size={18} />
                  {loading ? 'Running...' : 'Run Simulation'}
                </button>

                <button
                  onClick={handleReset}
                  disabled={loading}
                  className="btn-secondary flex items-center gap-2"
                >
                  <RotateCcw size={18} />
                  Reset
                </button>
              </div>
            </div>

            {/* Instructions */}
            <div className="card bg-gray-50">
              <h3 className="font-semibold mb-3">How it works</h3>
              <ol className="space-y-2 text-sm text-gray-600 list-decimal list-inside">
                <li>Choose Auto (scenario-based) or Manual mode</li>
                <li>Auto: Select a predefined startup scenario</li>
                <li>Manual: Add your own emails and tasks</li>
                <li>LLM parses emails for urgency, sentiment, action</li>
                <li>RL agent makes decisions and receives rewards</li>
                <li>View results: scores, threads, actions taken</li>
              </ol>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div>
            <ResultsDisplay results={results} loading={loading} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
          StartupOps AI Simulator - Built with Next.js, FastAPI, and LLM parsing
        </div>
      </footer>
    </div>
  );
}
