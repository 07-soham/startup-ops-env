import React from 'react';
import { SimulationResponse } from '../lib/api';
import { CheckCircle, AlertCircle, Clock, TrendingUp, Award } from 'lucide-react';

interface ResultsDisplayProps {
  results: SimulationResponse | null;
  loading: boolean;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, loading }) => {
  if (loading) {
    return (
      <div className="card flex flex-col items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
        <p className="text-gray-600">Running simulation...</p>
        <p className="text-sm text-gray-500">This may take a few seconds</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="card flex flex-col items-center justify-center py-12 text-gray-400">
        <Clock size={48} className="mb-4" />
        <p>Run a simulation to see results</p>
      </div>
    );
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case 'high':
        return 'text-green-600 bg-green-50';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50';
      default:
        return 'text-red-600 bg-red-50';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default:
        return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <span className="text-green-500">😊</span>;
      case 'negative':
        return <span className="text-red-500">😞</span>;
      default:
        return <span className="text-gray-500">😐</span>;
    }
  };

  // Group emails by thread
  const emailsByThread = results.emails.reduce((acc, email) => {
    if (!acc[email.thread_id]) {
      acc[email.thread_id] = [];
    }
    acc[email.thread_id].push(email);
    return acc;
  }, {} as Record<string, typeof results.emails>);

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Award className="text-blue-600" />
            Simulation Results
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(results.confidence)}`}>
            {results.confidence} Confidence
          </span>
        </div>

        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Overall Score</div>
            <div className="text-2xl font-bold text-blue-600">
              {(results.scores.overall_score * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Email Score</div>
            <div className="text-2xl font-bold text-green-600">
              {(results.scores.email_score * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Task Score</div>
            <div className="text-2xl font-bold text-purple-600">
              {(results.scores.task_score * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Total Reward</div>
            <div className="text-2xl font-bold text-orange-600">
              {results.total_reward.toFixed(1)}
            </div>
          </div>
        </div>

        <pre className="bg-gray-50 p-4 rounded-lg text-sm whitespace-pre-wrap">{results.summary}</pre>
      </div>

      {/* Thread Grouping */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <TrendingUp size={20} />
          Parsed Emails by Thread
        </h3>

        <div className="space-y-4">
          {Object.entries(emailsByThread).map(([threadId, emails]) => (
            <div key={threadId} className="border border-gray-200 rounded-lg p-4">
              <div className="font-medium text-gray-700 mb-2 flex items-center gap-2">
                <span className="bg-gray-100 px-2 py-1 rounded text-sm">{threadId}</span>
                <span className="text-sm text-gray-500">({emails.length} emails)</span>
              </div>

              <div className="space-y-2">
                {emails.map((email, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border ${getUrgencyColor(email.urgency)}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getSentimentIcon(email.sentiment)}
                        <span className="font-medium capitalize">{email.sentiment}</span>
                        <span className="text-sm opacity-75">({email.urgency} urgency)</span>
                      </div>
                      {email.escalation_level > 0 && (
                        <span className="flex items-center gap-1 text-red-600 text-sm">
                          <AlertCircle size={14} />
                          Escalation Level {email.escalation_level}
                        </span>
                      )}
                    </div>
                    <p className="text-sm opacity-90 line-clamp-2">{email.text}</p>
                    {email.requires_action && (
                      <div className="mt-2 flex items-center gap-1 text-xs">
                        <CheckCircle size={12} />
                        <span>Requires Action</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions Taken */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Actions Taken ({results.actions.length})</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 px-2">Step</th>
                <th className="text-left py-2 px-2">Action</th>
                <th className="text-left py-2 px-2">Reward</th>
              </tr>
            </thead>
            <tbody>
              {results.actions.slice(0, 10).map((action, idx) => (
                <tr key={idx} className="border-b hover:bg-gray-50">
                  <td className="py-2 px-2">{action.step}</td>
                  <td className="py-2 px-2">{action.action.type}</td>
                  <td className={`py-2 px-2 ${action.reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {action.reward >= 0 ? '+' : ''}{action.reward.toFixed(2)}
                  </td>
                </tr>
              ))}
              {results.actions.length > 10 && (
                <tr>
                  <td colSpan={3} className="py-2 text-center text-gray-500">
                    ...and {results.actions.length - 10} more actions
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
